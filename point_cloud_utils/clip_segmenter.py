import os
import numpy as np

import torch
from autoencoder.model import Autoencoder
import open_clip
from open_clip import tokenizer
import open3d as o3d
from point_cloud_utils.filling_util import get_bbox_point_mask
from cuml.cluster import DBSCAN as cuDBSCAN


class ClipSegmenter:
    def __init__(self, source_path, device="cuda:0"):
        self.source_path = source_path
        self.device = device
        self.encoder_dims = [256, 128, 64, 32, 16, 3]
        self.decoder_dims = [16, 32, 64, 128, 256, 512]
        self.clip_features512 = None
        print('开始加载CLIP模型...')
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
        )
        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.max_indices = None
        self.max_similarities = None
        print('CLIP 模型加载完成...')


    def load_features512(self, features3, batch_size=500000):
        model = Autoencoder(self.encoder_dims, self.decoder_dims).to("cuda:0")
        ckpt_path = os.path.join(self.source_path, 'model_ckpt', 'best_ckpt.pth')
        decoder_checkpoint = torch.load(ckpt_path)
        model.load_state_dict(decoder_checkpoint)
        model.eval()

        decoded_features = []
        with torch.no_grad():
            for i in range(0, features3.shape[0], batch_size):
                batch = features3[i:i + batch_size].to("cuda:0")
                decoded = model.decode(batch)
                decoded_features.append(decoded.cpu())  # 移到CPU防止占满GPU
                del batch, decoded
                torch.cuda.empty_cache()

        self.clip_features512 = torch.cat(decoded_features, dim=0)
        del model
        torch.cuda.empty_cache()
        print("512位CLIP特征生成完成")

    @torch.no_grad()
    # 计算所有高斯核的相似度
    def compute_similarity_normalized(self,
                                      positive_prompt,
                                      negative_prompt=["object", "things", "stuff", "texture"]
                                      ):
        assert positive_prompt is not None, "positive prompt must provide."
        assert self.clip_features512 is not None, "Please load 512 features first"
        prompt = [positive_prompt] + negative_prompt
        print(f'prompt is {prompt}, please check the positive prompt must be the first...')
        text_tokens = tokenizer.tokenize(prompt).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ self.clip_features512.cpu().numpy().T
        similarity = similarity.transpose()

        # 找到每个粒子最相似的特征索引
        self.max_indices = np.argmax(similarity, axis=1)
        # 找到每个粒子的最大相似度
        self.max_similarities = similarity[np.arange(similarity.shape[0]), self.max_indices]
        self.max_similarities -= self.max_similarities.min()
        self.max_similarities /= self.max_similarities.max()
        # del self.clip_model
        del self.clip_features512
        torch.cuda.empty_cache()  # 清理缓存
        return self.max_indices, self.max_similarities

    def get_cluster_labels(self, points, ratio, min_cluster_size):
        cp = points.copy()

        # 计算最小边界和最大边界
        min_bound = np.min(points, axis=0)  # 沿着轴0（即所有点）的每一列计算最小值
        max_bound = np.max(points, axis=0)  # 沿着轴0（即所有点）的每一列计算最大值
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cp)
        #
        eps = np.min(max_bound - min_bound) / ratio
        db = cuDBSCAN(eps=eps, min_samples=min_cluster_size)
        labels_gpu = db.fit_predict(points)
        # min_cluster_size 固定为 1000
        # labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_cluster_size, print_progress=True))  # eps 是点之间的最大距离
        print(f'labels is {labels_gpu}')
        return np.array(labels_gpu)

    # threshold: 相似度大于此值的认为是同一类别
    def get_segment_mask(self, all_points, threshold=0.75, ratio=5, min_cluster_size=3000, grid_n=10):
        if self.max_similarities is None or self.max_indices is None:
            raise ValueError("Similarities and indices have not been computed yet.")
        target_mask = (self.max_indices == 0) & (self.max_similarities > threshold)
        similar_mask = torch.from_numpy(target_mask).squeeze(0)
        seg_point_cloud = all_points[similar_mask]

        # 获取目标部分的聚类标签
        cluster_labels = self.get_cluster_labels(seg_point_cloud.detach().cpu().numpy(), ratio, min_cluster_size)

        cluster_label_mask = torch.from_numpy(cluster_labels == 0).squeeze(0)
        # cluster_mask = torch.zeros(all_points.shape[0], dtype=torch.bool)
        # cluster_mask[similar_mask] = cluster_label_mask  # 将 mask_tensor 的有效部分填充到 full_mask 中
        cluster_points = seg_point_cloud[cluster_label_mask]
        # 找到 seg_part 的最小值和最大值
        min_values = cluster_points.min(dim=0).values  # 最小坐标 (3,)
        max_values = cluster_points.max(dim=0).values  # 最大坐标 (3,)

        # 对于每个点，检查它是否在包围盒内
        # 分割出的点是否在包围盒内的mask
        mask_inside_bbox = (
                (all_points[:, 0] >= min_values[0]) & (all_points[:, 0] <= max_values[0]) &
                (all_points[:, 1] >= min_values[1]) & (all_points[:, 1] <= max_values[1]) &
                (all_points[:, 2] >= min_values[2]) & (all_points[:, 2] <= max_values[2])
        )
        # 包围盒内部的所有点
        points_inside_bbox = all_points[mask_inside_bbox]

        grid_inside_mask = get_bbox_point_mask(
            cluster_points,
            points_inside_bbox,
            max_values[0] - min_values[0],
            max_values[1] - min_values[1],
            max_values[2] - min_values[2],
            grid_n=grid_n
        )

        full_mask = torch.zeros(all_points.shape[0], dtype=torch.bool).to("cuda:0")
        # 4. 将包围盒内的点标记为 True
        full_mask[mask_inside_bbox] = grid_inside_mask

        return full_mask
