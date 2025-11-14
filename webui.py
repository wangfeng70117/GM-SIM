import os
import subprocess
import numpy as np

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')

import sys
import random
import time
import math

import torch
import viser
import viser.transforms as tf
from arguments import PipelineParams, ModelParams, OptimizationParams, get_combined_args
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from utils.camera_utils import qvec2rotmat
from scene.cameras import Simple_Camera
from gaussian_renderer import render
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import point_cloud_utils
from PIL import Image
from utils.graphics_utils import *
from utils.transformation_utils import *
from mpm_engine.export_utils import *
import traceback
import warp as wp

wp.init()
wp.config.print_launches = True


class WebUI:
    def __init__(self, dataset, opt, pipe, cfg) -> None:
        self.ckpt_path = cfg.ckpt_name
        self.source_path = cfg.source_path
        self.pipe = pipe
        self.opt = opt

        self.encoder_dims = [256, 128, 64, 32, 16, 3]
        self.decoder_dims = [16, 32, 64, 128, 256, 512]

        self.gaussian = GaussianModel(sh_degree=3)
        self.scene = Scene(dataset, self.gaussian, init_gaussian=False, shuffle=False)
        self.cameras = self.scene.getTrainCameras()

        self.port = 8088
        self.server = viser.ViserServer(port=self.port)

        self.render_camera = None

        # self.pre_cov = False
        # self.rotated_cov = None
        checkpoint = os.path.join(args.model_path, args.ckpt_name)
        (self.model_params, self.load_iteration) = torch.load(checkpoint)
        self.gaussian.training_setup(opt)
        self.gaussian.restore(self.model_params, args, mode='test')

        self.background_tensor = torch.tensor(
            [1, 1, 1], dtype=torch.float32, device="cuda"
        )
        # CLIP分割模型初始化
        self.init_clip_segmenter(args.source_path)

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            for i in range(30):
                self.make_one_camera_pos_frame(i)

        self.pc_xyz = self.gaussian.get_xyz.detach().cpu().numpy()
        self.pc_color = self.features2color(self.gaussian.get_language_feature)
        self.pc_handle = self.server.add_point_cloud(
            "Point cloud",
            points=self.pc_xyz,
            colors=self.pc_color,
            point_size=0.003,
            point_shape="circle",
            visible=True
        )

        self.reset_scene = self.server.add_gui_button("Reset Scene", visible=True)

        self.show_option = self.server.add_gui_dropdown(
            "ShowOption",
            ("Image", "Feature", "Depth", "Point Cloud")
        )

        self.full_mask = None

        with self.server.add_gui_folder("Segment Params"):
            self.text_prompt = self.server.add_gui_text(
                "Prompt",
                "Green Apple",
                visible=True)

            # 计算每个高斯核和输入文本的相似度
            self.calculate_similarity = self.server.add_gui_button(
                "Calculate Similarity",
                visible=True
            )

            self.seg_threshold = self.server.add_gui_text(
                "SegThreshold",
                "0.7",
                visible=True
            )

            self.eps = self.server.add_gui_text(
                "eps",
                "5",
                visible=True
            )
            self.cluster_num = self.server.add_gui_text(
                "cluster_num",
                "3000",
                visible=True
            )

            self.grid_n = self.server.add_gui_text(
                "grid_n",
                "5",
                visible=True
            )
            self.delete_choice = self.server.add_gui_checkbox(
                "Delete Choice Object",
                initial_value=False,
                visible=True
            )

            self.segment = self.server.add_gui_button(
                "Segment",
                visible=True
            )

            self.save_mask = self.server.add_gui_button(
                "Save Mask",
                visible=True
            )

        with self.server.add_gui_folder("Ground Estimate"):
            self.init_ground_estimator()
            self.calculate_ground = self.server.add_gui_button(
                "CalculateGround",
                visible=True
            )

            self.inverse_rotate = self.server.add_gui_button(
                "Inverse Rotated",
                visible=True
            )

        self.origin_frame = self.server.add_frame(
            name="origin axis",
            position=(0, 0, 0),
            axes_length=50,
            visible=True
        )

        self.server.add_grid(
            "origin",
            width=50,
            height=50,
            width_segments=50,
            height_segments=50
        )

        with self.server.add_gui_folder("Move Segmented Gaussians"):
            self.x_slider = self.server.add_gui_slider(
                "Vertical Slider",
                min=-10,
                max=10,
                step=0.1,
                initial_value=0
            )

            self.y_slider = self.server.add_gui_slider(
                "Vertical Slider",
                min=-10,
                max=10,
                step=0.1,
                initial_value=0
            )
            self.z_slider = self.server.add_gui_slider(
                "Vertical Slider",
                min=-10,
                max=10,
                step=0.1,
                initial_value=0
            )

            self.move = self.server.add_gui_button(
                "Move",
                visible=True
            )

            self.result_name = self.server.add_gui_text(
                "Result Name",
                "",
                visible=True
            )
            self.save_result = self.server.add_gui_button(
                "Save Moved Gaussians",
                visible=True
            )

        self.save_renders = self.server.add_gui_button("Save Renders")

        @self.reset_scene.on_click
        def _(event: viser.GuiEvent):
            self.gaussian.restore(self.model_params, args, mode='test')
            self.init_clip_segmenter(self.source_path)
            self.reload_pc()
            print("Reset Scene Done.")

        @self.calculate_similarity.on_click
        def _(_):
            self.calculate_similarities()
            bins = np.linspace(np.min(self.max_similarities), np.max(self.max_similarities), 21)  # 生成20个区间
            # 计算每个区间内的粒子数量
            counts, bin_edges = np.histogram(self.max_similarities, bins=bins)
            # 输出每个区间和对应的粒子数量
            for i in range(len(counts)):
                print(f"相似度范围: [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}) -> 粒子数量: {counts[i]}")

            norm = Normalize(vmin=np.min(self.max_similarities), vmax=np.max(self.max_similarities))
            cmap = cm.plasma  # 使用合适的 colormap
            colors = cmap(norm(self.max_similarities))
            rgb_colors = colors[:, :3]
            # assert self.pc_handle.colors.shape[0] == rgb_colors.shape[0]
            self.pc_color = rgb_colors
            self.reload_pc()
            print("Done")

        @self.segment.on_click
        def _(_):
            full_mask = self.clip_segmenter.get_segment_mask(self.gaussian.get_xyz.clone(),
                                                             threshold=float(self.seg_threshold.value),
                                                             ratio=float(self.eps.value),
                                                             min_cluster_size=int(self.cluster_num.value),
                                                             grid_n=int(self.grid_n.value)
                                                             )
            self.full_mask = full_mask.cpu().numpy()

            if self.delete_choice.value:
                self.gaussian.delete_points(~self.full_mask)
            else:
                self.gaussian.delete_points(self.full_mask)

            self.reload_pc()

        @self.save_mask.on_click
        def _(_):
            torch.save(self.full_mask, os.path.join(args.model_path, f"{self.text_prompt.value}.pt"))
            print(f"saved in {args.model_path}/{self.text_prompt.value}.pt")

        @self.calculate_ground.on_click
        def _(_):
            print("calculate ground")
            pcd = self.gaussian.get_xyz.detach().clone()
            all_points = pcd.cpu().numpy()
            rotation_matrix, distance, _ = self.ground_estimator.estimate(all_points, 0.05)

            device = "cuda:0"
            rotation_matrix = torch.tensor(rotation_matrix).to(device).to(dtype=pcd.dtype)
            self.rotation_matrix = rotation_matrix
            distance = torch.tensor(distance).to(device)
            self.rotation_matrix = rotation_matrix
            self.distance = distance
            # 进行矩阵乘法和加法
            new_pcd = self.ground_estimator.rotate_point_cloud(pcd, rotation_matrix, distance)
            self.pc_xyz = new_pcd.cpu().numpy()
            self.reload_pc()
            self.gaussian.reset_position(new_pcd)

            for frame in self.frames:
                position = frame.position
                position_tensor = torch.tensor(position, dtype=torch.float32).to(device)
                new_position = torch.matmul(position_tensor, rotation_matrix.T)
                new_position = new_position + distance

                wxyz = frame.wxyz
                origin_rotation_matrix = quaternion_to_rotation_matrix(wxyz)
                # 原相机朝向
                origin_dir = camera_orientation_from_rotation_matrix(origin_rotation_matrix)
                new_dir = get_new_camera_orientation(rotation_matrix.cpu().numpy(), origin_dir)
                new_rotation_matrix = create_camera_coordinate_system(new_dir)
                frame.position = new_position.cpu().numpy()
                frame.wxyz = rotation_matrix_to_quaternion(new_rotation_matrix)

            print("done")

        @self.inverse_rotate.on_click
        def _(_):
            new_pos = self.ground_estimator.inverse_rotate_point_cloud(
                torch.tensor(self.pc_xyz).cpu(),
                self.rotation_matrix.cpu(),
                self.distance.cpu()
            )
            self.pc_xyz = new_pos.cpu().numpy()
            self.gaussian.reset_position(torch.tensor(self.pc_xyz, dtype=torch.float32, device="cuda:0"))
            self.reload_pc()

            with torch.no_grad():
                self.frames = []
                random.seed(0)
                for i in range(30):
                    self.make_one_camera_pos_frame(i)
            print("Inverse rotate done.")


        @self.move.on_click
        def _(event: viser.GuiEvent):
            masked_points = self.pc_xyz[self.full_mask]
            masked_points[:, 0] += self.x_slider.value
            masked_points[:, 1] += self.y_slider.value
            masked_points[:, 2] += self.z_slider.value
            self.pc_xyz[self.full_mask] = masked_points
            self.reload_pc()
            self.gaussian.reset_position(torch.tensor(self.pc_xyz, dtype=torch.float32, device="cuda:0"))
            print("Move Done.")


        @self.save_renders.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            render_image = client.camera.get_render(height=1440, width=2560)
            image = Image.fromarray(render_image)
            image.save('scene_render.png')
            print(f'图像已保存到{"scene_render.png"}')

        @self.save_result.on_click
        def _(_):
            ckpt_name = self.result_name.value
            if ckpt_name == '':
                print("分割结果名称不能为空！")
                return
            else:
                self.gaussian.reset_position(torch.tensor(self.pc_xyz, dtype=torch.float32, device="cuda:0"))
                # self.gaussian.reset_position(self.pc_handle.points)
                name = args.model_path + "/chkpnt_" + ckpt_name + ".pth"
                torch.save((self.gaussian.capture(opt.include_feature), self.load_iteration), name)
                print(f"保存分割后的模型至{name}")

    @torch.no_grad()
    def reload_pc(self):
        print(f'正在重置点云...')
        # self.pc_xyz = self.gaussian.get_xyz.cpu().numpy()
        #
        # self.pc_color = self.features2color(self.gaussian.get_language_feature)
        self.pc_handle.points = self.pc_xyz
        self.pc_handle.colors = self.pc_color
        self.pc_handle = self.server.add_point_cloud(
            "Point cloud",
            points=self.pc_xyz,
            colors=self.pc_color,
            point_size=0.003,
            point_shape="circle",
            visible=True
        )
        print(f'点云重置完成，共{len(self.pc_xyz)}个点')

    def init_clip_segmenter(self, source_path):
        self.clip_segmenter = point_cloud_utils.ClipSegmenter(source_path)

    def init_ground_estimator(self):
        self.ground_estimator = point_cloud_utils.GroundEstimator(0.001)

    def features2color(self, features):
        return (features.cpu().detach().numpy() * 255).astype(np.uint8)

    def calculate_similarities(self):
        self.clip_segmenter.load_features512(self.gaussian.get_language_feature)
        self.max_indices, self.max_similarities = self.clip_segmenter.compute_similarity_normalized(
            self.text_prompt.value)

    def render(self, cam):
        render_pkg = simulation_render(cam, self.gaussian, self.background_tensor, None)
        image, language_feature, render_depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["language_feature_image"],
            render_pkg['render_depth'],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        return image, language_feature, render_depth

    @property
    def get_camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        client = list(self.server.get_clients().values())[0]
        camera = client.camera
        R = tf.SO3(camera.wxyz).as_matrix()
        T = -R.T @ camera.position
        focal_y = self.cameras[0].FoVy
        focal_x = 2 * math.atan(math.tan(focal_y / 2))
        width = self.cameras[0].image_width
        height = self.cameras[0].image_height
        return Simple_Camera(R, T, focal_x, focal_y, height, width, "", 0)

    @torch.no_grad()
    def update_viewer(self):
        if self.get_camera is None:
            image, features, depth = self.render(self.cameras[0])
        else:
            image, features, depth = self.render(self.get_camera)

        if self.show_option.value == "Image":
            if self.pc_handle is not None:
                self.pc_handle.visible = False
            out = image.clamp(0, 1)
            out = (out * 255).to(torch.uint8).cpu().to(torch.uint8)
            self.server.set_background_image(out.cpu().moveaxis(0, -1).numpy().astype(np.uint8),
                                             format="jpeg")
        elif self.show_option.value == "Feature":
            if self.pc_handle is not None:
                self.pc_handle.visible = False
            min_val = features.min()
            max_val = features.max()
            normalized_features = (features - min_val) / (max_val - min_val)
            out = (normalized_features * 255).to(torch.uint8).cpu().to(torch.uint8)
            self.server.set_background_image(out.cpu().moveaxis(0, -1).numpy().astype(np.uint8),
                                             format="jpeg")
        elif self.show_option.value == "Depth":
            if self.pc_handle is not None:
                self.pc_handle.visible = False
            min_depth = depth.min()
            max_depth = depth.max()
            normalized_depth = (depth - min_depth) / (max_depth - min_depth)
            # 将深度从单通道扩展为三通道（将深度值复制到三个通道）
            depth_rgb = normalized_depth.repeat(3, 1, 1)  # 变为 (3, h, w)
            # 将归一化后的深度值转换为 [0, 255] 范围，并转换为 uint8 类型
            out = (depth_rgb * 255).to(torch.uint8).cpu()
            # 将维度从 (3, h, w) 转换为 (h, w, 3) 以适应图像展示
            out = out.moveaxis(0, -1)
            self.server.set_background_image(out.numpy().astype(np.uint8), format="jpeg")
        elif self.show_option.value == "Point Cloud":
            image, features, depth = self.render(self.get_camera)
            self.server.set_background_image(
                np.ones((image.shape[1], image.shape[2], 3), dtype=np.uint8) * 255, format="jpeg")
            self.pc_handle.visible = True

    @torch.no_grad()
    def make_one_camera_pos_frame(self, idx):
        camera = self.cameras[idx]
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(camera.qvec), camera.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()
        frame = self.server.add_frame(
            f'/colmap/frame_{idx}',
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=True
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            print("frame.on_click")
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):
            def begin_trans(client):
                assert client is not None
                # 当前的旋转，缩放坐标
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                # 目标frame的旋转，缩放
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target
                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            # 将视角放到点击的frame那里
            self.begin_call = begin_trans

    def render_loop(self):
        try:
            while True:
                self.update_viewer()
                time.sleep(1e-3)
        except Exception as e:
            print("Exception occurred:", e)
            traceback.print_exc()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_name', type=str, required=True)
    op = OptimizationParams(parser)
    model = ModelParams(parser)
    pip = PipelineParams(parser)
    args = get_combined_args(parser)
    webui = WebUI(model.extract(args), op.extract(args), pip.extract(args), args)
    webui.render_loop()
