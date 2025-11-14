import torch

from utils.graphics_utils import *
import open3d as o3d


class GroundEstimator:
    def __init__(self, distance_threshold=0.01):
        self.distance_threshold = distance_threshold

    def estimate(self, points, distance_threshold=0.01):
        print("生成OPEN3D点云...")
        cp = points.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cp)
        # images:
        #
        # bench:
        # plane_model = [-0.07386721, -0.64615339,  0.75962453, -9.8945006 ]
        # 计算inlier数量，Inlier判定方式：计算所有点与随机点计算的平面之间的距离d，如果d<D(距离阈值)，判断为内点。
        # distance_threshold：inlier的最大距离阈值
        # ransac_n：随机采样的平面点数
        # num_iterations：RANSAC最小迭代次数
        # 返回值：plane_model：平面模型，即个平面方程系数（a,b,c,d），作为一个平面，对于平面上每个点(x,y,z)，我们有ax+by+cz+d=0。
        # inliers ：内点索引
        print("正在估计地面...")
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=4,
                                                 num_iterations=5000)
        # plane_model = [0.08137693, 0.72764097, 0.68111409, -2.4632256]
        # figurines
        # plane_model = [-0.01839567,  0.98137685,  0.19120952, -2.18772933]
        # tennis
        # plane_model = [0.08621219, 0.84106547, 0.53401905, -2.32042855]
        # bench
        # plane_model = [0.01494593, 0.751696, 0.65934039, -7.45432073]
        # kitchen
        # plane_model = [0.07627377,  0.73139238,  0.67767802, -2.46518187]
        # foot ball
        # plane_model = [0.00384885, 0.92897756, 0.37011602, -3.74186395]
        # # foot ball wo loss
        # plane_model = [3.62686866e-04, 9.27901066e-01, 3.72826341e-01, - 3.73786500e+00]
        # standing statue
        # plane_model = [-0.02075966,  0.99557452, -0.09165375, -1.30955032]

        print(f'预测结束，平面模型为：{plane_model}')
        # plane_model = [0.08247857, 0.72924579, 0.67926274, -2.46336072]
        # 计算空间原点到平面之间的距离
        plane_normal = tuple(plane_model[:3])
        plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)

        z_axis = np.array([0, 0, -1.0])

        # a, b, c, d = plane_model
        # distances = np.abs(np.dot(points, plane_normal) + d) / np.linalg.norm(plane_normal)
        # # 筛选出距离平面小于 max_distance 的点
        # close_points_indices = np.where(distances > 0.5)[0]

        return rotation_matrix_from_vectors(plane_normal, z_axis), point_to_plane_distance((0, 0, 0),
                                                                                           plane_model), plane_model

    def rotate_point_cloud(self, pcd, rotation_matrix, distance):
        new_pcd = torch.matmul(pcd, rotation_matrix.T)  # 点云与旋转矩阵相乘
        new_pcd += distance  # 加上位移
        return new_pcd

    def inverse_rotate_point_cloud(self, pcd, rotation_matrix, distance):
        new_pcd = pcd - distance
        new_pcd = torch.matmul(new_pcd, rotation_matrix)
        return new_pcd

    # 相机位置， 原相机旋转矩阵， 旋转矩阵
    def rotate_camera(self, position, camera_rotation, rotation_matrix, distance):
        new_position = position @ rotation_matrix.T
        new_position = new_position + distance
        # 原相机朝向
        origin_dir = camera_orientation_from_rotation_matrix(camera_rotation)
        new_dir = get_new_camera_orientation(rotation_matrix, origin_dir)
        new_rotation_matrix = create_camera_coordinate_system(new_dir)
        return new_rotation_matrix, new_position
