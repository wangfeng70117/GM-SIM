import os
import subprocess
import numpy as np
import torch
import tqdm

# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from scene import Scene, GaussianModel
from mpm_engine.mpm_solver import MPM_Simulator_WARP
from mpm_engine.export_utils import *
import warp as wp
import point_cloud_utils
from scene.cameras import Simple_Camera
import torchvision
from argparse import ArgumentParser
from arguments import PipelineParams, ModelParams, OptimizationParams, get_combined_args
from utils.transformation_utils import *
from point_cloud_utils.filling_util import *
from point_cloud_utils.filling import *
import time

wp.init()
wp.config.verify_cuda = True
# wp.config.print_launches = True

import taichi as ti

ti.init(arch=ti.gpu)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_gaussian(path, opt, args):
    gaussian = GaussianModel(sh_degree=3)
    (model_params, _) = torch.load(path)
    gaussian.training_setup(opt)
    gaussian.restore(model_params, args, mode='test')
    return gaussian


def load_params_from_gs(
        pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    print(f'pipe.compute_cov3D_python: {pipe.compute_cov3D_python}')
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
        pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


def get_particle_volume(pos, grid_n: int, grid_dx: float, unifrom: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)
    # 颗粒状物质让体积一致
    if unifrom:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def combine_positions(position, sim_pos, unselected_pos, region_mask):
    # 创建一个新的张量，用于存储合并后的位置
    combined_pos = position.clone()  # 使用 clone 以避免修改原始 position

    # 将 sim_pos 放回到原位置中对应的地方
    combined_pos[region_mask] = sim_pos

    # unselected_pos 已经在原位置中，因此无需改变。
    # 但如果你想验证它，可以再确认一下
    combined_pos[~region_mask] = unselected_pos

    return combined_pos


def simulation(dataset, opt, pipe, args):
    gaussian = load_gaussian(os.path.join(args.model_path, args.ckpt_name), opt, args)
    scene = Scene(dataset, gaussian, init_gaussian=False, shuffle=False)

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True

    params = load_params_from_gs(gaussian, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # -----------------分割---------------
    # clip_segmenter = point_cloud_utils.ClipSegmenter(args.source_path)
    # clip_segmenter.load_features512(gaussian.get_language_feature)
    # _, _ = clip_segmenter.compute_similarity_normalized(args.label)
    # obj_mask = clip_segmenter.get_segment_mask(init_pos, threshold=0.75, ratio=30, min_cluster_size=3000, grid_n=5)
    # torch.save(obj_mask, "sim_apple.pt")
    obj_mask = torch.load(os.path.join(args.model_path, "Green Apple.pt"))
    # 需要移动的物体，先获得mask，再移动，
    # -----------------------------------

    # --------------旋转点云，相机至XOY水平面--------------
    cameras = scene.getTrainCameras()
    ground_estimator = point_cloud_utils.GroundEstimator(0.001)
    rotation_matrix, distance, plane_model = ground_estimator.estimate(init_pos.detach().cpu().numpy(), 0.005)
    print(f'distance is {distance}, plan_model is {plane_model}')
    rotation_matrix = torch.tensor(rotation_matrix).to("cuda:0").to(dtype=init_pos.dtype)
    distance = torch.tensor(distance).to("cuda:0")
    # 旋转点云到水平面
    rotated_pos = ground_estimator.rotate_point_cloud(init_pos, rotation_matrix, distance)
    # 相机随着点云旋转
    cam = cameras[11]
    # cam = cameras[59]
    view_point = Simple_Camera(R=cam.R, T=cam.T, FoVx=cam.FoVx,
                               FoVy=cam.FoVy,
                               h=cam.image_height, w=cam.image_width, image_name="", uid=0)
    # -------------------------------------------------

    # -----------------设置仿真区域-----------------------

    obj_points = rotated_pos[obj_mask]
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(obj_points, "./log/obj_points.ply")
    min_vals = obj_points.min(dim=0).values
    max_vals = obj_points.max(dim=0).values
    # 计算每个轴的长度
    lengths = max_vals - min_vals
    extended_min_vals = min_vals - lengths
    extended_max_vals = max_vals + lengths
    # 在目标点云周围扩展N个单位内的所有点
    region_mask = (rotated_pos[:, 0] >= extended_min_vals[0]) & (rotated_pos[:, 0] <= extended_max_vals[0]) & \
                  (rotated_pos[:, 1] >= extended_min_vals[1]) & (rotated_pos[:, 1] <= extended_max_vals[1]) & \
                  (rotated_pos[:, 2] >= extended_min_vals[2]) & (rotated_pos[:, 2] <= extended_max_vals[2])
    # 仿真区域内的所有点
    # region_points
    region_points = rotated_pos[region_mask]
    unselected_pos = rotated_pos[~region_mask]
    if args.debug:
        particle_position_tensor_to_ply(region_points, "./log/region_points.ply")

    # 判断每个点是否需要动，不需要动的点让他所在的网格速度设置为0
    is_in_region = torch.isin(region_points, obj_points)
    region_sim_mask = torch.all(is_in_region, dim=1)

    # car_mask = torch.load("yellow toy dog.pt")
    # car_point = rotated_pos[car_mask]
    # car_is_in_region = torch.isin(region_points, car_point)
    # car_region_mask = torch.all(car_is_in_region, dim=1)
    #
    # egg_tart_mask = torch.load("snack package.pt")
    # egg_tart_point = rotated_pos[egg_tart_mask]
    # egg_tart_is_in_region = torch.isin(region_points, egg_tart_point)
    # egg_tart_region_mask = torch.all(egg_tart_is_in_region, dim=1)
    #
    # region_sim_mask = region_sim_mask | egg_tart_region_mask
    region_sim_mask = region_sim_mask
    sim_points = region_points[region_sim_mask]
    if args.debug:
        particle_position_tensor_to_ply(sim_points, "./log/sim_points.ply")
    # ------------------------------------

    # --------------缩放------------
    # 将仿真区域内的缩放到MPM仿真区域
    scaled_region_pos, scale, origin_center = rescale_points(region_points)
    print(f'scale is {scale}')
    # 将点云位置设置到0.1-0.9
    scaled_region_pos = scaled_region_pos + torch.tensor([0.5, 0.5, 0.5], device="cuda")
    print(f'torch.max(position) is {torch.max(scaled_region_pos, dim=0)}, '
          f'torch.min(position) is {torch.min(scaled_region_pos, dim=0)}')

    if args.debug:
        particle_position_tensor_to_ply(scaled_region_pos, "./log/scaled_region_pos.ply")

    # 仿真区域内协方差矩阵的旋转和缩放
    region_cov = init_cov[region_mask]
    region_cov = apply_cov_rotations(region_cov, rotation_matrix)
    region_cov = scale * scale * region_cov

    print(f'MPM仿真的粒子个数为{scaled_region_pos.shape[0]}')
    # only green apple
    # material_params = {
    #     "E": 1.2e4,
    #     "nu": 0.3,
    #     "material": "jelly",
    #     #          上方
    #     "g": [0.0, 0.0, -5.0],
    #     "density": 200.0,
    #     "n_grid": 80,
    #     "grid_lim": 1,
    #     "rpic_damping": 0.0,
    # }

    # Sim dog and apple and snack
    material_params = {
        "E": 1.2e4,
        "nu": 0.2,
        "material": "jelly",
        #          上方
        "g": [0.0, 0.0, -5.0],
        "density": 200.0,
        "n_grid": 100,
        "grid_lim": 1,
        "rpic_damping": 0.0,
    }
    # volume_tensor = torch.ones(region_points.shape[0]) * 2.5e-8
    # 未填充的高斯数

    volume_tensor = get_particle_volume(
        scaled_region_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device="cuda:0")
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(scaled_region_pos, "./log/scaled_pos.ply")
    mpm_solver = MPM_Simulator_WARP(10, n_grid=material_params["n_grid"])
    mpm_solver.load_initial_data_from_torch(
        scaled_region_pos,
        volume_tensor,
        region_sim_mask,
        region_cov,
    )
    #
    # material_params = {
    #     "E": 2000,
    #     "nu": 0.2,
    #     "material": "sand",
    #     "friction_angle": 30,
    #     "g": [0.0, 0.0, -2.9],
    #     "density": 200.0,
    # }
    mpm_solver.init_particle_mark(region_sim_mask)
    mpm_solver.set_parameters_dict(material_params)
    mpm_solver.finalize_mu_lam()
    mpm_solver.add_surface_collider((0.005, 0.0, 0.0), (1.0, 0.0, 0.0), "sticky", 0.0)
    mpm_solver.add_surface_collider((0.0, 0.005, 0.0), (0.0, 1.0, 0.0), "sticky", 0.0)
    mpm_solver.add_surface_collider((0.0, 0.0, 0.005), (0.0, 0.0, 1.0), "sticky", 0.0)

    mpm_solver.add_surface_collider((0.995, 0.0, 0.0), (-1.0, 0.0, 0.0), "sticky", 0.0)
    mpm_solver.add_surface_collider((0.0, 0.995, 0.0), (0.0, -1.0, 0.0), "sticky", 0.0)
    mpm_solver.add_surface_collider((0.0, 0.0, 0.995), (0.0, 0.0, -1.0), "sticky", 0.0)

    # ply_directory_to_save = os.path.join("./sim_results", "ply_files")
    image_directory_to_save = os.path.join('./sim_result', 'figurines_moved6_all')
    os.umask(0)
    os.makedirs(image_directory_to_save, 0o777, exist_ok=True)
    start_time = time.time()
    for k in tqdm.tqdm(range(0, 1000)):
        mpm_solver.p2g2p(k, 0.001, device="cuda:0")
        # if k % 20 == 0:
        #     save_data_at_frame(
        #         mpm_solver, ply_directory_to_save, k, save_to_ply=True
        #     )
        if k % 3 == 0:
            new_sim_pos = mpm_solver.export_particle_x_to_torch().clone()
            rescaled_pos = restore_points(new_sim_pos, scale, origin_center)

            rotated_pos[region_mask] = rescaled_pos
            rotated_pos[~region_mask] = unselected_pos
            new_pos = ground_estimator.inverse_rotate_point_cloud(rotated_pos, rotation_matrix, distance)

            region_cov3D = mpm_solver.export_particle_cov_to_torch()
            region_cov3D = region_cov3D.view(-1, 6)
            cov3D = region_cov3D / (scale * scale)
            region_cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrix)
            init_cov[region_mask] = region_cov3D

            gaussian.reset_position(new_pos)
            render_pkg = simulation_render(view_point, gaussian, background, pre_cov=init_cov)
            # render_pkg = simulation_render(view_point, gaussian, background)
            image, language_feature, render_depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["language_feature_image"], render_pkg['render_depth'], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            torchvision.utils.save_image(image, os.path.join(image_directory_to_save, '{0:05d}'.format(k) + ".png"))
    end_time = time.time()
    eplase = end_time - start_time
    print(f'time is {eplase}')
    print(f'fps is {int(1000 / eplase)}')
    mpm_solver.print_time_profile()


if __name__ == "__main__":
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    model = ModelParams(parser)
    pip = PipelineParams(parser)
    parser.add_argument('--ckpt_name', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--particle_filling', action='store_true', default=False)
    args = get_combined_args(parser)

    simulation(model.extract(args), op.extract(args), pip.extract(args), args)
