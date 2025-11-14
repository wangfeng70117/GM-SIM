import numpy as np
import os
import torch
import math
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel


def particle_position_to_ply(mpm_solver, filename):
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.mpm_state.particle_x.numpy()
    min_pos = position.min(axis=0)  # 沿第 0 轴求最小值
    max_pos = position.max(axis=0)  # 沿第 0 轴求最大值
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:
        header = f"""ply
        format binary_little_endian 1.0
        element vertex {num_particles}
        property float x
        property float y
        property float z
        end_header
        """
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)


def save_data_at_frame(mpm_solver, dir_name, frame, save_to_ply=True):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)

    fullfilename = dir_name + "/sim_" + str(frame).zfill(10)

    if save_to_ply:
        particle_position_to_ply(mpm_solver, fullfilename + ".ply")


def spherical_to_cartesian(theta, phi):
    """将球面坐标 (theta, phi) 转换为笛卡尔坐标 (x, y, z)"""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])



# 将点云数据缩放至[-0.4, 0.4]的范围
def rescale_points(position, scale_ratio=0.8):
    min_pos = torch.min(position, dim=0)[0]
    max_pos = torch.max(position, dim=0)[0]
    # 计算点云的最大范围差
    max_diff = torch.max(max_pos - min_pos)
    # 计算点云中心的位置
    center = (min_pos + max_pos) / 2.0
    center = center.to(device="cuda:0")
    # 计算缩放比例，将点云的范围缩放至[-0.4, 0.4]
    scale = scale_ratio / max_diff
    new_position_tensor = (position - center) * scale
    return new_position_tensor, scale, center


# 将缩放的点云位置恢复到原始位置
def restore_points(position, scale, center):
    """
    还原点云位置（反向缩放和平移操作）。

    Args:
        position (torch.Tensor): 经变换后的点云位置，形状为 [N, 3]。
        scale (float): 缩放因子。
        center (torch.Tensor): 点云的中心位置。

    Returns:
        torch.Tensor: 还原后的点云位置。
    """
    # 还原平移操作：减去之前加的平移量
    position = position - torch.tensor([0.5, 0.5, 0.5], device="cuda")

    # 还原缩放操作：除以缩放因子
    position = position / scale

    # 恢复点云的中心位置
    position = position + center

    return position

def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp


def simulation_render(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, pre_cov=None,
                      override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pre_cov is None:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    else:
        cov3D_precomp = pre_cov


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    language_feature_precomp = None
    if pc._language_feature is not None:
        language_feature_precomp = pc.get_language_feature

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # start_time = time.time()

    rendered_image, depth, language_feature_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    return {"render": rendered_image,
            "render_depth": depth,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}

def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)