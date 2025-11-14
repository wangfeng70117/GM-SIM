import torch
import warp as wp
import numpy as np
from mpm_engine.mpm_utils import *


@wp.kernel
def mark_seg_grid(
        particles: wp.array(dtype=wp.vec3),
        grid_dx: float,
        grid_dy: float,
        grid_dz: float,
        grid: wp.array(dtype=int, ndim=3)
):
    p = wp.tid()
    pos = particles[p]
    base_pos_x = wp.int((pos[0] / grid_dx) - 0.5)
    base_pos_y = wp.int((pos[1] / grid_dy) - 0.5)
    base_pos_z = wp.int((pos[2] / grid_dz) - 0.5)

    grid[base_pos_x, base_pos_y, base_pos_z] = 1


@wp.kernel
def calculate_mask(
        particles: wp.array(dtype=wp.vec3),
        grid_dx: float,
        grid_dy: float,
        grid_dz: float,
        grid_mask: wp.array(dtype=int, ndim=3),
        mask: wp.array(dtype=int)
):
    p = wp.tid()
    pos = particles[p]
    base_pos_x = wp.int((pos[0] / grid_dx) - 0.5)
    base_pos_y = wp.int((pos[1] / grid_dy) - 0.5)
    base_pos_z = wp.int((pos[2] / grid_dz) - 0.5)
    if grid_mask[base_pos_x, base_pos_y, base_pos_z]== 1:
        mask[p] = 1

#
# @wp.kernel
# def calculate_mask(
#     particles: wp.array(dtype=wp.vec3),  # 包围盒点数组
#     grid_dx: float,                     # 网格步长 x
#     grid_dy: float,                     # 网格步长 y
#     grid_dz: float,                     # 网格步长 z
#     grid_mask: wp.array(dtype=int, ndim=3),     # 网格标记
#     mask: wp.array(dtype=int),          # 结果 mask
# ):
#     p = wp.tid()
#     pos = particles[p]
#     base_pos_x = wp.int((pos[0] / grid_dx) - 0.5)
#     base_pos_y = wp.int((pos[1] / grid_dy) - 0.5)
#     base_pos_z = wp.int((pos[2] / grid_dz) - 0.5)
#     if grid_mask[base_pos_x, base_pos_y, base_pos_z] == 1:
#         mask[p] = 1


# 从包围盒的所有点中，找到簇中点所在区域的点
def get_bbox_point_mask(
        cluster_points: wp.array(dtype=wp.vec3),
        bbox_points,
        grid_length_x,
        grid_length_y,
        grid_length_z,
        grid_n=64
):
    grid_dx = grid_length_x / grid_n
    grid_dy = grid_length_y / grid_n
    grid_dz = grid_length_z / grid_n
    print(f'grid_dx is {grid_dx}, grid_dy is {grid_dy}, grid_dz is {grid_dz}')
    grid: wp.array(dtype=int, ndim=3)
    grid = wp.zeros(
        shape=(grid_n, grid_n, grid_n),
        dtype=int,
    )
    print(torch.min(cluster_points, dim=0))
    min_values = bbox_points.min(dim=0).values
    cluster_points = cluster_points.sub(min_values)
    bbox_points = bbox_points.sub(min_values)
    # 簇中的所有点
    cluster_particles = torch2warp_vec3(cluster_points)
    # 包围盒中的所有点
    bbox_particles = torch2warp_vec3(bbox_points)

    # mask 初始化
    mask = wp.zeros(shape=(bbox_points.shape[0]), dtype=int)
    wp.launch(kernel=mark_seg_grid, dim=cluster_particles.shape[0],
              inputs=[cluster_particles, grid_dx, grid_dy, grid_dz, grid])

    wp.launch(kernel=calculate_mask, dim=bbox_particles.shape[0],
              inputs=[bbox_particles, grid_dx, grid_dy, grid_dz, grid, mask])
    mask = wp.to_torch(mask)
    # # 确保输入到 kernel 中的 grid 具有正确的形状
    # wp.launch(kernel=mark_cluster_grid, dim=cluster_particles.shape[0], inputs=[cluster_particles, grid_dx, grid_dy, grid_dz, grid])
    #
    # wp.launch(kernel=calculate_mask, dim=bbox_particles.shape[0], inputs=[bbox_particles, grid_dx, grid_dy, grid_dz, grid, mask])
    #
    # # 将 mask 转回 PyTorch
    # mask = wp.to_torch(mask)
    return mask == 1
