#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    try:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
    except Exception as e:
        print(e)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# 计算两个向量的夹角
def vector_angle(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    return theta


# 计算点到平面的距离
def point_to_plane_distance(point, plane):
    x, y, z = point
    A, B, C, D = plane
    numerator = np.abs(A * x + B * y + C * z + D)
    denominator = np.sqrt(A ** 2 + B ** 2 + C ** 2)
    distance = numerator / denominator
    return distance


# 1. 向量A到向量B的旋转矩阵
def rotation_matrix_from_vectors(vec1, vec2):
    """计算从vec1到vec2的旋转矩阵"""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    axis = np.cross(vec1, vec2)
    angle = np.arccos(np.dot(vec1, vec2))

    if np.linalg.norm(axis) == 0:
        return np.eye(3)  # 如果两个向量是相同的，返回单位矩阵

    axis = axis / np.linalg.norm(axis)

    # 使用Rodrigues旋转公式
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array([
        [cos_angle + axis[0] ** 2 * (1 - cos_angle),
         axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
         axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],

        [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
         cos_angle + axis[1] ** 2 * (1 - cos_angle),
         axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],

        [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
         axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
         cos_angle + axis[2] ** 2 * (1 - cos_angle)]
    ])
    return R


# 根据相机的朝向创建相机坐标系(旋转矩阵)
def create_camera_coordinate_system(view_direction):
    # Define an arbitrary up vector (typically Y-axis in world space)
    up_vector = np.array([0.0, 0.0, -1.0])

    # Ensure the up vector is not collinear with the view direction
    if np.allclose(view_direction, up_vector) or np.allclose(view_direction, -up_vector):
        up_vector = np.array([1.0, 0.0, 0.0])  # Change up vector if necessary

    # Calculate the X-axis (right direction) via cross product
    x_axis = np.cross(up_vector, view_direction)
    x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize to unit vector

    # Calculate the Y-axis via cross product (ensure it's orthogonal)
    y_axis = np.cross(view_direction, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize to unit vector

    # Rotation matrix that transforms world coordinates to camera coordinates
    rotation_matrix = np.vstack([x_axis, y_axis, view_direction]).T
    return rotation_matrix


def rotation_matrix_from_vectors(vec1, vec2):
    """计算从vec1到vec2的旋转矩阵"""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    axis = np.cross(vec1, vec2)
    angle = np.arccos(np.dot(vec1, vec2))

    if np.linalg.norm(axis) == 0:
        return np.eye(3)  # 如果两个向量是相同的，返回单位矩阵

    axis = axis / np.linalg.norm(axis)

    # 使用Rodrigues旋转公式
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array([
        [cos_angle + axis[0] ** 2 * (1 - cos_angle),
         axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
         axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],

        [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
         cos_angle + axis[1] ** 2 * (1 - cos_angle),
         axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],

        [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
         axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
         cos_angle + axis[2] ** 2 * (1 - cos_angle)]
    ])

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
    - R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    - quaternion (tuple): A tuple (w, x, y, z) representing the quaternion.
    """
    # Ensure the matrix is 3x3
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix.")

    # Compute the trace of the matrix
    trace = np.trace(R)

    # Compute the quaternion components based on the trace
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the largest diagonal element and compute the quaternion accordingly
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    # Return the quaternion (w, x, y, z)
    return (w, x, y, z)


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
    - q (tuple): A quaternion (w, x, y, z).

    Returns:
    - R (numpy.ndarray): A 3x3 rotation matrix.
    """
    w, x, y, z = q

    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    return R


# 根据旋转矩阵计算相机的朝向
def camera_orientation_from_rotation_matrix(R):
    """
    从旋转矩阵计算相机的朝向（即相机的视线方向）。
    Args:
    - R (numpy.ndarray): 3x3 旋转矩阵。
    Returns:
    - orientation (numpy.ndarray): 相机的朝向（一个单位向量，指向相机的视线方向）。
    """
    # 假设相机朝向是旋转矩阵的第三列
    orientation = R[:, 2]  # 获取第三列（Z轴方向）

    # 返回单位化的朝向向量
    return orientation / np.linalg.norm(orientation)


# 根据旋转矩阵和原始的相机朝向计算旋转后的相机朝向
def get_new_camera_orientation(R_pos, d_orig):
    # 计算旋转后的朝向
    d_new = np.dot(R_pos, d_orig)
    # 返回单位化后的朝向
    return d_new / np.linalg.norm(d_new)

