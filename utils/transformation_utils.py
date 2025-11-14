import torch
import numpy as np


def transform2origin(position_tensor):
    # 计算最小位置和最大位置
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    # 计算最大差值,表示点云在所有方向上的最大跨度
    max_diff = torch.max(max_pos - min_pos)
    # 计算点云的中心位置
    original_mean_pos = (min_pos + max_pos) / 2.0
    # 计算缩放比例,将点云的范围缩放至1
    scale = 1.0 / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


# 根据旋转角度和旋转轴计算新的旋转矩阵
def generate_rotation_matrix(degree, axis):
    # 角度转化为弧度
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    # 旋转角度   旋转轴
    assert len(degrees) == len(axises)

    # 生成的旋转矩阵
    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices)
    return get_uppder_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices.T)
    return get_uppder_from_mat(cov_tensor)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )

