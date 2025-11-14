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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # TODO 变更条件
            if orig_h > 1080:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    gt_feature_map = torch.from_numpy(cam_info.feature_map).permute(2, 0, 1) if cam_info.feature_map is not None else None
    gray_mask = torch.from_numpy(cam_info.gray_mask) if cam_info.gray_mask is not None else None
    gt_depth = torch.from_numpy(cam_info.depth) if cam_info.depth is not None else None

    loaded_mask = None


    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, qvec=cam_info.qvec,
                  image=gt_image, feature_map=gt_feature_map, gt_alpha_mask=loaded_mask, gray_mask=gray_mask,
                  depth=gt_depth, image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


# 将四元数转化为旋转矩阵
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def manual_slerp(quat1, quat2, t):
    """
    Manual implementation of Spherical Linear Interpolation (SLERP) for quaternions.
    """
    dot_product = np.dot(quat1, quat2)

    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot_product < 0.0:
        quat1 = -quat1
        dot_product = -dot_product

    # Clamp value to stay within domain of acos()
    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta_0 = np.arccos(dot_product)  # angle between input vectors
    theta = theta_0 * t  # angle between v0 and result

    quat2 = quat2 - quat1 * dot_product
    quat2 = quat2 / np.linalg.norm(quat2)

    return quat1 * np.cos(theta) + quat2 * np.sin(theta)


def interpolate_se3(T1, T2, t):
    """
    Interpolates between two SE(3) poses.

    :param T1: First SE(3) matrix.
    :param T2: Second SE(3) matrix.
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated SE(3) matrix.
    """
    if np.isclose(T1 - T2, 0).all():
        return T1
    # Decompose matrices into rotation (as quaternion) and translation
    rot1, trans1 = T1[:3, :3], T1[:3, 3]
    rot2, trans2 = T2[:3, :3], T2[:3, 3]
    quat1, quat2 = R.from_matrix(rot1).as_quat(), R.from_matrix(rot2).as_quat()

    # Spherical linear interpolation (SLERP) for rotation
    # Manual SLERP for rotation
    interp_quat = manual_slerp(quat1, quat2, t)
    interp_rot = R.from_quat(interp_quat).as_matrix()

    # Linear interpolation for translation
    interp_trans = interp1d([0, 1], np.vstack([trans1, trans2]), axis=0)(t)

    # Recompose SE(3) matrix
    T_interp = np.eye(4)
    T_interp[:3, :3] = interp_rot
    T_interp[:3, 3] = interp_trans

    return T_interp


def interpolate_camera_se3(view1, view2, t):
    """
    Interpolates between two Camera poses.

    :param view1: First Camera.
    :param view2: Second Camera.
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated Camera.
    """
    view = deepcopy(view1)
    view.world_view_transform = torch.tensor(
        interpolate_se3(
            view1.world_view_transform.cpu().numpy().T,
            view2.world_view_transform.cpu().numpy().T,
            t).T
    ).cuda().float()
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    return view


def get_single_view(views, idx):
    if isinstance(idx, int):
        return views[idx]
    if idx.isnumeric():
        idx = int(idx)
        return views[idx]
    else:
        raise NotImplementedError("Got unsupported view index {}".format(idx))


def get_current_view(views, start_idx, end_idx, t):
    start_view = get_single_view(views, start_idx)
    end_view = get_single_view(views, end_idx)
    current_view = interpolate_camera_se3(start_view, end_view, t)
    return current_view
