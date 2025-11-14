import numpy as np
import torch
import torch.nn.functional as F

class CameraInfo:
    def __init__(self, fx, fy, cx, cy, w, h, near_plane, far_plane) -> None:
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h
        self.near_plane = near_plane
        self.far_plane = far_plane

    def downsample(self, scale):
        self.fx /= scale
        self.fy /= scale
        self.cx /= scale
        self.cy /= scale
        self.w //= scale
        self.h //= scale

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = self.w / self.h

    def upsample(self, scale):
        self.fx *= scale
        self.fy *= scale
        self.cx *= scale
        self.cy *= scale
        self.w *= int(scale)
        self.h *= int(scale)

    def get_frustum(self, c2w):
        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = self.far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = self.near_plane * lookat
        far_point = self.far_plane * lookat
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(up, far_point + half_hside * right)

        up_normal = torch.cross(far_point + half_vside * up, right)
        down_normal = torch.cross(right, far_point - half_vside * up)

        pts = [near_point + t, far_point + t, t, t, t, t]
        normals = [
            near_normal,
            far_normal,
            left_normal,
            right_normal,
            up_normal,
            down_normal,
        ]

        pts = torch.stack(pts, dim=0)
        normals = torch.stack(normals, dim=0)
        normals = F.normalize(normals, dim=-1)

        return normals, pts


    def camera_space_to_pixel_space(self, pts):
        if pts.shape[1] == 3:
            pts = pts[:, :2] / pts[:, 2:]

        assert pts.shape[1] == 2

        pts[:, 0] = pts[:, 0] * self.fx + self.cx
        pts[:, 1] = pts[:, 1] * self.fy + self.cy
        if isinstance(pts, np.ndarray):
            pts = pts.astype(np.int32)
        elif isinstance(pts, torch.Tensor):
            pts = pts.to(torch.int32)

        return pts

    @classmethod
    def from_fov_camera(cls, fov, aspect, resolution, near_plane, far_plane):
        W = resolution
        H = int(resolution / aspect)
        cx = W / 2
        cy = H / 2
        fx = cx / np.tan(fov / 2)
        fy = cy / np.tan(fov / 2)

        return cls(fx, fy, cx, cy, W, H, near_plane, far_plane)
