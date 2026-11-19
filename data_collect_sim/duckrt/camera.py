"""Head camera: model edits, intrinsics, extrinsics, pixel <-> ground projection.

Conventions
* Pixels: (u, v) with u to the right, v down, pixel centres at integer coordinates
  (pixel (0, 0) covers [-0.5, 0.5]^2), as OpenCV.
* Camera frame (returned extrinsics): OpenCV, x right, y down, z forward.
  MuJoCo cameras look along -z with y up; converted with diag(1, -1, -1).
* Robot frame: origin = trunk position projected on z = 0, x = trunk heading (yaw only), z up.
* The ground is the plane z = 0 in world.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from duckrt.messages import CameraExtrinsics, FrameMsg

if TYPE_CHECKING:
    from duckrt.backend_sim import SimBackend

HEAD_CAM = "head_cam"
TOP_CAM = "top_cam"
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


# -- model edits ----------------------------------------------------------------------------
def head_cam_local_pose(site_pos, site_quat, pos_offset, pitch_down_deg) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose in the mount body from the site frame (site +x forward, +z up)."""
    th = math.radians(pitch_down_deg)
    fwd = np.array([math.cos(th), 0.0, -math.sin(th)])
    up = np.array([math.sin(th), 0.0, math.cos(th)])
    right = np.array([0.0, -1.0, 0.0])
    R_site_cam = np.column_stack([right, up, -fwd])  # MuJoCo camera axes in site coords
    q_sc = np.zeros(4)
    mujoco.mju_mat2Quat(q_sc, R_site_cam.flatten())
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(site_quat, float), q_sc)
    R_site = np.zeros(9)
    mujoco.mju_quat2Mat(R_site, np.asarray(site_quat, float))
    pos = np.asarray(site_pos, float) + R_site.reshape(3, 3) @ np.asarray(pos_offset, float)
    return pos, q


def add_cameras(spec: mujoco.MjSpec, camera_cfg: dict, layout: dict | None) -> None:
    hc = camera_cfg["head_camera"]
    body = spec.body(hc["mount_body"])
    site = next(s for s in spec.sites if s.name == hc["mount_site"])
    pos, quat = head_cam_local_pose(site.pos, site.quat, hc["pos_offset_m"], hc["pitch_down_deg"])
    cam = body.add_camera()
    cam.name = HEAD_CAM
    cam.pos = pos
    cam.quat = quat
    cam.fovy = float(hc["fovy_deg"])

    tc = camera_cfg["top_cam"]
    if layout is not None:
        ext = layout["_arena_size"]
        centre = layout["_arena_centre"]
    else:
        ext, centre = tc["default_extent_m"], [0.0, 0.0]
    half_w = ext[0] / 2 + tc["margin_m"]
    half_h = ext[1] / 2 + tc["margin_m"]
    aspect = tc["width"] / tc["height"]
    half_fov = math.radians(tc["fovy_deg"]) / 2
    z = max(half_h, half_w / aspect) / math.tan(half_fov)
    top = spec.worldbody.add_camera()
    top.name = TOP_CAM
    top.pos = [centre[0], centre[1], z]
    top.quat = [1.0, 0.0, 0.0, 0.0]  # looks down -z, image up = +y (world)
    top.fovy = float(tc["fovy_deg"])


# -- intrinsics / extrinsics ----------------------------------------------------------------
def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole K for a MuJoCo camera (fovy is vertical, square pixels)."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0 - 0.5], [0.0, f, height / 2.0 - 0.5], [0.0, 0.0, 1.0]])


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def robot_frame(base_pos: np.ndarray, base_yaw: float) -> np.ndarray:
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return _T(R, np.array([base_pos[0], base_pos[1], 0.0]))


def extrinsics_from_data(model, data, cam_name: str, base_pos, base_yaw) -> CameraExtrinsics:
    cid = model.camera(cam_name).id
    R_mj = data.cam_xmat[cid].reshape(3, 3)
    T_wc = _T(R_mj @ _MJ_TO_CV, data.cam_xpos[cid].copy())
    T_wr = robot_frame(base_pos, base_yaw)
    return CameraExtrinsics(T_world_cam=T_wc, T_robot_cam=np.linalg.inv(T_wr) @ T_wc, T_world_robot=T_wr)


# -- projection -----------------------------------------------------------------------------
def pixel_to_ground(u: float, v: float, K: np.ndarray, T_world_cam: np.ndarray, ground_z: float = 0.0):
    """Intersect the pixel ray with the plane z = ground_z (world). Returns the 3-D world
    point, or None if the ray does not point down to the plane (horizon / above)."""
    d_cam = np.linalg.solve(K, np.array([u, v, 1.0]))
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    d = R @ d_cam
    if d[2] >= -1e-9:
        return None
    s = (ground_z - t[2]) / d[2]
    if s <= 0:
        return None
    return t + s * d


def ground_to_pixel(p_world, K: np.ndarray, T_world_cam: np.ndarray, width: int | None = None,
                    height: int | None = None):
    """Project a world point to pixels. Returns (u, v) or None if behind the camera (or,
    when width/height are given, outside the image)."""
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    p_cam = R.T @ (np.asarray(p_world, float) - t)
    if p_cam[2] <= 1e-9:
        return None
    uvw = K @ p_cam
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    if width is not None and height is not None and not (-0.5 <= u <= width - 0.5 and -0.5 <= v <= height - 0.5):
        return None
    return float(u), float(v)


def world_to_robot(p_world, T_world_robot: np.ndarray) -> np.ndarray:
    T = np.linalg.inv(T_world_robot)
    return T[:3, :3] @ np.asarray(p_world, float) + T[:3, 3]


# -- runtime wrapper ------------------------------------------------------------------------
class HeadCamera:
    def __init__(self, camera_cfg: dict, backend: "SimBackend"):
        hc = camera_cfg["head_camera"]
        self.backend = backend
        self.width, self.height, self.fps = int(hc["width"]), int(hc["height"]), float(hc["fps"])
        self.fovy = float(backend.model.cam_fovy[backend.model.camera(HEAD_CAM).id])
        self.K = intrinsics(self.fovy, self.width, self.height)

    def extrinsics(self) -> CameraExtrinsics:
        """Camera pose at the current sim time (world and robot frame)."""
        b = self.backend
        rd = b.sync_render_data()
        return extrinsics_from_data(b.model, rd, HEAD_CAM, b.base_pos(), b.base_yaw())

    def capture(self) -> FrameMsg:
        b = self.backend
        img = b.render(HEAD_CAM, self.width, self.height)
        ext = extrinsics_from_data(b.model, b.rdata, HEAD_CAM, b.base_pos(), b.base_yaw())
        return FrameMsg(t_exposure=b.time, image=img, head_q=b.head_q(), imu_quat=b.imu_quat(),
                        cam_extrinsics=ext, meta={"K": self.K, "fovy_deg": self.fovy})
