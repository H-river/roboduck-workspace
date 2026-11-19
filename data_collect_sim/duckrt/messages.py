"""Messages between the executor layer, the runtime and recorders.

Frames: robot frame x forward, y left, z up. Units: m, rad, s (acc in m/s^2).
Quaternions are (w, x, y, z).

* CommandMsg  executor -> runtime, ~10 Hz.
* StateMsg    runtime -> anyone, every policy tick. Only quantities the real robot can
              measure (IMU + servos) plus the runtime's own mode / fall flag.
* FrameMsg    head camera frame with what is needed to project pixels to the ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(str, Enum):
    WALK_FLAT = "WALK_FLAT"
    WALK_SLOPE = "WALK_SLOPE"
    WALK_ROUGH = "WALK_ROUGH"
    STAND = "STAND"
    KICK = "KICK"
    CELEBRATE = "CELEBRATE"
    GET_UP = "GET_UP"


@dataclass(frozen=True)
class CommandMsg:
    t_send: float                       # sender clock (sim time in simulation) [s]
    mode: Mode
    cmd: tuple[float, float, float]     # (vx [m/s], vy [m/s], wz [rad/s]) in the robot frame
    seq: int
    # Executor takes over the head while standing: extra downward camera tilt [deg] relative
    # to the default head pose (split over neck_pitch / head_pitch). None = policy drives the head.
    head_pitch_down_deg: float | None = None
    # KICK only: ball position (x, y) [m] in the robot frame, measured once before the kick
    # (kick policies with a ball observation; ignored by blind kicks).
    kick_ball_xy: tuple[float, float] | None = None


@dataclass
class StateMsg:
    t: float
    imu_quat: np.ndarray        # (4,) BNO08x game-rotation quaternion, sensor frame (wxyz)
    gyro: np.ndarray            # (3,) rad/s, sensor frame
    acc: np.ndarray             # (3,) m/s^2, sensor frame (specific force)
    joint_pos: np.ndarray       # (14,) rad, OBSERVATION_DOF_ORDER
    joint_vel: np.ndarray       # (14,) rad/s
    mode: Mode
    fall_flag: bool


@dataclass
class CameraExtrinsics:
    """Camera pose as 4x4 homogeneous transforms (camera frame: OpenCV, x right, y down,
    z forward). T_world_cam maps camera coords to world coords; T_robot_cam maps them to the
    robot frame (origin = trunk position projected on z = 0, x = trunk heading, z up)."""

    T_world_cam: np.ndarray
    T_robot_cam: np.ndarray
    T_world_robot: np.ndarray


@dataclass
class FrameMsg:
    t_exposure: float
    image: np.ndarray                   # (H, W, 3) uint8 RGB
    head_q: np.ndarray                  # (4,) neck_pitch, head_pitch, head_yaw, head_roll [rad]
    imu_quat: np.ndarray                # (4,) same convention as StateMsg.imu_quat
    cam_extrinsics: CameraExtrinsics
    meta: dict = field(default_factory=dict)
