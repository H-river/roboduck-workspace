"""Observation construction, driven entirely by the policy card's obs_layout.

Inputs are what the real robot measures: the BNO08x quaternion and gyro in the sensor
frame, servo positions / velocities, plus the runtime's last action and command. The
IMU terms use the deploy formulas (microduck/src/observer.py:76-85), which equal the
training terms mounted_body_* (tasks/mdp.py:142-179) when the sensor frame is
body (x) conj(BNO080_MOUNT_QUAT) -- checked by scripts/check_parity.py.
"""

from __future__ import annotations

import numpy as np

from duckrt.policy_card import PolicyCard
from duckrt.upstream import quat_apply_inverse

_UP = (0.0, 0.0, 1.0)


class ObsBuilder:
    def __init__(self, card: PolicyCard):
        self.layout = [(t["name"], int(t["dim"])) for t in card.obs_layout]
        self.scales = card.obs_scales
        self.default_q = card.default_q.astype(np.float64)
        n = len(card.joint_names)
        expected = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "joint_pos": n,
            "joint_vel": n,
            "actions": n,
            "command": 3,
            "kick_ball_xy": 2,   # kick v2: ball (x, y) in the robot frame, constant over the kick
        }
        for name, dim in self.layout:
            if name not in expected:
                raise ValueError(f"unsupported obs term {name!r} in card {card.name}")
            if expected[name] != dim:
                raise ValueError(f"obs term {name!r}: card dim {dim} != {expected[name]}")
        if card.obs_history != 1:
            raise NotImplementedError("obs_history > 1 is not used by any current policy")
        self.dim = sum(d for _, d in self.layout)
        self.extra_terms = {name for name, _ in self.layout} & {"kick_ball_xy"}

    def build(
        self,
        imu_quat: np.ndarray,
        gyro: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
        command: np.ndarray,
        extras: dict | None = None,
    ) -> np.ndarray:
        terms = {
            "base_ang_vel": lambda: np.asarray(gyro, dtype=np.float64),
            "projected_gravity": lambda: np.asarray(quat_apply_inverse(imu_quat, _UP)),
            "joint_pos": lambda: np.asarray(joint_pos, dtype=np.float64) - self.default_q,
            "joint_vel": lambda: np.asarray(joint_vel, dtype=np.float64),
            "actions": lambda: np.asarray(last_action, dtype=np.float64),
            "command": lambda: np.asarray(command, dtype=np.float64),
            "kick_ball_xy": lambda: np.asarray(extras["kick_ball_xy"], dtype=np.float64),
        }
        parts = []
        for name, _ in self.layout:
            v = terms[name]()
            s = self.scales.get(name)
            parts.append(v if s is None else v * s)
        return np.concatenate(parts).astype(np.float32)
