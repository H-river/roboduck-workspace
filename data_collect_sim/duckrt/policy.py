"""ONNX policy wrapper: card-driven observation -> action -> joint targets."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from duckrt.obs import ObsBuilder
from duckrt.policy_card import PolicyCard


class OnnxPolicy:
    def __init__(self, card: PolicyCard):
        if card.action["type"] != "joint_pos_offset":
            raise NotImplementedError(f"action type {card.action['type']!r}")
        self.card = card
        self.obs_builder = ObsBuilder(card)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(card.onnx), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self._input_name = inp.name
        if inp.shape[-1] != self.obs_builder.dim:
            raise ValueError(f"{card.name}: ONNX input {inp.shape} != card obs dim {self.obs_builder.dim}")
        n = len(card.joint_names)
        if out.shape[-1] != n:
            raise ValueError(f"{card.name}: ONNX output {out.shape} != {n} joints")
        meta = self.session.get_modelmeta().custom_metadata_map
        if "joint_names" in meta and meta["joint_names"].split(",") != card.joint_names:
            raise ValueError(f"{card.name}: ONNX joint_names differ from the card")
        self.default_q = card.default_q.copy()
        self.scale = card.action_scale
        self.last_action = np.zeros(n, dtype=np.float32)
        self.last_obs: np.ndarray | None = None
        self.extra_obs = self.obs_builder.extra_terms
        self.extras: dict = {}

    @property
    def period(self) -> float:
        return 1.0 / self.card.control_hz

    def reset(self) -> None:
        """Clear the action history (the only internal state of these policies)."""
        self.last_action[:] = 0.0
        self.last_obs = None

    def set_extra(self, name: str, value) -> None:
        """Value for an extra observation term (e.g. kick_ball_xy); None -> the card's default."""
        if value is None:
            value = self.card.task.get("obs_defaults", {}).get(name)
            if value is None:
                raise ValueError(f"{self.card.name}: no value and no default for {name}")
        self.extras[name] = np.asarray(value, dtype=np.float64)

    def act(
        self,
        imu_quat: np.ndarray,
        gyro: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        """One policy tick. Returns joint position targets (OBSERVATION_DOF_ORDER)."""
        obs = self.obs_builder.build(imu_quat, gyro, joint_pos, joint_vel, self.last_action, command, self.extras)
        action = self.session.run(None, {self._input_name: obs[None, :]})[0][0]
        self.last_obs = obs
        self.last_action = action.astype(np.float32)
        return self.default_q + self.scale * action.astype(np.float64)
