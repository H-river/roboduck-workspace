"""BAM XL330 actuator for bare MuJoCo (CPU, numpy).

Port of bam.mjlab.BamActuator (rhoban/bam @ 0007411bd82c48f1bc12ae8382f98e4d8b879c95,
site-packages/bam/mjlab.py), i.e. the actuator the walk / kick policies were trained
with in mjlab, plus mjlab's command DelayBuffer semantics
(site-packages/mjlab/utils/buffers/delay_buffer.py, actuator/actuator.py:270-302).

Not bam.mujoco.MujocoController: that one is a different friction implementation
(rewrites dof_frictionloss / dof_damping each step) and is what microduck's
MuJoCoController uses -- see NOTES.md §0.1-4.

The motor model itself (parameters, error_gain, max_pwm, armature) is imported from bam;
only the per-step torque computation is re-implemented, vectorized over joints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np
from bam.model import Model, load_model


@dataclass
class ActuatorCfg:
    motor_name: str
    model: str
    kp_fw: float
    vin: float
    vin_drop_gain: float | None
    vin_min: float | None
    max_current: float | None
    vin_max_for_forcerange: float
    delay: dict = field(default_factory=lambda: {"mode": "fixed", "steps": 0})
    # What the delay line holds right after a reset: "hold" = the current joint positions
    # (servos keep their pose), "zeros" = 0 rad for every joint, which is what mjlab does
    # (its reset writes the zero-initialised joint_pos_target into the buffer; NOTES §1).
    reset_target: str = "hold"

    @classmethod
    def from_dict(cls, d: dict) -> "ActuatorCfg":
        return cls(**d)


class CommandDelay:
    """mjlab DelayBuffer with update_period=0, hold_prob=0: every physics step the target
    is appended and a lag in [min, max] is drawn; the target from `lag` steps ago is used.
    History starts backfilled with the first target after a reset."""

    def __init__(self, mode: str, min_steps: int = 0, max_steps: int = 0, steps: int = 0, rng=None):
        if mode == "fixed":
            min_steps = max_steps = int(steps)
        elif mode != "uniform":
            raise ValueError(f"delay mode {mode!r}")
        self.min_lag, self.max_lag = int(min_steps), int(max_steps)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.history: deque[np.ndarray] = deque(maxlen=self.max_lag + 1)
        self.last_lag = 0

    def reset(self) -> None:
        self.history.clear()

    def push(self, target: np.ndarray) -> np.ndarray:
        self.history.append(np.array(target, dtype=np.float64))
        lag = self.min_lag if self.min_lag == self.max_lag else int(self.rng.integers(self.min_lag, self.max_lag + 1))
        self.last_lag = lag
        lag = min(lag, len(self.history) - 1)  # backfill: before enough history, use the first target
        return self.history[-1 - lag]


class BamActuator:
    """Joint-target -> torque for a group of identical BAM-modelled servos."""

    def __init__(self, cfg: ActuatorCfg, joint_names: list[str], rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.bam: Model = load_model(motor_name=cfg.motor_name, model=cfg.model)
        self.bam.actuator.kp = cfg.kp_fw
        self.kp_fw = float(cfg.kp_fw)
        self.kt = float(self.bam.kt.value)
        self.R = float(self.bam.R.value)
        self.error_gain = float(self.bam.actuator.error_gain)
        self.max_pwm = float(self.bam.actuator.max_pwm)
        self.armature = float(self.bam.actuator.get_extra_inertia())
        self.force_limit = cfg.vin_max_for_forcerange * self.kt / self.R
        self.delay = CommandDelay(rng=rng, **cfg.delay)
        n = len(self.joint_names)
        self._prev_motor_torque = np.zeros(n)
        self.last_motor_torque = np.zeros(n)
        self.last_output_torque = np.zeros(n)
        self.last_target = np.zeros(n)
        self.last_vin = cfg.vin
        self._bound = False
        self.reset_stats()

    # -- model edits (bam/mjlab.py:186-235) ---------------------------------------------
    def edit_spec(self, spec: mujoco.MjSpec) -> None:
        """Turn the XML position actuators into torque motors and hand friction to BAM."""
        targets = set(self.joint_names)
        found = set()
        for act in spec.actuators:
            if act.target in targets:
                act.set_to_motor()
                act.forcelimited = True
                act.forcerange = (-self.force_limit, self.force_limit)
                act.gear = [1.0, 0, 0, 0, 0, 0]
                found.add(act.target)
        missing = targets - found
        if missing:
            raise ValueError(f"no XML actuator for joints {sorted(missing)}")
        for joint in spec.joints:
            if joint.name in targets:
                joint.armature = self.armature
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data
        jids = [model.joint(n).id for n in self.joint_names]
        self.qpos_adr = model.jnt_qposadr[jids].copy()
        self.dof_adr = model.jnt_dofadr[jids].copy()
        self.ctrl_ids = np.array([model.actuator(n).id for n in self.joint_names])
        for i, j in zip(self.ctrl_ids, jids):
            if model.actuator_trnid[i, 0] != j:
                raise ValueError("actuator / joint mismatch")
        self.eff_inertia = 1.0 / model.dof_invweight0[self.dof_adr]
        self.dt = float(model.opt.timestep)
        self._bound = True

    def reset(self) -> None:
        """Clear the delay line and carried torque, then run one compute with the reset target
        (cfg.reset_target), as mjlab's reset does through scene.write_data_to_sim(): that seeds
        the delay line and the carried torque used by the voltage-drop model. Call after the
        reset state has been written and mj_forward'ed."""
        self.delay.reset()
        self._prev_motor_torque[:] = 0.0
        if self._bound:
            if self.cfg.reset_target == "hold":
                self.apply(self.data.qpos[self.qpos_adr].copy())
            elif self.cfg.reset_target == "zeros":
                self.apply(np.zeros(len(self.joint_names)))
            else:
                raise ValueError(f"reset_target {self.cfg.reset_target!r}")

    def set_kp(self, kp_fw: float) -> None:
        self.kp_fw = float(kp_fw)

    def reset_stats(self) -> None:
        """Per-physics-step statistics for tests: peak |output torque|, steps at the current limit."""
        n = len(self.joint_names)
        self.stat_peak_torque = np.zeros(n)
        self.stat_current_limited = np.zeros(n, dtype=np.int64)
        self.stat_steps = 0

    # -- friction budget (bam/mjlab.py:_compute_friction_budget) -------------------------
    def _friction_budget(self, motor_torque, external_torque, stribeck_coeff):
        b = self.bam
        fl = np.full_like(motor_torque, b.friction_base.value)
        if b.stribeck:
            fl = fl + stribeck_coeff * b.friction_stribeck.value
        if b.load_dependent:
            if b.directional:
                fl = fl + np.abs(external_torque * b.load_friction_external.value
                                 - motor_torque * b.load_friction_motor.value)
                if b.stribeck:
                    fl = fl + stribeck_coeff * np.abs(
                        external_torque * b.load_friction_external_stribeck.value
                        - motor_torque * b.load_friction_motor_stribeck.value)
                    if b.quadratic:
                        abs_ext, abs_mot = np.abs(external_torque), np.abs(motor_torque)
                        drive = (abs_mot > abs_ext).astype(np.float64)
                        quad = (drive * b.load_friction_external_quad.value * abs_ext**2
                                + (1.0 - drive) * b.load_friction_motor_quad.value * abs_mot**2)
                        fl = fl + stribeck_coeff * quad
            else:
                gearbox = np.abs(external_torque - motor_torque)
                fl = fl + b.load_friction_base.value * gearbox
                if b.stribeck:
                    fl = fl + stribeck_coeff * b.load_friction_stribeck.value * gearbox
        return fl

    # -- one physics step (bam/mjlab.py:429-525) -----------------------------------------
    def apply(self, q_target: np.ndarray) -> np.ndarray:
        """Compute torques from the (delayed) targets and the current data; write ctrl."""
        assert self._bound
        b, d = self.bam, self.data
        target = self.delay.push(q_target)
        vin = self.cfg.vin
        if self.cfg.vin_drop_gain is not None:
            vin = vin - self.cfg.vin_drop_gain * np.abs(self._prev_motor_torque).sum()
            if self.cfg.vin_min is not None:
                vin = max(vin, self.cfg.vin_min)
        q = d.qpos[self.qpos_adr]
        qd = d.qvel[self.dof_adr]

        duty = np.clip((target - q) * self.kp_fw * self.error_gain, -self.max_pwm, self.max_pwm)
        motor_torque = self.kt * (vin * duty) / self.R - (self.kt**2) * qd / self.R
        if self.cfg.max_current is not None:
            lim = self.cfg.max_current * self.kt
            self.stat_current_limited += np.abs(motor_torque) >= lim
            motor_torque = np.clip(motor_torque, -lim, lim)

        external_torque = -d.qfrc_bias[self.dof_adr]
        abs_vel = np.abs(qd)
        if b.stribeck:
            stribeck_coeff = np.exp(-np.power(abs_vel / b.dtheta_stribeck.value, b.alpha.value))
        else:
            stribeck_coeff = np.zeros_like(qd)
        budget = self._friction_budget(motor_torque, external_torque, stribeck_coeff) + b.friction_viscous.value * abs_vel

        tau_stop = (self.eff_inertia / self.dt) * qd + (motor_torque - external_torque)
        friction = -np.sign(tau_stop) * np.minimum(np.abs(tau_stop), budget)
        out = motor_torque + friction

        self._prev_motor_torque = motor_torque
        self.last_motor_torque = motor_torque
        self.last_output_torque = out
        self.last_target = target
        self.last_vin = vin
        self.stat_peak_torque = np.maximum(self.stat_peak_torque, np.abs(out))
        self.stat_steps += 1
        d.ctrl[self.ctrl_ids] = out
        return out
