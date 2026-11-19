"""Simulation runtime: one sim clock for the executor (10 Hz), the policy (its own rate)
and the head camera (15 fps).

Per control tick (mjlab order, NOTES §0.2): read state -> switcher -> `decimation` physics
steps (camera frames are rendered at the first physics step at/after their scheduled time,
from a kinematics-only copy) -> one mj_forward -> on_tick(state, frame, command).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from duckrt.backend_sim import SimBackend
from duckrt.messages import CommandMsg, FrameMsg, Mode, StateMsg
from duckrt.policy import OnnxPolicy
from duckrt.policy_card import load_card
from duckrt.switcher import Switcher

log = logging.getLogger("duckrt.runtime")

OnTick = Callable[[StateMsg, FrameMsg | None, CommandMsg | None], None]


class SimRuntime:
    def __init__(
        self,
        sim_cfg: dict,
        roles: dict[str, str | None] | None = None,
        layout: dict | None = None,
        camera_cfg: dict | None = None,
        render_head: bool = False,
        seed: int = 0,
        on_tick: OnTick | None = None,
        extra_spec_fn=None,
    ):
        self.cfg = sim_cfg
        self.backend = SimBackend(sim_cfg, layout=layout, camera_cfg=camera_cfg, seed=seed, extra_spec_fn=extra_spec_fn)
        # role (walk_flat / stand / kick) -> policy; sim.yaml runtime.policies, overridable
        roles = dict(sim_cfg["runtime"]["policies"] if roles is None else roles)
        self.policies = {role: OnnxPolicy(load_card(card)) for role, card in roles.items() if card}
        m = self.backend.model
        ranges = m.jnt_range[[m.joint(n).id for n in self.backend.joint_names]].copy()
        self.switcher = Switcher(sim_cfg, self.policies, joint_ranges=ranges)
        self.camera = None
        if camera_cfg is not None:
            from duckrt.camera import HeadCamera

            self.camera = HeadCamera(camera_cfg, self.backend)
        self.render_head = render_head and self.camera is not None
        self.on_tick = on_tick
        self.upper_period = 1.0 / float(sim_cfg["runtime"]["upper_hz"])
        od = sim_cfg.get("observation", {}).get("delay_steps", {})
        self.obs_delay_imu = int(od.get("imu", 0))
        self.obs_delay_jvel = int(od.get("joint_vel", 0))
        self.reset(seed)

    def reset(self, seed: int | None = None, start=None) -> None:
        self.backend.reset(seed, start=start)
        self.switcher.reset()
        self.last_cmd_msg: CommandMsg | None = None
        self.last_frame: FrameMsg | None = None
        self.next_frame_t = 0.0
        self.ticks = 0
        self.frames = 0
        self.seq = 0
        self._state_hist: list[StateMsg] = []

    @property
    def t(self) -> float:
        return self.backend.time

    def decimation(self, policy: OnnxPolicy) -> int:
        n = policy.period / self.backend.physics_dt
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"{policy.card.name}: period {policy.period} is not a multiple of dt")
        return int(round(n))

    # -- executor side --------------------------------------------------------------------
    def submit(self, msg: CommandMsg) -> None:
        self.last_cmd_msg = msg
        self.switcher.submit(msg, self.t)

    def send(self, mode: Mode, cmd=(0.0, 0.0, 0.0), head_pitch_down_deg: float | None = None,
             kick_ball_xy: tuple[float, float] | None = None) -> CommandMsg:
        """Convenience: build and submit a CommandMsg stamped with the sim clock."""
        self.seq += 1
        msg = CommandMsg(t_send=self.t, mode=mode, cmd=tuple(float(c) for c in cmd), seq=self.seq,
                         head_pitch_down_deg=head_pitch_down_deg,
                         kick_ball_xy=None if kick_ball_xy is None else tuple(float(v) for v in kick_ball_xy))
        self.submit(msg)
        return msg

    # -- one control tick -----------------------------------------------------------------
    def read_state(self) -> StateMsg:
        """Robot-side measurements. Optional sensor latency (sim.yaml observation.delay_steps,
        in control ticks): IMU quat/gyro/acc and joint velocities from that many ticks ago."""
        s = self.backend.read_state(self.switcher.mode, self.switcher.fall_flag)
        dmax = max(self.obs_delay_imu, self.obs_delay_jvel)
        if dmax == 0:
            return s
        self._state_hist.append(s)
        del self._state_hist[:-(dmax + 1)]
        old_imu = self._state_hist[max(0, len(self._state_hist) - 1 - self.obs_delay_imu)]
        old_jv = self._state_hist[max(0, len(self._state_hist) - 1 - self.obs_delay_jvel)]
        return StateMsg(t=s.t, imu_quat=old_imu.imu_quat, gyro=old_imu.gyro, acc=old_imu.acc,
                        joint_pos=s.joint_pos, joint_vel=old_jv.joint_vel, mode=s.mode, fall_flag=s.fall_flag)

    def tick(self) -> tuple[StateMsg, FrameMsg | None]:
        b = self.backend
        state = self.read_state()
        targets = self.switcher.tick(b.time, state.imu_quat, state.gyro, state.joint_pos, state.joint_vel)
        frame = None
        for _ in range(self.decimation(self.switcher.active)):
            if self.render_head and b.time >= self.next_frame_t - 1e-9:
                frame = self.camera.capture()
                self.last_frame = frame
                self.frames += 1
                self.next_frame_t += 1.0 / self.camera.fps
            b.step_physics(targets)
        b.forward()
        self.ticks += 1
        if self.on_tick is not None:
            self.on_tick(state, frame, self.last_cmd_msg)
        return state, frame

    def run(
        self,
        seconds: float,
        command_fn: Callable[[float], tuple[Mode, Any]] | None = None,
        per_tick: Callable[["SimRuntime"], None] | None = None,
    ) -> None:
        """Advance `seconds` of sim time. command_fn(t) -> (mode, cmd) is polled at upper_hz."""
        t_end = self.t + seconds
        next_cmd = self.t
        while self.t < t_end - 1e-9:
            if command_fn is not None and self.t >= next_cmd - 1e-9:
                mode, cmd = command_fn(self.t)
                self.send(mode, cmd)
                next_cmd += self.upper_period
            self.tick()
            if per_tick is not None:
                per_tick(self)
