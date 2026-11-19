"""Mode switcher: which policy drives the robot, with what command.

Runs on the robot side (same code in sim and on hardware). Inputs: CommandMsg from the
executor layer and the robot's own measurements; output: joint targets every control tick.

* WALK_FLAT  walk policy with the (clipped) velocity command.
* STAND      the `stand` policy with zero command if one is registered (configs/sim.yaml
             runtime.policies), else the walk policy with zero command.
* KICK       one-shot: stand (as STAND) for `settle_s` -> play the kick policy for
             `duration_s` -> stand `settle_s` -> back to the previous mode. New commands are
             ignored for the whole sequence.
* other modes are registered as not implemented: warning, mode kept, command not applied
  (and the message does not feed the watchdog).

Watchdog: no accepted CommandMsg for `watchdog_s` -> velocity command zeroed, mode kept.
Fall flag: body up-vector z (from the IMU) below `up_z_min` for `hold_s`.
Head takeover: in STAND, CommandMsg.head_pitch_down_deg replaces the policy's targets for
neck_pitch / head_pitch with scripted ones (the policy keeps seeing its own last action and the
measured head angles, as in rl_train's training wrapper); ignored with a warning in other modes.
Tilt guard (as microduck/src/moves/walk.py:101-107): past `tilt_guard_up_z`, skip the
policy and hold default_q.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from duckrt.messages import CommandMsg, Mode
from duckrt.policy import OnnxPolicy
from duckrt.upstream import imu_quat_to_body

log = logging.getLogger("duckrt.switcher")

IMPLEMENTED_BASE = {Mode.WALK_FLAT, Mode.STAND}


@dataclass
class KickSequence:
    phase: str            # settle_before | play | settle_after
    phase_t0: float
    return_mode: Mode


def body_up_z(imu_quat: np.ndarray) -> float:
    w, x, y, z = imu_quat_to_body(tuple(float(v) for v in imu_quat))
    return 1.0 - 2.0 * (x * x + y * y)


class Switcher:
    def __init__(self, sim_cfg: dict, policies: dict[str, OnnxPolicy], joint_ranges: np.ndarray | None = None):
        """joint_ranges: (n_joints, 2) in the cards' joint order (head takeover clipping)."""
        self.joint_ranges = joint_ranges
        rt = sim_cfg["runtime"]
        self.watchdog_s = float(rt["watchdog_s"])
        self.fall_up_z = float(rt["fall"]["up_z_min"])
        self.fall_hold_s = float(rt["fall"]["hold_s"])
        self.tilt_guard_up_z = float(rt["tilt_guard_up_z"])
        self.turn_eps = float(rt["turn_in_place_eps"])
        self.head_cfg = rt.get("head_override")
        self.policies = policies
        if "walk_flat" not in policies:
            raise ValueError("walk_flat policy is required")
        self.implemented = set(IMPLEMENTED_BASE)
        if "kick" in policies:
            self.implemented.add(Mode.KICK)
        self.reset()

    # -- state ----------------------------------------------------------------------------
    def reset(self, mode: Mode = Mode.STAND) -> None:
        self.mode = mode
        self.cmd = np.zeros(3)
        self.raw_cmd = np.zeros(3)
        self.last_cmd_time: float | None = None
        self.last_seq = -1
        self.watchdog_tripped = False
        self.fall_flag = False
        self._tilt_since: float | None = None
        self.kick: KickSequence | None = None
        self.active: OnnxPolicy | None = None
        self.guard_active = False
        self.head_pitch_down_deg: float | None = None
        self.kick_ball_xy: tuple[float, float] | None = None
        self.events: list[tuple[float, str]] = []
        for p in self.policies.values():
            p.reset()

    def _event(self, t: float, msg: str, level=logging.INFO) -> None:
        self.events.append((t, msg))
        log.log(level, "t=%.3f %s", t, msg)

    # -- commands -------------------------------------------------------------------------
    def clip_command(self, cmd: np.ndarray, policy: OnnxPolicy) -> np.ndarray:
        r, rules = policy.card.command_range, policy.card.command_rules
        vx = float(np.clip(cmd[0], *r["vx"]))
        vy = float(np.clip(cmd[1], *r["vy"]))
        wz_lo, wz_hi = r["wz"]
        if "turn_in_place_wz_max" in rules and abs(vx) < self.turn_eps and abs(vy) < self.turn_eps:
            wz_lo, wz_hi = -rules["turn_in_place_wz_max"], rules["turn_in_place_wz_max"]
        return np.array([vx, vy, float(np.clip(cmd[2], wz_lo, wz_hi))])

    def submit(self, msg: CommandMsg, now: float) -> None:
        if msg.seq <= self.last_seq:
            self._event(now, f"stale CommandMsg seq {msg.seq} <= {self.last_seq} ignored", logging.WARNING)
            return
        self.last_seq = msg.seq
        mode = Mode(msg.mode)
        if self.kick is not None:
            return  # one-shot behaviour in progress: commands ignored
        if mode not in self.implemented:
            self._event(now, f"mode {mode.value} not implemented; keeping {self.mode.value}", logging.WARNING)
            return
        self.last_cmd_time = now
        if self.watchdog_tripped:
            self._event(now, "watchdog cleared")
            self.watchdog_tripped = False
        hp = msg.head_pitch_down_deg
        if hp is not None and mode != Mode.STAND:
            self._event(now, f"head takeover only in STAND; ignored in {mode.value}", logging.WARNING)
            hp = None
        if (hp is None) != (self.head_pitch_down_deg is None):
            self._event(now, f"head takeover {'on (%.1f deg)' % hp if hp is not None else 'off'}")
        self.head_pitch_down_deg = hp
        if mode == Mode.KICK:
            self.kick_ball_xy = msg.kick_ball_xy
            self.raw_cmd = np.zeros(3)
            self.kick = KickSequence("settle_before", now, return_mode=self.mode if self.mode != Mode.KICK else Mode.STAND)
            self._event(now, f"KICK requested from {self.mode.value}: settle_before")
            self.mode = Mode.KICK
            return
        if mode != self.mode:
            self._event(now, f"mode {self.mode.value} -> {mode.value}")
            self.mode = mode
        self.raw_cmd = np.asarray(msg.cmd, dtype=np.float64)

    # -- per control tick -----------------------------------------------------------------
    def _select(self, now: float) -> tuple[OnnxPolicy, np.ndarray]:
        walk = self.policies["walk_flat"]
        stand = self.policies.get("stand", walk)
        if self.kick is not None:
            k = self.kick
            kick_pol = self.policies["kick"]
            settle = float(kick_pol.card.handoff.get("settle_s", 0.5))
            duration = kick_pol.card.duration_s or kick_pol.card.task.get("episode_length_s", 5.0)
            el = now - k.phase_t0
            if k.phase == "settle_before" and el >= settle:
                k.phase, k.phase_t0 = "play", now
                self._event(now, f"KICK play ({duration:.2f} s)")
            elif k.phase == "play" and el >= duration:
                k.phase, k.phase_t0 = "settle_after", now
                self._event(now, "KICK settle_after")
            elif k.phase == "settle_after" and el >= settle:
                self._event(now, f"KICK done -> {k.return_mode.value}")
                self.mode = k.return_mode
                self.kick = None
                self.raw_cmd = np.zeros(3)
                self.last_cmd_time = now
                return (stand if k.return_mode == Mode.STAND else walk), np.zeros(3)
            if k.phase == "play":
                return kick_pol, np.zeros(3)
            return stand, np.zeros(3)
        if self.mode == Mode.STAND:
            return stand, np.zeros(3)
        return walk, self.raw_cmd

    def tick(self, now: float, imu_quat, gyro, joint_pos, joint_vel) -> np.ndarray:
        # watchdog
        if self.kick is None and self.last_cmd_time is not None and now - self.last_cmd_time > self.watchdog_s:
            if not self.watchdog_tripped:
                self._event(now, f"watchdog: no command for {now - self.last_cmd_time:.3f} s -> zero velocity", logging.WARNING)
                self.watchdog_tripped = True
            self.raw_cmd = np.zeros(3)

        # fall flag / tilt guard from the IMU
        up_z = body_up_z(imu_quat)
        if up_z < self.fall_up_z:
            self._tilt_since = now if self._tilt_since is None else self._tilt_since
            if not self.fall_flag and now - self._tilt_since >= self.fall_hold_s - 1e-9:
                self.fall_flag = True
                self._event(now, f"fall_flag set (up_z {up_z:.2f})", logging.WARNING)
        else:
            self._tilt_since = None
            if self.fall_flag:
                self._event(now, "fall_flag cleared")
            self.fall_flag = False

        policy, cmd = self._select(now)
        if policy is not self.active:
            policy.reset()
            if self.active is not None:
                self._event(now, f"policy {self.active.card.name} -> {policy.card.name} (history reset)")
            self.active = policy
        self.cmd = self.clip_command(cmd, policy)

        self.guard_active = up_z < self.tilt_guard_up_z
        if self.guard_active:
            return policy.default_q.copy()
        if self.kick is not None and self.kick.phase == "play" and "kick_ball_xy" in policy.extra_obs:
            policy.set_extra("kick_ball_xy", self.kick_ball_xy)
        targets = policy.act(imu_quat, gyro, joint_pos, joint_vel, self.cmd)
        if self.head_pitch_down_deg is not None and self.mode == Mode.STAND and self.kick is None:
            targets = self.apply_head_override(targets, policy, self.head_pitch_down_deg)
        return targets

    def apply_head_override(self, targets: np.ndarray, policy: OnnxPolicy, pitch_down_deg: float) -> np.ndarray:
        """Scripted head targets: tilt the camera down by pitch_down_deg from the default pose."""
        hc = self.head_cfg
        names = policy.card.joint_names
        x = np.radians(min(max(pitch_down_deg, 0.0), hc["max_deg"]))
        idx = [names.index(j) for j in hc["joints"]]
        lims = self.joint_ranges[idx]
        d0 = policy.default_q[idx]
        out = targets.copy()
        share = np.array([x / 2, x / 2])
        for _ in range(2):  # move what one joint cannot take to the other
            q = d0 + np.asarray(hc["signs"]) * share
            qc = np.clip(q, lims[:, 0], lims[:, 1])
            spill = np.abs(q - qc)
            if spill.sum() < 1e-12:
                break
            share = share - spill + spill[::-1]
        out[idx] = np.clip(d0 + np.asarray(hc["signs"]) * share, lims[:, 0], lims[:, 1])
        return out

    @property
    def period(self) -> float:
        p = self.active or self.policies["walk_flat"]
        return p.period
