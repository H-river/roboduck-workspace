"""MDP pieces for the v2 walk task (rewards, observations, metrics, terrain, actions, symmetry).

Everything here is new code for rl_train; upstream mjlab_microduck is only imported.
See rl_train/NOTES.md §2 for why each piece exists.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from tensordict import TensorDict

_ROBOT = SceneEntityCfg("robot")


# -- rewards --------------------------------------------------------------------------------
def _zero_cmd(env, command_name: str, threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < threshold


def stand_still(env, std: float, command_name: str = "twist", command_threshold: float = 0.01,
                asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Zero command: exp(-(vx^2 + vy^2 + wz^2) / std^2) of the trunk; 0 otherwise.

    The velocity-tracking kernels (std 0.32 / 0.71) barely distinguish 0.1 m/s from standing
    still; this term does, so stopping quickly after a zero command pays off."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_link_lin_vel_b
    w = asset.data.root_link_ang_vel_b
    err = v[:, 0] ** 2 + v[:, 1] ** 2 + w[:, 2] ** 2
    return _zero_cmd(env, command_name, command_threshold).float() * torch.exp(-err / std**2)


def track_yaw_rate(env, std: float, command_name: str = "twist", asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(wz - wz_cmd)^2 / std^2). Yaw only: mjlab's track_angular_velocity also penalises the
    roll / pitch rates, so a tight std there is swamped by normal gait sway (NOTES §2.3)."""
    cmd = env.command_manager.get_command(command_name)
    wz = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-((wz - cmd[:, 2]) ** 2) / std**2)


# -- critic observation ---------------------------------------------------------------------
def foot_force_z(env, sensor_name: str) -> torch.Tensor:
    """Per-foot vertical contact force, sign * log1p(|fz|) (replaces foot_contact_forces in the
    critic: the full world-frame vector cannot be mirrored without the yaw)."""
    f = env.scene[sensor_name].data.force[..., 2]
    return torch.sign(f) * torch.log1p(torch.abs(f))


# -- training-log metrics ---------------------------------------------------------------------
def yaw_symmetry_metrics(env, command_name: str = "twist", asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Logs (env.extras['log']): yaw rate while commanded straight (vx > 0.1, vy = wz = 0),
    turn gains for left / right commands (|wz_cmd| > 0.2) and their ratio, standing speed."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    wz = asset.data.root_link_ang_vel_b[:, 2]
    v = asset.data.root_link_lin_vel_b
    log = env.extras.setdefault("log", {})
    straight = (cmd[:, 0] > 0.1) & (cmd[:, 1].abs() < 1e-3) & (cmd[:, 2].abs() < 1e-3)
    if straight.any():
        log["Metrics/straight_yaw_rate"] = wz[straight].mean()
        log["Metrics/straight_yaw_rate_abs"] = wz[straight].abs().mean()
    left, right = cmd[:, 2] > 0.2, cmd[:, 2] < -0.2
    if left.any() and right.any():
        gl = (wz[left] / cmd[left, 2]).mean()
        gr = (wz[right] / cmd[right, 2]).mean()
        log["Metrics/turn_gain_left"], log["Metrics/turn_gain_right"] = gl, gr
        log["Metrics/turn_gain_ratio"] = gl / gr.clamp(min=1e-3)
    zero = _zero_cmd(env, command_name, 0.01)
    if zero.any():
        log["Metrics/standing_speed"] = (torch.norm(v[zero, :2], dim=1) + wz[zero].abs()).mean()
    return torch.zeros(env.num_envs, device=env.device)


# -- terrain: planar pyramid ------------------------------------------------------------------
@dataclass(kw_only=True)
class HfPlanarPyramidCfg(SubTerrainCfg):
    """Square pyramid with four PLANAR faces (mjlab's HfPyramidSloped is a bilinear tent).

    height = slope * (half - pf/2) * clip((half - max(|x|, |y|)) / (half - pf/2), 0, 1); the
    rim is at 0 so neighbouring flat patches line up. inverted=True gives a pit (rim 0,
    floor at -h). Slope (rise / run) interpolated by difficulty in slope_range."""

    slope_range: tuple[float, float]
    platform_width: float = 1.0
    inverted: bool = False
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.0005

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        body = spec.body("terrain")
        slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])
        n = int(round(self.size[0] / self.horizontal_scale)) + 1
        half = self.size[0] / 2
        xs = np.linspace(-half, half, n)
        d = np.maximum(np.abs(xs)[:, None], np.abs(xs)[None, :])
        run = max(half - self.platform_width / 2, 1e-6)
        frac = np.clip((half - d) / run, 0.0, 1.0)
        h = slope * run
        z = h * frac if not self.inverted else -h * frac
        noise = np.rint(z / self.vertical_scale).astype(np.int32)
        lo, hi = noise.min(), noise.max()
        rng_z = max(hi - lo, 1)
        normalized = (noise - lo) / rng_z
        max_height = rng_z * self.vertical_scale
        field = spec.add_hfield(name=f"hfield_{uuid.uuid4().hex}", size=[half, half, max_height, max(max_height, 0.05)],
                                nrow=n, ncol=n, userdata=normalized.flatten().astype(np.float32).tolist())
        z_off = lo * self.vertical_scale
        material = color_by_height(spec, noise, uuid.uuid4().hex, normalized)
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                             pos=[half, half, z_off], material=material)
        origin = np.array([half, half, (hi if not self.inverted else lo) * self.vertical_scale])
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)])


# -- head takeover during zero commands ---------------------------------------------------------
@dataclass(kw_only=True)
class HeadOverrideJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action whose head targets are sometimes scripted, as the executor will do
    while the robot stands: camera tilted down by U(pitch_down_deg_range) for U(duration_s_range),
    started with rate start_rate_per_s while the command is zero, cancelled when it is not.
    The policy's own head outputs are ignored meanwhile; observations are unchanged (measured
    joint angles, the policy's raw last action) -- same as data_collect_sim's switcher."""

    head_joint_names: tuple[str, ...] = ("neck_pitch", "head_pitch")
    signs: tuple[float, ...] = (-1.0, 1.0)      # camera tilt = -d(neck_pitch) + d(head_pitch)
    pitch_down_deg_range: tuple[float, float] = (15.0, 28.0)
    duration_s_range: tuple[float, float] = (2.0, 4.0)
    start_rate_per_s: float = 0.25
    command_name: str = "twist"
    command_threshold: float = 0.01

    def build(self, env):
        return HeadOverrideJointPositionAction(self, env)


class HeadOverrideJointPositionAction(JointPositionAction):
    def __init__(self, cfg: HeadOverrideJointPositionActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        names = list(self._target_names)
        self._head_local = torch.tensor([names.index(n) for n in cfg.head_joint_names], device=self.device)
        ent = self._entity
        jids = [ent.joint_names.index(n) for n in cfg.head_joint_names]
        lim = ent.data.joint_pos_limits[0, jids]  # (2, 2)
        self._lo, self._hi = lim[:, 0].clone(), lim[:, 1].clone()
        self._d0 = ent.data.default_joint_pos[0, jids].clone()
        self._signs = torch.tensor(cfg.signs, device=self.device, dtype=torch.float32)
        self.remaining = torch.zeros(self.num_envs, device=self.device)
        self.target = self._d0.repeat(self.num_envs, 1)

    def _targets(self, pitch_rad: torch.Tensor) -> torch.Tensor:
        share = torch.stack([pitch_rad / 2, pitch_rad / 2], dim=1)
        for _ in range(2):  # whatever one joint cannot take goes to the other
            q = self._d0 + self._signs * share
            qc = torch.minimum(torch.maximum(q, self._lo), self._hi)
            spill = (q - qc).abs()
            share = share - spill + spill.flip(1)
        return torch.minimum(torch.maximum(self._d0 + self._signs * share, self._lo), self._hi)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        cfg: HeadOverrideJointPositionActionCfg = self.cfg
        dt = self._env.step_dt
        zero = _zero_cmd(self._env, cfg.command_name, cfg.command_threshold)
        self.remaining = torch.where(zero, (self.remaining - dt).clamp(min=0.0), torch.zeros_like(self.remaining))
        start = zero & (self.remaining <= 0) & (torch.rand(self.num_envs, device=self.device) < cfg.start_rate_per_s * dt)
        if start.any():
            k = int(start.sum())
            lo, hi = cfg.pitch_down_deg_range
            pitch = torch.deg2rad(torch.empty(k, device=self.device).uniform_(lo, hi))
            self.target[start] = self._targets(pitch)
            self.remaining[start] = torch.empty(k, device=self.device).uniform_(*cfg.duration_s_range)
        active = self.remaining > 0
        if active.any():
            pa = self._processed_actions.clone()
            pa[:, self._head_local] = torch.where(active[:, None], self.target, pa[:, self._head_local])
            self._processed_actions = pa
        log = self._env.extras.setdefault("log", {})
        log["Metrics/head_override_fraction"] = active.float().mean()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self.remaining[:] = 0.0
        else:
            self.remaining[env_ids] = 0.0


# -- mirror symmetry (rsl_rl symmetry_cfg) -------------------------------------------------------
# Leg joints: q_R = -q_L under the sagittal mirror (rl_train/NOTES §0.5); head: neck/head pitch
# unchanged, head yaw / roll negated.
JOINT_MIRROR = {
    "left_hip_yaw": ("right_hip_yaw", -1), "left_hip_roll": ("right_hip_roll", -1), "left_hip_pitch": ("right_hip_pitch", -1),
    "left_knee": ("right_knee", -1), "left_ankle": ("right_ankle", -1),
    "neck_pitch": ("neck_pitch", 1), "head_pitch": ("head_pitch", 1), "head_yaw": ("head_yaw", -1), "head_roll": ("head_roll", -1),
}
for _l, (_r, _s) in list(JOINT_MIRROR.items()):
    JOINT_MIRROR.setdefault(_r, (_l, _s))

PSEUDO = (-1.0, 1.0, -1.0)   # angular velocity in a frame rotated about y only (body, IMU mount, imu site)
VECTOR = (1.0, -1.0, 1.0)    # true vectors (gravity, linear velocity)
TERM_SIGNS = {
    "base_ang_vel": PSEUDO, "projected_gravity": VECTOR, "base_lin_vel": VECTOR,
    "command": (1.0, -1.0, -1.0),
}
FOOT_PAIR_TERMS = {"foot_height", "foot_air_time", "foot_contact", "foot_force_z"}
JOINT_TERMS = {"joint_pos", "joint_vel", "actions"}

_CACHE: dict[int, dict] = {}


def _joint_perm(names: list[str], device):
    idx = [names.index(JOINT_MIRROR[n][0]) for n in names]
    sign = [float(JOINT_MIRROR[n][1]) for n in names]
    return torch.tensor(idx, device=device), torch.tensor(sign, device=device)


def _group_maps(env, device) -> dict:
    key = id(env)
    if key in _CACHE:
        return _CACHE[key]
    base = env.unwrapped
    joints = list(base.scene["robot"].joint_names)
    jperm, jsign = _joint_perm(joints, device)
    om = base.observation_manager
    maps = {}
    for group, terms in om.active_terms.items():
        idx, sign, off = [], [], 0
        for name, dims in zip(terms, om.group_obs_term_dim[group]):
            n = int(np.prod(dims))
            if name in TERM_SIGNS:
                assert n == 3, (group, name, n)
                idx += [off, off + 1, off + 2]
                sign += list(TERM_SIGNS[name])
            elif name in JOINT_TERMS:
                assert n == len(joints), (group, name, n)
                idx += (off + jperm).tolist()
                sign += jsign.tolist()
            elif name in FOOT_PAIR_TERMS:
                assert n % 2 == 0, (group, name, n)
                h = n // 2  # [left..., right...] per foot
                idx += list(range(off + h, off + n)) + list(range(off, off + h))
                sign += [1.0] * n
            else:
                raise KeyError(f"no mirror rule for observation term {group}/{name}")
            off += n
        maps[group] = (torch.tensor(idx, device=device), torch.tensor(sign, device=device))
    maps["_actions"] = (jperm, jsign)
    _CACHE[key] = maps
    return maps


def mirror_augment(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl_rl data_augmentation_func: returns [original; mirrored] along the batch."""
    device = obs.device if obs is not None else actions.device
    maps = _group_maps(env, device)
    obs_aug = act_aug = None
    if obs is not None:
        out = {}
        for k in obs.keys():
            x = obs[k]
            idx, sign = maps[k]
            out[k] = torch.cat([x, x[:, idx] * sign], dim=0)
        obs_aug = TensorDict(out, batch_size=[obs.batch_size[0] * 2], device=device)
    if actions is not None:
        jperm, jsign = maps["_actions"]
        act_aug = torch.cat([actions, actions[:, jperm] * jsign], dim=0)
    return obs_aug, act_aug
