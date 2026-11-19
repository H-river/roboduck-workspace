"""Walk v2 task configs: upstream Mjlab-Velocity-Microduck + the changes in a YAML config.

Base: mjlab_microduck.tasks.microduck_velocity_env_cfg.make_microduck_velocity_env_cfg (imported,
not copied). Every number that changes lives in rl_train/configs/walk_*.yaml; this file only
wires them in. A config may `inherit:` another one (deep merge).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab_microduck.tasks.mdp import UniformVelocityCommandWithRotationCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckVelocityRlCfg, make_microduck_velocity_env_cfg

from rl_tasks import mdp_v2

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: Path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c.get("inherit"):
        c = deep_merge(load_config(CONFIG_DIR / c["inherit"]), {k: v for k, v in c.items() if k != "inherit"})
    return c


@dataclass(kw_only=True)
class SymPpoCfg(RslRlPpoAlgorithmCfg):
    """mjlab's PPO cfg plus rsl_rl's symmetry_cfg (rsl_rl/algorithms/ppo.py:58)."""

    symmetry_cfg: dict[str, Any] | None = None
    entropy_coef: float = 0.01


def _command(old, c: dict) -> UniformVelocityCommandWithRotationCfg:
    kw = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)}
    cmd = UniformVelocityCommandWithRotationCfg(**kw)
    cmd.ranges = copy.deepcopy(old.ranges)
    cmd.ranges.lin_vel_x = tuple(c["lin_vel_x"])
    cmd.ranges.lin_vel_y = tuple(c["lin_vel_y"])
    cmd.ranges.ang_vel_z = tuple(c["ang_vel_z"])
    cmd.rel_standing_envs = c["rel_standing_envs"]
    cmd.rel_rotation_envs = c["rel_rotation_envs"]
    cmd.rotation_env_ang_vel_range = tuple(c["rotation_env_ang_vel_range"])
    cmd.rotation_min_ang_vel = c["rotation_min_ang_vel"]
    cmd.rel_forward_envs = c["rel_forward_envs"]
    cmd.resampling_time_range = tuple(c["resampling_time_range"])
    return cmd


def make_env_cfg(c: dict, play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.episode_length_s = c["train"].get("episode_length_s", cfg.episode_length_s)
    # upstream SCENE_CFG has num_envs = 1 (cfg:93); training size comes from the config
    cfg.scene.num_envs = 1 if play else int(c["train"]["num_envs"])
    # per-world contact / constraint buffers (mujoco_warp's heuristic grows them ~3x / 16x as soon as
    # the model has a height field, which does not fit 4096 envs in 12 GB; HANDOFF_NOTES step 1)
    for k in ("nconmax", "njmax"):
        if (c.get("sim") or {}).get(k) is not None:
            setattr(cfg.sim, k, int(c["sim"][k]))

    # commands: a real cfg object instead of the upstream lambda (rl_train/NOTES §0.2)
    cfg.commands["twist"] = _command(cfg.commands["twist"], c["commands"])
    if play:
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_rotation_envs = 0.0

    # rewards: weights / params from the config; new terms by name
    new_terms = {"stand_still": mdp_v2.stand_still, "track_yaw_rate": mdp_v2.track_yaw_rate}
    for name, r in (c.get("rewards") or {}).items():
        if name in cfg.rewards:
            if "weight" in r:
                cfg.rewards[name].weight = r["weight"]
            cfg.rewards[name].params.update(r.get("params", {}))
        else:
            cfg.rewards[name] = RewardTermCfg(func=new_terms[name], weight=r["weight"], params=dict(r.get("params", {})))
    cfg.curriculum = {}  # the staged curriculum is replaced by the config's fixed values

    # critic: vertical foot forces instead of the 3-D world-frame vector (mirrorable)
    crit = cfg.observations["critic"].terms
    if "foot_contact_forces" in crit:
        del crit["foot_contact_forces"]
        crit["foot_force_z"] = ObservationTermCfg(func=mdp_v2.foot_force_z, params={"sensor_name": "feet_ground_contact"})

    cfg.metrics["yaw_symmetry"] = MetricsTermCfg(func=mdp_v2.yaw_symmetry_metrics, params={})

    # actuator randomization
    a = c.get("actuator") or {}
    if a:
        robot = cfg.scene.entities["robot"]
        act = robot.articulation.actuators[0]
        kw = {}
        if "vin_range" in a:
            kw["vin_range"] = tuple(a["vin_range"])
        if "vin_drop_gain_range" in a:
            kw["vin_drop_gain_range"] = tuple(a["vin_drop_gain_range"])
        if "delay_steps" in a:
            kw["delay_min_lag"], kw["delay_max_lag"] = a["delay_steps"]
        robot.articulation = dataclasses.replace(robot.articulation, actuators=(dataclasses.replace(act, **kw),))

    # terrain
    t = c.get("terrain") or {"type": "plane"}
    if t["type"] == "slopes":
        s_max = math.tan(math.radians(t["max_slope_deg"]))
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=tuple(t["patch_size_m"]), border_width=t["border_width_m"], num_rows=t["num_rows"], num_cols=t["num_cols"],
            curriculum=False, difficulty_range=(0.0, 1.0), add_lights=True,
            sub_terrains={
                "flat": BoxFlatTerrainCfg(proportion=t["proportions"]["flat"]),
                "pyramid": mdp_v2.HfPlanarPyramidCfg(proportion=t["proportions"]["pyramid"], slope_range=(0.0, s_max),
                                                     platform_width=t["platform_width_m"]),
                "pyramid_inv": mdp_v2.HfPlanarPyramidCfg(proportion=t["proportions"]["pyramid_inv"], slope_range=(0.0, s_max),
                                                         platform_width=t["platform_width_m"], inverted=True),
            },
        )

    # fall check relative to the terrain under the trunk (upstream's is an absolute height, which is
    # wrong as soon as the terrain is not at z = 0; mdp_v2.trunk_clearance_below_minimum)
    fo = (c.get("terminations") or {}).get("fell_over")
    if fo and fo["type"] == "trunk_clearance":
        cfg.scene.sensors = tuple(cfg.scene.sensors) + (TerrainHeightSensorCfg(
            name="trunk_height_scan", frame=(ObjRef(type="body", name=fo["body"], entity="robot"),),
            pattern=RingPatternCfg.single_ring(radius=fo["ray_radius_m"], num_samples=fo["rays"]), ray_alignment="yaw",
            max_distance=fo["max_distance_m"], exclude_parent_body=True, include_geom_groups=(0,), reduction="mean",
            debug_vis=False),)
        cfg.terminations["fell_over"] = TerminationTermCfg(
            func=mdp_v2.trunk_clearance_below_minimum,
            params={"minimum_height": fo["minimum_height"], "sensor_name": "trunk_height_scan"})

    # head takeover while standing
    h = c.get("head_override")
    if h and not play:
        old = cfg.actions["joint_pos"]
        kw = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)}
        cfg.actions["joint_pos"] = mdp_v2.HeadOverrideJointPositionActionCfg(
            **kw, pitch_down_deg_range=tuple(h["pitch_down_deg_range"]), duration_s_range=tuple(h["duration_s_range"]),
            start_rate_per_s=h["start_rate_per_s"])
    return cfg


def make_rl_cfg(c: dict):
    rl = copy.deepcopy(MicroduckVelocityRlCfg)
    tr = c["train"]
    rl.max_iterations = tr["max_iterations"]
    rl.save_interval = tr["save_interval"]
    rl.experiment_name = tr["experiment_name"]
    rl.logger = "tensorboard"
    rl.upload_model = False
    sym = c.get("symmetry")
    alg = dataclasses.asdict(rl.algorithm)
    alg.pop("class_name", None)
    if sym and (sym.get("use_data_augmentation") or sym.get("use_mirror_loss")):
        alg["symmetry_cfg"] = {"use_data_augmentation": bool(sym.get("use_data_augmentation")),
                               "use_mirror_loss": bool(sym.get("use_mirror_loss")),
                               "mirror_loss_coeff": float(sym.get("mirror_loss_coeff", 0.0)),
                               "data_augmentation_func": mdp_v2.mirror_augment}
    rl.algorithm = SymPpoCfg(**alg)
    return rl
