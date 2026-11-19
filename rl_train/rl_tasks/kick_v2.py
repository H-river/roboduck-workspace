"""Kick v2 task configs.

Derived from /home/jiamo/microduck_ws/duck_soccer/training/ai_fange_kick/af_kick_cfg.py
(sha256 bf31e9660b134892bd44a9427d3dda32ea119c5be3d6624ce54838eb3940b938, the stage-1 kick;
not a git checkout). Same structure and stage-1 terms; the v2 changes (rl_train/NOTES.md §3)
are driven by rl_train/configs/kick_*.yaml:
  * robot: allcollisions geometry with the WALK joint ranges (hip_yaw +-0.175, as the runtime),
  * start: walk v2 standing pose (+ offsets from the config), part of the episodes with the head
    just released from a takeover (pitched down 15-28 deg),
  * ball: x / y ranges in the trunk yaw frame (one foot), physics from the config,
  * actor observation + kick_ball_xy: ball (x, y) at the start + noise, constant for the episode,
  * reward + lateral ball speed penalty (direction), + standing still after the kick.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import mjlab_microduck.robot.microduck_constants as AF
from mjlab_microduck.tasks.mdp import mounted_body_ang_vel, mounted_body_projected_gravity
from mjlab_microduck.tasks.microduck_velocity_env_cfg import BNO080_MOUNT_QUAT, MICRODUCK_IMU_BODY, SIM_CFG
from rl_tasks import pollen_kick_mdp as kick
from rl_tasks.walk_v2 import CONFIG_DIR

ROBOT_DIR = Path(AF.__file__).resolve().parent / "microduck"
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]
HEAD_BODY_NAMES = ("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)")


def robot_spec() -> mujoco.MjSpec:
    """allcollisions geometry, walk joint ranges (data_collect_sim does the same)."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "robot_allcollisions.xml"))
    ref = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "robot_walk.xml"))
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.range = ref.jnt_range[ref.joint(j.name).id].copy()
    return spec


def ball_spec_fn(b: dict):
    def fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(CONFIG_DIR / "ball_kick.xml"))
        body = spec.body("ball")
        g = body.geoms[0]
        r = float(b["radius_m"])
        g.size = [r, 0, 0]
        g.friction = list(b["friction"])
        body.mass = float(b["mass_kg"])
        i = 2.0 / 3.0 * b["mass_kg"] * r * r  # thin hollow sphere (ball.xml's choice)
        body.inertia = [i, i, i]
        body.pos = [0.3, 0.0, r]
        return spec
    return fn


# -- v2 MDP terms -------------------------------------------------------------------------
def _store(env, name: str, shape) -> torch.Tensor:
    if not hasattr(env, name):
        setattr(env, name, torch.zeros(env.num_envs, *shape, device=env.device))
    return getattr(env, name)


def reset_ball_v2(env, env_ids, x_range, y_range, obs_noise_xy, ball_radius, asset_name="ball"):
    """Ball at (x, y) ~ U(x_range) x U(y_range) in the trunk's yaw frame; store the kick direction
    and the ball observation (x, y) + U(+-obs_noise) -- constant for the episode (the executor
    measures it once before the kick)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot, ball = env.scene["robot"], env.scene[asset_name]
    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = torch.cos(yaw), torch.sin(yaw)
    n = len(env_ids)
    off = torch.stack([torch.empty(n, device=env.device).uniform_(*x_range),
                       torch.empty(n, device=env.device).uniform_(*y_range)], dim=1)
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    pose[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)
    kd = kick._ball_kick_dir(env)
    kd[env_ids, 0], kd[env_ids, 1] = c, s
    _store(env, "_kick_ball_obs", (2,))[env_ids] = off + (torch.rand(n, 2, device=env.device) * 2 - 1) * obs_noise_xy


def kick_ball_xy(env) -> torch.Tensor:
    """Actor observation: the ball (x, y) measured at the start (see reset_ball_v2)."""
    return _store(env, "_kick_ball_obs", (2,)).clone()


def ball_lateral_speed(env, asset_name: str = "ball") -> torch.Tensor:
    """|ball velocity perpendicular to the kick direction| (use with a negative weight)."""
    v = env.scene[asset_name].data.root_link_lin_vel_w[:, :2]
    kd = kick._ball_kick_dir(env)
    return torch.abs(v[:, 0] * -kd[:, 1] + v[:, 1] * kd[:, 0])


def still_after(env, start_s: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """After start_s into the episode: exp(-(vx^2 + vy^2 + wz^2) / std^2) of the trunk."""
    a = env.scene[asset_cfg.name]
    v, w = a.data.root_link_lin_vel_b, a.data.root_link_ang_vel_b
    on = (env.episode_length_buf.float() * env.step_dt) >= start_s
    return on.float() * torch.exp(-(v[:, 0] ** 2 + v[:, 1] ** 2 + w[:, 2] ** 2) / std**2)


def offset_joints(env, env_ids, offsets: dict, head_takeover_prob: float, pitch_down_deg_range):
    """After the standing reset: move toward the walk-v2 standing pose (offsets x U(0, 1); the stage-1
    kick only fires from near HOME, NOTES §3.0); for a fraction of envs put
    the head in a takeover pose (camera pitched down; neck_pitch -x/2, head_pitch +x/2, clipped)."""
    if env_ids is None or len(env_ids) == 0:
        return
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    q = robot.data.joint_pos[env_ids].clone()
    blend = torch.rand(len(env_ids), device=env.device)  # anywhere between HOME and the walk-v2 standing pose
    for n, dq in offsets.items():
        q[:, names.index(n)] += blend * dq
    take = torch.rand(len(env_ids), device=env.device) < head_takeover_prob
    if take.any():
        x = torch.deg2rad(torch.empty(int(take.sum()), device=env.device).uniform_(*pitch_down_deg_range))
        jn, jh = names.index("neck_pitch"), names.index("head_pitch")
        lim = robot.data.joint_pos_limits[0]
        d0 = robot.data.default_joint_pos[0]
        qn = (d0[jn] - x / 2)
        qh = (d0[jh] + x / 2)
        spill_n = (lim[jn, 0] - qn).clamp(min=0)
        spill_h = (qh - lim[jh, 1]).clamp(min=0)
        q[take, jn] = (qn - spill_h).clamp(lim[jn, 0], lim[jn, 1])
        q[take, jh] = (qh + spill_n).clamp(lim[jh, 0], lim[jh, 1])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


# -- config ---------------------------------------------------------------------------------
def make_kick_env_cfg(c: dict, play: bool = False):
    k = c["kick"]
    support_foot = "left" if k["foot"] == "right" else "right"
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce",
        num_slots=1, track_air_time=True)
    support_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^{support_foot}_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="netforce", num_slots=1)
    self_cfg = ContactSensorCfg(
        name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"), fields=("found",),
        reduce="none", num_slots=1)

    robot_cfg = EntityCfg(spec_fn=robot_spec, init_state=AF.HOME_FRAME, collisions=(AF.FULL_COLLISION,),
                          articulation=EntityArticulationInfoCfg(actuators=(copy.deepcopy(AF.actuators),),
                                                                 soft_joint_pos_limit_factor=0.9))
    ball = c["ball"]
    ball_cfg = EntityCfg(spec_fn=ball_spec_fn(ball), init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, ball["radius_m"])))

    cfg = make_velocity_env_cfg()
    cfg.sim = copy.deepcopy(SIM_CFG)
    cfg.scene.entities = {"robot": robot_cfg, "ball": ball_cfg}
    cfg.scene.sensors = (feet_ground_cfg, support_cfg, self_cfg)
    cfg.scene.num_envs = 1 if play else int(c["train"]["num_envs"])
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = c["train"]["episode_length_s"]
    cfg.sim.nconmax = 50
    cfg.sim.njmax = 1500

    a = cfg.actions["joint_pos"]
    assert isinstance(a, JointPositionActionCfg)
    a.scale = 1.0

    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                 "foot_swing_height", "foot_slip", "pose", "soft_landing"]:
        cfg.rewards.pop(name, None)
    R = c["rewards"]
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(func=kick.ball_forward_velocity, weight=R["ball_forward_velocity"],
                                                         params={"asset_name": "ball", "max_speed": k["target_speed"]})
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(func=kick.ball_speed_overshoot_penalty, weight=R["ball_speed_overshoot"],
                                                        params={"asset_name": "ball", "target_speed": k["target_speed"]})
    cfg.rewards["ball_lateral_speed"] = RewardTermCfg(func=ball_lateral_speed, weight=R["ball_lateral_speed"],
                                                      params={"asset_name": "ball"})
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(func=kick.single_foot_grounded_reward, weight=R["support_foot_grounded"],
                                                         params={"sensor_name": support_cfg.name})
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(func=kick.pose_target_match, weight=R["pose_stand_legs"],
                                                   params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(func=kick.pose_target_match, weight=R["pose_stand_neck"],
                                                   params={"std": 0.3, "joint_indices": _NECK_JOINTS, "target_overrides": None})
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = R["upright"]
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(func=kick.height_target_gaussian, weight=R["height_stand"],
                                                params={"std": 0.04, "target_height": k["stand_z"],
                                                        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["still_after_kick"] = RewardTermCfg(func=still_after, weight=R["still_after_kick"],
                                                    params={"start_s": k["still_after_s"], "std": 0.05})
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_cfg.name})

    # observations: walk layout (51) + kick_ball_xy (2); IMU terms as in the walk task
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    imu = SceneEntityCfg("robot", body_names=(MICRODUCK_IMU_BODY,))
    for grp in ("actor", "critic"):
        g = cfg.observations[grp].terms["projected_gravity"] = copy.deepcopy(cfg.observations[grp].terms["projected_gravity"])
        g.func, g.params = mounted_body_projected_gravity, {"mount_quat": BNO080_MOUNT_QUAT, "asset_cfg": imu}
        av = cfg.observations[grp].terms["base_ang_vel"] = copy.deepcopy(cfg.observations[grp].terms["base_ang_vel"])
        av.func, av.params = mounted_body_ang_vel, {"mount_quat": BNO080_MOUNT_QUAT, "asset_cfg": imu}
    act = cfg.observations["actor"].terms
    act["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    act["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    for t in ("base_ang_vel", "projected_gravity"):
        act[t].delay_min_lag, act[t].delay_max_lag, act[t].delay_update_period = 0, 1, 64
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = copy.deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = copy.deepcopy(passive_excluded)
    act["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    act["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
    act["joint_vel"].delay_min_lag, act["joint_vel"].delay_max_lag = 0, 1
    cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
    act["joint_pos"].params["biased"] = True
    cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    act["kick_ball_xy"] = ObservationTermCfg(func=kick_ball_xy)
    cfg.observations["critic"].terms["kick_ball_xy"] = ObservationTermCfg(func=kick_ball_xy)
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(func=kick.ball_pos_in_base, params={"asset_name": "ball"})
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(func=kick.ball_vel_in_base, params={"asset_name": "ball"})

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = kick.VelocityCommandCommandOnlyCfg(**vars(command))

    cfg.terminations["nan_state"] = TerminationTermCfg(func=kick.robot_state_is_nan, time_out=False)

    cfg.events["reset_action_history"] = EventTermCfg(func=kick.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot_collision", "right_foot_collision")
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=kick.set_random_ground_state, mode="reset",
        params={"face_down_prob": 0.0, "face_up_prob": 0.0, "sitting_prob": 0.0, "standing_prob": 1.0,
                "sitting_tilt_max": math.radians(5), "standing_z_min": 0.11, "standing_z_max": 0.12})
    st = c["start"]
    cfg.events["offset_joints"] = EventTermCfg(
        func=offset_joints, mode="reset",
        params={"offsets": dict(st.get("joint_offsets") or {}), "head_takeover_prob": st["head_takeover_prob"],
                "pitch_down_deg_range": tuple(st["pitch_down_deg_range"])})
    cfg.events["reset_ball"] = EventTermCfg(
        func=reset_ball_v2, mode="reset",
        params={"x_range": tuple(k["ball_x_range"]), "y_range": tuple(k["ball_y_range"]),
                "obs_noise_xy": k["ball_obs_noise_m"], "ball_radius": ball["radius_m"], "asset_name": "ball"})
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, "asset_cfg": SceneEntityCfg("robot")})
    cfg.events["randomize_com"] = EventTermCfg(
        func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                                                 "operation": "add", "ranges": (-0.003, 0.003)})
    cfg.events["randomize_head_com"] = EventTermCfg(
        func=dr.body_ipos, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                                                 "operation": "add", "ranges": (-0.003, 0.003)})
    cfg.events["randomize_armature"] = EventTermCfg(
        func=dr.joint_armature, mode="reset", params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                                                      "operation": "scale", "ranges": (0.9, 1.1)})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=kick.reward_weight,
        params={"reward_name": "action_rate_l2",
                "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2}, {"step": 750 * 24, "weight": -0.4},
                                  {"step": 1000 * 24, "weight": -0.6}, {"step": 1250 * 24, "weight": -0.8},
                                  {"step": 1500 * 24, "weight": -1.0}]})
    cfg.curriculum["com_range"] = CurriculumTermCfg(
        func=kick.com_range_curriculum,
        params={"event_name": "randomize_com", "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                                                                {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=kick.push_curriculum,
        params={"event_name": "push_robot",
                "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                                {"step": 500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                                {"step": 1000 * 24, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}]})
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def make_kick_rl_cfg(c: dict):
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    tr = c["train"]
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
                                       num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                       gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
        experiment_name=tr["experiment_name"], save_interval=tr["save_interval"], num_steps_per_env=24,
        max_iterations=tr["max_iterations"], logger="tensorboard", upload_model=False)
