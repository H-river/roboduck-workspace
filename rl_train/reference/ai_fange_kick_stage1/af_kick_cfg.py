"""AI-FanGe Microduck ball-kick task: Pollen's kick (microduck_rl
`microduck_ball_kick_env_cfg.py`) moved onto the AI-FanGe robot.

Mirrored from Pollen's kick (same file, same values):
  * ball: microduck_rl `robot/microduck/ball.xml` (70 mm, 15 g, friction
    0.5 / 0.005 / 0.0001), copied verbatim as ball.xml next to this file;
  * spawn: centre (0.09, -0.042) in the robot's yaw frame +-0.015 m per axis,
    i.e. the same kickable-zone definition our bridge's KickCfg is measured
    against; right foot kicks; standing-only start (z 0.11-0.12, +-5 deg tilt,
    joints HOME +-0.05);
  * rewards (ball speed capped at 1.0 m/s, overshoot penalty, support foot
    grounded, HOME pose legs/neck, upright, stand height, action rate, body
    ang vel, angular momentum, self collisions), curricula, pushes, CoM /
    head-CoM / mass-inertia / armature DR, encoder bias, 5 s episodes.

Taken from AI-FanGe instead (their walk's conventions, so their runtime can run it):
  * robot: their `robot_allcollisions.xml` (11 colliding geoms, the same set
    as Pollen's kick model `robot_groundcontact.xml`) + their HOME_FRAME,
    FULL_COLLISION and BamActuatorCfg (kp 125, 1.75 A, vin 7-8 V, delay 3-6);
  * sim: their SIM_CFG (dt 0.005, 10 / 20 / 100 iterations), decimation 4;
  * actor obs (51-D, = their walk's layout): mounted gyro, mounted up-gravity
    (BNO080_MOUNT_QUAT, trunk_base), joint pos, joint vel, last action,
    3-D twist command (held ~0); their noise and 0-1 step delays.

Dropped (Pollen-only machinery): the 10 zero-padded head/body command slots
(61-D layout), the IMU-misalignment obs DR (their walk has none), Pollen's
friction-budget BAM DR (their stock BAM has no such fields).
"""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjlab.managers import (CurriculumTermCfg, EventTermCfg, ObservationTermCfg,  # noqa: E402
                            RewardTermCfg, TerminationTermCfg)
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.velocity import mdp  # noqa: E402
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg  # noqa: E402
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noqa: E402

import mjlab_microduck.robot.microduck_constants as AF  # noqa: E402  (AI-FanGe, 9a11a40)
from mjlab_microduck.tasks.mdp import (mounted_body_ang_vel,  # noqa: E402
                                       mounted_body_projected_gravity)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (BNO080_MOUNT_QUAT,  # noqa: E402
                                                              MICRODUCK_IMU_BODY, SIM_CFG)

import pollen_kick_mdp as kick  # noqa: E402

TASK_ID = "Mjlab-BallKick-Right-AiFange"
KICK_FOOT = "right"
EPISODE_LENGTH_S = 5.0
BALL_RADIUS = 0.035
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_POS_NOISE_XY = 0.015
BALL_TARGET_SPEED = 1.0
STAND_Z = 0.115
_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]
# Pollen microduck_velocity_env_cfg.HEAD_BODY_NAMES (same body names on AF)
HEAD_BODY_NAMES = ("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)")
COM_RANGE, HEAD_COM_RANGE = 0.003, 0.003
MASS_INERTIA_RANGE = (0.95, 1.05)
ARMATURE_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
PUSH_INTERVAL_S = (3.0, 6.0)
PUSH_RANGE = (-0.3, 0.3)

AF_ALLCOLL_XML = Path(AF.__file__).resolve().parent / "microduck" / "robot_allcollisions.xml"
BALL_XML = Path(__file__).resolve().parent / "ball.xml"


def _robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(AF_ALLCOLL_XML))


def _ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(BALL_XML))


AF_KICK_ROBOT_CFG = EntityCfg(
    spec_fn=_robot_spec,
    init_state=AF.HOME_FRAME,
    collisions=(AF.FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(actuators=(AF.actuators,),
                                           soft_joint_pos_limit_factor=0.9),
)
BALL_CFG = EntityCfg(spec_fn=_ball_spec,
                     init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)))


def make_af_ball_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    support_foot = "left" if KICK_FOOT == "right" else "right"
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$",
                             entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    support_foot_ground_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^{support_foot}_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1)
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",), reduce="none", num_slots=1)

    cfg = make_velocity_env_cfg()
    cfg.sim = deepcopy(SIM_CFG)                       # AI-FanGe sim settings
    cfg.scene.entities = {"robot": AF_KICK_ROBOT_CFG, "ball": BALL_CFG}
    cfg.scene.sensors = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = 50                              # Pollen: ball contacts on top of the robot's
    # Pollen's kick ran with njmax 1500 (its env.yaml) and logged no overflow;
    # AI-FanGe's SIM_CFG leaves njmax unset, which overflowed (965k "nefc
    # overflow" warnings in the first launch, i.e. dropped contact constraints).
    cfg.sim.njmax = 1500

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                 "foot_swing_height", "foot_slip", "pose", "soft_landing"]:
        cfg.rewards.pop(name, None)
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(
        func=kick.ball_forward_velocity, weight=12.0,
        params={"asset_name": "ball", "max_speed": BALL_TARGET_SPEED})
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=kick.ball_speed_overshoot_penalty, weight=-4.0,
        params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED})
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=kick.single_foot_grounded_reward, weight=2.0,
        params={"sensor_name": support_foot_ground_cfg.name})
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=kick.pose_target_match, weight=2.0,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=kick.pose_target_match, weight=1.0,
        params={"std": 0.3, "joint_indices": _NECK_JOINTS, "target_overrides": None})
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=kick.height_target_gaussian, weight=1.0,
        params={"std": 0.04, "target_height": STAND_Z,
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    # ── observations: AI-FanGe walk layout (51-D actor) ─────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    imu = SceneEntityCfg("robot", body_names=(MICRODUCK_IMU_BODY,))
    for grp in ("actor", "critic"):
        g = cfg.observations[grp].terms["projected_gravity"] = deepcopy(cfg.observations[grp].terms["projected_gravity"])
        g.func, g.params = mounted_body_projected_gravity, {"mount_quat": BNO080_MOUNT_QUAT, "asset_cfg": imu}
        a = cfg.observations[grp].terms["base_ang_vel"] = deepcopy(cfg.observations[grp].terms["base_ang_vel"])
        a.func, a.params = mounted_body_ang_vel, {"mount_quat": BNO080_MOUNT_QUAT, "asset_cfg": imu}
    act = cfg.observations["actor"].terms
    act["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    act["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    for t in ("base_ang_vel", "projected_gravity"):
        act[t].delay_min_lag, act[t].delay_max_lag, act[t].delay_update_period = 0, 1, 64
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)
    act["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    act["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
    act["joint_vel"].delay_min_lag, act["joint_vel"].delay_max_lag = 0, 1
    cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
    act["joint_pos"].params["biased"] = True
    cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=kick.ball_pos_in_base, params={"asset_name": "ball"})
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=kick.ball_vel_in_base, params={"asset_name": "ball"})

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
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
    cfg.events["reset_ball"] = EventTermCfg(
        func=kick.reset_ball_in_front_of_foot, mode="reset",
        params={"offset": (BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if KICK_FOOT == "right" else BALL_OFFSET_ABS_Y),
                "noise_xy": BALL_POS_NOISE_XY, "ball_radius": BALL_RADIUS, "asset_name": "ball"})
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(0.5, 1.0) if play else PUSH_INTERVAL_S,
        params={"velocity_range": {"x": PUSH_RANGE, "y": PUSH_RANGE}, "asset_cfg": SceneEntityCfg("robot")})
    cfg.events["randomize_com"] = EventTermCfg(
        func=dr.body_ipos, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add", "ranges": (-COM_RANGE, COM_RANGE)})
    cfg.events["randomize_head_com"] = EventTermCfg(
        func=dr.body_ipos, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add", "ranges": (-HEAD_COM_RANGE, HEAD_COM_RANGE)})
    cfg.events["randomize_armature"] = EventTermCfg(
        func=dr.joint_armature, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale", "ranges": ARMATURE_RANGE})
    lo, hi = MASS_INERTIA_RANGE
    cfg.events["randomize_mass_inertia"] = EventTermCfg(
        func=dr.pseudo_inertia, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=kick.reward_weight,
        params={"reward_name": "action_rate_l2",
                "weight_stages": [{"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2},
                                  {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6},
                                  {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})
    cfg.curriculum["com_range"] = CurriculumTermCfg(
        func=kick.com_range_curriculum,
        params={"event_name": "randomize_com",
                "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                                 {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})
    cfg.curriculum["head_com_range"] = CurriculumTermCfg(
        func=kick.com_range_curriculum,
        params={"event_name": "randomize_head_com",
                "range_stages": [{"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                                 {"step": 1000 * 24, "range": 0.01}]})
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=kick.push_curriculum,
        params={"event_name": "push_robot",
                "push_stages": [{"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                                {"step": 500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                                {"step": 1000 * 24, "velocity_range": {"x": PUSH_RANGE, "y": PUSH_RANGE}}]})
    return cfg


AF_KICK_RL_CFG = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                   entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
                                   learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
                                   desired_kl=0.01, max_grad_norm=1.0),
    wandb_project="ai_fange_kick", experiment_name="ai_fange_ball_kick_right",
    run_name="ai_fange_ball_kick_right", save_interval=250, num_steps_per_env=24,
    max_iterations=10_000,
)


def register() -> str:
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
    register_mjlab_task(task_id=TASK_ID, env_cfg=make_af_ball_kick_env_cfg(),
                        play_env_cfg=make_af_ball_kick_env_cfg(play=True),
                        rl_cfg=AF_KICK_RL_CFG, runner_cls=VelocityOnPolicyRunner)
    return TASK_ID
