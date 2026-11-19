"""Record a reference rollout from the mjlab training environment (DR off).

Runs with mjlab_microduck's own venv (it needs mjlab/torch/warp); it does not import
duckrt. Called by scripts/check_parity.py:

    ../mjlab_microduck/.venv/bin/python scripts/parity_mjlab_rollout.py \
        --onnx outputs/policies/walk_flat_model1300.onnx --cmd 0.2 0 0 --steps 200 --lag 4 \
        --vin 7.5 --drop-gain 0.1 --out outputs/parity/mjlab_vx0.2.npz

Environment = Mjlab-Velocity-Microduck play config with every randomization removed:
pushes / friction / encoder bias / CoM / armature / joint-friction events deleted,
obs noise off (play) and obs delays 0, BAM vin and drop gain fixed, actuator lag fixed,
reset pose fixed (x = y = yaw = 0, z = 0.12), command fixed (written into the command
term directly, because the task's command builder ignores cfg edits -- see NOTES).
The policy is the exported ONNX evaluated with onnx.reference (same network as our side).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Microduck"
DR_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com",
             "dof_armature_randomization", "dof_friction_randomization")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.2, 0.0, 0.0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lag", type=int, default=4)
    ap.add_argument("--vin", type=float, default=7.5)
    ap.add_argument("--drop-gain", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--z", type=float, default=0.12, help="reset height of the trunk")
    ap.add_argument("--gravity", type=float, default=-9.81, help="gravity z (0 = floating, no contacts)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1000.0
    for name in DR_EVENTS:
        cfg.events.pop(name, None)
    cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (args.z, args.z), "yaw": (0.0, 0.0)}
    cfg.sim.mujoco.gravity = (0.0, 0.0, args.gravity)
    for term in cfg.observations["actor"].terms.values():
        term.delay_min_lag = 0
        term.delay_max_lag = 0
    robot = cfg.scene.entities["robot"]
    act = robot.articulation.actuators[0]
    act = dataclasses.replace(act, vin_range=(args.vin, args.vin), vin_drop_gain_range=(args.drop_gain, args.drop_gain),
                              delay_min_lag=args.lag, delay_max_lag=args.lag)
    robot.articulation = dataclasses.replace(robot.articulation, actuators=(act,))
    cfg.observations["actor"].enable_corruption = False

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    obs_dict, _ = env.reset()
    term = env.command_manager.get_term("twist")
    cmd = torch.tensor(args.cmd, dtype=torch.float32, device=env.device)

    def pin_command():
        term.vel_command_b[:] = cmd
        term.is_standing_env[:] = False
        term.is_heading_env[:] = False
        term.is_world_env[:] = False
        term.time_left[:] = 1e9

    pin_command()
    # compute() would return the cached post-reset obs (observation_manager.py:304-311);
    # compute_group re-evaluates with the pinned command (obs delays are 0, no history).
    obs = env.observation_manager.compute_group("actor")

    ev = ReferenceEvaluator(args.onnx)
    data = env.sim.data
    bam_act = env.scene["robot"]._custom_actuators[0]
    rec = {k: [] for k in ("qpos", "qvel", "obs", "action", "command", "time", "prev_motor_torque")}
    for _ in range(args.steps + 1):
        o = obs[0].cpu().numpy().astype(np.float32)
        a = ev.run(None, {"obs": o[None, :]})[0][0]
        rec["qpos"].append(data.qpos[0].cpu().numpy().astype(np.float64))
        rec["qvel"].append(data.qvel[0].cpu().numpy().astype(np.float64))
        rec["obs"].append(o)
        rec["action"].append(a.astype(np.float32))
        rec["command"].append(term.vel_command_b[0].cpu().numpy().copy())
        rec["time"].append(float(env.episode_length_buf[0].item()) * env.step_dt)
        rec["prev_motor_torque"].append(bam_act._prev_motor_torque[0].cpu().numpy().astype(np.float64))
        pin_command()
        obs_dict, *_ = env.step(torch.from_numpy(a[None, :]).to(env.device))
        obs = obs_dict["actor"]

    jn = list(env.scene["robot"].joint_names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import mujoco
    mujoco.mj_saveModel(env.sim.mj_model, str(out.with_suffix(".mjb")))
    np.savez(out, **{k: np.asarray(v) for k, v in rec.items()}, joint_names=np.array(jn),
             lag=args.lag, vin=args.vin, drop_gain=args.drop_gain, physics_dt=env.physics_dt,
             decimation=cfg.decimation, mujoco_version=__import__("mujoco").__version__)
    print(f"[mjlab] saved {out} ({args.steps} steps, joints {jn[:2]}..., device {args.device})")
    env.close()


if __name__ == "__main__":
    main()
