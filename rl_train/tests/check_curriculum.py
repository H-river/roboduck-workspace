"""Does the upstream staged curriculum reach the velocity-command sampler? (runs in mjlab_microduck/.venv, GPU)

Mjlab-Velocity-Microduck widens the command ranges at iteration 3000 (vx +-0.25 -> +-0.35, rotation-only
|wz| up to 0.75 -> 1.5) through set_command_velocity(), which edits env.command_manager.get_term_cfg("twist").
Upstream built the command term with `command.build = lambda env, _cmd=command: ...`, which bound the cfg
object from BEFORE the manager's deepcopy, so those edits never reached the sampler (HANDOFF_NOTES step 1).

This builds the task with the stage moved to env step --stage-step and short command resampling, steps it
with zero actions, and compares the commands actually sampled before and after the stage.
--buggy re-installs the upstream lambda (negative control: the check must then FAIL).

    cd rl_train && ../mjlab_microduck/.venv/bin/python tests/check_curriculum.py [--buggy] [--out result.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.mdp import UniformVelocityCommandWithRotation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-Velocity-Microduck")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--stage-step", type=int, default=100, help="env step at which the stage is applied")
    ap.add_argument("--steps", type=int, default=400, help="env steps in total")
    ap.add_argument("--resample-s", type=float, nargs=2, default=[0.2, 0.5])
    ap.add_argument("--buggy", action="store_true", help="re-install the upstream lambda (negative control)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    stages = cfg.curriculum["staged_curriculum"].params["stages"]
    stages[0]["step"] = args.stage_step
    cmd = cfg.commands["twist"]
    cmd.resampling_time_range = tuple(args.resample_s)
    if args.buggy:
        # as upstream: the lambda binds the cfg object that make_*_env_cfg() returns (the registered
        # one); the env is built from a deep copy of it (load_env_cfg / tyro do the copy)
        cmd.build = lambda env, _cmd=cmd: UniformVelocityCommandWithRotation(_cmd, env)
        cfg = copy.deepcopy(cfg)

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    term = env.command_manager._terms["twist"]
    managed = env.command_manager.get_term_cfg("twist")
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    before = {"vx": 0.0, "wz": 0.0}
    after = {"vx": 0.0, "wz": 0.0}
    settle = int(args.resample_s[1] / env.step_dt) + 1  # every env has resampled once after the stage
    for _ in range(args.steps):
        env.step(zero)
        n = env.common_step_counter
        if n < args.stage_step:  # the stage fires inside the reset at n == stage_step
            d = before
        elif n > args.stage_step + settle:
            d = after
        else:
            continue
        c = env.command_manager.get_command("twist").abs().amax(0).tolist()
        d["vx"], d["wz"] = max(d["vx"], c[0]), max(d["wz"], c[2])
    res = {
        "task": args.task, "buggy_lambda": args.buggy, "stage_step": args.stage_step, "steps": args.steps,
        "term_uses_managed_cfg": term.cfg is managed,
        "managed_cfg_after": {"lin_vel_x": list(managed.ranges.lin_vel_x), "rotation_env_ang_vel_range": list(managed.rotation_env_ang_vel_range)},
        "sampler_cfg_after": {"lin_vel_x": list(term.cfg.ranges.lin_vel_x), "rotation_env_ang_vel_range": list(term.cfg.rotation_env_ang_vel_range)},
        "sampled_max_abs_before": before, "sampled_max_abs_after": after,
    }
    # rotation-only envs: |wz| <= 0.75 before the stage, up to 1.5 after it (vx is not used: world-frame
    # and forward-only envs blur its bound)
    res["curriculum_reaches_sampler"] = bool(before["wz"] <= 0.76 and after["wz"] > 0.8)
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
