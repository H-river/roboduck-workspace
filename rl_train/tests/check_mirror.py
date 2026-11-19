"""Mirror-map check: env 1 is put in the exact mirror state of env 0 (base pose / velocity,
joints, command, last action), both are stepped with mirrored actions, and env 1's actor and
critic observations must equal mirror(env 0's). DR events off, obs noise off, lags fixed.

    ../mjlab_microduck/.venv/bin/python tests/check_mirror.py
"""
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from rl_tasks import mdp_v2  # noqa: E402
from tensordict import TensorDict  # noqa: E402

cfg = load_env_cfg("RoboDuck-walk_v2_r1", play=True)
cfg.scene.num_envs = 2
for k in ("push_robot", "foot_friction", "encoder_bias", "base_com", "dof_armature_randomization", "dof_friction_randomization"):
    cfg.events.pop(k, None)
cfg.events["reset_base"].params["pose_range"] = {"x": (0, 0), "y": (0, 0), "z": (0.12, 0.12), "yaw": (0, 0)}
for g in ("actor", "critic"):
    for t in cfg.observations[g].terms.values():
        t.delay_min_lag = t.delay_max_lag = 0
robot = cfg.scene.entities["robot"]
act = robot.articulation.actuators[0]
robot.articulation = dataclasses.replace(robot.articulation, actuators=(dataclasses.replace(
    act, vin_range=(7.5, 7.5), vin_drop_gain_range=(0.1, 0.1), delay_min_lag=4, delay_max_lag=4),))
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
wenv = RslRlVecEnvWrapper(env)
env.reset()
maps = mdp_v2._group_maps(wenv, "cuda:0")
jperm, jsign = maps["_actions"]
r = env.scene["robot"]
d = env.sim.data
term = env.command_manager.get_term("twist")

# a non-trivial asymmetric state for env 0, its mirror for env 1
torch.manual_seed(0)
qa, va = r.indexing.free_joint_q_adr, r.indexing.free_joint_v_adr
jq, jv = r.indexing.joint_q_adr, r.indexing.joint_v_adr
q = d.qpos.clone(); v = d.qvel.clone()
q[0, qa[0:3]] += torch.tensor([0.0, 0.02, 0.0], device=q.device)
quat = torch.tensor([0.995, 0.03, -0.05, 0.08], device=q.device); quat /= quat.norm()
q[0, qa[3:7]] = quat
q[0, jq] += 0.1 * torch.randn(len(jq), device=q.device)
v[0, va] = torch.tensor([0.1, 0.05, -0.02, 0.3, -0.2, 0.4], device=q.device)
v[0, jv] = 0.5 * torch.randn(len(jv), device=q.device)
# mirror: y -> -y; quat (w,x,y,z) -> (w,-x,y,-z); linear vel (world) (x,-y,z); angular vel (local) (-x, y, -z)
q[1, qa[0:3]] = q[0, qa[0:3]] * torch.tensor([1.0, -1.0, 1.0], device=q.device) + (q[1, qa[0:3]] - q[0, qa[0:3]]) * 0  # same env origin
q[1, qa[3:7]] = q[0, qa[3:7]] * torch.tensor([1.0, -1.0, 1.0, -1.0], device=q.device)
q[1, jq] = q[0, jq][jperm] * jsign
v[1, va[0:3]] = v[0, va[0:3]] * torch.tensor([1.0, -1.0, 1.0], device=q.device)
v[1, va[3:6]] = v[0, va[3:6]] * torch.tensor([-1.0, 1.0, -1.0], device=q.device)
v[1, jv] = v[0, jv][jperm] * jsign
origins = env.scene.env_origins
q[1, qa[0:2]] = origins[1, :2] + (q[0, qa[0:2]] - origins[0, :2]) * torch.tensor([1.0, -1.0], device=q.device)
d.qpos[:] = q; d.qvel[:] = v
cmd = torch.tensor([[0.2, 0.05, 0.3]], device=q.device)
term.vel_command_b[0] = cmd[0]; term.vel_command_b[1] = cmd[0] * torch.tensor([1.0, -1.0, -1.0], device=q.device)
term.time_left[:] = 1e9; term.is_standing_env[:] = False; term.is_heading_env[:] = False; term.is_world_env[:] = False
a0 = 0.2 * torch.randn(1, 14, device=q.device)
acts = torch.cat([a0, a0[:, jperm] * jsign])
env.sim.forward()
worst = {}
per_step = []
mag = {}
for step in range(5):
    obs, *_ = env.step(acts)
    for g in ("actor", "critic"):
        o = obs[g]
        mo, _ = mdp_v2.mirror_augment(wenv, TensorDict({g: o[0:1]}, batch_size=[1]), None)
        err = (mo[g][1] - o[1]).abs()
        names = env.observation_manager.active_terms[g]
        dims = [int(torch.tensor(x).prod()) for x in env.observation_manager.group_obs_term_dim[g]]
        off = 0
        for n, k in zip(names, dims):
            worst[(g, n)] = max(worst.get((g, n), 0.0), float(err[off:off + k].max()))
            mag[(g, n)] = max(mag.get((g, n), 0.0), float(o[0, off:off + k].abs().max()))
            off += k
        if g == "actor":
            per_step.append(float(err.max()))
    if step == 0:
        # negative control: flip head_yaw's mirror sign -> must be detected
        bad_maps = {k: (v[0], v[1].clone()) for k, v in maps.items()}
        names_a = env.observation_manager.active_terms["actor"]
        j = list(env.scene["robot"].joint_names).index("head_yaw")
        off = sum(int(torch.tensor(x).prod()) for x in env.observation_manager.group_obs_term_dim["actor"][:names_a.index("joint_vel")])
        bad_maps["actor"][1][off + j] *= -1
        o = obs["actor"]
        mo = o[0:1][:, bad_maps["actor"][0]] * bad_maps["actor"][1]
        print(f"negative control (head_yaw sign flipped in joint_vel): error {float((mo[0] - o[1]).abs()[off + j]):.3e} "
              f"vs correct map {float(worst[('actor', 'joint_vel')]):.3e} at step 1")
for (g, n), e in worst.items():
    print(f"{g:6s} {n:18s} max |mirror(obs0) - obs1| over 5 steps = {e:.2e}   (max |obs0| {mag[(g, n)]:.2e}, ratio {e / max(mag[(g, n)], 1e-9):.1%})")
print("actor max error per step:", [f"{x:.1e}" for x in per_step])
bad = {k: e for k, e in worst.items() if e > 0.1 * max(mag[k], 1e-3)}
print("RESULT:", "PASS (every term within 10 % of its magnitude; sign errors would be ~100-200 %)" if not bad else f"FAIL {bad}")
