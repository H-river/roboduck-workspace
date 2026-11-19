"""Do envs on uneven terrain terminate right after spawning? (runs in mjlab_microduck/.venv, GPU)

Builds the task with 512 envs, steps it 25 times with zero actions and counts terminations per spawn height
(below / at / above z = 0), plus the trunk clearance seen by the terrain-relative fall check. Run it for every
new terrain config: the first r3 run lost 20 % of its envs to an absolute-height fall check (HANDOFF_NOTES step 1).

    cd rl_train && ../mjlab_microduck/.venv/bin/python tests/check_terrain_termination.py RoboDuck-walk_v2_r3
"""
import sys
from pathlib import Path

ws = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ws / "rl_train"))
import torch
import rl_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = sys.argv[1] if len(sys.argv) > 1 else "RoboDuck-walk_v2_r3"
cfg = load_env_cfg(task)
cfg.scene.num_envs = 512
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
robot = env.scene["robot"]
origins = env.scene.env_origins
print("terminations:", list(env.termination_manager.active_terms))
print("fell_over cfg:", env.termination_manager.get_term_cfg("fell_over").func.__name__, env.termination_manager.get_term_cfg("fell_over").params)
print("origin z: min %.3f max %.3f" % (origins[:, 2].min(), origins[:, 2].max()))
z = robot.data.root_link_pos_w[:, 2]
print("root z after reset: min %.3f max %.3f; root z - origin z: min %.3f max %.3f" % (z.min(), z.max(), (z - origins[:, 2]).min(), (z - origins[:, 2]).max()))
zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
dead = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
for i in range(25):
    _, _, term, trunc, _ = env.step(zero)
    dead |= term.bool()
oz = origins[:, 2]
for name, m in (("origin z < -0.02", oz < -0.02), ("|origin z| <= 0.02", oz.abs() <= 0.02), ("origin z > 0.02", oz > 0.02)):
    n = int(m.sum())
    print(f"{name}: {n} envs, terminated within 25 steps (zero action): {int((dead & m).sum())}")
if "trunk_height_scan" in env.scene.sensors:
    h = env.scene["trunk_height_scan"].data.heights[:, 0]
    rz = robot.data.root_link_pos_w[:, 2]
    flat = oz.abs() <= 0.02
    print("trunk clearance now: min %.3f max %.3f; flat envs: clearance - root z max |diff| %.4f" % (h.min(), h.max(), (h[flat] - rz[flat]).abs().max()))
