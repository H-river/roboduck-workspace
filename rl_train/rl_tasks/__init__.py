"""RoboDuck phase-2 training tasks. Importing this package registers one mjlab task per
rl_train/configs/{walk,kick}_*.yaml, named RoboDuck-<file stem>.

Upstream mjlab_microduck tasks (Mjlab-Velocity-Microduck) stay registered through their own
entry point; nothing in mjlab_microduck/ is modified.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from rl_tasks.walk_v2 import CONFIG_DIR, load_config, make_env_cfg, make_rl_cfg

for _p in sorted(CONFIG_DIR.glob("walk_*.yaml")):
    _c = load_config(_p)
    register_mjlab_task(task_id=f"RoboDuck-{_p.stem}", env_cfg=make_env_cfg(_c), play_env_cfg=make_env_cfg(_c, play=True),
                        rl_cfg=make_rl_cfg(_c), runner_cls=VelocityOnPolicyRunner)

from rl_tasks.kick_v2 import make_kick_env_cfg, make_kick_rl_cfg  # noqa: E402

for _p in sorted(CONFIG_DIR.glob("kick_*.yaml")):
    _c = load_config(_p)
    register_mjlab_task(task_id=f"RoboDuck-{_p.stem}", env_cfg=make_kick_env_cfg(_c), play_env_cfg=make_kick_env_cfg(_c, play=True),
                        rl_cfg=make_kick_rl_cfg(_c), runner_cls=VelocityOnPolicyRunner)
