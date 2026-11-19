# AI-FanGe kick, stage 1 — reference copy

The kick policy in `data_collect_sim/outputs/policies/kick_af_stage1_model3999.onnx` (card
`data_collect_sim/configs/policies/kick.yaml`) was trained outside this workspace, in
`~/microduck_ws` (duck_soccer commit `c519306`, 2026-10-06; run `runs/ai_fange_kick/models/2026-10-06_13-14-13_af_kick_stage1`,
task `Mjlab-BallKick-Right-AiFange`, 2048 envs x 4000 iterations). Copied here on 2026-11-20 so the
repo is self-contained:

- `af_kick_cfg.py`, `ball.xml`, `launch_ai_fange_kick.py`, `launch_stage1.sh`: training config and launchers, verbatim
  (`pollen_kick_mdp.py`, which they import, is `rl_train/rl_tasks/pollen_kick_mdp.py`).
- `params/env.yaml`, `params/agent.yaml`: the resolved config dumped by mjlab at the start of the run.

Not runnable from here as-is (paths point into `~/microduck_ws`); kick v2 (`rl_train/rl_tasks/kick_v2.py`,
`rl_train/configs/kick_v2_*.yaml`) is the maintained version. The checkpoints (`model_*.pt`) are not in git:
see HANDOFF.md, "没提交的文件".
