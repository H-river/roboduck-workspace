#!/usr/bin/env bash
# AI-FanGe kick, stage 1 (mirrors Pollen's kick_train: 2048 envs, 4000 iterations).
cd /home/jiamo/microduck_ws/runs/ai_fange_kick
export MUJOCO_GL=egl AF_KICK_GPU_FRACTION=0.30
exec /home/jiamo/microduck_ws/ai_fange_mjlab/.venv/bin/python \
  /home/jiamo/microduck_ws/duck_soccer/training/ai_fange_kick/launch_ai_fange_kick.py \
  --env.scene.num-envs 2048 --agent.max-iterations 4000 --agent.logger tensorboard \
  --agent.upload-model False --agent.experiment-name /home/jiamo/microduck_ws/runs/ai_fange_kick/models \
  --agent.run-name af_kick_stage1
