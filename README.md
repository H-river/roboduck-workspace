# RoboDuck workspace

RL skill training (walk, kick) for the AI-FanGe Microduck replica, plus the bare-MuJoCo acceptance
environment that decides whether a trained policy is good enough.

**Start with [HANDOFF.md](HANDOFF.md)** (中文): status, 10-minute setup, how to train, how to judge results, known pitfalls.

```bash
git clone --recursive https://github.com/H-river/roboduck-workspace.git && cd roboduck-workspace
scripts/setup_workspace.sh
cd data_collect_sim && .venv/bin/python scripts/accept.py --walk walk_flat --stand walk_flat_2600 --kick kick --quick
```

Based on [AI-FanGe/Microduck-build-tutorial](https://github.com/AI-FanGe/Microduck-build-tutorial) @ `4967821`
(`mjlab_microduck/` and `microduck/` are submodules carrying the upstream history plus our commits; see `patches/`).
Private repository: the robot meshes are CC BY-NC-SA 4.0 and the deploy runtime is GPL-3.0 (HANDOFF.md §8).
