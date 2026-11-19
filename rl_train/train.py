"""Train an rl_train task with mjlab's own training entry point (runs in mjlab_microduck/.venv).

    cd rl_train/runs
    ../../mjlab_microduck/.venv/bin/python ../train.py RoboDuck-walk_v2_r1 --agent.seed 1 --agent.run-name r1_s1

Logs go to ./logs/rsl_rl/<experiment_name>/ under the current directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rl_tasks  # noqa: E402,F401  registers RoboDuck-* tasks
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
