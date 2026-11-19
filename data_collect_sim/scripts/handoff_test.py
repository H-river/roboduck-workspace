"""Handoff test: walk -> stand -> kick -> stand -> walk, 20 cycles in one run (configs/acceptance.yaml kick.handoff).

    uv run python scripts/handoff_test.py [--walk walk_flat] [--kick kick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt import evaluation  # noqa: E402
from duckrt.config import output_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", default="walk_flat", help="walk card (also stands)")
    ap.add_argument("--kick", default="kick")
    args = ap.parse_args()
    res = evaluation.kick_handoff(args.walk, args.kick)
    out = output_dir("handoff_test") / f"{Path(args.walk).stem}__{Path(args.kick).stem}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"cycles {res['cycles']}, falls {res['falls']}, kicks that hit {res['kick_success']}")
    print(f"{'PASS' if res['pass'] else 'FAIL'} -> {out}")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
