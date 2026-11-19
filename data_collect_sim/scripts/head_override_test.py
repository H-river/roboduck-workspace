"""head_override acceptance test for a walk policy card (definition: configs/acceptance.yaml, rl_train/NOTES.md §1.1).

    uv run python scripts/head_override_test.py [--card walk_flat]
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
    ap.add_argument("--card", default="walk_flat", help="policy card name or path (walk role; also used to stand)")
    args = ap.parse_args()
    res = evaluation.TESTS["head_override"](args.card)
    out = output_dir("head_override_test") / f"{Path(args.card).stem}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("trials",)}, indent=1, default=float)[:3000])
    print(f"\n{'PASS' if res.get('pass') else 'FAIL'} -> {out}")
    sys.exit(0 if res.get("pass") else 1)


if __name__ == "__main__":
    main()
