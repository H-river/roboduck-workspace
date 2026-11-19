"""Evaluate walk checkpoints against the walk acceptance gates (configs/acceptance.yaml).

checkpoint (.pt) -> ONNX (rl_train/export_onnx.py in mjlab's venv) -> policy card
(make_policy_cards.make_walk) -> full walk suite (duckrt.evaluation) -> comparison table.
Candidates are evaluated in parallel worker processes.

    uv run python scripts/eval_candidates.py --run ../mjlab_microduck/logs/rsl_rl/mjlab_microduck_velocity/2026-10-16_17-43-09 --every 200
    uv run python scripts/eval_candidates.py --run ../rl_train/runs/<run> --task Mjlab-Velocity-Microduck-v2 --every 500 --tag v2r1
    uv run python scripts/eval_candidates.py --onnx ../microduck/src/agents/walk.onnx --env-yaml <params/env.yaml>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt.config import CONFIG_DIR, ROOT, output_dir  # noqa: E402
from duckrt.testing import md_table  # noqa: E402

WS = ROOT.parent
MJLAB_PY = WS / "mjlab_microduck" / ".venv" / "bin" / "python"
EXPORT = WS / "rl_train" / "export_onnx.py"
CAND_ONNX = ROOT / "outputs" / "policies" / "candidates"
CAND_CARDS = CONFIG_DIR / "policies" / "candidates"
GB_PER_WORKER = 1.3   # measured RSS of one evaluation worker (EGL renderer + model + onnxruntime)
RESERVE_GB = 2.0      # keep this much RAM free for the desktop / training


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def iteration(p: Path) -> int:
    return int(re.search(r"model_(\d+)\.pt$", p.name).group(1))


def export(ckpt: Path, task: str, out: Path) -> Path:
    if out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime:
        return out
    subprocess.run([str(MJLAB_PY), str(EXPORT), "--task", task, "--checkpoint", str(ckpt.resolve()), "--out", str(out)],
                   check=True, cwd=EXPORT.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    return out


def make_card(name: str, onnx: Path, ckpt: Path | None, run_dir: Path, train_cfg_py: Path) -> Path:
    import make_policy_cards as mpc

    src = {
        "onnx": str(onnx), "checkpoint": str(ckpt) if ckpt else str(onnx),
        "agent_yaml": str(run_dir / "params" / "agent.yaml"), "env_yaml": str(run_dir / "params" / "env.yaml"),
        "train_cfg_py": str(train_cfg_py),
        "robot_constants_py": str(WS / "mjlab_microduck/src/mjlab_microduck/robot/microduck_constants.py"),
        "mjcf": str(WS / "mjlab_microduck/src/mjlab_microduck/robot/microduck/robot_walk.xml"),
        "mjlab_velocity_command_py": str(WS / "mjlab_microduck/.venv/lib/python3.13/site-packages/mjlab/tasks/velocity/mdp/velocity_command.py"),
    }
    out = CAND_CARDS / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = mpc.make_walk(src, name, out_path=out) if ckpt else mpc.make_walk({**src, "checkpoint": str(onnx)}, name, out_path=out)
    out.write_text(text)
    return out


def _worker(args) -> tuple[str, dict, float]:
    name, card, which = args
    from duckrt.evaluation import walk_suite

    t0 = time.time()
    try:
        res = walk_suite(str(card), which)
    except Exception as e:  # keep the other candidates going
        res = {"error": repr(e), "all_pass": False}
    return name, res, time.time() - t0


def row(name: str, r: dict) -> list:
    if "error" in r:
        return [name, "ERROR " + r["error"][:60]] + [""] * 9
    g = lambda k: r.get(k, {})  # noqa: E731
    rnd, st, sy, rp, ho, cam = g("random"), g("stop"), g("symmetry"), g("ramp"), g("head_override"), g("camera")
    ok = lambda b: "✔" if b else ("–" if b is None else "✘")  # noqa: E731
    turns = sy.get("turns", [])
    res5 = (rp.get("results") or {})
    return [
        name,
        f"{ok(rnd.get('pass'))} {rnd.get('no_fall', '-')}/{rnd.get('n', '-')}",
        f"{ok(st.get('pass'))} {st.get('n_pass', '-')}/{st.get('n', '-')} max {st.get('max_stop_s', float('nan')):.2f}s/{100 * st.get('max_drift_m', float('nan')):.1f}cm" if st else "",
        f"{ok(sy.get('straight_pass'))} {max((abs(w) for w in sy.get('straight_yaw_rates', [float('nan')])), default=float('nan')):.3f}" if sy else "",
        " ".join(f"{ok(t['pass'])}{t['ratio']:.2f}" for t in turns),
        " ".join(f"{k.replace('_', '° ')} {v['success']}/{v['n']}" for k, v in res5.items()),
        f"{ok(ho.get('pass'))} {ho.get('n_pass', '-')}/{ho.get('n', '-')}" if ho else "",
        f"{ok(cam.get('pass'))}" if cam else "",
        f"{cam.get('stand_minus_walk_pitch_deg', float('nan')):+.1f}°" if cam else "",
        "**PASS**" if r.get("all_pass") else "",
    ]


HEADER = ["candidate", "random 20×60s", "stop (n, worst time / drift)", "straight |ωz| max", "±ωz ratio (in place, walking)",
          "ramps (heading hold)", "head takeover", "2 m flag", "stand−walk cam pitch", "all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=[], help="run dir with model_*.pt and params/")
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--min-iter", type=int, default=0)
    ap.add_argument("--pt", type=Path, nargs="*", default=[], help="explicit checkpoints (their run dir supplies params/)")
    ap.add_argument("--onnx", type=Path, nargs="*", default=[], help="ONNX files without checkpoint (need --env-yaml run dir)")
    ap.add_argument("--onnx-run", type=Path, default=WS / "mjlab_microduck/checkpoints/walk_stable_5s",
                    help="run dir whose params/ describe the --onnx files")
    ap.add_argument("--task", default="Mjlab-Velocity-Microduck")
    ap.add_argument("--train-cfg-py", type=Path, default=WS / "mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py")
    ap.add_argument("--tests", nargs="*", default=None, help="subset of: random stop symmetry ramp head_override camera")
    ap.add_argument("--workers", type=int, default=4, help="upper bound; also capped by available RAM")
    ap.add_argument("--tag", default="scan")
    ap.add_argument("--only", nargs="*", default=None, help="evaluate only these candidate names")
    ap.add_argument("--resume", action="store_true", help="keep results already in results.json, skip those candidates")
    args = ap.parse_args()

    cands: list[tuple[str, Path | None, Path, Path]] = []  # name, ckpt, onnx target, run dir
    for run in args.run:
        for ck in sorted(run.glob("model_*.pt"), key=iteration):
            it = iteration(ck)
            if it % args.every == 0 and it >= args.min_iter:
                name = f"{run.name}_it{it}"
                cands.append((name, ck, CAND_ONNX / f"{name}.onnx", run))
    for ck in args.pt:
        name = f"{ck.parent.name}_it{iteration(ck)}"
        cands.append((name, ck, CAND_ONNX / f"{name}.onnx", ck.parent))
    for ox in args.onnx:
        cands.append((ox.stem, None, ox, args.onnx_run))

    t0 = time.time()
    print(f"[export] {sum(c[1] is not None for c in cands)} checkpoints")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda c: export(c[1], args.task, c[2]) if c[1] is not None else c[2], cands))
    cards = [(name, make_card(name, onnx, ck, run, args.train_cfg_py)) for name, ck, onnx, run in cands]
    print(f"[export+cards] {time.time() - t0:.0f} s; evaluating {len(cards)} candidates on {args.workers} workers")

    out = output_dir("candidates", args.tag)
    rpath = out / "results.json"
    results = json.loads(rpath.read_text()) if args.resume and rpath.exists() else {}
    todo = [(n, c) for n, c in cards if n not in results and (args.only is None or n in args.only)]
    workers = max(1, min(args.workers, len(todo) or 1, int((mem_available_gb() - RESERVE_GB) / GB_PER_WORKER)))
    print(f"[eval] {len(todo)} to evaluate with {workers} workers ({mem_available_gb():.1f} GB available)")
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, (n, c, args.tests)): n for n, c in todo}
        for f in as_completed(futs):
            try:
                name, res, dt = f.result()
            except BrokenProcessPool as e:  # a worker died (e.g. OOM): record, do not hang
                name, res, dt = futs[f], {"error": f"worker died: {e!r}", "all_pass": False}, 0.0
            results[name] = res
            print(f"  {name}: {'PASS' if res.get('all_pass') else 'fail'} ({dt:.0f} s)", flush=True)
            rpath.write_text(json.dumps(results, indent=2, default=float))
    order = [n for n, _ in cards if n in results]
    md = [f"# eval_candidates — {args.tag}", "", f"gates: configs/acceptance.yaml; {len(order)} candidates; {time.time() - t0:.0f} s", "",
          md_table(HEADER, [row(n, results[n]) for n in order])]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
