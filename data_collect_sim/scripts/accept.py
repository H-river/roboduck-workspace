"""一键验收：给一个 walk 策略（可选 stand、kick），跑完 configs/acceptance.yaml 里的全部测试，输出合格 / 不合格报告。

    uv run python scripts/accept.py --walk <onnx | checkpoint.pt | card.yaml | card 名> [--stand <...>] [--kick <...>]
        [--config <训练 run 目录>] [--kick-config <kick 训练 run 目录>] [--task <mjlab 任务名>]
        [--quick] [--out outputs/acceptance/<name>/] [--tests random stop ...] [--workers N] [--no-videos]

输入：
- card（`configs/policies/*.yaml` 或它的名字，如 `walk_flat`）：直接用。
- checkpoint（`model_N.pt`）：它所在的 run 目录要有 `params/env.yaml`、`params/agent.yaml`。在 mjlab 的 venv 里
  导出 ONNX（rl_train/export_onnx.py），再生成 policy card。任务名从 run 目录名推断（`..._<rl_train 配置名>_s<seed>`），
  推断不了就用 --task。
- ONNX：policy card 需要训练配置。来源依次是 --config（walk / stand）或 --kick-config（kick）、ONNX 所在目录、
  ONNX 的上一级目录（run 目录下的 `onnx/`）。都找不到就报错。

门槛只在 configs/acceptance.yaml 里；--quick 合并它的 `quick:` 段（规模变小，门槛不变）。
输出：<out>/report.md、report.json、cards/、policies/（导出的 ONNX）、videos/（只有失败用例）。
退出码：0 合格，1 不合格，2 输入错误。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from duckrt import report  # noqa: E402
from duckrt.config import CONFIG_DIR, OUTPUT_DIR, ROOT, load_yaml  # noqa: E402

WS = ROOT.parent
MJLAB_VENV = WS / "mjlab_microduck" / ".venv"
EXPORT = WS / "rl_train" / "export_onnx.py"
RL_CONFIGS = WS / "rl_train" / "configs"
WALK_TESTS = ["random", "stop", "symmetry", "tracking", "ramp", "head_override", "camera", "switch"]
KICK_TESTS = ["kick_window", "kick_head_down", "kick_handoff"]


class InputError(SystemExit):
    def __init__(self, msg: str):
        print(f"[accept] 输入错误：{msg}", file=sys.stderr)
        super().__init__(2)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


# -- inputs -------------------------------------------------------------------------------------
def run_dir_of(path: Path | None) -> Path | None:
    """A run directory (has params/env.yaml) from a run dir, its params/ dir or its env.yaml."""
    if path is None:
        return None
    p = Path(path).resolve()
    for cand in (p, p.parent, p.parent.parent):
        if (cand / "params" / "env.yaml").exists() and (cand / "params" / "agent.yaml").exists():
            return cand
    return None


def rl_config_stem(run_dir: Path) -> str | None:
    """rl_train config whose name is in the run dir name (`<date>_<config>_s<seed>`, rl_train/launch_round.sh)."""
    stems = sorted((p.stem for p in RL_CONFIGS.glob("*.yaml")), key=len, reverse=True)
    for s in stems:
        if re.search(rf"_{re.escape(s)}(_s\d+)?$", run_dir.name):
            return s
    return None


def infer_task(run_dir: Path) -> str | None:
    from make_policy_cards import load_env_yaml  # mjlab's dumps carry python/tuple tags

    agent = load_env_yaml(run_dir / "params" / "agent.yaml")
    if agent.get("experiment_name") == "mjlab_microduck_velocity":
        return "Mjlab-Velocity-Microduck"
    s = rl_config_stem(run_dir)
    return f"RoboDuck-{s}" if s else None


def export_onnx(ckpt: Path, task: str, out: Path) -> Path:
    if out.exists() and out.stat().st_mtime > ckpt.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    py = MJLAB_VENV / "bin" / "python"
    if not py.exists():
        raise InputError(f"导出 checkpoint 需要 mjlab 的 venv：{py} 不存在（先跑 scripts/setup_workspace.sh）")
    r = subprocess.run([str(py), str(EXPORT), "--task", task, "--checkpoint", str(ckpt), "--out", str(out)],
                       cwd=EXPORT.parent, capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "MUJOCO_GL": "egl"})
    if r.returncode != 0:
        raise InputError(f"导出 ONNX 失败（{ckpt}，任务 {task}）：\n{r.stderr[-2000:]}")
    return out


def card_sources(onnx: Path, ckpt: Path | None, run_dir: Path) -> dict:
    site = sorted((MJLAB_VENV / "lib").glob("python3*/site-packages/mjlab/tasks/velocity/mdp/velocity_command.py"))
    if not site:
        raise InputError(f"生成 card 需要 mjlab 源码（{MJLAB_VENV}/lib/python3*/site-packages/mjlab）；先跑 scripts/setup_workspace.sh")
    return {
        "onnx": str(onnx), "checkpoint": str(ckpt or onnx),
        "agent_yaml": str(run_dir / "params" / "agent.yaml"), "env_yaml": str(run_dir / "params" / "env.yaml"),
        "train_cfg_py": str(WS / "mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py"),
        "robot_constants_py": str(WS / "mjlab_microduck/src/mjlab_microduck/robot/microduck_constants.py"),
        "mjcf": str(WS / "mjlab_microduck/src/mjlab_microduck/robot/microduck/robot_walk.xml"),
        "mjlab_velocity_command_py": str(site[0]),
    }


def resolve_policy(arg: str, kind: str, config: Path | None, task: str | None, out: Path,
                   rl_config: str | None = None) -> tuple[Path, Path | None]:
    """-> (card path, checkpoint or None)."""
    import make_policy_cards as mpc

    p = Path(arg)
    if not p.exists():
        named = CONFIG_DIR / "policies" / f"{arg}.yaml"
        if named.exists():
            return named, None
        raise InputError(f"找不到 {arg}（既不是文件，也不是 configs/policies/ 里的 card 名）")
    p = p.resolve()
    if p.suffix in (".yaml", ".yml"):
        return p, None
    ckpt = None
    if p.suffix == ".pt":
        ckpt, run_dir = p, run_dir_of(p.parent)
        if run_dir is None:
            raise InputError(f"{p} 所在目录没有 params/env.yaml、params/agent.yaml，无法确定训练配置")
        task = task or (f"RoboDuck-{rl_config}" if rl_config else infer_task(run_dir))
        if task is None:
            raise InputError(f"从 run 目录名 {run_dir.name} 推断不出任务名，请用 --task 指定（例如 RoboDuck-walk_v2_r3）")
        it = re.search(r"model_(\d+)\.pt$", p.name)
        name = f"{run_dir.name}_it{it.group(1) if it else p.stem}"
        onnx = export_onnx(p, task, out / "policies" / f"{name}.onnx")
    elif p.suffix == ".onnx":
        run_dir = run_dir_of(config) if config else (run_dir_of(p.parent))
        if run_dir is None:
            raise InputError(f"输入是 ONNX（{p}），生成 policy card 需要训练配置来源：用 "
                             f"{'--kick-config' if kind == 'kick' else '--config'} 指定训练 run 目录（含 params/env.yaml、"
                             "params/agent.yaml），或者直接传 checkpoint（.pt）/ 已有的 card（.yaml）")
        onnx, name = p, p.stem
    else:
        raise InputError(f"不认识的输入类型：{p}（支持 .onnx / .pt / .yaml / card 名）")
    card = out / "cards" / f"{name}.yaml"
    card.parent.mkdir(parents=True, exist_ok=True)
    src = card_sources(onnx, ckpt, run_dir)
    if kind == "walk":
        card.write_text(mpc.make_walk(src, name, out_path=card))
    else:
        stem = rl_config or rl_config_stem(run_dir)
        if stem is None or not stem.startswith("kick"):
            raise InputError(f"kick 输入 {p}：从 run 目录名 {run_dir.name} 找不到 rl_train/configs/kick_*.yaml；"
                             "stage-1 kick 请直接传它的 card（configs/policies/kick.yaml）")
        card.write_text(mpc.make_kick_v2({**src, "kick_config_yaml": str(RL_CONFIGS / f"{stem}.yaml")}, name, out_path=card))
    return card, ckpt


def card_info(card: Path) -> dict:
    from duckrt.config import resolve

    c = load_yaml(card)
    onnx = resolve(c["onnx"])
    return {"card": os.path.relpath(card, ROOT), "onnx": os.path.relpath(onnx, ROOT), "onnx_sha256": sha256(onnx)}


# -- jobs ---------------------------------------------------------------------------------------
def _worker_init(log_dir: str) -> None:
    """Worker stderr -> <out>/logs/worker_<pid>.log (keeps warnings and the EGL teardown noise of
    exiting renderers out of the console; test exceptions are caught and reported anyway)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fd = os.open(os.path.join(log_dir, f"worker_{os.getpid()}.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)


def _job(args: tuple) -> tuple[str, dict, float]:
    """Worker: one test (or one ramp angle). Imported here so the parent never builds a renderer."""
    jid, test, cards, quick, kw = args
    from duckrt import evaluation as ev

    t0 = time.time()
    acc = ev.acceptance(quick=quick)
    try:
        if test in ev.TESTS:
            res = ev.TESTS[test](cards["walk"], acc, stand_card=cards.get("stand"), **kw)
        else:
            res = ev.KICK_TESTS[test](cards["walk"], cards["kick"], acc, stand_card=cards.get("stand"), **kw)
    except Exception as e:  # keep the other tests going; reported as a failed item
        import traceback

        res = {"error": f"{e!r}\n{traceback.format_exc()[-1500:]}", "pass": False}
    return jid, res, time.time() - t0


def _video_job(args: tuple) -> dict:
    """Worker: re-run one failed trial with the chase camera recording."""
    test, key, cards, quick, path = args
    from duckrt import evaluation as ev
    from duckrt.config import load_yaml as ly

    acc = ev.acceptance(quick=quick)
    vc = acc["videos"]
    cam = ly(CONFIG_DIR / "camera.yaml")["chase_cam"]
    out = {"test": test, "key": key, "file": os.path.relpath(path, Path(path).parents[1]), "reproduced": None}
    try:
        with ev.recording(path, vc["fps"], cam):
            if test == "symmetry":
                c = acc["walk"]["symmetry"]
                ev.walk_clip(cards["walk"], key["cmd"], c["warmup_s"] + c["straight"]["duration_s"], seed=key["seed"])
            elif test in ev.TESTS:
                kw = {"angles": [key[0]]} if test == "ramp" else {}
                r = ev.TESTS[test](cards["walk"], acc, stand_card=cards.get("stand"), only=[key], **kw)
                out["reproduced"] = r.get("pass") is False
            else:
                r = ev.KICK_TESTS[test](cards["walk"], cards["kick"], acc, stand_card=cards.get("stand"), only=[key])
                out["reproduced"] = not any(g["success"] for g in r.get("grid", []))
    except Exception as e:
        out["error"] = repr(e)
    return out


def failed_keys(test: str, r: dict, acc: dict) -> list:
    """Trial keys of failed trials (for re-runs with video), worst first where it matters."""
    if "error" in r:
        return []
    if test == "random":
        return [f["seed"] for f in r["falls"]]
    if test == "stop":
        return [[t["ci"], t["seed"]] for t in sorted(r["trials"], key=lambda t: -(t["stop_s"] or 99.0)) if not t["pass"]]
    if test == "ramp":
        gate = [float(a) for a in acc["walk"]["ramp"]["gate_angles_deg"]]
        return [[v["angle"], v["direction"], i] for v in r["results"].values() if float(v["angle"]) in gate for i in v["failed_runs"]]
    if test in ("head_override",):
        return [t["seed"] for t in r["trials"] if not t["pass"]]
    if test == "switch":
        return [t["trial"] for t in r["trials"] if not t["pass"]]
    if test == "symmetry" and not r.get("straight_pass", True):
        s = acc["walk"]["symmetry"]["straight"]
        worst = max(range(len(r["straight_yaw_rates"])), key=lambda i: abs(r["straight_yaw_rates"][i]))
        return [{"cmd": s["cmd"], "seed": worst}]
    if test == "kick_window":
        return [g["cell"] for g in r["grid"] if not g["success"]]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--stand", help="STAND 用的策略（默认：walk 策略零命令）")
    ap.add_argument("--kick")
    ap.add_argument("--config", type=Path, help="walk / stand 输入为 ONNX 时的训练 run 目录")
    ap.add_argument("--kick-config", type=Path, help="kick 输入为 ONNX 时的训练 run 目录")
    ap.add_argument("--task", help="导出 checkpoint 用的 mjlab 任务名（默认从 run 目录推断）")
    ap.add_argument("--rl-config", help="rl_train/configs/ 里的配置名（默认从 run 目录名推断；训练时改过 RUN_NAME 就要给）")
    ap.add_argument("--quick", action="store_true", help="缩小规模（acceptance.yaml 的 quick:），门槛不变")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--tests", nargs="*", help=f"只跑这些：{' '.join(WALK_TESTS + KICK_TESTS)}")
    ap.add_argument("--workers", type=int, help="进程数上限（默认 acceptance.yaml runner.workers）；还受可用内存限制")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()

    unknown = set(args.tests or []) - set(WALK_TESTS + KICK_TESTS)
    if unknown:
        raise InputError(f"不认识的测试：{sorted(unknown)}（可选：{' '.join(WALK_TESTS + KICK_TESTS)}）")
    t0 = time.time()
    name = args.name or "+".join(Path(a).stem for a in (args.walk, args.stand, args.kick) if a) + ("_quick" if args.quick else "")
    out = (args.out or OUTPUT_DIR / "acceptance" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    kick_rl = args.rl_config if args.rl_config and args.rl_config.startswith("kick") else None
    walk_rl = args.rl_config if args.rl_config and not kick_rl else None
    walk_card, _ = resolve_policy(args.walk, "walk", args.config, args.task, out, walk_rl)
    cards["walk"] = str(walk_card)
    if args.stand:
        cards["stand"] = str(resolve_policy(args.stand, "walk", args.config, args.task, out, walk_rl)[0])
    if args.kick:
        cards["kick"] = str(resolve_policy(args.kick, "kick", args.kick_config, None, out, kick_rl)[0])

    from duckrt.evaluation import acceptance

    acc = acceptance(quick=args.quick)
    tests = [t for t in WALK_TESTS + (KICK_TESTS if args.kick else []) if args.tests is None or t in args.tests]
    jobs = []
    for t in tests:
        if t == "ramp":  # the slowest test: one job per angle
            r = acc["walk"]["ramp"]
            jobs += [(f"ramp@{a:g}", "ramp", cards, args.quick, {"angles": [a]}) for a in r["gate_angles_deg"] + r["record_angles_deg"]]
        else:
            jobs.append((t, t, cards, args.quick, {}))
    rn = acc["runner"]
    workers = max(1, min(args.workers or rn["workers"], len(jobs), int((mem_available_gb() - rn["reserve_gb"]) / rn["gb_per_worker"])))
    print(f"[accept] {name}: {len(jobs)} jobs on {workers} workers ({mem_available_gb():.1f} GB available){' [quick]' if args.quick else ''}",
          flush=True)

    results: dict[str, dict] = {}
    timing: dict[str, float] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init, initargs=(str(out / "logs"),)) as ex:
        futs = {ex.submit(_job, j): j[0] for j in jobs}
        for f in as_completed(futs):
            try:
                jid, res, sec = f.result()
            except BrokenProcessPool as e:  # a worker died (e.g. OOM): record, do not hang
                jid, res, sec = futs[f], {"error": f"worker died: {e!r}", "pass": False}, 0.0
            timing[jid] = round(sec, 1)
            print(f"  {jid}: {'ERROR' if 'error' in res else ('pass' if res.get('pass') else ('recorded' if res.get('pass') is None else 'FAIL'))}"
                  f" ({sec:.0f} s)", flush=True)
            if jid.startswith("ramp@"):
                agg = results.setdefault("ramp", {"results": {}, "pass": None})
                if "error" in res:
                    results["ramp"] = {**agg, "error": res["error"]}
                else:
                    agg["results"].update(res["results"])
            else:
                results[jid] = res
    if "ramp" in results and "error" not in results["ramp"]:
        gates = [v for v in results["ramp"]["results"].values() if float(v["angle"]) in [float(a) for a in acc["walk"]["ramp"]["gate_angles_deg"]]]
        results["ramp"]["pass"] = all(v["success"] == v["n"] for v in gates) if gates else None

    items = report.build_items(results, acc)
    verdict = report.verdict(items)

    videos = []
    if not args.no_videos:
        vjobs = []
        for test, r in results.items():
            for i, key in enumerate(failed_keys(test, r, acc)[: acc["videos"]["max_per_test"]]):
                ks = re.sub(r"[^0-9A-Za-z_.-]+", "_", json.dumps(key))[:40].strip("_")
                vjobs.append((test, key, cards, args.quick, out / "videos" / f"{test}_{ks}.mp4"))
        if vjobs:
            print(f"[accept] {len(vjobs)} failure videos", flush=True)
            with ProcessPoolExecutor(min(workers, len(vjobs)), mp_context=ctx, initializer=_worker_init,
                                     initargs=(str(out / "logs"),)) as ex:
                videos = [f.result() for f in as_completed([ex.submit(_video_job, v) for v in vjobs])]
            videos.sort(key=lambda v: (v["test"], v["file"]))

    notes = []
    if args.kick and "kick_handoff" in results and results["kick_handoff"].get("falls"):
        notes.append("kick_handoff 是一次连续运行，失败不单独录视频；需要时用 scripts/kick_test.py 看。")
    rep = {
        "name": name, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "quick": args.quick,
        "verdict": verdict, "items": items, "inputs": {k: card_info(Path(v)) for k, v in cards.items()},
        "acceptance_sha256": sha256(CONFIG_DIR / "acceptance.yaml"), "elapsed_s": time.time() - t0, "workers": workers,
        "job_seconds": timing, "videos": videos, "notes": notes, "results": results,
    }
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float) + "\n")
    (out / "report.md").write_text(report.render_md(rep))
    print(report.render_md(rep).split("\n")[0])
    print(f"[accept] {out / 'report.md'} ({rep['elapsed_s']:.0f} s)")
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
