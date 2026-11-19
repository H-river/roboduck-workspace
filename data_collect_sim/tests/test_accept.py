"""scripts/accept.py and duckrt/report.py: gates come from configs/acceptance.yaml, report format,
a known-bad policy is judged not acceptable, ONNX without a training config is a clear input error."""

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from duckrt import report
from duckrt.config import CONFIG_DIR, ROOT
from duckrt.evaluation import acceptance

ACCEPT = [sys.executable, str(ROOT / "scripts" / "accept.py")]
ENV = {**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}


def _stop_result(stop_s: float, drift_m: float) -> dict:
    t = {"ci": 0, "cmd": [0.2, 0, 0], "clipped": [0.2, 0, 0], "seed": 0, "stop_s": stop_s, "drift_m": drift_m, "fell": False, "pass": True}
    return {"trials": [t], "max_stop_s": stop_s, "max_drift_m": drift_m, "n_pass": 1, "n": 1, "pass": True}


def _kick_result(rate: float, bias: float) -> dict:
    n = 10
    grid = [{"success": i < round(rate * n), "dir_err_deg": bias, "fell": False, "cell": i} for i in range(n)]
    return {"n": n, "success_rate": rate, "bias_deg": bias, "mean_abs_err_deg": abs(bias), "falls": 0, "grid": grid}


def _item(items, iid):
    return next(i for i in items if i["id"] == iid)


def test_gates_read_from_config():
    acc = acceptance()
    res = {"stop": _stop_result(0.55, 0.01), "kick_window": _kick_result(0.85, 2.0),
           "symmetry": {"straight_yaw_rates": [0.02, -0.025], "turns": [], "lateral": {"v_pos": 0.1, "v_neg": -0.1, "ratio": 1.0}}}
    items = report.build_items(res, acc)
    assert _item(items, "stop")["pass"] and _item(items, "kick_rate")["pass"] and _item(items, "kick_bias")["pass"]
    assert _item(items, "straight_yaw")["pass"]

    tight = copy.deepcopy(acc)  # same metrics, stricter thresholds -> every verdict flips
    tight["walk"]["stop"]["max_stop_s"] = 0.5
    tight["kick"]["gate_success_rate"] = 0.9
    tight["kick"]["gate_mean_abs_dir_deg"] = 1.5
    tight["walk"]["symmetry"]["straight"]["max_abs_yaw_rate"] = 0.02
    items = report.build_items(res, tight)
    assert not any(_item(items, k)["pass"] for k in ("stop", "kick_rate", "kick_bias", "straight_yaw"))
    assert "0.5 s" in _item(items, "stop")["threshold"] and "90%" in _item(items, "kick_rate")["threshold"]


def test_quick_shrinks_sizes_keeps_gates():
    full, quick = acceptance(), acceptance(quick=True)
    assert quick["walk"]["random"]["seeds"] < full["walk"]["random"]["seeds"]
    for path in (("walk", "stop", "max_stop_s"), ("walk", "stop", "max_drift_m"), ("walk", "symmetry", "turn", "ratio_range"),
                 ("walk", "symmetry", "straight", "max_abs_yaw_rate"), ("kick", "gate_success_rate"), ("kick", "gate_mean_abs_dir_deg")):
        a, b = full, quick
        for k in path:
            a, b = a[k], b[k]
        assert a == b, path


def test_no_gate_literals_in_code():
    """Gate thresholds of acceptance.yaml must not appear as numeric literals in the test / verdict code."""
    gates = {0.6, 0.03, 0.8, 1.25, 3.0, 0.07, 0.7, 1.3, 0.05, 10.0, 0.3}
    for f in ("duckrt/evaluation.py", "duckrt/report.py", "scripts/accept.py"):
        consts = {n.value for n in ast.walk(ast.parse((ROOT / f).read_text())) if isinstance(n, ast.Constant)
                  and isinstance(n.value, float)}
        assert not (consts & gates), f"{f}: {sorted(consts & gates)}"


def test_report_format():
    acc = acceptance()
    res = {"random": {"no_fall": 19, "n": 20, "falls": [{"seed": 3, "t": 12.0}], "pass": False}, "stop": _stop_result(0.4, 0.01)}
    items = report.build_items(res, acc)
    rep = {"name": "unit", "created": "now", "quick": False, "verdict": report.verdict(items), "items": items,
           "inputs": {"walk": {"card": "c.yaml", "onnx": "p.onnx", "onnx_sha256": "0" * 64}}, "acceptance_sha256": "1" * 64,
           "elapsed_s": 1.0, "workers": 1, "videos": [], "notes": [], "results": res}
    md = report.render_md(rep)
    assert md.splitlines()[0].startswith("# 验收结论：**不合格**") and "random" in md.splitlines()[0]
    header = next(line for line in md.splitlines() if line.startswith("| 技能"))
    assert header == "| 技能 | 项目 | 数值 | 门槛 | 结果 |"
    rows = [line for line in md.splitlines() if line.startswith("| walk")]
    assert len(rows) == 2 and all(line.count("|") == 6 for line in rows)
    assert rep["verdict"] == {"pass": False, "n_gated": 2, "n_failed": 1, "failed": ["random"]}
    json.loads(json.dumps(rep))  # report.json is plain JSON
    for it in items:
        assert set(it) == {"skill", "id", "name", "value", "threshold", "pass", "test"}


def _run_accept(tmp_path, card, name):
    out = tmp_path / name
    r = subprocess.run(ACCEPT + ["--walk", str(card), "--quick", "--tests", "random", "--no-videos", "--workers", "1", "--out", str(out)],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    return r, (json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else None)


@pytest.mark.slow
def test_known_bad_policy_fails(tmp_path):
    """The same policy with its actions multiplied by 20 (card action.scale) must be judged not acceptable."""
    good = CONFIG_DIR / "policies" / "walk_flat_2600.yaml"
    card = yaml.safe_load(good.read_text())
    card["action"]["scale"] = float(card["action"]["scale"]) * 20.0
    card["name"] = "walk_flat_2600_x20"
    bad = tmp_path / "walk_flat_2600_x20.yaml"
    bad.write_text(yaml.safe_dump(card, sort_keys=False))

    r, rep = _run_accept(tmp_path, bad, "bad")
    assert r.returncode == 1, r.stderr[-2000:]
    assert rep["verdict"]["pass"] is False and _item(rep["items"], "random")["pass"] is False
    r, rep = _run_accept(tmp_path, good, "good")   # control: the unscaled policy passes the same test
    assert r.returncode == 0, r.stderr[-2000:]
    assert _item(rep["items"], "random")["pass"] is True


def test_onnx_without_config_is_an_input_error(tmp_path):
    onnx = tmp_path / "lonely.onnx"
    onnx.write_bytes((ROOT / "outputs" / "policies" / "walk_flat_model2600.onnx").read_bytes())
    r = subprocess.run(ACCEPT + ["--walk", str(onnx), "--quick", "--out", str(tmp_path / "x")], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "--config" in r.stderr
