"""Acceptance verdicts and reports (scripts/accept.py, scripts/compare.py).

build_items() turns the raw metrics of duckrt.evaluation into one row per acceptance item and decides
pass / fail from the thresholds in the `acc` dict (configs/acceptance.yaml) -- never from numbers in
code. Items with pass = None are recorded only (no gate).
"""

from __future__ import annotations

import math
from typing import Any

from duckrt.testing import md_table

PASS, FAIL, RECORD = "✔", "✘", "记录"


def _item(skill: str, iid: str, name: str, value: str, threshold: str, ok: bool | None, test: str) -> dict:
    return {"skill": skill, "id": iid, "name": name, "value": value, "threshold": threshold,
            "pass": None if ok is None else bool(ok), "test": test}


def _err(skill: str, iid: str, name: str, test: str, r: dict) -> dict:
    return _item(skill, iid, name, f"ERROR: {r['error'][:200]}", "测试必须能跑完", False, test)


def _f(x, nd=3) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def walk_items(res: dict[str, Any], acc: dict) -> list[dict]:
    w = acc["walk"]
    out: list[dict] = []
    if "random" in res:
        r, c = res["random"], w["random"]
        if "error" in r:
            out.append(_err("walk", "random", "随机命令不摔", "random", r))
        else:
            out.append(_item("walk", "random", f"{r['n']} 个 seed × {c['duration_s']:g} s 随机命令",
                             f"{r['no_fall']}/{r['n']} 不摔", f"{r['n']}/{r['n']} 不摔", r["n"] > 0 and r["no_fall"] == r["n"], "random"))
    if "stop" in res:
        r, c = res["stop"], w["stop"]
        if "error" in r:
            out.append(_err("walk", "stop", "切零命令后停住", "stop", r))
        else:
            ok_t = [(not t["fell"]) and t["stop_s"] is not None and t["stop_s"] <= c["max_stop_s"] + 1e-9
                    and t["drift_m"] is not None and t["drift_m"] < c["max_drift_m"] for t in r["trials"]]
            worst = max((t["stop_s"] if t["stop_s"] is not None else math.inf) for t in r["trials"])
            drift = max((t["drift_m"] or 0.0) for t in r["trials"])
            out.append(_item("walk", "stop", f"切零命令后停住（{len(c['commands'])} 个命令 × {c['seeds']} seed）",
                             f"{sum(ok_t)}/{len(ok_t)} 通过；最长 {'未停住' if math.isinf(worst) else _f(worst, 2) + ' s'}，最大漂移 {drift * 100:.1f} cm",
                             f"每次 ≤ {c['max_stop_s']:g} s 停住，之后 {c['drift_window_s']:g} s 内漂移 < {c['max_drift_m'] * 100:g} cm",
                             len(ok_t) > 0 and all(ok_t), "stop"))
    if "symmetry" in res:
        r, c = res["symmetry"], w["symmetry"]
        if "error" in r:
            out.append(_err("walk", "symmetry", "对称性", "symmetry", r))
        else:
            s = c["straight"]
            ws = r["straight_yaw_rates"]
            out.append(_item("walk", "straight_yaw", f"直走偏航（vx = {s['cmd'][0]:g}，ωz = 0，{len(ws)} 个 seed）",
                             f"max |ωz| = {max(abs(x) for x in ws):.3f} rad/s",
                             f"每个 seed |ωz| < {s['max_abs_yaw_rate']:g} rad/s", all(abs(x) < s["max_abs_yaw_rate"] for x in ws), "symmetry"))
            lo, hi = c["turn"]["ratio_range"]
            for t in r["turns"]:
                kind = "原地" if abs(t["cmd"][0]) < 1e-9 and abs(t["cmd"][1]) < 1e-9 else "边走边"
                out.append(_item("walk", f"turn_ratio_{'inplace' if kind == '原地' else 'walking'}",
                                 f"±ωz 转速比（{kind}转，命令 {t['cmd']}）",
                                 f"{t['ratio']:.2f}（+{t['w_pos']:.3f} / {t['w_neg']:.3f} rad/s）", f"{lo:g}–{hi:g}",
                                 lo <= t["ratio"] <= hi, "symmetry"))
            lat = r["lateral"]
            out.append(_item("walk", "lateral_ratio", f"±vy 速度比（命令 {c['lateral']['cmd']}）",
                             f"{lat['ratio']:.2f}（+{lat['v_pos']:.3f} / {lat['v_neg']:.3f} m/s）", "只记录", None, "symmetry"))
    if "tracking" in res:
        r = res["tracking"]
        if "error" in r:
            out.append(_err("walk", "tracking", "命令跟踪", "tracking", r))
        else:
            e = [[abs(x) for x in row["err"]] for row in r["rows"]]
            out.append(_item("walk", "tracking", f"命令跟踪误差（{len(e)} 档，明细见下表）",
                             f"max |Δvx| {max(x[0] for x in e):.3f}，|Δvy| {max(x[1] for x in e):.3f}，|Δωz| {max(x[2] for x in e):.3f}",
                             "只记录", None, "tracking"))
    if "ramp" in res:
        r, c = res["ramp"], w["ramp"]
        if "error" in r:
            out.append(_err("walk", "ramp", "坡道", "ramp", r))
        else:
            for k, v in r["results"].items():
                gate = float(v["angle"]) in [float(a) for a in c["gate_angles_deg"]]
                d = "上坡" if v["direction"] == "up" else "下坡"
                out.append(_item("walk", f"ramp_{k}", f"{v['angle']:g}° 坡{d}（航向保持）", f"{v['success']}/{v['n']}（摔 {v['falls']}）",
                                 f"{v['n']}/{v['n']}" if gate else "只记录", (v["success"] == v["n"]) if gate else None, "ramp"))
    if "head_override" in res:
        r, c = res["head_override"], w["head_override"]
        if "error" in r:
            out.append(_err("walk", "head_override", "头部接管", "head_override", r))
        else:
            st, wa = c["stable"], c["walk_after"]
            tr = r["trials"]
            ok = [(not t["fell"]) and t["max_tilt_deg"] < st["max_tilt_deg"] and t["max_disp_m"] < st["max_displacement_m"]
                  and wa["min_ratio"] <= t["ratio"] <= wa["max_ratio"] for t in tr]
            ratios = [t["ratio"] for t in tr]
            out.append(_item("walk", "head_override", f"站立时头部接管低头 {c['pitch_down_deg']:g}°、保持 {c['hold_s']:g} s",
                             f"{sum(ok)}/{len(ok)}；最大倾斜 {max(t['max_tilt_deg'] for t in tr):.1f}°，"
                             f"起步速度比 {min(ratios):.2f}–{max(ratios):.2f}",
                             f"{len(ok)}/{len(ok)}（倾斜 < {st['max_tilt_deg']:g}°、位移 < {st['max_displacement_m'] * 100:g} cm、"
                             f"起步速度比 {wa['min_ratio']:g}–{wa['max_ratio']:g}）", len(ok) > 0 and all(ok), "head_override"))
    if "camera" in res:
        r, c = res["camera"], w["camera"]
        if "error" in r:
            out.append(_err("walk", "camera", "相机", "camera", r))
        else:
            f = r["flags"][f"{c['gate_flag_m']:g}"]
            out.append(_item("walk", "flag_visible", f"站立时 {c['gate_flag_m']:g} m 处旗垫在画面内",
                             "完全可见" if f["fully_visible"] else f"不完全可见（{f['corners_in_image']}/4 个角）", "完全可见",
                             f["fully_visible"], "camera"))
            out.append(_item("walk", "cam_pitch_diff", "站立与行走（vx 0.2）的相机俯角差",
                             f"{r['stand_minus_walk_pitch_deg']:+.1f}°（站 {r['stand_pitch_deg']:.1f}°，走 {_f(r['walk_pitch_deg'], 1)}°）",
                             "只记录", None, "camera"))
    if "switch" in res:
        r, c = res["switch"], w["switch"]
        if "error" in r:
            out.append(_err("walk↔stand", "switch", "热切换", "switch", r))
        else:
            ok = [not t["fell"] for t in r["trials"]]
            out.append(_item("walk↔stand", "switch", f"walk ↔ stand 热切换（{len(ok)} 次 站→走→站→走）",
                             f"{sum(ok)}/{len(ok)} 不摔；最大倾斜 {max(t['max_tilt_deg'] for t in r['trials']):.1f}°",
                             f"{len(ok)}/{len(ok)} 不摔", len(ok) > 0 and all(ok), "switch"))
    return out


def kick_items(res: dict[str, Any], acc: dict) -> list[dict]:
    k = acc["kick"]
    out: list[dict] = []
    if "kick_window" in res:
        r = res["kick_window"]
        if "error" in r:
            out.append(_err("kick", "kick_window", "踢球窗口", "kick_window", r))
        else:
            w = k["window"]
            succ = sum(g["success"] for g in r["grid"])
            out.append(_item("kick", "kick_rate", f"球在 x ∈ [{w['x'][0]:g}, {w['x'][1]:g}] m、y 在踢球脚 ±{w['y_offset'][1]:g} m 内",
                             f"{r['success_rate'] * 100:.0f}%（{succ}/{r['n']}，摔 {r['falls']}）",
                             f"≥ {k['gate_success_rate'] * 100:g}%（成功 = 球速 ≥ {k['success']['min_ball_speed']:g} m/s 且方向偏差 ≤ {k['success']['max_dir_err_deg']:g}°）",
                             r["success_rate"] >= k["gate_success_rate"], "kick_window"))
            b = r["bias_deg"]
            out.append(_item("kick", "kick_bias", "踢出方向平均偏差（成功的那些）", "无成功的踢球" if math.isnan(b) else f"{b:+.1f}°",
                             f"|平均偏差| < {k['gate_mean_abs_dir_deg']:g}°", (not math.isnan(b)) and abs(b) < k["gate_mean_abs_dir_deg"],
                             "kick_window"))
    if "kick_head_down" in res:
        r = res["kick_head_down"]
        if "error" in r:
            out.append(_err("kick", "kick_head_down", "低头测球后再踢", "kick_head_down", r))
        else:
            out.append(_item("kick", "kick_head_down", f"头部接管低头 {k['head_down_then_kick']['pitch_down_deg']:g}° 后再踢",
                             f"低头后 {r['head_down_rate'] * 100:.0f}% vs 直接踢 {r['direct_rate'] * 100:.0f}%", "低头后成功率 ≥ 直接踢",
                             r["head_down_rate"] >= r["direct_rate"], "kick_head_down"))
    if "kick_handoff" in res:
        r = res["kick_handoff"]
        if "error" in r:
            out.append(_err("kick", "kick_handoff", "walk → stand → kick → stand → walk", "kick_handoff", r))
        else:
            out.append(_item("kick", "kick_handoff", "walk → stand → kick → stand → walk",
                             f"{r['cycles'] - r['falls']}/{r['cycles']} 不摔（踢中 {r['kick_success']}）", f"{r['cycles']}/{r['cycles']} 不摔",
                             r["falls"] == 0, "kick_handoff"))
    return out


def build_items(res: dict[str, Any], acc: dict) -> list[dict]:
    return walk_items(res, acc) + kick_items(res, acc)


def verdict(items: list[dict]) -> dict:
    gated = [i for i in items if i["pass"] is not None]
    failed = [i for i in gated if not i["pass"]]
    return {"pass": bool(gated) and not failed, "n_gated": len(gated), "n_failed": len(failed), "failed": [i["id"] for i in failed]}


def render_md(rep: dict) -> str:
    v = rep["verdict"]
    head = (f"# 验收结论：**合格**（{v['n_gated']} 项门槛全部通过）" if v["pass"]
            else f"# 验收结论：**不合格**（{v['n_gated']} 项门槛中 {v['n_failed']} 项不通过：{', '.join(v['failed'])}）")
    lines = [head, ""]
    if rep.get("quick"):
        lines += ["> **--quick 模式**：规模缩小（configs/acceptance.yaml 的 `quick:`），门槛不变。只用于训练中挑 checkpoint，不能作为定版依据。", ""]
    lines += [f"- 名称：`{rep['name']}`；时间：{rep['created']}；总耗时 {rep['elapsed_s']:.0f} s（{rep['workers']} 个进程）"]
    for role, p in rep["inputs"].items():
        lines += [f"- {role}：card `{p['card']}`，ONNX `{p['onnx']}`（sha256 {p['onnx_sha256'][:16]}…）"]
    lines += [f"- 门槛：`configs/acceptance.yaml`（sha256 {rep['acceptance_sha256'][:16]}…）", ""]
    rows = [[i["skill"], i["name"], i["value"], i["threshold"], RECORD if i["pass"] is None else (PASS if i["pass"] else FAIL)]
            for i in rep["items"]]
    lines += [md_table(["技能", "项目", "数值", "门槛", "结果"], rows), ""]
    tr = rep["results"].get("tracking")
    if tr and "rows" in tr:
        lines += ["## 命令跟踪（只记录）", "",
                  md_table(["命令 [vx, vy, ωz]", "裁剪后", "实际平均", "误差", "摔倒"],
                           [[r["cmd"], r["clipped"], r["mean"], r["err"], "是" if r["fell"] else ""] for r in tr["rows"]]), ""]
    if rep.get("videos"):
        lines += ["## 失败用例视频", ""]
        lines += [f"- `{x['file']}`：{x['test']} {x['key']}" + ("" if x.get("reproduced") is None else
                  f"（单独重放{'复现了失败' if x['reproduced'] else '没有复现失败，视频仅供参考'}）") for x in rep["videos"]]
        lines += [""]
    if rep.get("notes"):
        lines += ["## 说明", ""] + [f"- {n}" for n in rep["notes"]] + [""]
    return "\n".join(lines)
