"""并排对比多个验收报告（scripts/accept.py 写的 report.json）。

    uv run python scripts/compare.py outputs/acceptance/baseline outputs/acceptance/<new> [...] [--out compare.md]

参数可以是 report.json 或它所在的目录。每行一个验收项，每列一个报告：数值 + ✔ / ✘（只记录的项没有符号）。
第一个报告当基线：门槛项里基线通过、新报告不通过的，标 ⚠ 退步。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duckrt.report import FAIL, PASS  # noqa: E402
from duckrt.testing import md_table  # noqa: E402


def load(p: Path) -> dict:
    p = p / "report.json" if p.is_dir() else p
    rep = json.loads(p.read_text())
    rep["_path"] = str(p)
    return rep


def compare(reports: list[dict]) -> str:
    order: list[str] = []
    names: dict[str, tuple[str, str]] = {}
    for r in reports:
        for it in r["items"]:
            if it["id"] not in names:
                order.append(it["id"])
                names[it["id"]] = (it["skill"], it["name"])
    by = [{it["id"]: it for it in r["items"]} for r in reports]
    header = ["技能", "项目"] + [f"{r['name']}{' (quick)' if r.get('quick') else ''}" for r in reports]
    rows = [["", "**总结论**"] + [("**合格**" if r["verdict"]["pass"] else f"**不合格**（{r['verdict']['n_failed']} 项）") for r in reports]]
    for iid in order:
        row = list(names[iid])
        base = by[0].get(iid)
        for k, b in enumerate(by):
            it = b.get(iid)
            if it is None:
                row.append("—")
                continue
            mark = "" if it["pass"] is None else (PASS if it["pass"] else FAIL)
            worse = k > 0 and base is not None and base["pass"] is True and it["pass"] is False
            row.append(f"{it['value']} {mark}{' ⚠ 退步' if worse else ''}")
        rows.append(row)
    lines = ["# 验收对比", "", "报告：" + "；".join(f"`{r['_path']}`" for r in reports), ""]
    if any(r.get("quick") for r in reports) and not all(r.get("quick") for r in reports):
        lines += ["> 注意：quick 和完整模式的报告混在一起比，规模不同。", ""]
    lines += [md_table(header, rows), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    md = compare([load(p) for p in args.reports])
    if args.out:
        args.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
