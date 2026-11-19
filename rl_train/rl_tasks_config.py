"""Config loading without mjlab (used by data_collect_sim's card generator)."""

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_kick_config(path) -> dict:
    path = Path(path)
    c = yaml.safe_load(path.read_text())
    if c.get("inherit"):
        c = _merge(load_kick_config(path.parent / c["inherit"]), {k: v for k, v in c.items() if k != "inherit"})
    return c
