"""Config loading. Relative paths in configs resolve against the data_collect_sim root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load_sim_cfg(path: str | Path = CONFIG_DIR / "sim.yaml") -> dict[str, Any]:
    return load_yaml(path)


def output_dir(*parts: str) -> Path:
    d = OUTPUT_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
