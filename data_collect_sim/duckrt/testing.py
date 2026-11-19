"""Small helpers shared by the test scripts (videos, foot contacts, tables)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402


class VideoWriter:
    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def add(self, frame: np.ndarray) -> None:
        self.w.append_data(frame)

    def close(self) -> None:
        self.w.close()


def foot_floor_contacts(backend, foot_geoms: list[str], floor_names=("floor",)) -> list[bool]:
    m, d = backend.model, backend.data
    gids = [m.geom(g).id for g in foot_geoms]
    floor = {m.geom(n).id for n in floor_names if _exists(m, n)}
    hit = [False] * len(gids)
    for i in range(d.ncon):
        c = d.contact[i]
        for k, g in enumerate(gids):
            if (c.geom1 == g and (not floor or c.geom2 in floor)) or (c.geom2 == g and (not floor or c.geom1 in floor)):
                hit[k] = True
    return hit


def _exists(m, name: str) -> bool:
    try:
        m.geom(name)
        return True
    except KeyError:
        return False


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"
