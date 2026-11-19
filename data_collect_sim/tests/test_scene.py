"""Scene modules / layouts: schema validation, overlap checks, privileged queries."""

import copy

import numpy as np
import pytest

from duckrt.backend_sim import SimBackend
from duckrt.config import load_sim_cfg
from duckrt.policy_card import load_card
from duckrt.scene.builder import LayoutError, load_layout
from duckrt.scene.modules import LAYOUT_DIR, ModuleError

LAYOUTS = sorted(p.stem for p in LAYOUT_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", LAYOUTS)
def test_layout_builds(name):
    b = SimBackend(load_sim_cfg(), layout=load_layout(name))
    ids = {s["id"] for s in b.module_states()}
    assert "arena" in ids
    assert not b.has_nan()


def test_ball_defaults_from_kick_card():
    b = SimBackend(load_sim_cfg(), layout=load_layout("demo_v1"))
    ball = b.module_state("ball")
    kc = load_card("kick").task["ball"]
    assert ball["params"]["radius_m"] == pytest.approx(kc["radius"])
    assert b.model.body_mass[b.model.body("mod_ball").id] == pytest.approx(kc["mass"])
    assert ball["pos"][2] == pytest.approx(kc["radius"], abs=1e-6)


def test_goal_line_site_and_flag_centre():
    b = SimBackend(load_sim_cfg(), layout=load_layout("demo_v1"))
    goal = b.module_state("goal")
    assert "mod_goal_goal_line" in goal["sites"]
    assert np.allclose(b.module_state("flag_A")["sites"]["mod_flag_A_centre"][:2], [1.05, -0.8])


def _bad(layout_mod):
    lay = load_layout("flat_single_flag")
    layout_mod(lay)
    return lambda: SimBackend(load_sim_cfg(), layout=lay)


def test_missing_required_param():
    def mod(lay):
        lay["modules"].append({"id": "o", "module": "box_20cm", "pos": [1.0, 0.8], "params": {"size_m": None}})
    with pytest.raises(ModuleError):
        _bad(mod)()


def test_out_of_range_param():
    with pytest.raises(ModuleError, match="out of range"):
        SimBackend(load_sim_cfg(), layout=load_layout("ramp_test", {"ramp": {"slope_length_m": 0.3}}))


def test_unknown_param():
    with pytest.raises(ModuleError, match="unknown parameters"):
        SimBackend(load_sim_cfg(), layout=load_layout("ramp_test", {"ramp": {"slope": 5}}))


def test_overlap_rejected():
    def mod(lay):
        lay["modules"].append({"id": "o", "module": "box_20cm", "pos": [0.6, 0.1]})
    with pytest.raises(LayoutError, match="overlaps"):
        _bad(mod)()


def test_robot_start_clearance():
    def mod(lay):
        lay["modules"].append({"id": "o", "module": "box_20cm", "pos": [-0.8, 0.0]})
    with pytest.raises(LayoutError, match="robot start"):
        _bad(mod)()
