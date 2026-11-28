"""Tests for src/units.py. Run from the repo root:  python -m pytest tests/test_units.py -v -s

The data checks use files written by scripts/data_collection.py and are skipped
when a file is missing.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.features import BuildingIndex
from src.units import (
    AHU_MIN_STATIC_PRESSURE_PA,
    DAMPER_FLOOR_NATIVE,
    ActionLayout,
    ObsUnits,
    clock_from_obs,
)

CASES = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_2", "atlanta_3"),
    ("building_train_4", "atlanta_5"),
    ("building_train_5", "atlanta_6"),
    ("building_train_7", "atlanta_8"),
    ("building_train_8", "atlanta_1"),
]


def data_path(building, weather, steps=600):
    return Path(f"data/raw/{building}__{weather}__pid_water__{steps}.npz")


@pytest.mark.parametrize("building,weather", CASES)
def test_action_layout_partition(load_scenario, building, weather):
    _, meta = load_scenario(building, weather)
    al = ActionLayout(meta)
    parts = [al.ahu_temp, al.ahu_pressure, [al.hws_dp], al.ahu_recirc, [al.hws_temp],
             al.zone_damper, al.zone_reheat]
    all_idx = np.concatenate([np.atleast_1d(p) for p in parts])
    assert sorted(all_idx.tolist()) == list(range(meta["n_continuous_action"]))
    assert (al.native_hi > al.native_lo).all()


@pytest.mark.parametrize("building,weather", CASES)
def test_roundtrips(load_scenario, building, weather):
    _, meta = load_scenario(building, weather)
    al, ou = ActionLayout(meta), ObsUnits(meta)
    rng = np.random.default_rng(0)
    cont = rng.uniform(-1, 1, al.n_cont).astype(np.float32)
    assert np.allclose(al.to_normalized(al.to_native(cont)), cont, atol=1e-5)
    obs = rng.normal(size=ou.n_obs)
    assert np.allclose(ou.to_normalized(ou.to_native(obs)), obs, atol=1e-9)
    assert np.allclose(al.to_native(np.full(al.n_cont, -1.0)), al.native_lo)
    assert np.allclose(al.to_native(np.full(al.n_cont, 1.0)), al.native_hi)


@pytest.mark.parametrize("building,weather", CASES)
def test_recorded_episode_relations(load_scenario, building, weather):
    path = data_path(building, weather)
    if not path.exists():
        pytest.skip(f"missing {path}")
    _, meta = load_scenario(building, weather)
    names = meta["observation_field_names"]
    bi, al, ou = BuildingIndex(meta), ActionLayout(meta), ObsUnits(meta)
    d = np.load(path)
    obs, cont, disc = d["obs"], d["cont"], d["disc"]
    assert obs.shape[1] == len(names) and len(obs) == len(cont) + 1

    nat = ou.to_native(obs)
    cnat = al.to_native(cont)

    # physical ranges after conversion
    temp = nat[:, bi.zone_obs["temp"]]
    assert 250 < temp.min() and temp.max() < 330, (temp.min(), temp.max())
    damper = nat[:, bi.zone_obs["damper"]]
    reheat = nat[:, bi.zone_obs["reheat"]]
    assert damper.min() >= DAMPER_FLOOR_NATIVE - 1e-4 and damper.max() <= 1 + 1e-4
    assert reheat.min() >= -1e-4 and reheat.max() <= 1 + 1e-4

    # the action sent at step t is visible in the observation at step t+1
    sp = bi.ahu_obs("supply_air_temperature_setpoint")
    assert np.allclose(nat[1:, sp], cnat[:, al.ahu_temp], atol=1e-2)
    assert np.allclose(damper[1:], np.maximum(cnat[:, al.zone_damper], DAMPER_FLOOR_NATIVE), atol=1e-4)
    assert np.allclose(reheat[1:], cnat[:, al.zone_reheat], atol=1e-4)

    # hot-water run fields are normalized {-1, 1}; AHU run command is raw {0, 1}
    hws_cmd = nat[1:, names.index("hws_supervisor_run_command")]
    assert np.array_equal(hws_cmd, disc[:, -1].astype(float))

    # static pressure floor (only verified on building_train_1)
    psp = bi.ahu_obs("supply_air_static_pressure_setpoint")
    on = disc[:, : al.num_ahus] == 1
    min_on = nat[1:, psp][on].min()
    print(f"{building}: min static pressure setpoint while AHU on = {min_on:.1f} Pa")
    if True:
        expected = np.where(on, np.maximum(cnat[:, al.ahu_pressure], AHU_MIN_STATIC_PRESSURE_PA), 0.0)
        assert np.allclose(nat[1:, psp], expected, atol=1.0)

    # clock: hour advances 5 minutes per step
    hours = np.array([clock_from_obs(o, names)[0] for o in obs[:60]])
    step = ((np.diff(hours) + 12) % 24) - 12
    assert np.allclose(step, 5 / 60, atol=1e-3)
