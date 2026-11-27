"""Shared fixture: load a scenario once per test session (loading takes about a minute)."""
from pathlib import Path

import pytest

from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment

SCENARIOS = Path("smart_buildings/training_scenarios")
_cache = {}


@pytest.fixture(scope="session")
def load_scenario():
    def _load(building, weather):
        key = (building, weather)
        if key not in _cache:
            config = SCENARIOS / building / weather / "config.gin"
            env = load_hybrid_environment(config)
            meta = create_building_meta_data(env, config.parent)
            _cache[key] = (env, meta)
        return _cache[key]

    return _load
