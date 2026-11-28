"""Run the shipped policies through the interface contract.

python -m pytest tests/test_policy_contract.py -v -s

A failure here for a shipped baseline is information, not necessarily a bug in
your code: it means the evaluator tolerates something the strict check does not
(for example values slightly outside [-1, 1]). Send the message to the team.
"""
from pathlib import Path

import numpy as np
import pytest

from smart_buildings.evaluate import POLICY_MAP, initialize_policy
from src.policy_contract import check_policy

CASES = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_2", "atlanta_3"),
]


@pytest.mark.parametrize("building,weather", CASES)
@pytest.mark.parametrize("policy_key", ["transfer", "pid_water"])
def test_shipped_policy_follows_contract(load_scenario, building, weather, policy_key):
    path = Path(f"data/raw/{building}__{weather}__pid_water__600.npz")
    if not path.exists():
        pytest.skip(f"missing {path}")
    env, meta = load_scenario(building, weather)
    obs = np.load(path)["obs"][:60]

    def make():
        return initialize_policy(POLICY_MAP[policy_key], meta, env.action_spec())

    # strict_dtype=False: the evaluator casts dtypes itself (validate_action)
    assert check_policy(make, meta, obs, strict_dtype=False)
