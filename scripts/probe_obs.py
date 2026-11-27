"""Probe real observations through SimulationTracker, the same path a policy uses.

Run from the repo root:
    python scripts/probe_obs.py --building building_train_1 --weather fairbanks_1
"""
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    POLICY_MAP,
    SimulationTracker,
    create_building_meta_data,
    get_policy_action,
    initialize_policy,
    load_hybrid_environment,
    validate_action,
)


def to_vec(x):
    if isinstance(x, tuple):
        x = x[0]
    return np.asarray(x, dtype=np.float64).ravel()


def group_table(names, obs):
    groups = OrderedDict()
    for i, name in enumerate(names):
        groups.setdefault(re.sub(r"\d+", "{n}", name), []).append(i)
    print(f"{'field group':62s} {'n':>4s} {'min':>10s} {'mean':>10s} {'max':>10s}")
    for pattern, idxs in groups.items():
        v = obs[idxs]
        print(f"{pattern:62s} {len(idxs):4d} {v.min():10.3f} {v.mean():10.3f} {v.max():10.3f}")


def show_fields(names, obs, keys):
    idx = {n: i for i, n in enumerate(names)}
    for key in keys:
        if key in idx:
            print(f"  [{idx[key]:3d}] {key:55s} = {obs[idx[key]]:.4f}")
        else:
            print(f"  (missing) {key}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--building", required=True)
    p.add_argument("--weather", required=True)
    p.add_argument("--policy", default="pid_water")
    p.add_argument("--steps", type=int, default=12)  # 12 steps = 1 hour
    args = p.parse_args()

    config = Path("smart_buildings/training_scenarios") / args.building / args.weather / "config.gin"
    env = load_hybrid_environment(config)
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0

    meta = create_building_meta_data(env, config.parent)
    names = meta["observation_field_names"]
    policy = initialize_policy(POLICY_MAP[args.policy], meta, env.action_spec())

    tracker.reset()
    raw = tracker.last_obs
    obs = to_vec(raw)
    print(f"raw obs: type={type(raw).__name__}, dtype={getattr(raw, 'dtype', None)}, "
          f"shape={getattr(raw, 'shape', None)}")
    print(f"len(obs)={len(obs)}, len(names)={len(names)}, all finite={np.isfinite(obs).all()}")

    zone_keys = [
        "VAV_1_zone_air_temperature_sensor",
        "VAV_1_zone_air_cooling_temperature_setpoint_sensor",
        "VAV_1_zone_air_heating_temperature_setpoint_sensor",
        "VAV_1_supply_air_damper_percentage_command",
        "VAV_1_reheat_valve_percentage_command",
        "VAV_1_supply_air_flowrate_setpoint",
        "ahs_ahu_1_outside_air_temperature_sensor",
        "ahs_ahu_1_supply_air_static_pressure_sensor",
        "ahs_ahu_1_supply_air_temperature_sensor",
        "hws_supply_water_temperature_sensor",
        "hws_differential_pressure_sensor",
        "hod_cos_000", "hod_sin_000", "dow_cos_000", "dow_sin_000",
        "comfort_mode_now", "comfort_mode_soon", "num_occupants",
    ]

    print("\n===== OBS AT RESET =====")
    group_table(names, obs)
    print("\n--- key fields at reset ---")
    show_fields(names, obs, zone_keys)

    for t in range(args.steps):
        action = validate_action(get_policy_action(policy, obs), env.action_spec())
        if t == 0:
            for k, v in action.items():
                print(f"\naction[{k}]: dtype={v.dtype}, shape={v.shape}")
        obs, reward, terminated, truncated, _ = tracker.step(action)
        obs = to_vec(obs)
        if terminated or truncated:
            break

    print(f"\n===== OBS AFTER {t + 1} STEPS (last reward={reward}) =====")
    group_table(names, obs)
    print("\n--- key fields after steps ---")
    show_fields(names, obs, zone_keys)


if __name__ == "__main__":
    main()