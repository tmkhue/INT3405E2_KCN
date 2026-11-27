"""Collect one rollout episode as a raw .npz file.

Run from the repo root:
    python scripts/data_collection.py --building building_train_1 --weather fairbanks_1 --steps 600
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.evaluate import (
    POLICY_MAP,
    SimulationTracker,
    create_building_meta_data,
    get_policy_action,
    initialize_policy,
    load_hybrid_environment,
    validate_action,
)
from src.features import BuildingIndex, zone_band_kelvin, zone_temp_kelvin


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def to_vec(x):
    if isinstance(x, tuple):
        x = x[0]
    return np.asarray(x, dtype=np.float32).ravel()


def append_manifest(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def collect(args):
    if args.policy not in POLICY_MAP:
        raise SystemExit(f"unknown policy {args.policy}; choices: {sorted(POLICY_MAP)}")

    config = Path("smart_buildings/training_scenarios") / args.building / args.weather / "config.gin"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.policy if args.perturb <= 0 else f"{args.policy}_p{int(args.perturb * 100):02d}_s{args.seed}"
    out_path = out_dir / f"{args.building}__{args.weather}__{tag}__{args.steps}.npz"
    if out_path.exists() and not args.force:
        print(f"skip, already exists: {out_path}")
        return

    env = load_hybrid_environment(config)
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0

    meta = create_building_meta_data(env, config.parent)
    idx = BuildingIndex(meta)
    action_spec = env.action_spec()
    policy = initialize_policy(POLICY_MAP[args.policy], meta, action_spec)

    rng = np.random.default_rng(args.seed)
    n_zone = idx.num_zones
    off_damper = np.zeros(n_zone, dtype=np.float32)
    off_reheat = np.zeros(n_zone, dtype=np.float32)

    tracker.reset()
    obs = to_vec(tracker.last_obs)
    obs_list, cont_list, disc_list, rew_list, pert_list = [obs], [], [], [], []
    status = "ok"
    t0 = time.time()

    for t in range(args.steps):
        action = validate_action(get_policy_action(policy, obs), action_spec)
        cont = np.array(action["continuous_action"], dtype=np.float32)
        disc = np.array(action["discrete_action"], dtype=np.int32)

        perturbed = False
        if args.perturb > 0:
            if t % args.hold == 0:
                off_damper = rng.normal(0.0, args.perturb, n_zone).astype(np.float32)
                off_reheat = rng.normal(0.0, args.perturb, n_zone).astype(np.float32)
            lo, hi = zone_band_kelvin(obs, idx)
            temp = zone_temp_kelvin(obs, idx)
            safe = bool(np.all((temp > lo - args.margin) & (temp < hi + args.margin)))
            if safe:
                cont[idx.zone_damper_action] = np.clip(cont[idx.zone_damper_action] + off_damper, -1.0, 1.0)
                cont[idx.zone_reheat_action] = np.clip(cont[idx.zone_reheat_action] + off_reheat, -1.0, 1.0)
                perturbed = True
            action = validate_action({"continuous_action": cont, "discrete_action": disc}, action_spec)
            cont = np.array(action["continuous_action"], dtype=np.float32)
            disc = np.array(action["discrete_action"], dtype=np.int32)

        obs_next, reward, terminated, truncated, _ = tracker.step(action)
        obs = to_vec(obs_next)

        obs_list.append(obs)
        cont_list.append(cont)
        disc_list.append(disc)
        rew_list.append(float(reward))
        pert_list.append(perturbed)

        if not np.isfinite(obs).all():
            status = "nonfinite_obs"
            break
        if terminated or truncated:
            status = "ended_early"
            break
        if (t + 1) % 1000 == 0:
            print(f"  step {t + 1}/{args.steps}  elapsed {time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    n = len(rew_list)
    if n == 0:
        status = "no_steps"
    else:
        np.savez_compressed(
            out_path,
            obs=np.stack(obs_list),
            cont=np.stack(cont_list),
            disc=np.stack(disc_list),
            reward=np.array(rew_list, dtype=np.float32),
            perturbed=np.array(pert_list, dtype=bool),
        )
        info = {
            "building": args.building, "weather": args.weather, "policy": args.policy,
            "steps": n, "perturb": args.perturb, "hold": args.hold, "margin": args.margin,
            "seed": args.seed, "git_sha": git_sha(), "status": status,
            "total_reward": float(np.sum(rew_list)),
            "num_ahus": idx.num_ahus, "num_zones": idx.num_zones,
            "zone_ids": idx.zone_ids,
            "observation_field_names": list(meta["observation_field_names"]),
        }
        out_path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    append_manifest(out_dir / "manifest.csv", {
        "file": out_path.name, "building": args.building, "weather": args.weather,
        "policy": args.policy, "perturb": args.perturb, "steps": n,
        "total_reward": float(np.sum(rew_list)) if n else 0.0,
        "seconds": round(elapsed, 1), "status": status, "git_sha": git_sha(),
    })
    print(f"done: {out_path.name}  steps={n}  status={status}  "
          f"total_reward={np.sum(rew_list):.4f}  time={elapsed:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--building", required=True)
    p.add_argument("--weather", required=True)
    p.add_argument("--policy", default="pid_water")
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--perturb", type=float, default=0.0, help="std of zone damper/reheat offsets, normalized units")
    p.add_argument("--hold", type=int, default=12, help="steps an offset is held before resampling")
    p.add_argument("--margin", type=float, default=1.0, help="Kelvin outside band where perturbation switches off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    collect(p.parse_args())


if __name__ == "__main__":
    main()