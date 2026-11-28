"""Run data_collection.py over many scenarios in parallel.

Examples (repo root):
    python scripts/run_collection.py --split train --policies pid_water --workers 4 --dry-run
    python scripts/run_collection.py --split train --policies pid_water pid --workers 4
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_jobs(splits, which, policies):
    names = list(splits) if which == "all" else [which]
    jobs = []
    for split in names:
        for building, weathers in splits[split].items():
            for weather in weathers:
                for policy in policies:
                    jobs.append((split, building, weather, policy))
    return jobs


def run_job(job, args, log_dir):
    split, building, weather, policy = job
    cmd = [
        sys.executable, str(ROOT / "scripts" / "data_collection.py"),
        "--building", building, "--weather", weather, "--policy", policy,
        "--steps", str(args.steps), "--out-dir", args.out_dir,
        "--perturb", str(args.perturb), "--seed", str(args.seed),
    ]
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[var] = "1"
    log_path = log_dir / f"{building}__{weather}__{policy}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return job, proc.returncode, time.time() - t0


def rebuild_manifest(out_dir):
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "file": p.with_suffix(".npz").name, "building": d["building"],
            "weather": d["weather"], "policy": d["policy"], "perturb": d["perturb"],
            "steps": d["steps"], "total_reward": d["total_reward"],
            "status": d["status"], "git_sha": d["git_sha"],
        })
    if rows:
        path = Path(out_dir) / "manifest_all.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} episodes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--policies", nargs="+", default=["pid_water"])
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--perturb", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.split in ("test", "all") and not args.dry_run:
        print("WARNING: test split is meant to be run once after the model is frozen.")

    splits = yaml.safe_load(open(ROOT / "configs" / "splits.yaml", encoding="utf-8"))
    jobs = build_jobs(splits, args.split, args.policies)
    print(f"{len(jobs)} jobs, {args.workers} workers, steps={args.steps}")
    if args.dry_run:
        for j in jobs:
            print("  ", *j)
        return

    log_dir = ROOT / "logs" / "collect"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed, t_start = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, j, args, log_dir) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            job, code, secs = fut.result()
            tag = "ok" if code == 0 else f"FAILED({code})"
            print(f"[{n}/{len(jobs)}] {job[1]} {job[2]} {job[3]}: {tag} in {secs / 60:.1f} min")
            if code != 0:
                failed.append(job)
    print(f"all done in {(time.time() - t_start) / 60:.1f} min, failed: {len(failed)}")
    for j in failed:
        print("  failed:", *j)
    rebuild_manifest(args.out_dir)


if __name__ == "__main__":
    main()