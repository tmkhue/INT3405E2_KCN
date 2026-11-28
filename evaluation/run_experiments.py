
import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FULL_STEPS = 8640


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Invalid name")
    return value


def git_info(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True
    ).strip()


def save_metadata(path, metadata):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    temp.replace(path)


def run_experiment(args):
    output_dir = RESULTS_DIR / args.experiment_id

    config = (
        ROOT / "smart_buildings"
        / "training_scenarios"
        / args.building
        / args.weather
        / "config.gin"
    )

    if not config.is_file():
        raise FileNotFoundError(config)

    if output_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {output_dir}"
        )

    command = [
        sys.executable,
        "-u",
        "-m", "smart_buildings.evaluate",
        "--policies", args.policy,
        "--buildings", args.building,
        "--weathers", args.weather,
        "--steps", str(args.steps),
        "--results-dir", str(output_dir),
    ]

    metadata = {
        "experiment_id": args.experiment_id,
        "policy": args.policy,
        "building": args.building,
        "weather": args.weather,
        "requested_steps": args.steps,
        "git_commit": git_info("rev-parse", "HEAD"),
        "git_dirty": bool(
            git_info("status", "--porcelain")
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(
            config.read_bytes()
        ).hexdigest(),
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "command": command,
        "status": "running",
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    metadata_file = output_dir / "run_metadata.json"
    log_file = output_dir / "run.log"

    save_metadata(metadata_file, metadata)

    start = time.perf_counter()

    print(f"Running: {args.experiment_id}", flush=True)
    print(f"Scenario: {args.building}/{args.weather}")
    print(f"Policy: {args.policy}")
    print(f"Steps: {args.steps}")
    print(f"Log: {log_file}", flush=True)

    try:
        with log_file.open("w", encoding="utf-8") as f:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        metadata["return_code"] = result.returncode

        if result.returncode != 0:
            raise RuntimeError("Evaluation failed")

        summary_path = (
            output_dir / args.building
            / args.weather / args.policy
            / "summary.json"
        )

        if set(output_dir.rglob("summary.json")) != {summary_path}:
            raise ValueError("Unexpected or missing summary")

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        steps = summary["total_steps"]

        if type(steps) is not int or steps != args.steps:
            raise ValueError("Incorrect step count")

        if summary["policy"] != args.policy:
            raise ValueError("Policy mismatch")

        reward = float(summary["total_reward"])

        if not math.isfinite(reward) or reward > 0:
            raise ValueError("Invalid reward")

        metadata["total_reward"] = reward
        metadata["total_steps"] = steps
        metadata["status"] = "success"

        metadata["score"] = (
            abs(reward / FULL_STEPS)
            if steps == FULL_STEPS else None
        )

    except KeyboardInterrupt:
        metadata["status"] = "interrupted"

    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)

    finally:
        metadata["runtime_seconds"] = (
            time.perf_counter() - start
        )

        metadata["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        save_metadata(metadata_file, metadata)

    print("\nEXPERIMENT FINISHED")
    print(f"Status: {metadata['status']}")
    print(f"Reward: {metadata.get('total_reward')}")
    print(f"Runtime: {metadata['runtime_seconds']:.2f}s")

    return 0 if metadata["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible HVAC experiments"
    )

    parser.add_argument(
        "--experiment-id", type=safe_name, required=True
    )
    parser.add_argument(
        "--policy", type=safe_name, default="pid_water"
    )
    parser.add_argument(
        "--building", type=safe_name, required=True
    )
    parser.add_argument(
        "--weather", type=safe_name, required=True
    )
    parser.add_argument(
        "--steps", type=int, choices=[288, 8640],
        default=288
    )

    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
