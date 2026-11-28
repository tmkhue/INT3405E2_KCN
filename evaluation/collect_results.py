import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "experiment_results.csv"
FULL_EPISODE_STEPS = 8640


def collect_results():
    rows = []

    for file_path in sorted(RESULTS_DIR.rglob("summary.json")):

        # Read experiment information from the directory
        parts = file_path.relative_to(RESULTS_DIR).parts

        if len(parts) != 5:
            print(f"Unexpected directory structure: {file_path}")
            continue

        experiment_id, building, weather, policy, _ = parts

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            total_reward = float(data["total_reward"])
            total_steps = int(data["total_steps"])

            if total_steps <= 0 or not math.isfinite(total_reward):
                raise ValueError("Invalid steps or reward")

            if data["policy"] != policy:
                raise ValueError("Policy name mismatch")

            if total_steps == FULL_EPISODE_STEPS:
                run_type = "full"
                score = abs(total_reward / FULL_EPISODE_STEPS)
            elif total_steps == 288:
                run_type = "smoke"
                score = None
            else:
                run_type = "partial"
                score = None

            row = {
                "experiment_id": experiment_id,
                "building": building,
                "weather": weather,
                "policy": policy,
                "run_type": run_type,
                "total_steps": total_steps,
                "total_reward": total_reward,
                "avg_reward": total_reward / total_steps,
                "score": score,
                "avg_occupancy": data.get("avg_occupancy"),
                "avg_zone_temp_c": (
                    data["avg_zone_temp"] - 273.15
                ),
                "productivity_regret": data.get(
                    "total_normalized_productivity_regret"
                ),
                "energy_cost": data.get(
                    "total_normalized_energy_cost"
                ),
                "carbon_emission": data.get(
                    "total_normalized_carbon_emission"
                ),
                "seconds_per_step": data.get(
                    "seconds_per_step"
                ),
                "num_ahus": data.get("num_ahus"),
                "status": "parsed",
                "source_file": file_path.as_posix(),
            }

            rows.append(row)

        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Error reading {file_path}: {error}")

    return pd.DataFrame(rows)


def main():
    df = collect_results()

    if df.empty:
        print("No valid experiment results found.")
        return

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nEXPERIMENT RESULTS")
    print("=" * 70)

    columns = [
        "experiment_id",
        "building",
        "weather",
        "policy",
        "run_type",
        "total_steps",
        "total_reward",
        "score",
    ]

    print(df[columns].to_string(index=False))
    print(f"\nCollected {len(df)} experiment results.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
