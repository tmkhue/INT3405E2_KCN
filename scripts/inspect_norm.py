import argparse
import pprint
from pathlib import Path

from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data

parser = argparse.ArgumentParser()
parser.add_argument("--building", required=True)
parser.add_argument("--weather", required=True)
args = parser.parse_args()

config = Path("smart_buildings/training_scenarios") / args.building / args.weather / "config.gin"
env = load_hybrid_environment(config)
meta = create_building_meta_data(env, config.parent)

for key in ["comfort_temp_window", "eco_temp_window",
            "observation_normalization_metadata", "action_normalization_metadata"]:
    print(f"\n=== {key} ===")
    pprint.pprint(meta[key], width=100, compact=True)