import argparse
from pathlib import Path

from smart_buildings.evaluate import (
    load_hybrid_environment,
    create_building_meta_data,
)


def describe(value):
    """Briefly describe a value by its type and length."""
    if isinstance(value, dict):
        return f"dict, {len(value)} keys"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}, length {len(value)}"
    if hasattr(value, "shape"):
        return f"array, shape {value.shape}"
    return f"{type(value).__name__}: {value}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", required=True)
    parser.add_argument("--weather", required=True)
    args = parser.parse_args()

    config = Path("smart_buildings/training_scenarios") / args.building / args.weather / "config.gin"
    print(f"Loading: {config}")
    env = load_hybrid_environment(config)
    meta = create_building_meta_data(env, config.parent)

    
    print("\n=== METADATA KEYS ===")
    for key in meta:
        print(f"  {key:30s} -> {describe(meta[key])}")

    
    print("\n=== ACTION INDICES ===")
    for key in ["temp_indices", "pressure_indices", "recirculation_indices", "water_indices"]:
        print(f"  {key}: {meta.get(key)}")
    zai = meta.get("zone_action_indices", {})
    first_vav = next(iter(zai), None)
    print(f"  zone_action_indices: {len(zai)} VAVs; example {first_vav} -> {zai.get(first_vav)}")

    
    print("\n=== AHU -> ZONE ASSIGNMENTS ===")
    for ahu, zones in meta["ahu_zone_assignments"].items() if isinstance(
        meta["ahu_zone_assignments"], dict
    ) else enumerate(meta["ahu_zone_assignments"]):
        print(f"  {ahu}: {len(zones)} zones, example {list(zones)[:3]}")

    
    K = len(meta["ahu_zone_assignments"])
    V = len(zai)
    n_cont = meta["n_continuous_action"]
    n_disc = meta["n_discrete_action"]
    n_obs = len(meta["observation_field_names"])
    print("\n=== VALIDATION ===")
    print(f"  K={K}, V={V}, obs={n_obs}")
    print(f"  continuous: 3K+2+2V = {3*K+2+2*V}  (actual {n_cont})  ->", "OK" if 3*K+2+2*V == n_cont else "MISMATCH")
    print(f"  discrete  : K+1     = {K+1}  (actual {n_disc})  ->", "OK" if K+1 == n_disc else "MISMATCH")
    print(f"  zone_temp_indices: {len(meta['zone_temp_indices'])} (compared with V={V})")

    
    out = Path("logs") / f"obs_fields_{args.building}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, name in enumerate(meta["observation_field_names"]):
            f.write(f"{i}\t{name}\n")
    print(f"\nSaved {n_obs} field names to {out}")


if __name__ == "__main__":
    main()