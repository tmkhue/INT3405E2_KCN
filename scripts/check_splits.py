"""Kiểm tra configs/splits.yaml khớp training_set.yaml và các thư mục scenario."""
from pathlib import Path

import yaml

splits = yaml.safe_load(open("configs/splits.yaml", encoding="utf-8"))
official = yaml.safe_load(open("smart_buildings/training_set.yaml", encoding="utf-8"))["scenarios"]
official_pairs = {(s["building"], s["weather"]) for s in official}

pairs = {}
for split, buildings in splits.items():
    pairs[split] = {(b, w) for b, ws in buildings.items() for w in ws}
    print(f"{split:5s}: {len(buildings)} buildings, {len(pairs[split])} scenarios")

all_pairs = set().union(*pairs.values())
total = sum(len(p) for p in pairs.values())

assert total == len(all_pairs), "có scenario nằm ở nhiều split"
assert all_pairs == official_pairs, f"lệch: {all_pairs ^ official_pairs}"
bsets = [set(b for b, _ in p) for p in pairs.values()]
assert len(set().union(*bsets)) == sum(len(s) for s in bsets), "một tòa nằm ở nhiều split"
for b, w in all_pairs:
    assert (Path("smart_buildings/training_scenarios") / b / w / "config.gin").exists(), (b, w)

print(f"OK: {len(all_pairs)} scenarios, match training_set.yaml, no buildings shared between splits")