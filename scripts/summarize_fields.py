import re
import sys
from collections import OrderedDict

building = sys.argv[1]
groups = OrderedDict()

with open(f"logs/obs_fields_{building}.txt", encoding="utf-8") as f:
    for line in f:
        idx, name = line.rstrip("\n").split("\t", 1)
        pattern = re.sub(r"\d+", "{n}", name)
        groups.setdefault(pattern, []).append((int(idx), name))

print(f"{len(groups)} field groups:\n")
for pattern, items in groups.items():
    first_idx, first_name = items[0]
    print(f"{len(items):4d} x  {pattern}   (e.g. [{first_idx}] {first_name})")