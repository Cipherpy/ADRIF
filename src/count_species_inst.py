#!/usr/bin/env python3
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/reshma/ADRIF/ADRIF/model_data")
DATA_YAML = ROOT / "data.yaml"

# -----------------------------
# Load class names from data.yaml
# -----------------------------
with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

names = data.get("names")
if isinstance(names, dict):
    class_names = [names[i] for i in sorted(names.keys())]
else:
    class_names = list(names)

nc = len(class_names)

print(f"\n✅ Loaded {nc} classes from {DATA_YAML}")
for i, n in enumerate(class_names):
    print(f"  {i}: {n}")

# -----------------------------
# Count INSTANCES per class from labels
# -----------------------------
def count_instances_per_class(labels_dir: Path):
    """
    Counts INSTANCE-level occurrences per class.
    Each line in a YOLO label file corresponds to one instance.
    """
    per_class = defaultdict(int)
    label_files = sorted(labels_dir.glob("*.txt"))

    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                if 0 <= cid < nc:
                    per_class[cid] += 1

    return per_class, len(label_files)

splits = ["train", "val", "test"]

overall_instances = defaultdict(int)

print("\n==============================")
print("🧮 Instance counts per species (from labels)")
print("==============================\n")

for split in splits:
    labels_dir = ROOT / split / "labels"
    if not labels_dir.exists():
        print(f"⚠️ Missing labels folder: {labels_dir}")
        continue

    per_class, n_label_files = count_instances_per_class(labels_dir)

    # add to overall
    for cid, v in per_class.items():
        overall_instances[cid] += v

    total_instances_split = sum(per_class.values())

    print(f"--- {split.upper()} ---")
    print(f"Label files:      {n_label_files}")
    print(f"Total instances:  {total_instances_split}\n")

    for cid in range(nc):
        print(f"{class_names[cid]:25s}: {per_class.get(cid, 0)}")
    print()

print("=== OVERALL (train+val+test) ===")
print(f"Total instances overall: {sum(overall_instances.values())}\n")
for cid in range(nc):
    print(f"{class_names[cid]:25s}: {overall_instances.get(cid, 0)}")
print()
