#!/usr/bin/env python3
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/reshma/ADRIF/ADRIF/model_data")
DATA_YAML = ROOT / "data.yaml"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}

# -----------------------------
# Load class names from data.yaml
# -----------------------------
with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

# YOLO datasets usually keep names as list; sometimes dict {0:"a",1:"b"}
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
# Helpers
# -----------------------------
def count_images_per_class_from_labels(labels_dir: Path):
    """
    Counts IMAGE-level presence per class.
    For each label file -> collect unique class IDs -> add 1 per class.
    """
    per_class = defaultdict(int)
    label_files = sorted(labels_dir.glob("*.txt"))

    for lf in label_files:
        classes_in_image = set()
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                if 0 <= cid < nc:
                    classes_in_image.add(cid)

        for cid in classes_in_image:
            per_class[cid] += 1

    return per_class, len(label_files)

def count_images_in_folder(images_dir: Path):
    return len([p for p in images_dir.iterdir() if p.suffix in IMG_EXTS])

# -----------------------------
# Process splits
# -----------------------------
splits = ["train", "val", "test"]

split_counts = {}
overall = defaultdict(int)

print("\n==============================")
print("📊 Images-per-species (from labels)")
print("==============================\n")

for split in splits:
    labels_dir = ROOT / split / "labels"
    images_dir = ROOT / split / "images"

    if not labels_dir.exists():
        print(f"⚠️ Missing labels folder: {labels_dir}")
        continue

    per_class, n_label_files = count_images_per_class_from_labels(labels_dir)
    n_images = count_images_in_folder(images_dir) if images_dir.exists() else None

    split_counts[split] = per_class

    # add to overall
    for cid, v in per_class.items():
        overall[cid] += v

    print(f"--- {split.upper()} ---")
    print(f"Label files (images with labels): {n_label_files}")
    if n_images is not None:
        print(f"Images in images/:              {n_images}")
        if n_images != n_label_files:
            print("⚠️ Note: image count != label count (some images may be unlabeled or missing labels).")

    for cid in range(nc):
        print(f"{class_names[cid]:25s}: {per_class.get(cid, 0)}")
    print()

# -----------------------------
# Overall summary
# -----------------------------
print("=== OVERALL (train+val+test) ===")
for cid in range(nc):
    print(f"{class_names[cid]:25s}: {overall.get(cid, 0)}")
print()
