import os
import json
from pathlib import Path
from PIL import Image

# ================== CONFIG ==================
VIA_JSON = "/home/reshma/ADRIF/OOD_test/via_project_23Feb2026_14h46m_json.json"  # <-- your uploaded file
IMAGES_DIR = "/home/reshma/ADRIF/OOD_test/images"           # <-- change to your images folder
OUT_LABELS_DIR = "/home/reshma/ADRIF/OOD_test/labels"       # <-- output labels folder

# If you want to limit to specific species (set to None to keep all)
KEEP_CLASSES = None
# Example:
# KEEP_CLASSES = {"Balaenoptera edeni", "Balaenoptera musculus", "Stenella coeruleoalba"}

# Which VIA attribute contains the class label
CLASS_ATTR_KEY = "mammal"   # in your JSON it's "mammal"
# ===========================================

os.makedirs(OUT_LABELS_DIR, exist_ok=True)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rect_to_yolo(x, y, w, h, img_w, img_h):
    # x,y,w,h are in pixels; returns normalized YOLO xc,yc,bw,bh
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return clamp01(xc), clamp01(yc), clamp01(bw), clamp01(bh)

# -------- Load VIA JSON --------
with open(VIA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect all class names (for stable class_id mapping)
all_class_names = set()
for k, entry in data.items():
    for region in entry.get("regions", []):
        attrs = region.get("region_attributes", {}) or {}
        cls_name = attrs.get(CLASS_ATTR_KEY)
        if cls_name:
            if KEEP_CLASSES is None or cls_name in KEEP_CLASSES:
                all_class_names.add(cls_name)

classes = sorted(all_class_names)
class_to_id = {c: i for i, c in enumerate(classes)}

# Save classes.txt (YOLO class order)
classes_txt = os.path.join(OUT_LABELS_DIR, "classes.txt")
with open(classes_txt, "w", encoding="utf-8") as f:
    for c in classes:
        f.write(c + "\n")

print(f"✅ Found {len(classes)} classes")
print(f"✅ Saved class list: {classes_txt}")

# -------- Convert each image annotations --------
missing_images = []
written = 0
skipped = 0

for k, entry in data.items():
    filename = entry.get("filename")
    if not filename:
        continue

    img_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(img_path):
        missing_images.append(filename)
        continue

    # read image size
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    yolo_lines = []

    for region in entry.get("regions", []):
        shape = region.get("shape_attributes", {}) or {}
        attrs = region.get("region_attributes", {}) or {}

        cls_name = attrs.get(CLASS_ATTR_KEY)
        if not cls_name:
            skipped += 1
            continue
        if KEEP_CLASSES is not None and cls_name not in KEEP_CLASSES:
            skipped += 1
            continue

        if shape.get("name") != "rect":
            # only rect supported here (matches your JSON)
            skipped += 1
            continue

        x = float(shape.get("x", 0))
        y = float(shape.get("y", 0))
        w = float(shape.get("width", 0))
        h = float(shape.get("height", 0))

        # ignore empty boxes
        if w <= 1 or h <= 1:
            skipped += 1
            continue

        cls_id = class_to_id[cls_name]
        xc, yc, bw, bh = rect_to_yolo(x, y, w, h, img_w, img_h)

        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # write label file (even if empty -> create empty txt)
    out_txt = os.path.join(OUT_LABELS_DIR, Path(filename).stem + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    written += 1

print(f"\n✅ Wrote labels for {written} images into: {OUT_LABELS_DIR}")
print(f"ℹ️ Skipped regions (missing class / not rect / filtered / tiny): {skipped}")

if missing_images:
    print(f"\n⚠️ Missing images ({len(missing_images)}). Example:")
    for m in missing_images[:10]:
        print("  -", m)

# -------- Optional: write a YOLO data.yaml next to labels folder --------
data_yaml_path = os.path.join(Path(OUT_LABELS_DIR).parent, "data_ood.yaml")
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {Path(OUT_LABELS_DIR).parent}\n")
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("test: images\n")
    f.write("names:\n")
    for i, c in enumerate(classes):
        f.write(f"  {i}: {c}\n")

print(f"\n✅ Wrote data.yaml: {data_yaml_path}")