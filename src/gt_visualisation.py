#!/usr/bin/env python3
"""
Visualize YOLO format bounding box annotations.

✔ Reads images from train/images
✔ Reads labels from train/labels
✔ Draws bounding boxes (THICK)
✔ Draws class name text (NO background rectangle)
✔ Saves annotated images to output folder
"""

from pathlib import Path
import cv2
import numpy as np

# ============================
# CONFIGURATION
# ============================
images_dir = Path("/home/reshma/ADRIF/ADRIF/model_data_old/train/images")
labels_dir = Path("/home/reshma/ADRIF/ADRIF/model_data_old/train/labels")
output_dir = Path("/home/reshma/ADRIF/ADRIF/model_data_old/visualized_annotations1")

class_names = [
    "Globicephala macrorhynchus",
    "Stenella longirostris",
    "Balaenoptera musculus",
]

# How thick you want the bounding boxes
BOX_THICKNESS = 6  # <- increase/decrease as needed (e.g., 4, 6, 8)

# Text settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
TEXT_THICKNESS = 3

# ============================
# SETUP
# ============================
output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]

if len(image_files) == 0:
    raise ValueError("No images found.")

# consistent random colors per class
np.random.seed(42)
class_colors = {
    i: tuple(np.random.randint(0, 255, 3).tolist())
    for i in range(len(class_names))
}

# ============================
# VISUALIZATION
# ============================
for img_path in image_files:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    label_path = labels_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    lines = label_path.read_text().strip().splitlines()
    if not lines:
        continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        # YOLO normalized -> pixels
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Box color per class
        box_color = class_colors.get(cls_id, (0, 255, 0))

        # Draw THICK bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        # Label text (class name only; change if you want ID also)
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        label_text = class_name  # or: f"ID:{cls_id} | {class_name}"

        # Put text WITHOUT background rectangle
        # Ensure text is inside image:
        (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, TEXT_THICKNESS)
        tx = x1
        ty = y1 - 10
        if ty - th < 0:  # if goes above image, move inside
            ty = y1 + th + 10

        # Choose a text color different from the box (here: bright yellow)
        text_color = (0, 255, 255)

        cv2.putText(
            img,
            label_text,
            (tx, ty),
            FONT,
            FONT_SCALE,
            text_color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    save_path = output_dir / img_path.name
    cv2.imwrite(str(save_path), img)
    print(f"Saved: {save_path}")

print("\n✅ Done! All annotated images saved.")
