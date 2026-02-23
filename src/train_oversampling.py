#!/usr/bin/env python3
"""
train_oversampling.py

YOLO training with class-imbalance oversampling that fits your reality:

✅ Mostly SINGLE-CLASS per image (one species), but MULTIPLE instances in that image.
✅ Some species have very few images/instances → naive oversampling overfits.
✅ Solution:
   1) Compute per-image "dominant class" (usually the only class).
   2) Compute rarity from BOTH:
        - image counts per class (main)
        - instance counts per class (secondary)
   3) Build sampling weights with:
        - smoothing (power < 1)
        - capped oversampling ratio (max_ratio)
   4) Use WeightedRandomSampler for train loader only.
   5) Add strong-but-safe augmentation knobs to reduce memorization.

IMPORTANT for multi-GPU (DDP):
- Ultralytics DDP spawns a temp script that can't import classes from __main__.
- Put the trainer in an importable module (imbalance_trainer.py).
- This file is the launcher. Create the module below as instructed.

USAGE:
  python train_oversampling.py
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ---- CHANGE THESE PATHS ----
DATASET_ROOT = Path("/home/reshma/ADRIF/ADRIF/model_data")
DATA_YAML = DATASET_ROOT / "data.yaml"

# Multi-GPU list (DDP). If debugging, set to single GPU: [0]
DEVICES = [7]

# Model choice
MODEL_WEIGHTS = "yolo11m.pt"     # pretrained
# MODEL_WEIGHTS = "yolo11m.yaml" # true scratch init

# Import trainer from module (DDP-safe)
from imbalance_trainer import ImbalanceDetectionTrainer


def main():
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        device=DEVICES,
        workers=0,
        trainer=ImbalanceDetectionTrainer,

        # --------- anti-overfit augment settings (good defaults) ----------
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.10,
        copy_paste=0.10,
        erasing=0.40,
        hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,

        # regularization
        weight_decay=7e-4,
    )

    metrics = model.val(data=str(DATA_YAML), split="val")
    path = model.export(format="onnx")
    print(path)


if __name__ == "__main__":
    main()
