#!/usr/bin/env python3
"""
imbalance_trainer.py

DDP-safe custom trainer that injects WeightedRandomSampler with:
- dominant-class per image (single-class images)
- rarity from image counts + instance counts
- smoothing (power < 1) + cap (max_ratio) to avoid rare-class overfitting

Put this file in the SAME FOLDER as train_oversampling.py (or on PYTHONPATH).
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ultralytics.models.yolo.detect import DetectionTrainer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Dataset helpers (your layout)
# model_data/train/images + model_data/train/labels
# -----------------------------
def list_images(images_dir: Path):
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def label_path_from_image(img_path: Path):
    # .../train/images/... -> .../train/labels/... (preserves subfolders)
    parts = list(img_path.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
    except ValueError:
        pass
    return Path(*parts).with_suffix(".txt")


def read_classes(label_path: Path):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(float(line.split()[0])))
        except Exception:
            continue
    return out


def build_sampling_weights_capped(
    train_images,
    nc: int,
    alpha_img: float = 0.80,   # weight of image-count rarity
    alpha_inst: float = 0.20,  # weight of instance-count rarity
    power: float = 0.70,       # smoothing (<1 reduces oversampling extremes)
    max_ratio: float = 8.0,    # cap oversampling ratio
    eps: float = 1e-6,
):
    """
    Returns:
      weights_tensor (double)
      img_counts (float64)
      inst_counts (float64)
    """
    img_class = np.full(len(train_images), -1, dtype=int)
    img_counts = np.zeros(nc, dtype=np.float64)
    inst_counts = np.zeros(nc, dtype=np.float64)

    # pass 1: dominant class per image + instance counts
    for i, im in enumerate(train_images):
        cl = read_classes(label_path_from_image(im))
        if not cl:
            continue

        cl = np.array(cl, dtype=int)
        cl = cl[(cl >= 0) & (cl < nc)]
        if len(cl) == 0:
            continue

        # dominant class (usually unique)
        vals, cnts = np.unique(cl, return_counts=True)
        dom = int(vals[np.argmax(cnts)])
        img_class[i] = dom
        img_counts[dom] += 1

        # instance counts
        for c in cl:
            inst_counts[c] += 1

    # inverse frequencies
    inv_img = 1.0 / (img_counts + eps)
    inv_inst = 1.0 / (inst_counts + eps)

    # normalize (mean=1) so scaling is stable
    inv_img = inv_img / inv_img.mean()
    inv_inst = inv_inst / inv_inst.mean()

    # mix (mostly image rarity)
    inv = alpha_img * inv_img + alpha_inst * inv_inst

    # smooth to reduce overfit risk
    inv = inv ** power

    # cap oversampling ratio
    inv = np.clip(inv, a_min=1.0 / max_ratio, a_max=max_ratio)

    # per-image weights from dominant class
    w = np.ones(len(train_images), dtype=np.float64)
    for i, c in enumerate(img_class):
        if 0 <= c < nc:
            w[i] = float(inv[c])
        else:
            w[i] = 1.0

    w = np.clip(w, 1e-3, None)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.double), img_counts, inst_counts


class ImbalanceDetectionTrainer(DetectionTrainer):
    """
    DDP-safe trainer that uses WeightedRandomSampler for train dataloader.
    """

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode != "train":
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
            )

        # Ultralytics data.yaml train should be train/images (recommended).
        train_images_dir = Path(self.data["train"])
        if train_images_dir.is_dir() and train_images_dir.name != "images":
            train_images_dir = train_images_dir / "images"

        train_images = list_images(train_images_dir)
        if len(train_images) == 0:
            raise RuntimeError(f"No training images found in: {train_images_dir}")

        nc = int(self.data["nc"])

        # ---- core: capped+smoothed weights to reduce rare-class overfit ----
        weights, img_counts, inst_counts = build_sampling_weights_capped(
            train_images,
            nc=nc,
            alpha_img=0.80,
            alpha_inst=0.20,
            power=0.70,
            max_ratio=8.0,
        )

        # Print counts once (rank 0 / single GPU)
        if rank in (0, -1):
            print("\n[Oversampling] Per-class counts (train):")
            for c in range(nc):
                print(f"  class {c}: {int(img_counts[c])} images, {int(inst_counts[c])} instances")
            print("[Oversampling] Using capped+smoothed WeightedRandomSampler.\n")

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),  # one epoch draws ~N images (with replacement)
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # must be False when sampler is used
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
