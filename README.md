# ADRIF — Adaptive Distance-aware Refinement & Intelligent Fusion

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

ADRIF is a small research/utility codebase built around **Ultralytics YOLO** for **marine mammal object detection**, with:
- **Intelligent fusion** via **Weighted Box Fusion (WBF)** to combine predictions from multiple models.
- **Distance-aware refinement** (heuristic) that adapts post-processing based on whether detections appear *distant* (small boxes) or *normal* (larger boxes).
- Dataset utilities for **label conversion**, **visualization**, and **class-imbalance aware training**.

> **Note:** Most scripts currently use **hard-coded absolute paths** (e.g., `/home/reshma/...`). You’ll want to edit the config variables at the top of each script before running.

---

## Contents

- [Project idea](#project-idea)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Dataset format](#dataset-format)
- [Quickstart pipeline](#quickstart-pipeline)
- [Scripts](#scripts)
  - [Training](#training)
  - [Inference + export predictions](#inference--export-predictions)
  - [Fusion](#fusion)
  - [Distance-aware refinement](#distance-aware-refinement)
  - [Evaluation](#evaluation)
  - [Dataset utilities](#dataset-utilities)
- [Notes & gotchas](#notes--gotchas)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project idea

ADRIF combines two practical ideas for improving detection outputs:

### 1) Intelligent Fusion (WBF)
Fuse detections from multiple YOLO models (e.g., `yolo11n` + `yolo11s`) using **Weighted Box Fusion** to reduce duplicates and stabilize box placement.

### 2) Distance-aware Refinement (Adaptive Post-Processing)
Classify images as:
- **distant**: detections are mostly small (based on the **75th percentile** of predicted box area ratio),
- **normal**: detections are larger,

…and apply different overlap/containment thresholds when removing redundant nested boxes.

---

## Repository layout

```text
ADRIF/
├─ src/
│  ├─ train.py
│  ├─ train_oversampling.py
│  ├─ imbalance_trainer.py
│  ├─ nms_original_pred.py
│  ├─ wbf.py
│  ├─ yolo_wbf.py
│  ├─ sort_images.py
│  ├─ final.py
│  ├─ test_metrics.py
│  ├─ evaluate_ood.py
│  ├─ vgg2yolo.py
│  ├─ gt_visualisation.py
│  ├─ count_img_with_species.py
│  ├─ count_species_inst.py
│  └─ data_statistics.py
├─ LICENSE
└─ README.md 
```

## Installation

Create a virtual environment and install the dependencies used by the scripts:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install -U pip

# Core dependencies used in src/
pip install ultralytics torch opencv-python numpy ensemble-boxes pyyaml pillow


## Dataset Format
```text
model_data/
├─ data.yaml
├─ train/
├─ ├─ images/
├─ ├─ labels/ 
├─ val/
├─ ├─ images/
├─ ├─ labels/ 
├─ test/
├─ ├─ images/
├─ ├─ labels/
```