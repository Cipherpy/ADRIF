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
```


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
## Example `data.yaml`

```yaml
path: /absolute/path/to/model_data
train: train/images
val: val/images
test: test/images

names:
  0: Globicephala macrorhynchus
  1: Stenella longirostris
  2: Balaenoptera musculus
```

## Quickstart Pipeline

A typical end-to-end run looks like:

1. **Train YOLO**  
   (baseline or oversampling trainer)

2. **Run inference** on the test set  
   Export YOLO-format prediction `.txt` files

3. **Fuse predictions**  
   Combine outputs from multiple models using **Weighted Box Fusion (WBF)**

4. **Apply ADRIF distance-aware refinement**  
   Separate *distant* vs *normal* detections and refine overlapping boxes

5. **Evaluate performance**
   - In-distribution test set  
   - Out-of-Distribution (OOD) dataset


## Scripts

### Training

---

#### `src/train.py`

Basic **Ultralytics YOLO training**  
(uses `yolo11n.pt` by default)

```bash
python src/train.py
```
####`src/train_oversampling.py` + `src/imbalance_trainer.py`

Training with **class-imbalance aware oversampling** using a `WeightedRandomSampler`.

> ⚠️ The oversampling components are integrated and used within `train.py`.

#### Key ideas implemented:

- Determine a **per-image dominant class**
- Combine rarity from:
  - Image counts (**primary weighting**)
  - Instance counts (**secondary weighting**)
- Apply:
  - Smoothing (`power < 1`)
  - Upper cap (`max_ratio`)

These mechanisms help reduce overfitting to extremely rare classes while improving minority class representation during training.

## Inference + Export Predictions

---

### `src/predict.py`

Run prediction on a **single image** and save an annotated output.

```bash
python src/predict.py
```


### `src/nms_original_pred.py`

Run YOLO on a folder of test images and save:

- Annotated images  
- YOLO-format prediction `.txt` files (with confidence included)  
- A pickle file containing all predictions  

```bash
python src/nms_original_pred.py
```
## Fusion

### `src/wbf.py`

Fuse YOLO-format predictions from two models using **Weighted Box Fusion (WBF)**.

```bash
python src/wbf.py
```
## Distance-Aware Refinement

### `src/sort_images.py`

Splits images into `distant/` vs `normal/` using:

- The **75th percentile** of predicted bounding box area ratios
- A threshold `area_threshold_ratio`  
  (default: `0.005` → 0.5% of image area)

```bash
python src/sort_images.py
```
### `src/final.py`

Core **ADRIF post-processing**:

- Classifies each image as **distant** or **normal**
- Removes redundant contained boxes using different thresholds per group
- Writes filtered `.txt` predictions
- Copies images to corresponding output folders

```bash
python src/final.py
```