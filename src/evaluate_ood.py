from ultralytics import YOLO
import numpy as np

# ====== CONFIG ======
model_path = "/home/reshma/ADRIF/ADRIF/src/runs/detect/train/weights/best.pt"
ood_yaml   = "/home/reshma/ADRIF/OOD/dataset/data_ood.yaml"

CONF = 0.001     # keep low for mAP computation
IOU  = 0.5

# ====== LOAD MODEL ======
model = YOLO(model_path)

# ====== RUN EVALUATION ======
res = model.val(data=ood_yaml, split="test", conf=CONF, iou=IOU)

# ====== OVERALL METRICS ======
# In Ultralytics, box metrics are accessible via res.box
b = res.box

print("\n📊 OOD Evaluation (Overall)")
print(f"Precision (P):        {b.p:.4f}")
print(f"Recall (R):           {b.r:.4f}")
print(f"mAP@0.5 (AP50):       {b.map50:.4f}")
print(f"mAP@0.5:0.95 (mAP):   {b.map:.4f}")

# ====== PER-CLASS METRICS ======
# Per-class AP arrays:
#   b.ap50 : AP50 per class
#   b.ap   : AP50-95 per class
# Per-class P/R are not always exposed as arrays in every ultralytics build,
# so we compute per-class P/R from confusion matrix only if available.
# But AP50/AP are always the most standard per-class reporting.

names = model.names  # dict: id -> class name

ap50 = np.array(b.ap50) if b.ap50 is not None else None
ap   = np.array(b.ap)   if b.ap is not None else None

print("\n📋 Per-class Results (OOD)")
for i in range(len(names)):
    cname = names[i]
    ap50_i = ap50[i] if ap50 is not None and i < len(ap50) else float("nan")
    ap_i   = ap[i]   if ap is not None and i < len(ap) else float("nan")
    print(f"{cname:<28}  AP50: {ap50_i:.4f}  AP50-95: {ap_i:.4f}")

print("\n📁 Detailed outputs (plots, confusion matrix, etc.) saved to:")
print(res.save_dir)