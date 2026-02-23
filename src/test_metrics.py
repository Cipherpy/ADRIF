from ultralytics import YOLO
import os

# ====== CONFIG ======
model_path = "/home/reshma/marine_mammal/runs/detect/train17/weights/best.pt"   # Change to your trained model
test_data_yaml = "/home/reshma/marine_mammal/model_data/data.yaml"   # Should include path to test/images and class names

# Example YAML content:
# path: /home/reshma/marine_mammal/Actual_data1
# test: test/images
# names:
#   0: Globicephala macrorhynchus
#   1: Stenella longirostris
#   2: Balaenoptera musculus

# ====== LOAD MODEL ======
model = YOLO(model_path)

# ====== RUN TEST EVALUATION ======
results = model.val(data=test_data_yaml, split='test', conf=0.001, iou=0.5)

# ====== DISPLAY KEY METRICS ======
metrics = results.box
#print(metrics)
print("\n📊 Evaluation Metrics:")
print(f"Precision:        {metrics['precision']:.4f}")
print(f"Recall:           {metrics['recall']:.4f}")
print(f"mAP@0.5:          {metrics['map50']:.4f}")
print(f"mAP@0.5:0.95:     {metrics['map']:.4f}")

# Optional: per-class metrics
print("\n📋 Per-class Results:")
for i, name in model.names.items():
    print(f"{name:<25}  P: {metrics['per_class_precision'][i]:.4f}  R: {metrics['per_class_recall'][i]:.4f}  mAP50: {metrics['per_class_ap50'][i]:.4f}")

# ====== Output folder where plots (PR, confusion, F1 curves) are saved ======
print("\n📁 Results saved to:", results.save_dir)
