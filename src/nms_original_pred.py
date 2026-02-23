from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
import pickle

# ====== Load Model ======
model_path = "/home/reshma/marine_mammal/runs/detect/train15/weights/best.pt"
model = YOLO(model_path)

# ====== Test Data Directory ======
test_images_dir = "/home/reshma/marine_mammal/model_data/test/images"

# ====== Output Directories ======
image_output_dir = "/home/reshma/marine_mammal/Results/Study_4/yolo_11s/predicted_images_11s_0.20_0.55"
yolo_output_dir = "/home/reshma/marine_mammal/Results/Study_4/yolo_11s/predicted_yolo_txt_11s_0.20_0.55"
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(yolo_output_dir, exist_ok=True)

# ====== Class Labels (update if needed) ======
class_names = ['Globicephala macrorhynchus', 'Stenella longirostris', 'Balaenoptera musculus']

FONT, FSCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
GREEN, BLUE, WHITE = (0,255,0), (255,0,0), (255,255,255)

# ====== Gather Image Paths ======
image_files = [os.path.join(test_images_dir, f)
               for f in os.listdir(test_images_dir)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]


# ====== Run Prediction and Save Results ======
all_preds = []

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model.predict(img_path, conf=0.20, iou=0.55, save=False)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            h, w = img.shape[:2]
            base = os.path.splitext(os.path.basename(img_path))[0]
            yolo_txt_path = os.path.join(yolo_output_dir, base + ".txt")

            # Save YOLO format predictions
            with open(yolo_txt_path, "w") as f_txt:
                for cls, box, score in zip(classes, boxes, scores):
                    x1, y1, x2, y2 = box
                    xc = (x1 + x2) / 2 / w
                    yc = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f_txt.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.6f}\n")

                    # Draw bounding box
                    x1, y1, x2, y2 = [int(val) for val in box]
                    label = f"{class_names[cls] if cls < len(class_names) else f'Class {cls}'}: {score:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
                    cv2.putText(img, label, (x1, max(y1 - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

            all_preds.append((img_path, boxes, scores, classes))
        else:
            all_preds.append((img_path, [], [], []))

    # Save the annotated image
    filename = os.path.basename(img_path)
    save_path = os.path.join(image_output_dir, filename)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved annotated image: {save_path}")

# Save all_preds as pickle
with open("/home/reshma/marine_mammal/Results/Study_4/yolo_11s/preds_11s_0.20_0.55.pkl", "wb") as f:
    pickle.dump(all_preds, f)

print("\n✅ YOLO-format predictions saved to:", yolo_output_dir)
print("✅ Annotated images saved to:", image_output_dir)
print("✅ all_preds saved to all_preds.pkl")
print(f"Total test images processed: {len(all_preds)}")
