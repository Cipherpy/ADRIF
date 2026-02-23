import cv2
from ultralytics import YOLO

# ====== SETTINGS ======
MODEL_PATH = "/home/reshma/marine_mammal/runs/detect/train7/weights/best.pt"
IMAGE_PATH = "/home/reshma/ADRIF/test_OOD/Tr3_Balaenoptera edeni_0749.JPG"
OUT_PATH = "Tr3_Balaenoptera edeni_0749.jpg"

CONF_THRES = 0.50# <-- ch   ange this (e.g., 0.25, 0.5, 0.7)
IOU_THRES  = 0.3  # optional (NMS IoU threshold)
# ======================

# Load the model
model = YOLO(MODEL_PATH)

# Read the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Run prediction on the image (set confidence threshold here)
results = model.predict(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

# Draw results
for result in results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("No detections found with the current confidence threshold.")
        continue

    boxes_np = boxes.cpu().numpy()
    for box in boxes_np:
        r = box.xyxy[0].astype(int)          # [x1, y1, x2, y2]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        conf = float(box.conf[0])            # confidence score (0..1)

        print(f"Class: {class_name}, Conf: {conf:.3f}, Box: {r}")

        # Draw rectangle
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

        # Label text with confidence
        label = f"{class_name} {conf:.2f}"

        # Put label slightly above the box (clamp y to >= 0)
        x, y = r[0], max(0, r[1] - 10)
        cv2.putText(
            img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

# Save once
cv2.imwrite(OUT_PATH, img)
print(f"Saved: {OUT_PATH}")