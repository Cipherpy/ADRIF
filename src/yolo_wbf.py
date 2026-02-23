import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ====== SETTINGS ======
MODEL_PATH = "/home/reshma/marine_mammal/runs/detect/train7/weights/best.pt"
IMAGE_PATH = "/home/reshma/ADRIF/test_OOD/Tr3_Balaenoptera edeni_0749.JPG"
OUT_PATH   = "Tr3_Balaenoptera_edeni_0749_WBF.jpg"

CONF_THRES = 0.50
IOU_THRES  = 0.30       # YOLO NMS threshold

WBF_IOU    = 0.10      # lower this if boxes are not fusing (try 0.10–0.30)
WBF_SKIP   = 0.10

MERGE_IOU  = 0.15       # post-merge IoU threshold (try 0.05–0.25)
SCORE_MODE = "max"      # "max" or "mean"
# ======================


def iou_xyxy(a, b):
    # a,b = [x1,y1,x2,y2] normalized
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def merge_overlapping_same_class(boxes, scores, labels, iou_thr=0.15, score_mode="max"):
    """
    Merge overlapping boxes of same label into union box.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    boxes = [list(map(float, b)) for b in boxes]
    scores = list(map(float, scores))
    labels = list(map(int, labels))

    out_boxes, out_scores, out_labels = [], [], []

    for cls in sorted(set(labels)):
        idxs = [i for i, lab in enumerate(labels) if lab == cls]
        cls_boxes = [boxes[i] for i in idxs]
        cls_scores = [scores[i] for i in idxs]

        used = [False] * len(cls_boxes)

        for i in range(len(cls_boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(cls_boxes)):
                    if used[j]:
                        continue
                    if any(iou_xyxy(cls_boxes[j], cls_boxes[k]) >= iou_thr for k in cluster):
                        cluster.append(j)
                        used[j] = True
                        changed = True

            # union box
            xs1 = [cls_boxes[k][0] for k in cluster]
            ys1 = [cls_boxes[k][1] for k in cluster]
            xs2 = [cls_boxes[k][2] for k in cluster]
            ys2 = [cls_boxes[k][3] for k in cluster]
            mbox = [min(xs1), min(ys1), max(xs2), max(ys2)]

            cs = [cls_scores[k] for k in cluster]
            if score_mode == "mean":
                mscore = float(np.mean(cs))
            else:
                mscore = float(np.max(cs))

            out_boxes.append(mbox)
            out_scores.append(mscore)
            out_labels.append(cls)

    return out_boxes, out_scores, out_labels


# Load model + image
model = YOLO(MODEL_PATH)
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

H, W = img.shape[:2]

# YOLO predict
results = model.predict(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

# Collect detections for WBF input
boxes_list, scores_list, labels_list = [], [], []
raw_boxes_norm, raw_scores, raw_labels = [], [], []

for result in results:
    b = result.boxes
    if b is None or len(b) == 0:
        continue
    bnp = b.cpu().numpy()

    boxes_norm, scores, labels = [], [], []
    for box in bnp:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        bx = [x1 / W, y1 / H, x2 / W, y2 / H]
        boxes_norm.append(bx)
        scores.append(conf)
        labels.append(cls)

        raw_boxes_norm.append(bx)
        raw_scores.append(conf)
        raw_labels.append(cls)

    boxes_list.append(boxes_norm)
    scores_list.append(scores)
    labels_list.append(labels)

if len(boxes_list) == 0:
    print("No detections found.")
    cv2.imwrite(OUT_PATH, img)
    raise SystemExit

# ---- DEBUG: print IoU if there are exactly 2 same-class boxes ----
if len(raw_boxes_norm) >= 2:
    # find top-2 by score (often the two you see)
    top2 = np.argsort(raw_scores)[-2:]
    iou_val = iou_xyxy(raw_boxes_norm[top2[0]], raw_boxes_norm[top2[1]])
    print(f"DEBUG IoU(top2 boxes): {iou_val:.3f}  (WBF_IOU={WBF_IOU}, MERGE_IOU={MERGE_IOU})")

# ---- WBF ----
boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    iou_thr=WBF_IOU, skip_box_thr=WBF_SKIP
)

# ---- Post-merge (this is what will turn 2 into 1 if they overlap enough) ----
boxes, scores, labels = merge_overlapping_same_class(
    boxes, scores, labels, iou_thr=MERGE_IOU, score_mode=SCORE_MODE
)

# Draw final boxes
for box, score, label in zip(boxes, scores, labels):
    x1 = int(box[0] * W); y1 = int(box[1] * H)
    x2 = int(box[2] * W); y2 = int(box[3] * H)

    class_name = model.names[int(label)]
    print(f"FINAL → {class_name}, Conf: {score:.3f}")

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img, f"{class_name} {score:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
    )

cv2.imwrite(OUT_PATH, img)
print(f"✅ Saved: {OUT_PATH}")