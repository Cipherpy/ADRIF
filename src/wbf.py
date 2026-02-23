import os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# ==== Paths ====
pred_s_dir = "/home/reshma/marine_mammal/Results/Study_4/yolo_11n/predicted_yolo_txt_11n_0.15_0.6"
pred_n_dir = "/home/reshma/marine_mammal/Results/Study_4/yolo_11s/predicted_yolo_txt_11s_0.15_0.6"
#pred_m_dir="/home/reshma/marine_mammal/NMS/predicted_yolo_txt_11m"
output_dir = "/home/reshma/marine_mammal/Results/Study_4/wbf/yolo_0.15_0.6"
os.makedirs(output_dir, exist_ok=True)

# ==== WBF Parameters ====
iou_thr = 0.4
skip_thr = 0.1

# ==== Helper: Load YOLO-format predictions ====
def load_yolo_preds(pred_file):
    boxes, scores, labels = [], [], []
    if not os.path.exists(pred_file):
        return boxes, scores, labels
    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, xc, yc, w, h, conf = map(float, parts)
            # Convert YOLO center to xmin,ymin,xmax,ymax
            xmin = xc - w/2
            ymin = yc - h/2
            xmax = xc + w/2
            ymax = yc + h/2
            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(conf)
            labels.append(int(cls))
    return boxes, scores, labels

# ==== Process each image ====
all_files = set(os.listdir(pred_s_dir)).union(os.listdir(pred_n_dir))

for filename in all_files:
    if not filename.endswith('.txt'):
        continue

    pred_s_file = os.path.join(pred_s_dir, filename)
    pred_n_file = os.path.join(pred_n_dir, filename)
    #pred_m_file=os.path.join(pred_m_dir, filename)

    # Load boxes from each model
    boxes_s, scores_s, labels_s = load_yolo_preds(pred_s_file)
    boxes_n, scores_n, labels_n = load_yolo_preds(pred_n_file)
    #boxes_m, scores_m, labels_m = load_yolo_preds(pred_m_file)

    boxes_list = [boxes_s, boxes_n]
    scores_list = [scores_s, scores_n]
    labels_list = [labels_s, labels_n]

    # Handle empty predictions gracefully
    if all(len(b)==0 for b in boxes_list):
        continue

    # Run WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_thr
    )

    # Save in YOLO format (class_id xc yc w h conf)
    with open(os.path.join(output_dir, filename), "w") as f_out:
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            bw = xmax - xmin
            bh = ymax - ymin
            f_out.write(f"{int(label)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.6f}\n")

    print(f"✅ WBF done: {filename}")

print("\n✅ All images processed and saved to:", output_dir)
