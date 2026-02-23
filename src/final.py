import os
import cv2
import shutil
import numpy as np

# ========== PATHS ==========
image_folder = '/home/reshma/marine_mammal/model_data/test/images'
prediction_folder = '/home/reshma/marine_mammal/Results/Study_4/wbf/yolo_0.15_0.6'   # .txt files with YOLO format
distant_output_img = '/home/reshma/marine_mammal/Results/Study_5/distant_images'
normal_output_img = '/home/reshma/marine_mammal/Results/Study_5/normal_images'
distant_output_txt = '/home/reshma/marine_mammal/Results/Study_5/distant_predictions'
normal_output_txt = '/home/reshma/marine_mammal/Results/Study_5/normal_predictions'

# ========== THRESHOLDS ==========
area_threshold_ratio = 0.005  # 0.5%
distant_containment_thresh = 0.90
normal_containment_thresh = 0.95
size_ratio_limit = 0.7  # small box must be < 70% of big box to be removed

# ========== SETUP ==========
os.makedirs(distant_output_img, exist_ok=True)
os.makedirs(normal_output_img, exist_ok=True)
os.makedirs(distant_output_txt, exist_ok=True)
os.makedirs(normal_output_txt, exist_ok=True)

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h

# ========== MAIN LOOP ==========
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_folder, image_name)
    txt_name = os.path.splitext(image_name)[0] + '.txt'
    txt_path = os.path.join(prediction_folder, txt_name)

    if not os.path.exists(txt_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue
    height, width = img.shape[:2]
    img_area = height * width

    # Read and convert boxes
    raw_boxes = []
    area_ratios = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, xc, yc, w, h, conf = map(float, parts)
            box = yolo_to_xyxy(xc, yc, w, h, width, height)
            area = (box[2] - box[0]) * (box[3] - box[1])
            area_ratio = area / img_area
            area_ratios.append(area_ratio)
            raw_boxes.append({'cls': int(cls), 'xc': xc, 'yc': yc, 'w': w, 'h': h, 'conf': conf, 'box': box, 'area': area})

    if len(raw_boxes) == 0:
        continue

    # Determine image type
    p75 = np.percentile(area_ratios, 75)
    image_type = 'distant' if p75 < area_threshold_ratio else 'normal'
    containment_thresh = distant_containment_thresh if image_type == 'distant' else normal_containment_thresh

    # Apply overlap filtering
    keep_flags = [True] * len(raw_boxes)
    for i in range(len(raw_boxes)):
        for j in range(len(raw_boxes)):
            if i == j:
                continue
            small, big = (i, j) if raw_boxes[i]['area'] < raw_boxes[j]['area'] else (j, i)
            inter_area = intersection_area(raw_boxes[small]['box'], raw_boxes[big]['box'])
            containment_ratio = inter_area / raw_boxes[small]['area']
            if (containment_ratio > containment_thresh and
                raw_boxes[small]['area'] < size_ratio_limit * raw_boxes[big]['area'] and
                raw_boxes[small]['conf'] < raw_boxes[big]['conf']):
                keep_flags[small] = False

    # Filtered boxes
    final_boxes = [raw_boxes[i] for i in range(len(raw_boxes)) if keep_flags[i]]

    # Save image
    out_img_dir = distant_output_img if image_type == 'distant' else normal_output_img
    shutil.copy(image_path, os.path.join(out_img_dir, image_name))

    # Save prediction .txt
    out_txt_dir = distant_output_txt if image_type == 'distant' else normal_output_txt
    out_txt_path = os.path.join(out_txt_dir, txt_name)

    with open(out_txt_path, 'w') as f_out:
        for box in final_boxes:
            f_out.write(f"{box['cls']} {box['xc']} {box['yc']} {box['w']} {box['h']} {box['conf']}\n")

print("✅ All images and filtered prediction files saved successfully.")
