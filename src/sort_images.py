import os
import cv2
import shutil
import numpy as np

# ========== INPUT FOLDERS ==========
image_folder = '/home/reshma/marine_mammal/model_data/test/images'
prediction_folder = '/home/reshma/marine_mammal/Results/Study_4/wbf/yolo_0.15_0.6'  # YOLO txt files

# ========== OUTPUT FOLDERS ==========
distant_output = '/home/reshma/marine_mammal/Results/Study_5/distant_images'
normal_output = '/home/reshma/marine_mammal/Results/Study_5/normal_images'
area_threshold_ratio = 0.005  # 0.5%

# ========== CREATE OUTPUT FOLDERS ==========
os.makedirs(distant_output, exist_ok=True)
os.makedirs(normal_output, exist_ok=True)



# ========== PROCESS ==========
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.JPG')):
        continue

    image_path = os.path.join(image_folder, image_name)
    txt_path = os.path.join(prediction_folder, os.path.splitext(image_name)[0] + '.txt')

    if not os.path.exists(txt_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        continue

    height, width = img.shape[:2]
    img_area = width * height
    area_ratios = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            _, xc, yc, w, h, _ = map(float, parts)
            box_area = (w * width) * (h * height)
            area_ratio = box_area / img_area
            area_ratios.append(area_ratio)

    if len(area_ratios) == 0:
        continue  # no detections

    percentile_75 = np.percentile(area_ratios, 75)

    if percentile_75 < area_threshold_ratio:
        shutil.copy(image_path, os.path.join(distant_output, image_name))
    else:
        shutil.copy(image_path, os.path.join(normal_output, image_name))

print("✅ Images classified and copied using the 75th percentile rule.")