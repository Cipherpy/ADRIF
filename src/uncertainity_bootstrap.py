import os
import json
import random
import numpy as np
from collections import defaultdict

# ---------------------------
# Readers
# ---------------------------

def read_yolo_format(file_path, img_width, img_height):
    """
    Read YOLO format annotations (normalized coordinates).
    Format:
      GT:   class_id cx cy w h
      Pred: class_id cx cy w h conf

    Returns:
      list of [class_id, x1, y1, x2, y2] (GT)
      or   [class_id, x1, y1, x2, y2, conf] (Pred)
    """
    annotations = []
    if not os.path.exists(file_path):
        return annotations

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            if len(parts) >= 6:
                conf = float(parts[5])
                annotations.append([class_id, x1, y1, x2, y2, conf])
            else:
                annotations.append([class_id, x1, y1, x2, y2])

    return annotations


def read_pascal_voc_format(file_path):
    """
    Read Pascal VOC XML format annotations.
    Returns list of [class_name, x1, y1, x2, y2]
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        print("xml.etree.ElementTree not available for Pascal VOC format")
        return []

    annotations = []
    if not os.path.exists(file_path):
        return annotations

    tree = ET.parse(file_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        annotations.append([class_name, x1, y1, x2, y2])

    return annotations


def read_coco_format(file_path):
    """
    Read COCO JSON format annotations.
    Returns list of:
      GT:   [class_id, x1, y1, x2, y2]
      Pred: [class_id, x1, y1, x2, y2, conf]
    """
    annotations = []
    if not os.path.exists(file_path):
        return annotations

    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        # Ground truth style COCO
        for ann in data["annotations"]:
            class_id = ann["category_id"]
            x1, y1, w, h = ann["bbox"]
            x2, y2 = x1 + w, y1 + h
            annotations.append([class_id, x1, y1, x2, y2])

    elif isinstance(data, list):
        # Predictions list
        for ann in data:
            class_id = ann.get("category_id", ann.get("class_id", 0))
            x1, y1, w, h = ann["bbox"]
            x2, y2 = x1 + w, y1 + h
            conf = ann.get("score", ann.get("confidence", 1.0))
            annotations.append([class_id, x1, y1, x2, y2, float(conf)])

    return annotations


def read_csv_format(file_path):
    """
    Read CSV format annotations.
    Expected columns: filename, class_id/class_name, x1, y1, x2, y2, [confidence]
    Returns dict filename -> list of anns
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available for CSV format")
        return {}

    annotations_by_file = defaultdict(list)
    if not os.path.exists(file_path):
        return dict(annotations_by_file)

    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        filename = str(row["filename"])
        if "class_id" in df.columns:
            cls = row["class_id"]
        elif "class_name" in df.columns:
            cls = row["class_name"]
        else:
            cls = 0

        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])

        if "confidence" in df.columns:
            conf = float(row["confidence"])
            annotations_by_file[filename].append([cls, x1, y1, x2, y2, conf])
        else:
            annotations_by_file[filename].append([cls, x1, y1, x2, y2])

    return dict(annotations_by_file)


def auto_detect_format(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return "yolo"
    if ext == ".xml":
        return "pascal_voc"
    if ext == ".json":
        return "coco"
    if ext == ".csv":
        return "csv"
    return "unknown"


def read_annotations_from_folder(folder_path, format_type="auto", img_width=640, img_height=640):
    """
    Read all annotation files from a folder.
    Returns dict: image_id (base filename) -> list of annotations
    """
    annotations_by_file = defaultdict(list)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return dict(annotations_by_file)

    # CSV special case (single file)
    csv_file = os.path.join(folder_path, "annotations.csv")
    if os.path.exists(csv_file):
        return read_csv_format(csv_file)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        detected = auto_detect_format(file_path) if format_type == "auto" else format_type

        if detected == "yolo":
            anns = read_yolo_format(file_path, img_width, img_height)
        elif detected == "pascal_voc":
            anns = read_pascal_voc_format(file_path)
        elif detected == "coco":
            anns = read_coco_format(file_path)
        else:
            continue

        base_name = os.path.splitext(filename)[0]
        annotations_by_file[base_name] = anns

    return dict(annotations_by_file)


# ---------------------------
# Class mapping + normalization (image-wise)
# ---------------------------

def create_class_mapping(annotations_dict):
    """
    Create mapping from class names to IDs (only if string labels exist).
    If you already use numeric IDs everywhere (YOLO), this stays empty.
    """
    class_to_id = {}
    id_counter = 0
    for file_annotations in annotations_dict.values():
        for ann in file_annotations:
            cls = ann[0]
            if isinstance(cls, str) and cls not in class_to_id:
                class_to_id[cls] = id_counter
                id_counter += 1
    return class_to_id


def normalize_annotations_by_image(annotations_by_file, class_to_id=None, has_conf=False):
    """
    Normalize to:
      GT:   [class_id, x1, y1, x2, y2]
      Pred: [class_id, x1, y1, x2, y2, conf]
    Returns dict img_id -> list
    """
    out = {}
    for img_id, anns in annotations_by_file.items():
        norm = []
        for ann in anns:
            cls = ann[0]
            if class_to_id and isinstance(cls, str):
                cid = class_to_id[cls]
            else:
                cid = int(cls)

            if has_conf:
                if len(ann) >= 6:
                    norm.append([cid, float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5])])
                else:
                    norm.append([cid, float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4]), 1.0])
            else:
                norm.append([cid, float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4])])
        out[img_id] = norm
    return out


# ---------------------------
# Metrics
# ---------------------------

def calculate_iou(box1, box2):
    """IoU between [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_average_precision(precisions, recalls):
    """COCO-style area under interpolated PR curve."""
    if not precisions or not recalls:
        return 0.0
    recalls = [0.0] + list(recalls) + [1.0]
    precisions = [0.0] + list(precisions) + [0.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def pr_curve_for_class_imgwise(pred_by_img, gt_by_img, class_id, iou_threshold=0.5):
    """
    Correct PR curve: match predictions to GTs within the same image only.
    """
    # Collect GTs by image
    gt_cls = {}
    total_gt = 0
    for img, gts in gt_by_img.items():
        g = [gt for gt in gts if gt[0] == class_id]
        gt_cls[img] = g
        total_gt += len(g)

    if total_gt == 0:
        return [], [], []

    # Collect predictions with image id
    preds = []
    for img, ps in pred_by_img.items():
        for p in ps:
            if p[0] == class_id:
                preds.append((img, p[1:5], float(p[5])))

    preds.sort(key=lambda x: x[2], reverse=True)

    matched = {img: [False] * len(gt_cls[img]) for img in gt_cls.keys()}

    tp = fp = 0
    precisions, recalls, confidences = [], [], []

    for img, pbox, conf in preds:
        best_iou = 0.0
        best_j = -1
        gts_img = gt_cls.get(img, [])
        m_img = matched.get(img, [])

        for j, gt in enumerate(gts_img):
            if m_img[j]:
                continue
            iou = calculate_iou(pbox, gt[1:5])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j != -1:
            tp += 1
            matched[img][best_j] = True
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_gt if total_gt else 0.0

        precisions.append(precision)
        recalls.append(recall)
        confidences.append(conf)

    return precisions, recalls, confidences


def evaluate_detection_performance_imgwise(pred_by_img, gt_by_img, class_names=None, iou_thresholds=None):
    """
    Returns:
      overall_metrics: precision, recall, f1_score, mAP_0.5, mAP_0.5:0.95
      per_class_metrics: AP@0.5
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    # class set
    classes = set()
    for gts in gt_by_img.values():
        for gt in gts:
            classes.add(gt[0])
    for ps in pred_by_img.values():
        for p in ps:
            classes.add(p[0])
    all_classes = sorted(classes)

    if class_names is None:
        class_names = [f"Class_{cid}" for cid in all_classes]

    results = {"per_class_metrics": {}, "overall_metrics": {}}
    aps_per_threshold = defaultdict(list)

    # AP / mAP
    for iou_t in iou_thresholds:
        class_aps = []
        for cid in all_classes:
            prec, rec, _ = pr_curve_for_class_imgwise(pred_by_img, gt_by_img, cid, iou_threshold=iou_t)
            ap = calculate_average_precision(prec, rec)
            class_aps.append(ap)

            if abs(iou_t - 0.5) < 1e-12:
                cname = class_names[all_classes.index(cid)] if all_classes.index(cid) < len(class_names) else f"Class_{cid}"
                results["per_class_metrics"][cname] = {"ap": float(ap)}

        aps_per_threshold[iou_t] = class_aps

    results["overall_metrics"]["mAP_0.5"] = float(np.mean(aps_per_threshold[0.5])) if 0.5 in aps_per_threshold else 0.0
    all_aps = [ap for t in iou_thresholds for ap in aps_per_threshold[t]]
    results["overall_metrics"]["mAP_0.5:0.95"] = float(np.mean(all_aps)) if all_aps else 0.0

    # Precision/Recall/F1 at IoU=0.5 (greedy match, per image)
    tp = fp = fn = 0
    for img, gts in gt_by_img.items():
        ps = pred_by_img.get(img, [])
        ps_sorted = sorted(ps, key=lambda x: float(x[5]), reverse=True)
        used_gt = [False] * len(gts)

        for p in ps_sorted:
            cid = p[0]
            pbox = p[1:5]
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                if used_gt[j] or gt[0] != cid:
                    continue
                iou = calculate_iou(pbox, gt[1:5])
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j != -1:
                tp += 1
                used_gt[best_j] = True
            else:
                fp += 1

        fn += sum(1 for m in used_gt if not m)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    results["overall_metrics"]["precision"] = float(precision)
    results["overall_metrics"]["recall"] = float(recall)
    results["overall_metrics"]["f1_score"] = float(f1)

    return results


def print_results(results):
    print("=== Object Detection Performance Metrics ===\n")
    print("Overall Metrics:")
    for metric, value in results["overall_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print("\nPer-Class Average Precision (AP @ IoU 0.5):")
    for class_name, metrics in results["per_class_metrics"].items():
        print(f"  {class_name}: {metrics['ap']:.4f}")


# ---------------------------
# Bootstrap CI
# ---------------------------

def bootstrap_ci_from_folders(
    ground_truth_folder,
    predictions_folder,
    gt_format="yolo",
    pred_format="yolo",
    img_width=640,
    img_height=640,
    class_names=None,
    B=1000,
    seed=123,
):
    """
    Image-level bootstrap (resample images with replacement).
    Computes mean/std/95% CI for:
      precision, recall, f1_score, mAP_0.5, mAP_0.5:0.95
    plus per-class AP@0.5.
    """
    random.seed(seed)
    np.random.seed(seed)

    gt_ann = read_annotations_from_folder(ground_truth_folder, gt_format, img_width, img_height)
    pred_ann = read_annotations_from_folder(predictions_folder, pred_format, img_width, img_height)

    class_to_id = create_class_mapping({**gt_ann, **pred_ann})

    gt_by_img = normalize_annotations_by_image(gt_ann, class_to_id=class_to_id, has_conf=False)
    pred_by_img = normalize_annotations_by_image(pred_ann, class_to_id=class_to_id, has_conf=True)

    img_ids = sorted(list(gt_by_img.keys()))
    n = len(img_ids)
    if n == 0:
        raise ValueError("No GT images found to bootstrap.")

    # point estimate
    point = evaluate_detection_performance_imgwise(pred_by_img, gt_by_img, class_names=class_names)

    overall_keys = ["precision", "recall", "f1_score", "mAP_0.5", "mAP_0.5:0.95"]
    boot_overall = {k: [] for k in overall_keys}
    boot_perclass_ap = defaultdict(list)

    for _ in range(B):
        sampled = [img_ids[random.randrange(n)] for _ in range(n)]  # with replacement

        # make duplicates count independently by renaming IDs
        gt_bs = {}
        pred_bs = {}
        counts = defaultdict(int)
        for img in sampled:
            counts[img] += 1
            new_id = f"{img}__bs{counts[img]}"
            gt_bs[new_id] = gt_by_img.get(img, [])
            pred_bs[new_id] = pred_by_img.get(img, [])

        res = evaluate_detection_performance_imgwise(pred_bs, gt_bs, class_names=class_names)

        for k in overall_keys:
            boot_overall[k].append(res["overall_metrics"].get(k, 0.0))

        for cname, m in res["per_class_metrics"].items():
            boot_perclass_ap[cname].append(m["ap"])

    def summarize(arr):
        arr = np.asarray(arr, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    summary = {
        "point_estimate": point,
        "overall": {k: summarize(v) for k, v in boot_overall.items()},
        "per_class_ap_0.5": {c: summarize(v) for c, v in boot_perclass_ap.items()},
    }
    return summary


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    gt_folder = "/home/reshma/ADRIF/ADRIF/model_data/test/labels"
    pred_folder = "/home/reshma/ADRIF/ADRIF/Final_result/normal_predictions"

    class_names = [
        "Globicephala macrorhynchus",
        "Stenella longirostris",
        "Balaenoptera musculus",
    ]

    # Evaluate once
    gt_ann = read_annotations_from_folder(gt_folder, "yolo", 640, 640)
    pred_ann = read_annotations_from_folder(pred_folder, "yolo", 640, 640)
    class_to_id = create_class_mapping({**gt_ann, **pred_ann})

    gt_by_img = normalize_annotations_by_image(gt_ann, class_to_id=class_to_id, has_conf=False)
    pred_by_img = normalize_annotations_by_image(pred_ann, class_to_id=class_to_id, has_conf=True)

    results = evaluate_detection_performance_imgwise(pred_by_img, gt_by_img, class_names=class_names)
    print_results(results)

    # Bootstrap CI
    ci = bootstrap_ci_from_folders(
        ground_truth_folder=gt_folder,
        predictions_folder=pred_folder,
        gt_format="yolo",
        pred_format="yolo",
        img_width=640,
        img_height=640,
        class_names=class_names,
        B=1000,
        seed=123,
    )

    print("\n=== Bootstrap (Image-level) 95% CI ===")
    for k, s in ci["overall"].items():
        print(f"{k}: mean={s['mean']:.4f}, std={s['std']:.4f}, 95%CI=({s['ci95_low']:.4f}, {s['ci95_high']:.4f})")

    print("\nPer-class AP@0.5 bootstrap CI:")
    for cname, s in ci["per_class_ap_0.5"].items():
        print(f"{cname}: mean={s['mean']:.4f}, std={s['std']:.4f}, 95%CI=({s['ci95_low']:.4f}, {s['ci95_high']:.4f})")