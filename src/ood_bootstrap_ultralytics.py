# ood_bootstrap_ultralytics.py
import os
import glob
import yaml
import random
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
model_path = "/home/reshma/ADRIF/ADRIF/src/runs/detect/train/weights/best.pt"
ood_yaml   = "/home/reshma/ADRIF/OOD/dataset/data_ood.yaml"

IMG_SIZE = 640
CONF_PRED_EXPORT = 0.001   # export very low conf so PR/mAP can be computed
B = 1000
SEED = 123

# Where to export predictions (txt)
EXPORT_DIR = "/home/reshma/ADRIF/OOD/boot_eval/preds_txt"


# =========================
# Utilities: YOLO label/pred readers (image-wise)
# =========================
def read_yolo_txt(path, img_w=640, img_h=640, has_conf=False):
    """
    YOLO txt:
      GT:   cls cx cy w h
      Pred: cls cx cy w h conf
    returns list of:
      GT:   [cls, x1,y1,x2,y2]
      Pred: [cls, x1,y1,x2,y2,conf]
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            cx = float(p[1]) * img_w
            cy = float(p[2]) * img_h
            w  = float(p[3]) * img_w
            h  = float(p[4]) * img_h
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            if has_conf and len(p) >= 6:
                conf = float(p[5])
                out.append([cls, x1,y1,x2,y2, conf])
            elif has_conf:
                out.append([cls, x1,y1,x2,y2, 1.0])
            else:
                out.append([cls, x1,y1,x2,y2])
    return out

def load_yolo_folder_by_image(labels_dir, img_w=640, img_h=640, has_conf=False):
    """
    labels_dir contains per-image txt files.
    returns dict: image_id -> list of anns
    """
    d = {}
    for p in glob.glob(os.path.join(labels_dir, "*.txt")):
        img_id = os.path.splitext(os.path.basename(p))[0]
        d[img_id] = read_yolo_txt(p, img_w, img_h, has_conf=has_conf)
    return d


# =========================
# Metrics (image-wise matching)
# =========================
def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    u = a1 + a2 - inter
    return inter/u if u > 0 else 0.0

def ap_from_pr(prec, rec):
    if len(prec) == 0:
        return 0.0
    rec = [0.0] + list(rec) + [1.0]
    prec = [0.0] + list(prec) + [0.0]
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    area = 0.0
    for i in range(1, len(rec)):
        area += (rec[i]-rec[i-1]) * prec[i]
    return area

def pr_curve_class(pred_by_img, gt_by_img, cid, iou_thr):
    # GT per image
    gt_cls = {}
    total_gt = 0
    for img, gts in gt_by_img.items():
        g = [gt for gt in gts if gt[0] == cid]
        gt_cls[img] = g
        total_gt += len(g)
    if total_gt == 0:
        return [], []

    # preds list (img, box, conf)
    preds = []
    for img, ps in pred_by_img.items():
        for p in ps:
            if p[0] == cid:
                preds.append((img, p[1:5], float(p[5])))
    preds.sort(key=lambda x: x[2], reverse=True)

    matched = {img: [False]*len(gt_cls[img]) for img in gt_cls.keys()}
    tp = fp = 0
    prec, rec = [], []

    for img, pbox, conf in preds:
        best = 0.0
        best_j = -1
        gts = gt_cls.get(img, [])
        m = matched.get(img, [])
        for j, gt in enumerate(gts):
            if m[j]:
                continue
            v = iou(pbox, gt[1:5])
            if v > best:
                best = v
                best_j = j
        if best >= iou_thr and best_j != -1:
            tp += 1
            matched[img][best_j] = True
        else:
            fp += 1
        prec.append(tp/(tp+fp) if (tp+fp) else 0.0)
        rec.append(tp/total_gt if total_gt else 0.0)

    return prec, rec

def evaluate_imgwise(pred_by_img, gt_by_img, class_names, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    ncls = len(class_names)
    # APs per threshold per class
    aps = {t: [] for t in iou_thresholds}
    per_class_ap50 = {}

    for t in iou_thresholds:
        for cid in range(ncls):
            p, r = pr_curve_class(pred_by_img, gt_by_img, cid, t)
            ap = ap_from_pr(p, r)
            aps[t].append(ap)
            if abs(t-0.5) < 1e-12:
                per_class_ap50[class_names[cid]] = ap

    map50 = float(np.mean(aps[0.5])) if 0.5 in aps else 0.0
    map5095 = float(np.mean([ap for t in iou_thresholds for ap in aps[t]])) if iou_thresholds else 0.0

    # precision/recall/f1 at IoU=0.5 greedy per image
    TP = FP = FN = 0
    for img, gts in gt_by_img.items():
        ps = pred_by_img.get(img, [])
        ps_sorted = sorted(ps, key=lambda x: float(x[5]), reverse=True)
        used = [False]*len(gts)

        for p in ps_sorted:
            cid = p[0]
            pbox = p[1:5]
            best_i = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                if used[j] or gt[0] != cid:
                    continue
                v = iou(pbox, gt[1:5])
                if v >= 0.5 and v > best_i:
                    best_i = v
                    best_j = j
            if best_j != -1:
                TP += 1
                used[best_j] = True
            else:
                FP += 1

        FN += sum(1 for m in used if not m)

    precision = TP/(TP+FP) if (TP+FP) else 0.0
    recall = TP/(TP+FN) if (TP+FN) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0

    return {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mAP_0.5": map50,
            "mAP_0.5:0.95": map5095
        },
        "per_class_ap50": per_class_ap50
    }


# =========================
# Bootstrap
# =========================
def bootstrap(pred_by_img, gt_by_img, class_names, B=1000, seed=123):
    random.seed(seed)
    np.random.seed(seed)

    img_ids = sorted(gt_by_img.keys())
    n = len(img_ids)

    overall_keys = ["precision","recall","f1","mAP_0.5","mAP_0.5:0.95"]
    boot_overall = {k: [] for k in overall_keys}
    boot_ap50 = {c: [] for c in class_names}

    # point estimate
    point = evaluate_imgwise(pred_by_img, gt_by_img, class_names)

    for _ in range(B):
        sampled = [img_ids[random.randrange(n)] for _ in range(n)]
        # rename duplicates to make them count independently
        gt_bs = {}
        pred_bs = {}
        counts = defaultdict(int)
        for img in sampled:
            counts[img] += 1
            new_id = f"{img}__bs{counts[img]}"
            gt_bs[new_id] = gt_by_img[img]
            pred_bs[new_id] = pred_by_img.get(img, [])

        res = evaluate_imgwise(pred_bs, gt_bs, class_names)
        o = res["overall"]
        boot_overall["precision"].append(o["precision"])
        boot_overall["recall"].append(o["recall"])
        boot_overall["f1"].append(o["f1"])
        boot_overall["mAP_0.5"].append(o["mAP_0.5"])
        boot_overall["mAP_0.5:0.95"].append(o["mAP_0.5:0.95"])

        for c in class_names:
            boot_ap50[c].append(res["per_class_ap50"].get(c, 0.0))

    def summ(x):
        x = np.array(x, dtype=float)
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "ci_low": float(np.percentile(x, 2.5)),
            "ci_high": float(np.percentile(x, 97.5))
        }

    return point, {k: summ(v) for k,v in boot_overall.items()}, {c: summ(v) for c,v in boot_ap50.items()}


# =========================
# Export OOD predictions using Ultralytics
# =========================
def export_predictions_ultralytics(model_path, data_yaml, export_dir, conf=0.001, imgsz=640):
    """
    Exports per-image prediction txt files in YOLO format with confidence.
    """
    os.makedirs(export_dir, exist_ok=True)

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    # Ultralytics YAML can store split paths in different ways
    test_path = data.get("test", None)
    if test_path is None:
        raise ValueError("Could not find 'test' in YAML.")

    # test_path can be a directory or a txt listing images
    img_paths = []
    if isinstance(test_path, str) and os.path.isdir(test_path):
        img_paths = sorted(glob.glob(os.path.join(test_path, "*.*")))
    elif isinstance(test_path, str) and os.path.isfile(test_path) and test_path.endswith(".txt"):
        with open(test_path, "r") as f2:
            img_paths = [ln.strip() for ln in f2 if ln.strip()]
    else:
        # could be relative path
        base = os.path.dirname(data_yaml)
        cand = os.path.join(base, str(test_path))
        if os.path.isdir(cand):
            img_paths = sorted(glob.glob(os.path.join(cand, "*.*")))
        elif os.path.isfile(cand) and cand.endswith(".txt"):
            with open(cand, "r") as f2:
                img_paths = [ln.strip() for ln in f2 if ln.strip()]
        else:
            raise ValueError(f"Unrecognized test path in YAML: {test_path}")

    model = YOLO(model_path)
    # run predict and export txt + conf
    model.predict(
        source=img_paths,
        conf=conf,
        imgsz=imgsz,
        save=False,
        save_txt=True,
        save_conf=True,
        project=export_dir,
        name="preds",
        exist_ok=True,
        verbose=False
    )

    # Ultralytics writes to export_dir/preds/labels
    pred_labels_dir = os.path.join(export_dir, "preds", "labels")
    if not os.path.isdir(pred_labels_dir):
        raise RuntimeError(f"Prediction labels not found at: {pred_labels_dir}")

    return pred_labels_dir, model.names, data


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 1) Export predictions
    pred_labels_dir, names_dict, data = export_predictions_ultralytics(
        model_path=model_path,
        data_yaml=ood_yaml,
        export_dir=EXPORT_DIR,
        conf=CONF_PRED_EXPORT,
        imgsz=IMG_SIZE
    )

    # class_names in index order
    class_names = [names_dict[i] for i in range(len(names_dict))]

    # 2) Locate GT labels directory from YAML
    # Ultralytics YAML typically: path + test/images + test/labels
    base_path = data.get("path", "")
    test_split = data.get("test", "")
    # Most common: test: images/test (dir). Then labels are sibling "labels"
    # We'll try a few common options.
    gt_labels_dir = None
    candidates = []

    # If YAML has explicit labels path (rare)
    if "labels" in data:
        candidates.append(os.path.join(base_path, data["labels"]))

    # Derive from test path
    def add_candidate(p):
        if p and isinstance(p, str):
            candidates.append(p)
            if base_path:
                candidates.append(os.path.join(base_path, p))

    add_candidate(test_split)

    # If test points to images dir, labels usually replace /images/ with /labels/
    for c in list(candidates):
        if "images" in c:
            candidates.append(c.replace("images", "labels"))

    # If test is a txt list, labels usually live at: path/labels/test or similar
    if isinstance(test_split, str) and test_split.endswith(".txt"):
        candidates.append(os.path.join(base_path, "labels", "test"))
        candidates.append(os.path.join(os.path.dirname(os.path.join(base_path, test_split)), "..", "labels"))

    # pick first existing folder with txt files
    for c in candidates:
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.txt"))) > 0:
            gt_labels_dir = c
            break

    if gt_labels_dir is None:
        raise RuntimeError(
            "Could not auto-locate GT labels dir from YAML. "
            "Set gt_labels_dir manually in code."
        )

    print(f"\nGT labels:   {gt_labels_dir}")
    print(f"Pred labels: {pred_labels_dir}")

    # 3) Load GT + preds by image id
    gt_by_img = load_yolo_folder_by_image(gt_labels_dir, img_w=IMG_SIZE, img_h=IMG_SIZE, has_conf=False)
    pred_by_img = load_yolo_folder_by_image(pred_labels_dir, img_w=IMG_SIZE, img_h=IMG_SIZE, has_conf=True)

    # ensure we only evaluate images that have GT
    pred_by_img = {k: pred_by_img.get(k, []) for k in gt_by_img.keys()}

    # 4) Bootstrap
    point, overall_ci, ap50_ci = bootstrap(pred_by_img, gt_by_img, class_names, B=B, seed=SEED)

    # 5) Print as tables
    print("\n=== OOD Point Estimate ===")
    for k, v in point["overall"].items():
        print(f"{k:>12}: {v:.4f}")

    print("\n=== Overall Metrics (Bootstrap, image-level) ===")
    print("| Metric | Mean | Std | 95% CI low | 95% CI high |")
    print("|---|---:|---:|---:|---:|")
    for k in ["precision","recall","f1","mAP_0.5","mAP_0.5:0.95"]:
        s = overall_ci[k]
        print(f"| {k} | {s['mean']:.4f} | {s['std']:.4f} | {s['ci_low']:.4f} | {s['ci_high']:.4f} |")

    print("\n=== Per-class AP@0.5 (Bootstrap, image-level) ===")
    print("| Class | Mean | Std | 95% CI low | 95% CI high |")
    print("|---|---:|---:|---:|---:|")
    for c in class_names:
        s = ap50_ci[c]
        print(f"| {c} | {s['mean']:.4f} | {s['std']:.4f} | {s['ci_low']:.4f} | {s['ci_high']:.4f} |")