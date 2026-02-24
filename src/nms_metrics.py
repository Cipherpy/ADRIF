import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def read_yolo_format(file_path, img_width, img_height):
    """
    Read YOLO format annotations (normalized coordinates).
    Format: class_id center_x center_y width height [confidence]
    
    Returns list of [class_id, x1, y1, x2, y2, confidence] or [class_id, x1, y1, x2, y2]
    """
    annotations = []
    
    if not os.path.exists(file_path):
        return annotations
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to x1, y1, x2, y2
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            if len(parts) == 6:  # Has confidence score
                confidence = float(parts[5])
                annotations.append([class_id, x1, y1, x2, y2, confidence])
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
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        
        annotations.append([class_name, x1, y1, x2, y2])
        
    return annotations

def read_coco_format(file_path):
    """
    Read COCO JSON format annotations.
    Returns list of [class_id, x1, y1, x2, y2] for ground truth
    or [class_id, x1, y1, x2, y2, confidence] for predictions
    """
    annotations = []
    
    if not os.path.exists(file_path):
        return annotations
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different COCO structures
    if 'annotations' in data:  # Ground truth format
        for ann in data['annotations']:
            class_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            annotations.append([class_id, x1, y1, x2, y2])
    elif isinstance(data, list):  # Predictions format
        for ann in data:
            class_id = ann.get('category_id', ann.get('class_id', 0))
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            confidence = ann.get('score', ann.get('confidence', 1.0))
            annotations.append([class_id, x1, y1, x2, y2, confidence])
            
    return annotations

def read_csv_format(file_path):
    """
    Read CSV format annotations.
    Expected columns: filename, class_id/class_name, x1, y1, x2, y2, [confidence]
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available for CSV format")
        return {}
    
    annotations_by_file = defaultdict(list)
    
    if not os.path.exists(file_path):
        return annotations_by_file
        
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        filename = row['filename']
        
        # Handle both class_id and class_name columns
        if 'class_id' in df.columns:
            class_id = row['class_id']
        elif 'class_name' in df.columns:
            class_id = row['class_name']
        else:
            class_id = 0
            
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        if 'confidence' in df.columns:
            confidence = row['confidence']
            annotations_by_file[filename].append([class_id, x1, y1, x2, y2, confidence])
        else:
            annotations_by_file[filename].append([class_id, x1, y1, x2, y2])
            
    return dict(annotations_by_file)

def auto_detect_format(file_path):
    """Auto-detect annotation format based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        return 'yolo'
    elif ext == '.xml':
        return 'pascal_voc'
    elif ext == '.json':
        return 'coco'
    elif ext == '.csv':
        return 'csv'
    else:
        return 'unknown'

def read_annotations_from_folder(folder_path, format_type='auto', img_width=640, img_height=640):
    """
    Read all annotation files from a folder.
    
    Args:
        folder_path: Path to folder containing annotation files
        format_type: 'yolo', 'pascal_voc', 'coco', 'csv', or 'auto'
        img_width, img_height: Image dimensions (needed for YOLO format)
    
    Returns:
        Dictionary mapping filename to list of annotations
    """
    annotations_by_file = defaultdict(list)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return dict(annotations_by_file)
    
    # Handle CSV format (single file for all images)
    csv_file = os.path.join(folder_path, 'annotations.csv')
    if os.path.exists(csv_file):
        return read_csv_format(csv_file)
    
    # Handle individual annotation files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        # Auto-detect format if needed
        if format_type == 'auto':
            detected_format = auto_detect_format(file_path)
        else:
            detected_format = format_type
        
        # Read based on format
        if detected_format == 'yolo':
            annotations = read_yolo_format(file_path, img_width, img_height)
        elif detected_format == 'pascal_voc':
            annotations = read_pascal_voc_format(file_path)
        elif detected_format == 'coco':
            annotations = read_coco_format(file_path)
        else:
            print(f"Unknown format for file: {filename}")
            continue
        
        # Use filename without extension as key
        base_name = os.path.splitext(filename)[0]
        annotations_by_file[base_name] = annotations
    
    return dict(annotations_by_file)

def create_class_mapping(annotations_dict):
    """Create mapping from class names to class IDs."""
    class_to_id = {}
    id_counter = 0
    
    for file_annotations in annotations_dict.values():
        for ann in file_annotations:
            class_name = ann[0]
            if isinstance(class_name, str) and class_name not in class_to_id:
                class_to_id[class_name] = id_counter
                id_counter += 1
    
    return class_to_id

def normalize_annotations(annotations_dict, class_to_id=None):
    """
    Normalize annotations to consistent format with numeric class IDs.
    Returns flat list of all annotations.
    """
    normalized = []
    
    for file_annotations in annotations_dict.values():
        for ann in file_annotations:
            # Convert class names to IDs if needed
            if class_to_id and isinstance(ann[0], str):
                class_id = class_to_id[ann[0]]
                normalized_ann = [class_id] + ann[1:]
            else:
                normalized_ann = ann.copy()
            
            normalized.append(normalized_ann)
    
    return normalized

# Include all the metric calculation functions from the previous code
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_precision_recall_per_class(predictions, ground_truths, class_id, iou_threshold=0.5):
    """Calculate precision-recall curve for a specific class."""
    class_preds = [p for p in predictions if p[0] == class_id]
    class_gts = [gt for gt in ground_truths if gt[0] == class_id]
    
    if len(class_gts) == 0:
        return [], [], []
    
    class_preds.sort(key=lambda x: x[5], reverse=True)
    gt_matched = [False] * len(class_gts)
    
    precisions = []
    recalls = []
    confidences = []
    
    tp = 0
    fp = 0
    
    for pred in class_preds:
        pred_box = pred[1:5]
        confidence = pred[5]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(class_gts):
            if gt_matched[gt_idx]:
                continue
                
            gt_box = gt[1:5]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(class_gts)
        
        precisions.append(precision)
        recalls.append(recall)
        confidences.append(confidence)
    
    return precisions, recalls, confidences

def calculate_average_precision(precisions, recalls):
    """Calculate Average Precision (AP) using interpolated precision."""
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap

def evaluate_detection_performance(predictions, ground_truths, class_names=None, iou_thresholds=None):
    """Comprehensive evaluation of object detection performance."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    pred_classes = set(p[0] for p in predictions)
    gt_classes = set(gt[0] for gt in ground_truths)
    all_classes = sorted(pred_classes.union(gt_classes))
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in all_classes]
    
    results = {
        'per_class_metrics': {},
        'overall_metrics': {}
    }
    
    aps_per_threshold = defaultdict(list)
    
    for iou_thresh in iou_thresholds:
        class_aps = []
        
        for class_id in all_classes:
            precisions, recalls, confidences = calculate_precision_recall_per_class(
                predictions, ground_truths, class_id, iou_thresh
            )
            
            ap = calculate_average_precision(precisions, recalls)
            class_aps.append(ap)
            
            if abs(iou_thresh - 0.5) < 0.01:
                class_name = class_names[all_classes.index(class_id)] if class_id < len(class_names) else f"Class_{class_id}"
                results['per_class_metrics'][class_name] = {
                    'ap': ap,
                    'precision_curve': precisions,
                    'recall_curve': recalls,
                    'confidence_curve': confidences
                }
        
        aps_per_threshold[iou_thresh] = class_aps
    
    # Calculate overall metrics
    results['overall_metrics']['mAP_0.5'] = np.mean(aps_per_threshold[0.5]) if 0.5 in aps_per_threshold else 0
    
    all_aps = []
    for iou_thresh in iou_thresholds:
        all_aps.extend(aps_per_threshold[iou_thresh])
    results['overall_metrics']['mAP_0.5:0.95'] = np.mean(all_aps) if all_aps else 0
    
    # Calculate precision and recall at IoU 0.5
    tp = fp = fn = 0
    used_gt = set()
    
    pred_indices = sorted(range(len(predictions)), key=lambda i: predictions[i][5], reverse=True)
    
    for pred_idx in pred_indices:
        pred = predictions[pred_idx]
        pred_class, pred_box = pred[0], pred[1:5]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt or gt[0] != pred_class:
                continue
                
            gt_box = gt[1:5]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou and iou >= 0.5:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            tp += 1
            used_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len([gt for gt in ground_truths if ground_truths.index(gt) not in used_gt])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results['overall_metrics']['precision'] = precision
    results['overall_metrics']['recall'] = recall
    results['overall_metrics']['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return results

def print_results(results):
    """Print formatted results."""
    print("=== Object Detection Performance Metrics ===\n")
    
    print("Overall Metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nPer-Class Average Precision (AP @ IoU 0.5):")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  {class_name}: {metrics['ap']:.4f}")

# Main execution function
def evaluate_from_folders(ground_truth_folder, predictions_folder, 
                         gt_format='auto', pred_format='auto',
                         img_width=640, img_height=640, class_names=None):
    """
    Complete evaluation pipeline reading from folders.
    
    Args:
        ground_truth_folder: Path to ground truth annotations folder
        predictions_folder: Path to predictions folder
        gt_format: Ground truth format ('auto', 'yolo', 'pascal_voc', 'coco', 'csv')
        pred_format: Predictions format ('auto', 'yolo', 'pascal_voc', 'coco', 'csv')
        img_width, img_height: Image dimensions (for YOLO format)
        class_names: List of class names
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("Reading ground truth annotations...")
    gt_annotations = read_annotations_from_folder(
        ground_truth_folder, gt_format, img_width, img_height
    )
    
    print("Reading predictions...")
    pred_annotations = read_annotations_from_folder(
        predictions_folder, pred_format, img_width, img_height
    )
    
    # Create class mapping if needed
    all_annotations = {**gt_annotations, **pred_annotations}
    class_to_id = create_class_mapping(all_annotations)
    
    # Normalize annotations
    ground_truths = normalize_annotations(gt_annotations, class_to_id)
    predictions = normalize_annotations(pred_annotations, class_to_id)
    
    print(f"Found {len(ground_truths)} ground truth annotations")
    print(f"Found {len(predictions)} predictions")
    
    # Evaluate performance
    results = evaluate_detection_performance(predictions, ground_truths, class_names)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example: Evaluate YOLO format annotations
    results = evaluate_from_folders(
        ground_truth_folder="/home/reshma/marine_mammal/model_data/test/labels",
        predictions_folder="/home/reshma/marine_mammal/Results/Study_5/normal_predictions", 
        gt_format='yolo',
        pred_format='yolo',
        img_width=640,
        img_height=640,
        class_names=['Globicephala macrorhynchus',
    'Stenella longirostris',
    'Balaenoptera musculus']
    )
    
    print_results(results)