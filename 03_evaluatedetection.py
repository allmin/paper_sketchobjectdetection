import os
import json
import math
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = (
    "01_synthetic_data/"
    "05_clusters_08_classes_01_to_10_images_per_class_20_minimum_icons_"
    "00_to_00_percent_overlap_00_minimum_overlap_images_"
    "3.0_max_scaling_factor"
)

IOU_THRESHOLD = 0.5


# =============================================================================
# Geometry
# =============================================================================


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


# =============================================================================
# Matching Logic
# =============================================================================


def match_detections(gt_items, pred_items, iou_threshold):
    matched_gt = set()
    matched_pred = set()
    matches = []

    for p_idx, p in enumerate(pred_items):
        best_iou = 0.0
        best_g = None

        for g_idx, g in enumerate(gt_items):
            if g_idx in matched_gt:
                continue
            if p["class"] != g["class"]:
                continue

            i = iou(p["bounding_box"], g["bounding_box"])
            if i >= iou_threshold and i > best_iou:
                best_iou = i
                best_g = g_idx

        if best_g is not None:
            matched_pred.add(p_idx)
            matched_gt.add(best_g)
            matches.append(best_iou)

    tp = len(matches)
    fp = len(pred_items) - tp
    fn = len(gt_items) - tp

    return tp, fp, fn, matches


# =============================================================================
# Metrics
# =============================================================================


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# =============================================================================
# Main Evaluation
# =============================================================================


def evaluate():
    totals = {
        "icon": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
        "scene": {"tp": 0, "fp": 0, "fn": 0, "ious": []},
    }

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith("_detected.json"):
            continue

        base = fname.replace("_detected.json", "")
        pred_path = os.path.join(DATA_DIR, fname)
        gt_path = os.path.join(DATA_DIR, f"{base}_gt.json")

        if not os.path.exists(gt_path):
            print(f"⚠️ Missing GT for {base}, skipping")
            continue

        with open(pred_path, "r") as f:
            preds = json.load(f)

        with open(gt_path, "r") as f:
            gts = json.load(f)

        for elem_type in ["icon", "scene"]:
            pred_items = [p for p in preds if p["element_type"] == elem_type]
            gt_items = [g for g in gts if g["element_type"] == elem_type]

            tp, fp, fn, ious = match_detections(gt_items, pred_items, IOU_THRESHOLD)

            totals[elem_type]["tp"] += tp
            totals[elem_type]["fp"] += fp
            totals[elem_type]["fn"] += fn
            totals[elem_type]["ious"].extend(ious)

    # --- Report ---
    print("\n================= Evaluation Results =================\n")

    for elem_type in ["icon", "scene"]:
        tp = totals[elem_type]["tp"]
        fp = totals[elem_type]["fp"]
        fn = totals[elem_type]["fn"]

        p = precision(tp, fp)
        r = recall(tp, fn)
        f = f1(p, r)
        miou = (
            sum(totals[elem_type]["ious"]) / len(totals[elem_type]["ious"])
            if totals[elem_type]["ious"]
            else 0.0
        )

        print(f"{elem_type.upper()}")
        print("-" * 40)
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"Precision: {p:.3f}")
        print(f"Recall:    {r:.3f}")
        print(f"F1 Score:  {f:.3f}")
        print(f"Mean IoU:  {miou:.3f}")
        print()

    print("✅ Evaluation complete\n")


if __name__ == "__main__":
    evaluate()
