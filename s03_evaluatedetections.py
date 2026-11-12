"""
s04: evaluate the detected picture elements (from s02) against the ground truth, with the
semantic threshold calibrated in s03.

For every pipeline x proximity metric x semantic threshold, predictions are matched
one-to-one to ground-truth icons (see classes/evaluation.py), and per-image and
overall precision / recall are written. A super summary collects all runs.

Edit the SETTINGS section, then run:  python s04_evaluate_detections.py

Code layout:
  this file                      settings + main() (what to run, collect summary)
  classes/evaluation.py          similarity, matching, metrics, Evaluator
  classes/evaluation_plots.py    annotated images, IoU/similarity plot, sensitivity report
  classes/detection_metrics.py   optional mAP / mAR
"""

import json
import os
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd

from classes.evaluation import Evaluator, Matcher, SemanticSimilarity, data_folder_name, report_corrupt_files
from classes.evaluation_plots import threshold_sensitivity_report

# ===========================================================================
# SETTINGS
# ===========================================================================

# --- Data -------------------------------------------------------------------
#   True  -> real rich pictures: ground truth in 01_real_data/<folder>/*_gt.json
#            (made by s01b_prepare_real_data.py), results in 03_evaluation_real/
#   False -> synthetic batches in 01_synthetic_data/<batch>/, results in 03_evaluation/
# Every data folder (synthetic batch or real-data folder) is evaluated separately:
# results go to <results root>/<pipeline>/<folder>/ and the folder name ends each
# row name in super_summary.xlsx.
USE_REAL_DATA = True
SYNTHETIC_DATA_ROOT = "01_synthetic_data"
REAL_DATA_ROOT = "01_real_data"
DETECTIONS_ROOT = "02_detected_picture_elements"
DATA_ROOT = REAL_DATA_ROOT if USE_REAL_DATA else SYNTHETIC_DATA_ROOT
EVALUATION_ROOT = "03_evaluation_real" if USE_REAL_DATA else "03_evaluation"

# --- Pipelines and settings to evaluate --------------------------------------
# Default run on synthetic data: every pipeline x metric x threshold below.
PIPELINES = ["llm_gemma4-31b_icons_rgb_fullres", "kde_scene_llm_gemma4-31b_icons_rgb_fullres"]
PROXIMITY_METRICS = ['iou-10', 'iou-20', 'iou-30', 'iou-40', 'iou-50', 'iou-60', 'iou-70', 'iou-80', 'iou-90',
                     'centerdist-100', 'centerdist-200', 'centerdist-400', 'centerdist-600', 'centerdist-800']
SEMANTIC_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Only evaluate the images this reference pipeline also has predictions for, in
# the same data folder (so all pipelines are compared on the same images).
# None = all images. Not used on real data.
REF_PIPELINE = "kde_scene_llm_gemma4-31b_icons_ensemble_verify_ignore_big_rgb_fullres"

# Quick run (synthetic data only): TEST_PIPELINES at the paper setting.
TEST = True
TEST_PIPELINES = [f"kde_scene_llm_gemma4-31b_icons_ensemble_verify_{model}_only"
                  for model in ["ministral-3:8b", "llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "minicpm-v:8b"]]

# Paper setting, used for real data and for TEST.
PAPER_PROXIMITY_METRIC = 'iou-30'

# --- Semantic similarity threshold of the paper setting --------------------------
#   True  -> use the threshold calibrated and locked by s03_threshold_calibration.py
#            (THRESHOLD_LOCK). s04 stops if there is no lock, if the locked model is
#            CLIP (s04 scores labels with the Sentence-Transformer), or if
#            SEMANTIC_MODEL differs from the model s03 calibrated with; it warns if
#            the lock has not been checked on the held-out test images yet.
#            The value is rounded to 4 decimals (at most 0.00005 off).
#   False -> use MANUAL_SEMANTIC_THRESHOLD.
# Run s03 train and test BEFORE looking at s04 results, and don't change the
# threshold afterwards.
USE_LOCKED_THRESHOLD = True
THRESHOLD_LOCK = "03_evaluation/threshold_calibration/threshold_lock.json"
MANUAL_SEMANTIC_THRESHOLD = 0.375

# Pipelines in the paper's table: folder -> name used in the paper.
PAPER_PIPELINES = {
    "kde_scene_kde_icon_clip_icons_rgb_fullres": "SIGC_clip",
    "llm_gemma4-latest_icons_rgb_fullres": "GC_small",
    "llm_gemma4-31b_icons_rgb_fullres": "GC_big",
    "kde_scene_llm_gemma4-latest_icons_rgb_fullres": "SGC_small",
    "kde_scene_llm_gemma4-31b_icons_rgb_fullres": "SGC_big",
    "kde_scene_llm_gemma4-31b_icons_ensemble_verify_ignore_big_rgb_fullres": "SGC_big_ensemble",
}

# --- Matching ------------------------------------------------------------------
#   "hungarian" -> optimal one-to-one assignment among the pairs that pass the
#                  label and box thresholds: most matches first, then the highest
#                  total weight  SEMANTIC_WEIGHT * similarity + (1 - SEMANTIC_WEIGHT) * box proximity
#                  (IoU, or 1 - distance/threshold for centre distance).
#   "greedy"    -> the original first-fit matching, in prediction order.
MATCHING_METHOD = "hungarian"
SEMANTIC_WEIGHT = 0.5
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Optional COCO-style metrics (mAP, mAR, ...; needs torchmetrics) -------------
COMPUTE_DETECTION_METRICS = False

# --- Threshold sensitivity run -------------------------------------------------
#   True -> instead of the runs above, evaluate SENSITIVITY_PIPELINES at every
#           sensitivity threshold (see below) and write a report (precision /
#           recall per pipeline x threshold, ranks, Kendall's tau vs. the ranking
#           at the paper threshold, figures) to <EVALUATION_ROOT>/threshold_sensitivity/.
#           The usual super_summary.xlsx is left untouched.
# Thresholds and reference:
#   with USE_LOCKED_THRESHOLD: the locked threshold (reference) and the 5th, 25th,
#     75th and 95th percentiles of the thresholds chosen in s03's CV folds (if those
#     barely vary: the locked threshold -0.05, -0.025, +0.025 and +0.05);
#   otherwise: MANUAL_SENSITIVITY_THRESHOLDS around MANUAL_SEMANTIC_THRESHOLD (the
#     old bootstrap interval 0.344-0.474, 0.375, 0.398 and 0.443).
RUN_THRESHOLD_SENSITIVITY = False
MANUAL_SENSITIVITY_THRESHOLDS = [0.344, 0.375, 0.398, 0.443, 0.474]
SENSITIVITY_PROXIMITY = "iou-30"
SENSITIVITY_PIPELINES = PAPER_PIPELINES
# which summary rows to report (with-bbox matching; see classes.evaluation.summarize)
SENSITIVITY_METRICS = {
    "precision": "considering bbox mean. precision",
    "recall": "considering bbox mean. recall",
}

ELEMENT_TYPES = ['Icon']


# ===========================================================================
# RUN
# ===========================================================================

@dataclass
class SemanticThreshold:
    value: float              # threshold of the paper setting
    sensitivity: list         # thresholds for the sensitivity run (includes value)
    source: str               # where it came from, for the log


def resolve_semantic_threshold():
    """The paper-setting threshold: from the s03 lock (with checks) or the manual setting."""
    if not USE_LOCKED_THRESHOLD:
        sens = sorted(set(MANUAL_SENSITIVITY_THRESHOLDS) | {MANUAL_SEMANTIC_THRESHOLD})
        return SemanticThreshold(MANUAL_SEMANTIC_THRESHOLD, sens, "manual setting (MANUAL_SEMANTIC_THRESHOLD)")

    if not os.path.exists(THRESHOLD_LOCK):
        raise SystemExit(f"No {THRESHOLD_LOCK}: run 's03_threshold_calibration.py train' and 'test' first, "
                         f"or set USE_LOCKED_THRESHOLD = False to use MANUAL_SEMANTIC_THRESHOLD.")
    with open(THRESHOLD_LOCK) as f:
        lock = json.load(f)
    if lock["model"] != "sentence_transformer":
        raise SystemExit(f"The locked threshold belongs to the '{lock['model']}' model, but s04 scores labels with "
                         f"the Sentence-Transformer, so the threshold doesn't apply. Set USE_LOCKED_THRESHOLD = False, "
                         f"or add {lock['model']} scoring to classes/evaluation.py.")
    calibrated_with = lock.get("similarity_model")
    if calibrated_with and calibrated_with != SEMANTIC_MODEL:
        raise SystemExit(f"The threshold was calibrated with '{calibrated_with}', but SEMANTIC_MODEL is "
                         f"'{SEMANTIC_MODEL}'. A threshold only applies to the model it was calibrated on.")
    if not lock.get("test_runs"):
        print("WARNING: the locked threshold has not been checked on the held-out test images yet "
              "(run 's03_threshold_calibration.py test' before reporting results).")
    if lock.get("earlier_locks"):
        print("WARNING: earlier locks were evaluated on the test set before this one was made, so the test "
              "set is not untouched (see threshold_lock.json).")

    value = round(float(lock["threshold"]), 4)
    pct = lock.get("fold_threshold_percentiles")
    if pct:
        sens = sorted({round(float(pct[q]), 4) for q in ("5", "25", "75", "95")} | {value})
    else:  # lock written before the percentiles were recorded
        print("NOTE: the lock has no fold-threshold percentiles (re-run s03 train to add them); "
              "the sensitivity run uses MANUAL_SENSITIVITY_THRESHOLDS around the locked threshold.")
        sens = sorted(set(MANUAL_SENSITIVITY_THRESHOLDS) | {value})
    if len(sens) < 3:  # the CV folds (almost) all chose the same threshold
        steps = (-0.05, -0.025, 0.025, 0.05)
        print(f"NOTE: the CV fold thresholds barely vary ({sens}); the sensitivity run adds "
              f"{', '.join(f'{d:+g}' for d in steps)} around the locked threshold.")
        sens = sorted(set(sens) | {round(min(1.0, max(0.0, value + d)), 4) for d in steps})
    tested = len(lock.get("test_runs", []))
    source = (f"{THRESHOLD_LOCK} ({lock['model']}, locked {lock.get('locked_at', '?')}, "
              f"tested on held-out images {tested} time(s))")
    return SemanticThreshold(value, sens, source)


@dataclass
class RunPlan:
    pipelines: list
    proximity_metrics: list
    semantic_thresholds: list
    ref_pipeline: str = None
    summary_dir: str = None
    sensitivity: bool = False
    reference_threshold: float = None


def select_run_plan(threshold):
    """
    Which pipelines and settings to evaluate. In order of precedence:
      1. RUN_THRESHOLD_SENSITIVITY -> SENSITIVITY_PIPELINES x the sensitivity thresholds
      2. USE_REAL_DATA             -> PIPELINES at the paper setting, all images
      3. TEST                      -> TEST_PIPELINES at the paper setting
      4. otherwise                 -> PIPELINES x PROXIMITY_METRICS x SEMANTIC_THRESHOLDS
                                      (plus the paper threshold, if not in the list)
    threshold: SemanticThreshold from resolve_semantic_threshold().
    """
    if USE_REAL_DATA:
        plan = RunPlan(PIPELINES, [PAPER_PROXIMITY_METRIC], [threshold.value], ref_pipeline=None)
    elif TEST:
        plan = RunPlan(TEST_PIPELINES, [PAPER_PROXIMITY_METRIC], [threshold.value], ref_pipeline=REF_PIPELINE)
    else:
        grid = list(SEMANTIC_THRESHOLDS)
        if not any(np.isclose(grid, threshold.value)):
            grid = sorted(grid + [threshold.value])
        plan = RunPlan(PIPELINES, PROXIMITY_METRICS, grid, ref_pipeline=REF_PIPELINE)
    plan.summary_dir = EVALUATION_ROOT

    if RUN_THRESHOLD_SENSITIVITY:
        plan = RunPlan(list(SENSITIVITY_PIPELINES), [SENSITIVITY_PROXIMITY], list(threshold.sensitivity),
                       ref_pipeline=plan.ref_pipeline, summary_dir=f"{EVALUATION_ROOT}/threshold_sensitivity",
                       sensitivity=True)
    return plan


def report_data_folders(pipelines, data_dirs, ref_pipeline):
    """Say which data folders will be evaluated and which pipelines (or the
    reference pipeline) have no predictions for any of them."""
    folders = [data_folder_name(d) for d in data_dirs]
    print(f"{len(folders)} data folder(s) in {DATA_ROOT}/: {folders}")
    for p in list(pipelines) + ([ref_pipeline] if ref_pipeline else []):
        if not any(os.path.isdir(os.path.join(DETECTIONS_ROOT, p, f)) for f in folders):
            role = "reference pipeline" if p == ref_pipeline and p not in pipelines else "pipeline"
            print(f"WARNING: no predictions for {role} '{p}' in {DETECTIONS_ROOT}/{p}/<{'|'.join(folders)}>"
                  + (" - no images will pass the reference filter" if role == "reference pipeline"
                     else " - it will be skipped"))


def summary_row_name(pipeline, element_type, metric, threshold, method, data_dir):
    """Row name in super_summary.xlsx: pipeline_element|metric|threshold|method|data folder."""
    return f"{pipeline}_{element_type}|{metric}|{threshold}|{method}|{data_folder_name(data_dir)}"


def evaluate_all(plan, data_dirs, similarity, failures):
    """Run every pipeline x setting x data folder; returns [(row name, summary DataFrame)].
    Unreadable JSON files are collected in `failures` ({path: reason}) and their images skipped."""
    summaries = []
    for pipeline in plan.pipelines:
        similarity.current_pipeline = pipeline
        first_setting = True
        for metric in plan.proximity_metrics:
            for threshold in plan.semantic_thresholds:
                matcher = Matcher(similarity, threshold, metric, method=MATCHING_METHOD,
                                  semantic_weight=SEMANTIC_WEIGHT)
                evaluator = Evaluator(matcher, DETECTIONS_ROOT, EVALUATION_ROOT,
                                      write_pred_gt=first_setting,  # predictions/GT are the same for every setting
                                      detection_metrics=COMPUTE_DETECTION_METRICS,
                                      failures=failures)
                for element_type in ELEMENT_TYPES:
                    print(f"evaluating pipeline {pipeline} | {metric} | {threshold}")
                    for data_dir in data_dirs:
                        summary = evaluator.evaluate(data_dir, pipeline, plan.ref_pipeline, element_type)
                        if summary is None:
                            continue
                        name = summary_row_name(pipeline, element_type, metric, threshold, matcher.method, data_dir)
                        summaries.append((name, summary))
                        print("Evaluation complete\n")
                first_setting = False
    return summaries


def build_super_summary(summaries):
    """One row per evaluation ('pipeline' = row name), one column per summary metric."""
    metric_names = list(summaries[0][1]['metric'])
    rows = [list(summary['value']) for _, summary in summaries]
    table = pd.DataFrame(rows, columns=metric_names)
    table.insert(0, "pipeline", [name for name, _ in summaries])
    return table


def main():
    threshold = resolve_semantic_threshold()
    print(f"Semantic threshold {threshold.value:g} from {threshold.source}")
    plan = select_run_plan(threshold)
    data_dirs = sorted(p for p in glob(f"{DATA_ROOT}/*") if os.path.isdir(p))
    if plan.sensitivity:
        print(f"Threshold sensitivity run: {plan.pipelines} | {plan.proximity_metrics} | "
              f"thresholds {plan.semantic_thresholds}")
    elif TEST and not USE_REAL_DATA:
        print(plan.pipelines)
    report_data_folders(plan.pipelines, data_dirs, plan.ref_pipeline)

    similarity = SemanticSimilarity(SEMANTIC_MODEL)
    failures = {}
    summaries = evaluate_all(plan, data_dirs, similarity, failures)
    os.makedirs(plan.summary_dir, exist_ok=True)
    if not summaries:
        report_corrupt_files(failures, f"{plan.summary_dir}/failed_json.csv")
        raise SystemExit(f"Nothing was evaluated: check {DATA_ROOT}/ and {DETECTIONS_ROOT}/")

    super_summary = build_super_summary(summaries)
    super_summary.to_excel(f"{plan.summary_dir}/super_summary.xlsx", index=False)

    if plan.sensitivity:
        threshold_sensitivity_report(
            super_summary, plan.summary_dir, SENSITIVITY_PIPELINES, SENSITIVITY_METRICS,
            reference=threshold.value,
            interval=(min(threshold.sensitivity), max(threshold.sensitivity)),
        )
    else:
        # the threshold-annotation sample is drawn from this file: only refresh it in normal runs
        similarity.cache_frame().to_excel(f"{EVALUATION_ROOT}/similarity_cache.xlsx", index=False)
    report_corrupt_files(failures, f"{plan.summary_dir}/failed_json.csv")


if __name__ == "__main__":
    main()
