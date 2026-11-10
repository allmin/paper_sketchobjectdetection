"""
s03: evaluate the detected picture elements (from s02) against the ground truth.

For every pipeline x proximity metric x semantic threshold, predictions are matched
one-to-one to ground-truth icons (see classes/evaluation.py), and per-image and
overall precision / recall are written. A super summary collects all runs.

Edit the SETTINGS section, then run:  python s03_evaluatedetections.py

Code layout:
  this file                      settings + main() (what to run, collect summary)
  classes/evaluation.py          similarity, matching, metrics, Evaluator
  classes/evaluation_plots.py    annotated images, IoU/similarity plot, sensitivity report
  classes/detection_metrics.py   optional mAP / mAR
"""

import os
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd

from classes.evaluation import Evaluator, Matcher, SemanticSimilarity, data_folder_name
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
REF_PIPELINE = "kde_scene_llm_gemma4-31b_icons_ensemble_verify"

# Quick run (synthetic data only): TEST_PIPELINES at the paper setting.
TEST = True
TEST_PIPELINES = [f"kde_scene_llm_gemma4-31b_icons_ensemble_verify_{model}_only"
                  for model in ["ministral-3:8b", "llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "minicpm-v:8b"]]

# Paper setting, used for real data and for TEST.
PAPER_PROXIMITY_METRIC = 'iou-30'
PAPER_SEMANTIC_THRESHOLD = 0.375

# Pipelines in the paper's table: folder -> name used in the paper.
PAPER_PIPELINES = {
    "kde_scene_kde_icon_clip_icons": "SIGC_clip",
    "llm_gemma4-latest_icons": "GC_small",
    "llm_gemma4-31b_icons": "GC_big",
    "kde_scene_llm_gemma4-latest_icons": "SGC_small",
    "kde_scene_llm_gemma4-31b_icons": "SGC_big",
    "kde_scene_llm_gemma4-31b_icons_ensemble_verify": "SGC_big_ensemble",
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
#           threshold in SENSITIVITY_THRESHOLDS and write a report (precision /
#           recall per pipeline x threshold, ranks, Kendall's tau vs. the ranking
#           at SENSITIVITY_REFERENCE, figures) to <EVALUATION_ROOT>/threshold_sensitivity/.
#           The usual super_summary.xlsx is left untouched.
# Thresholds: the bootstrap 95% interval (0.344-0.474), the reported 0.375, the
# bagged/logistic estimate 0.398 and the frequent bootstrap pick 0.443
# (see s04_apps/bootstrap_threshold.py).
RUN_THRESHOLD_SENSITIVITY = False
SENSITIVITY_THRESHOLDS = [0.344, 0.375, 0.398, 0.443, 0.474]
SENSITIVITY_REFERENCE = 0.375
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
class RunPlan:
    pipelines: list
    proximity_metrics: list
    semantic_thresholds: list
    ref_pipeline: str = None
    summary_dir: str = None
    sensitivity: bool = False


def select_run_plan():
    """
    Which pipelines and settings to evaluate. In order of precedence:
      1. RUN_THRESHOLD_SENSITIVITY -> SENSITIVITY_PIPELINES x SENSITIVITY_THRESHOLDS
      2. USE_REAL_DATA             -> PIPELINES at the paper setting, all images
      3. TEST                      -> TEST_PIPELINES at the paper setting
      4. otherwise                 -> PIPELINES x PROXIMITY_METRICS x SEMANTIC_THRESHOLDS
    """
    if USE_REAL_DATA:
        plan = RunPlan(PIPELINES, [PAPER_PROXIMITY_METRIC], [PAPER_SEMANTIC_THRESHOLD], ref_pipeline=None)
    elif TEST:
        plan = RunPlan(TEST_PIPELINES, [PAPER_PROXIMITY_METRIC], [PAPER_SEMANTIC_THRESHOLD], ref_pipeline=REF_PIPELINE)
    else:
        plan = RunPlan(PIPELINES, PROXIMITY_METRICS, SEMANTIC_THRESHOLDS, ref_pipeline=REF_PIPELINE)
    plan.summary_dir = EVALUATION_ROOT

    if RUN_THRESHOLD_SENSITIVITY:
        thresholds = list(SENSITIVITY_THRESHOLDS)
        if not any(np.isclose(thresholds, SENSITIVITY_REFERENCE)):
            thresholds.append(SENSITIVITY_REFERENCE)
        plan = RunPlan(list(SENSITIVITY_PIPELINES), [SENSITIVITY_PROXIMITY], thresholds,
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


def evaluate_all(plan, data_dirs, similarity):
    """Run every pipeline x setting x data folder; returns [(row name, summary DataFrame)]."""
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
                                      detection_metrics=COMPUTE_DETECTION_METRICS)
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
    plan = select_run_plan()
    data_dirs = sorted(p for p in glob(f"{DATA_ROOT}/*") if os.path.isdir(p))
    if plan.sensitivity:
        print(f"Threshold sensitivity run: {plan.pipelines} | {plan.proximity_metrics} | "
              f"thresholds {plan.semantic_thresholds}")
    elif TEST and not USE_REAL_DATA:
        print(plan.pipelines)
    report_data_folders(plan.pipelines, data_dirs, plan.ref_pipeline)

    similarity = SemanticSimilarity(SEMANTIC_MODEL)
    summaries = evaluate_all(plan, data_dirs, similarity)
    if not summaries:
        raise SystemExit(f"Nothing was evaluated: check {DATA_ROOT}/ and {DETECTIONS_ROOT}/")

    os.makedirs(plan.summary_dir, exist_ok=True)
    super_summary = build_super_summary(summaries)
    super_summary.to_excel(f"{plan.summary_dir}/super_summary.xlsx", index=False)

    if plan.sensitivity:
        threshold_sensitivity_report(
            super_summary, plan.summary_dir, SENSITIVITY_PIPELINES, SENSITIVITY_METRICS,
            reference=SENSITIVITY_REFERENCE,
            interval=(min(SENSITIVITY_THRESHOLDS), max(SENSITIVITY_THRESHOLDS)),
        )
    else:
        # the threshold-annotation sample is drawn from this file: only refresh it in normal runs
        similarity.cache_frame().to_excel(f"{EVALUATION_ROOT}/similarity_cache.xlsx", index=False)


if __name__ == "__main__":
    main()
