"""
Evaluation of detected picture elements against ground truth (used by s03).

Reading order:
  1. Geometry            iou(), center_distance()
  2. Semantic similarity SemanticSimilarity: label similarity with a cache
  3. Loading             load_elements(), add_rescaled_boxes()
  4. Matching            Matcher: which (prediction, ground truth) pairs count as
                         true positives, with Hungarian or greedy assignment
  5. Metrics             image_metrics(), summarize()
  6. Evaluator           runs 3-5 over a data folder and writes the result files
"""

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from glob import glob

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from classes.evaluation_plots import draw_annotated_image, plot_iou_semantic_mapping

# Threshold belonging to each proximity metric name used in the settings.
# 'iou*': minimum IoU of the boxes; 'centerdist*': maximum centre distance in pixels.
PROXIMITY_THRESHOLDS = {
    'iou-10': 0.1, 'iou-20': 0.2, 'iou-30': 0.3, 'iou-40': 0.4, 'iou-50': 0.5,
    'iou-60': 0.6, 'iou-70': 0.7, 'iou-80': 0.8, 'iou-90': 0.9,
    'iou': 0.5, 'iou-min': 0.1, 'iou-max': 0.9,
    'centerdist': 800, 'centerdist-min': 0, 'centerdist-max': 1000, 'centerdist-threshold': 100,
    'centerdist-100': 100, 'centerdist-200': 200, 'centerdist-400': 400,
    'centerdist-600': 600, 'centerdist-800': 800,
}


# ---------------------------------------------------------------------------
# 1. Geometry (boxes are [x_min, y_min, x_max, y_max])
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    """Intersection over union of two boxes (0 when they don't overlap)."""
    inter_w = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(box_a, box_b):
    """Euclidean distance between the centres of two boxes, in pixels."""
    ax, ay = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    bx, by = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ---------------------------------------------------------------------------
# 2. Semantic similarity of labels
# ---------------------------------------------------------------------------

class SemanticSimilarity:
    """
    Cosine similarity of two labels' Sentence-Transformer embeddings, cached.

    One instance is shared by all evaluations, so the model is loaded once. Every
    computed pair is remembered with the pipeline that first produced it
    (`current_pipeline`); cache_frame() returns them for similarity_cache.xlsx,
    from which the threshold-annotation sample is drawn.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.current_pipeline = None
        self._cache = {}  # (text1, text2) -> {"text1", "text2", "similarity", "Pipeline"}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _compute(self, text1, text2):
        from sentence_transformers import util
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def similarity(self, text1, text2):
        """Similarity of two labels; 0.0 when either is missing or empty
        (e.g. "unlabelled" from majority voting), so such labels never match."""
        if not isinstance(text1, str) or not isinstance(text2, str) or not text1.strip() or not text2.strip():
            return 0.0
        key = (text1, text2)
        if key not in self._cache:
            self._cache[key] = dict(text1=text1, text2=text2, similarity=self._compute(text1, text2),
                                    Pipeline=self.current_pipeline)
        return self._cache[key]["similarity"]

    def cache_frame(self):
        return pd.DataFrame(list(self._cache.values()), columns=["text1", "text2", "similarity", "Pipeline"])


# ---------------------------------------------------------------------------
# 3. Loading ground truth and predictions
# ---------------------------------------------------------------------------

def load_elements(json_path):
    """
    Elements of one image (ground truth or predictions) as a DataFrame, one row
    per element. Columns are the keys of the first Icon; an image without icons
    gives an empty frame with the basic columns.
    """
    with open(json_path) as f:
        elements = json.load(f)
    icons = [e for e in elements if e['element_type'] == 'Icon']
    if not icons:  # e.g. a real image on which a pipeline detected nothing
        return pd.DataFrame({key: [] for key in ["represents", "bounding_box", "element_type"]})
    return pd.DataFrame({key: [e.get(key) for e in elements] for key in icons[0].keys()})


def add_rescaled_boxes(pred_df, image_width):
    """
    Add 'reconst_bounding_box': the prediction boxes scaled to the ground-truth
    image. Predictions are in the frame of the image the pipeline processed; new
    results store its size in "image_size", older ones were always 1000 px wide.
    """
    if not len(pred_df):
        pred_df['reconst_bounding_box'] = pred_df.get('bounding_box', [])
        return pred_df

    def rescale(row):
        size = row.get('image_size') if 'image_size' in row else None
        pred_width = size[0] if isinstance(size, (list, tuple)) and size else 1000
        factor = image_width / pred_width
        return [int(v * factor) for v in row['bounding_box']]

    pred_df['reconst_bounding_box'] = pred_df.apply(rescale, axis=1)
    return pred_df


@lru_cache(maxsize=300)
def open_image(path):
    from PIL import Image
    return Image.open(path)


# ---------------------------------------------------------------------------
# 4. Matching predictions to ground truth
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Outcome of matching one image. `pairs` lists the matched
    (prediction position, ground-truth position) pairs."""
    tp: pd.DataFrame
    fp: pd.DataFrame
    fn: pd.DataFrame
    pairs: list = field(default_factory=list)


def _rows_to_frame(rows):
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _has_label(row):
    label = row["represents"]
    return isinstance(label, str) and bool(label.strip())


class Matcher:
    """
    Decides which predictions match which ground-truth elements (one-to-one).

    A pair is a *candidate* when (see is_candidate):
      * both have the same element type,
      * for icons: label similarity >= semantic_threshold,
      * when boxes are used: IoU >= threshold ('iou*') or centre distance
        < threshold ('centerdist*').
    Candidates are then assigned:
      * "hungarian": optimal assignment; maximises the number of matches first,
        then the summed pair weight (see pair_scores). Order-independent.
      * "greedy":    first fit, in prediction order (the original method).
    """

    def __init__(self, similarity, semantic_threshold, proximity_metric,
                 method="hungarian", semantic_weight=0.5, use_rescaled_boxes=True):
        method = method.lower()
        if method not in ("hungarian", "greedy"):
            raise ValueError(f"Unknown matching method: {method}")
        self.similarity = similarity
        self.semantic_threshold = semantic_threshold
        self.proximity_metric = proximity_metric
        self.proximity_threshold = PROXIMITY_THRESHOLDS[proximity_metric]
        self.method = method
        self.semantic_weight = semantic_weight
        self.use_rescaled_boxes = use_rescaled_boxes

    # -- pair tests ---------------------------------------------------------

    def _pred_box(self, pred):
        return pred["reconst_bounding_box"] if self.use_rescaled_boxes else pred['bounding_box']

    def is_candidate(self, pred, gt, use_boxes):
        """True if the pair passes the element-type, label and (optionally) box tests."""
        if pred['element_type'] != gt['element_type']:
            return False  # icons and scenes are never compared
        if pred['element_type'] == 'Icon':
            if not (_has_label(pred) and _has_label(gt)):
                return False  # a missing/empty label never matches
            if self.similarity.similarity(pred["represents"], gt["represents"]) < self.semantic_threshold:
                return False
        if not use_boxes:
            return True
        pred_box, gt_box = self._pred_box(pred), gt["bounding_box"]
        if self.proximity_metric.startswith('iou'):
            return iou(pred_box, gt_box) >= self.proximity_threshold
        if self.proximity_metric.startswith('center'):
            return center_distance(tuple(pred_box), tuple(gt_box)) < self.proximity_threshold
        return True

    def pair_scores(self, pred, gt):
        """
        (weight, semantic_similarity, iou) of a pair. weight is in 0..1:
            semantic_weight * semantic similarity + (1 - semantic_weight) * box proximity
        box proximity = IoU for 'iou*' metrics, and 1 - distance / threshold
        (clipped to 0..1) for 'centerdist*' metrics. Scenes count as similarity 1.
        """
        pred_box, gt_box = self._pred_box(pred), gt["bounding_box"]
        sem = self.similarity.similarity(pred["represents"], gt["represents"]) \
            if pred['element_type'] == 'Icon' else 1.0
        overlap = iou(pred_box, gt_box)
        if self.proximity_metric.startswith('center') and self.proximity_threshold > 0:
            dist = center_distance(tuple(pred_box), tuple(gt_box))
            proximity = min(1.0, max(0.0, 1.0 - dist / self.proximity_threshold))
        else:
            proximity = overlap
        weight = self.semantic_weight * sem + (1 - self.semantic_weight) * proximity
        return weight, sem, overlap

    # -- assignment -----------------------------------------------------------

    def match(self, pred_df, gt_df, use_boxes):
        """Match one image's predictions to its ground truth -> MatchResult.
        True-positive rows hold the prediction's columns, the ground truth's columns
        prefixed with gt_ (gt_represents = its label) and, for Hungarian matching,
        match_weight / match_semantic / match_iou.
        Predictions without a label are never false positives."""
        gt_df = gt_df.assign(gt_represents=gt_df['represents'])
        preds = [row for _, row in pred_df.iterrows()]
        gts = [row for _, row in gt_df.iterrows()]
        if self.method == "hungarian":
            pairs = self._assign_hungarian(preds, gts, use_boxes)
        else:
            pairs = self._assign_greedy(preds, gts, use_boxes)

        tp = []
        for i, j, scores in pairs:
            # prediction columns as they are, ground-truth columns as gt_<name>
            gt_part = gts[j].drop(labels=['represents'])
            gt_part.index = [k if k == 'gt_represents' else f"gt_{k}" for k in gt_part.index]
            merged = pd.concat([preds[i], gt_part], axis=0)
            if scores is not None:
                merged["match_weight"], merged["match_semantic"], merged["match_iou"] = scores
            tp.append(merged.to_frame().T)
        matched_p = {i for i, _, _ in pairs}
        matched_g = {j for _, j, _ in pairs}
        fp = [p.to_frame().T for i, p in enumerate(preds) if i not in matched_p and p['represents']]
        fn = [g.to_frame().T for j, g in enumerate(gts) if j not in matched_g]
        return MatchResult(_rows_to_frame(tp), _rows_to_frame(fp), _rows_to_frame(fn),
                           [(i, j) for i, j, _ in pairs])

    def _assign_hungarian(self, preds, gts, use_boxes):
        """Optimal assignment over the candidate pairs -> [(i, j, scores)].

        Candidates cost -(BIG + weight), with weight in 0..1 and BIG larger than any
        possible total weight, so one extra match always beats a better weight:
        the number of matches is maximised first, the weight second. Other pairs
        cost 0 and are dropped if the solver assigns them anyway."""
        n_p, n_g = len(preds), len(gts)
        big = float(min(n_p, n_g) + 1)
        cost = np.zeros((n_p, n_g))
        scores = {}
        for i, pred in enumerate(preds):
            for j, gt in enumerate(gts):
                if self.is_candidate(pred, gt, use_boxes):
                    scores[(i, j)] = self.pair_scores(pred, gt)
                    cost[i, j] = -(big + scores[(i, j)][0])
        if not scores:
            return []
        rows, cols = linear_sum_assignment(cost)
        return [(i, j, scores[(i, j)]) for i, j in zip(rows, cols) if (i, j) in scores]

    def _assign_greedy(self, preds, gts, use_boxes):
        """Each prediction, in order, takes the first free candidate -> [(i, j, None)]."""
        taken, pairs = set(), []
        for i, pred in enumerate(preds):
            for j, gt in enumerate(gts):
                if self.is_candidate(pred, gt, use_boxes):
                    if j not in taken:
                        taken.add(j)
                        pairs.append((i, j, None))
                        break
        return pairs

    # -- diagnostics ----------------------------------------------------------

    def iou_semantic_mapping(self, pred_df, gt_df, pairs):
        """
        One row per prediction and per ground-truth element, with the IoU and
        semantic similarity of the pair it maps to (for the IoU/similarity plot):
          TP  the matched pair
          FP  an unmatched prediction and its closest ground truth (highest weight,
              thresholds ignored)
          FN  an unmatched ground truth and its closest prediction
        Unlabelled predictions are skipped (they are not counted as FP either);
        images without predictions or without ground truth give no rows.
        """
        preds = [row for _, row in pred_df.iterrows()]
        gts = [row for _, row in gt_df.iterrows()]
        if not preds or not gts:
            return pd.DataFrame([])

        cache = {}

        def scores(i, j):
            if (i, j) not in cache:
                cache[(i, j)] = self.pair_scores(preds[i], gts[j])
            return cache[(i, j)]

        rows = []

        def add(role, i, j):
            _, sem, overlap = scores(i, j)
            rows.append(dict(uid=gts[j].get("uid"), role=role, iou=overlap, semantic_similarity=sem,
                             pred_label=preds[i]["represents"], gt_label=gts[j]["represents"]))

        matched_p = {i for i, _ in pairs}
        matched_g = {j for _, j in pairs}
        labelled = [i for i, p in enumerate(preds) if p["represents"]]
        for i, j in pairs:
            add("TP", i, j)
        for i in labelled:
            if i not in matched_p:
                add("FP", i, max(range(len(gts)), key=lambda j: scores(i, j)[0]))
        for j in range(len(gts)):
            if j not in matched_g and labelled:
                add("FN", max(labelled, key=lambda i: scores(i, j)[0]), j)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def image_metrics(result):
    """Counts and precision/recall of one image (-1 when undefined)."""
    tp, fp, fn = len(result.tp), len(result.fp), len(result.fn)
    precision = tp / (tp + fp) if tp + fp else -1
    recall = tp / (tp + fn) if tp + fn else -1
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall)


def summarize(per_image):
    """
    Summary rows from per-image metrics, for each matching mode
    (per_image: {use_boxes: DataFrame}). "mean" = average over images (undefined
    images left out); "dataset" = from the summed counts.
    """
    rows = []
    for use_boxes, df in per_image.items():
        prefix = "" if use_boxes else "not "
        for metric in ['precision', 'recall']:
            valid = df[metric][df[metric] >= 0]
            rows.append((f"{prefix}considering bbox mean. {metric}", valid.mean() if len(valid) else -1))
        tp, fp, fn = df['tp'].sum(), df['fp'].sum(), df['fn'].sum()
        rows.append((f"{prefix}considering bbox dataset precision", tp / (tp + fp) if (tp + fp) > 0 else -1))
        rows.append((f"{prefix}considering bbox dataset recall", tp / (tp + fn) if (tp + fn) > 0 else -1))
    return pd.DataFrame(rows, columns=["metric", "value"])


# ---------------------------------------------------------------------------
# 6. Evaluator: one pipeline x one setting over a data folder
# ---------------------------------------------------------------------------

BOX_MODES = (True, False)  # match with boxes, then on labels only


def data_folder_name(data_dir):
    """Name of a data folder (a synthetic batch or a real-data folder)."""
    return os.path.basename(os.path.normpath(data_dir))


def _mode_suffix(use_boxes):
    return ' with bbox' if use_boxes else ' no bbox'


class Evaluator:
    """
    Evaluates one pipeline with one Matcher over a data folder and writes:
      <out>/annotated_images[_<method>]/{with_bbox,no_bbox}/<image>.png
      <out>/<metric>_sim-<threshold>[_<method>]/<element>/results.xlsx
      <out>/<metric>_sim-<threshold>[_<method>]/<element>/iou_semantic_mapping.png
      <out>/Pred-GT.xlsx                       (if write_pred_gt)
    where <out> = <evaluation_root>/<pipeline>/<data folder>, so every synthetic
    batch and every real-data folder has its own results.
    """

    def __init__(self, matcher, detections_root, evaluation_root, write_pred_gt=True, detection_metrics=False):
        self.matcher = matcher
        self.detections_root = detections_root
        self.evaluation_root = evaluation_root
        self.write_pred_gt = write_pred_gt
        self.detection_metrics = detection_metrics

    # -- paths ----------------------------------------------------------------

    @property
    def _method_tag(self):
        return "" if self.matcher.method == "greedy" else f"_{self.matcher.method}"

    def output_root(self, pipeline, data_dir):
        return f"{self.evaluation_root}/{pipeline}/{data_folder_name(data_dir)}"

    def results_folder(self, pipeline, data_dir, element_type):
        m = self.matcher
        return (f"{self.output_root(pipeline, data_dir)}/"
                f"{m.proximity_metric}_sim-{m.semantic_threshold}{self._method_tag}/{element_type}")

    def prediction_dir(self, pipeline, data_dir):
        """s02 writes predictions to <detections_root>/<pipeline>/<data folder>/."""
        return os.path.join(self.detections_root, pipeline, data_folder_name(data_dir))

    def prediction_path(self, pipeline, data_dir, gt_name):
        pred_dir = self.prediction_dir(pipeline, data_dir)
        path = os.path.join(pred_dir, gt_name.replace("_gt.json", "_detected.json"))
        return path if os.path.exists(path) else os.path.join(pred_dir, gt_name.replace("_gt", ""))

    def ground_truth_files(self, data_dir, ref_pipeline=None):
        """*_gt.json files of a data folder. With ref_pipeline, only the images that
        pipeline also has predictions for in the same data folder (so pipelines
        are compared on the same images)."""
        names = sorted(f for f in os.listdir(data_dir) if f.endswith("_gt.json"))
        if ref_pipeline:
            ref_files = glob(os.path.join(self.prediction_dir(ref_pipeline, data_dir), "*_detected.json"))
            ref_ids = {os.path.basename(p)[:-len('_detected.json')] for p in ref_files}
            kept = [n for n in names if n[:-len('_gt.json')] in ref_ids]
            if len(kept) < len(names):
                print(f"{data_folder_name(data_dir)}: evaluating the {len(kept)} of {len(names)} images that "
                      f"reference pipeline '{ref_pipeline}' also processed")
            names = kept
        return names

    # -- one image ------------------------------------------------------------

    def load_image(self, pipeline, data_dir, gt_name, element_type):
        """(gt_df, pred_df) of one image, or None when a file is missing."""
        gt_path = os.path.join(data_dir, gt_name)
        pred_path = self.prediction_path(pipeline, data_dir, gt_name)
        if not os.path.exists(pred_path):
            print(f"Missing predictions for {gt_name[:-len('.json')]}, skipping {pred_path}")
            return None
        uid = f"{data_dir}/{gt_name}"
        gt_df = load_elements(gt_path).assign(uid=uid)
        gt_df = gt_df[gt_df.element_type == element_type].copy()
        pred_df = load_elements(pred_path).assign(uid=uid)
        pred_df = pred_df[pred_df.element_type == element_type].copy()
        if self.matcher.use_rescaled_boxes:
            width, _ = open_image(gt_path.replace("_gt.json", ".png")).size
            add_rescaled_boxes(pred_df, width)
        return gt_df, pred_df

    # -- whole folder ---------------------------------------------------------

    def evaluate(self, data_dir, pipeline, ref_pipeline=None, element_type='Icon'):
        """Evaluate all images of data_dir; returns the summary DataFrame
        (columns metric, value), or None if no image had predictions."""
        out_root = self.output_root(pipeline, data_dir)
        per_image = {mode: [] for mode in BOX_MODES}
        results = {mode: [] for mode in BOX_MODES}
        mappings, all_gt, all_pred = [], [], []

        for gt_name in tqdm(self.ground_truth_files(data_dir, ref_pipeline)):
            loaded = self.load_image(pipeline, data_dir, gt_name, element_type)
            if loaded is None:
                continue
            gt_df, pred_df = loaded
            all_gt.append(gt_df)
            all_pred.append(pred_df)
            for use_boxes in BOX_MODES:
                result = self.matcher.match(pred_df, gt_df, use_boxes)
                if use_boxes:
                    mappings.append(self.matcher.iou_semantic_mapping(pred_df, gt_df, result.pairs))
                self._save_annotated_image(out_root, data_dir, gt_name, gt_df, result, use_boxes)
                per_image[use_boxes].append(dict(image=gt_name, **image_metrics(result)))
                results[use_boxes].append(result)

        if not all_gt:
            print(f"No ground-truth/prediction pairs for pipeline {pipeline} in {data_dir}, skipping")
            return None

        gt_all, pred_all = pd.concat(all_gt), pd.concat(all_pred)
        per_image = {mode: pd.DataFrame(rows) for mode, rows in per_image.items()}
        summary = summarize(per_image)
        if self.detection_metrics:
            from classes.detection_metrics import detection_metrics
            summary = pd.concat([summary, detection_metrics(
                gt_all, pred_all, self.matcher.semantic_threshold, self.matcher.similarity.model_name)])
        mapping = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()

        if self.write_pred_gt:
            with pd.ExcelWriter(f"{out_root}/Pred-GT.xlsx", engine="openpyxl") as writer:
                gt_all.to_excel(writer, sheet_name="Gt", index=False)
                pred_all.to_excel(writer, sheet_name="Pred", index=False)
        out_folder = self.results_folder(pipeline, data_dir, element_type)
        self._write_results(out_folder, per_image, results, summary, mapping)
        self._plot_mapping(out_folder, pipeline, mapping)
        return summary

    # -- outputs --------------------------------------------------------------

    def _save_annotated_image(self, out_root, data_dir, gt_name, gt_df, result, use_boxes):
        folder = f"{out_root}/annotated_images{self._method_tag}/{'with_bbox' if use_boxes else 'no_bbox'}"
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(data_dir, gt_name).replace("_gt.json", ".png")
        draw_annotated_image(open_image(image_path), gt_df, result.tp, result.fp, result.fn,
                             f"{folder}/{gt_name.replace('.json', '')}.png")

    def _write_results(self, out_folder, per_image, results, summary, mapping):
        os.makedirs(out_folder, exist_ok=True)
        out_path = f"{out_folder}/results.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for use_boxes in BOX_MODES:
                suffix = _mode_suffix(use_boxes)
                runs = results[use_boxes]
                pd.concat([r.tp for r in runs]).to_excel(writer, sheet_name=f"True Positives {suffix}", index=False)
                pd.concat([r.fp for r in runs]).to_excel(writer, sheet_name=f"False Positives {suffix}", index=False)
                pd.concat([r.fn for r in runs]).to_excel(writer, sheet_name=f"False Negatives {suffix}", index=False)
                per_image[use_boxes].to_excel(writer, sheet_name=f"Result {suffix}", index=False)
            summary.to_excel(writer, sheet_name="Result summary", index=False)
            mapping.to_excel(writer, sheet_name="IoU-semantic mapping", index=False)
        print(f"Saved detailed results to {out_path}")

    def _plot_mapping(self, out_folder, pipeline, mapping):
        m = self.matcher
        path = plot_iou_semantic_mapping(
            mapping,
            semantic_threshold=m.semantic_threshold,
            iou_threshold=m.proximity_threshold if m.proximity_metric.startswith("iou") else None,
            title=f"{pipeline} | {m.proximity_metric} | similarity {m.semantic_threshold:g} | {m.method}",
            out_path=f"{out_folder}/iou_semantic_mapping.png",
        )
        print(f"Saved IoU / semantic similarity plot to {path}")
