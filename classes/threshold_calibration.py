"""
Calibrating the semantic-similarity threshold on held-out data (used by s05).

Reading order:
  1. Collect     collect_pairs(): every prediction-ground-truth pair with enough
                 box overlap, from several pipelines, on images all of them processed
  2. Split       split_images(): images (not pairs) go to train or test
  3. Sample      sample_pairs(): the same number of pairs per pipeline and split;
                 label_pairs_to_annotate(): the unique label pairs, in random order
  4. Annotate    (s04 "Calibration annotation" page) -> annotations_<role>.csv
  5. Score       score_label_pairs(): Sentence-Transformer and CLIP similarity
  6. Train       train_analysis(): compare the models and choose the threshold with
                 repeated cross-validation in which folds are split by image
  7. Test        test_analysis(): apply the locked threshold to the held-out images

Pairs whose labels are identical (after lowercasing and trimming) are left out of
sampling: they score 1.0, so every threshold accepts them and they carry no
information about where the threshold should be. Their number is reported.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve

from classes.evaluation import (CorruptJSONError, add_rescaled_boxes, data_folder_name, iou, load_elements,
                                open_image)

SPLITS = ("train", "test")
ROLES = ("primary", "second")  # primary annotates everything, second only the agreement subset

# Annotation guideline, shown in the annotation app and in the Excel coding sheet.
# Agree on it before annotating starts; every coder gets the same text.
GUIDELINE = """
Decide whether the **detected label** names the same thing as the **ground-truth label**,
so that a coder would accept it as the granular code for that icon.

- **Similar:** synonyms and spelling variants (*sofa* / *couch*); the same object described in
  more or less detail (*red ball* / *ball*); a part or a closely related type with a similar
  appearance (*face* / *person*, *cow* / *bull*, *mobile phone*/*calculator*); a slightly general name (*animal* for *cat*).
- **Not similar:** a different object (*cat* / *watch*); a label that is logically similar, but so broad it would not count as the
  code (*round object* for *pizza*, ).
- Judge only the two labels without any presumtions
"""


def normalize_label(text):
    """Lowercase, trim and collapse spaces; used to recognise identical labels."""
    return " ".join(str(text).lower().split()) if isinstance(text, str) else ""


# ---------------------------------------------------------------------------
# 1. Collect pairs with enough box overlap
# ---------------------------------------------------------------------------

def _prediction_file(detections_root, pipeline, data_dir, gt_name):
    pred_dir = os.path.join(detections_root, pipeline, data_folder_name(data_dir))
    path = os.path.join(pred_dir, gt_name.replace("_gt.json", "_detected.json"))
    return path if os.path.exists(path) else os.path.join(pred_dir, gt_name.replace("_gt", ""))


def collect_pairs(data_root, detections_root, pipelines, min_iou=0.3, element_type="Icon"):
    """
    All (prediction, ground truth) pairs with IoU >= min_iou, before any matching
    and regardless of label similarity, from every pipeline in `pipelines`
    ({folder: name}). Only images that every pipeline processed are used, so the
    pipelines contribute pairs from the same images. Unlabelled predictions are
    left out (they can never match).
    Returns (pairs DataFrame, info dict).
    """
    rows = []
    info = {"data_folders": {}, "images_missing_a_pipeline": 0, "coverage": [], "max_iou": 0.0, "overlapping_pairs": 0,
            "failed_files": {}, "images_skipped_corrupt": 0}  # failed_files: {path: reason}
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data folder '{data_root}' not found (run from the project folder?)")
    data_dirs = sorted(p for p in (os.path.join(data_root, d) for d in os.listdir(data_root)) if os.path.isdir(p))
    for data_dir in data_dirs:
        folder = data_folder_name(data_dir)
        gt_names = sorted(f for f in os.listdir(data_dir) if f.endswith("_gt.json"))
        # how many ground-truth images each pipeline has predictions for (for diagnosing)
        found = {p: sum(os.path.exists(_prediction_file(detections_root, p, data_dir, g)) for g in gt_names)
                 for p in pipelines}
        info["coverage"].append(dict(data_folder=folder, gt_images=len(gt_names), **found))
        common = [g for g in gt_names
                  if all(os.path.exists(_prediction_file(detections_root, p, data_dir, g)) for p in pipelines)]
        info["images_missing_a_pipeline"] += len(gt_names) - len(common)
        if not common:
            continue
        used = 0
        for gt_name in common:
            image = gt_name[:-len("_gt.json")]
            # read all files of this image first: if any is unreadable, skip the whole
            # image so every pipeline still contributes pairs from the same images
            try:
                gt_df = load_elements(os.path.join(data_dir, gt_name))
                preds = {p: load_elements(_prediction_file(detections_root, p, data_dir, gt_name)) for p in pipelines}
            except CorruptJSONError as e:
                info["failed_files"][e.path] = e.reason
                info["images_skipped_corrupt"] += 1
                continue
            used += 1
            gt_df = gt_df[gt_df.element_type == element_type]
            width, _ = open_image(os.path.join(data_dir, gt_name).replace("_gt.json", ".png")).size
            for pipeline, name in pipelines.items():
                pred_df = preds[pipeline]
                pred_df = add_rescaled_boxes(pred_df[pred_df.element_type == element_type].copy(), width)
                for _, pred in pred_df.iterrows():
                    if not normalize_label(pred["represents"]):
                        continue
                    for _, gt in gt_df.iterrows():
                        overlap = iou(pred["reconst_bounding_box"], gt["bounding_box"])
                        info["max_iou"] = max(info["max_iou"], overlap)
                        info["overlapping_pairs"] += overlap > 0
                        if overlap >= min_iou:
                            rows.append(dict(
                                data_folder=folder, image=image, image_group=f"{folder}/{image}",
                                pipeline=pipeline, pipeline_name=name,
                                pred_label=pred["represents"], gt_label=gt["represents"], iou=overlap,
                                pred_box=list(pred["reconst_bounding_box"]), gt_box=list(gt["bounding_box"]),
                                identical_labels=normalize_label(pred["represents"]) == normalize_label(gt["represents"]),
                            ))
        if used:
            info["data_folders"][folder] = used
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------------------
# 2. Split images, 3. sample pairs
# ---------------------------------------------------------------------------

def split_images(image_groups, test_fraction=0.3, seed=0):
    """{image_group: 'train' | 'test'}. Images are shuffled and split within each
    data folder, so every batch is represented on both sides."""
    rng = random.Random(seed)
    by_folder = {}
    for g in sorted(set(image_groups)):
        by_folder.setdefault(g.split("/")[0], []).append(g)
    split = {}
    for folder, groups in sorted(by_folder.items()):
        rng.shuffle(groups)
        n_test = int(round(len(groups) * test_fraction))
        for k, g in enumerate(groups):
            split[g] = "test" if k < n_test else "train"
    return split


def sample_pairs(pairs, split, n_per_pipeline, seed=0):
    """
    Equal allocation: n_per_pipeline[split] pairs drawn at random from each
    pipeline's non-identical pairs on that side (all of them if there are fewer).
    Returns (sampled DataFrame, shortfalls list).
    """
    pairs = pairs.assign(split=pairs.image_group.map(split))
    candidates = pairs[~pairs.identical_labels]
    sampled, shortfalls = [], []
    for k, ((side, pipeline), group) in enumerate(sorted(candidates.groupby(["split", "pipeline"]), key=lambda kv: kv[0])):
        n = n_per_pipeline[side]
        if len(group) < n:
            shortfalls.append(f"{side}/{pipeline}: only {len(group)} pairs available, wanted {n}")
        sampled.append(group.sample(min(n, len(group)), random_state=seed + k))
    out = pd.concat(sampled, ignore_index=True) if sampled else candidates.iloc[:0]
    out.insert(0, "occurrence_id", [f"o{i:05d}" for i in range(len(out))])
    return out, shortfalls


def label_pairs_to_annotate(sampled, n_agreement=100, seed=0):
    """
    Unique (detected label, ground-truth label) pairs of the sample, in random
    order, with pair_id. A label pair that occurs several times is annotated once
    and the answer applies to every occurrence. `agreement_subset` marks
    n_agreement train pairs for the second annotator (Cohen's kappa).
    """
    keys = sampled[["pred_label", "gt_label"]].drop_duplicates().reset_index(drop=True)
    keys = keys.sample(frac=1, random_state=seed).reset_index(drop=True)
    keys.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(keys))])
    counts = sampled.groupby(["pred_label", "gt_label"]).size().rename("n_occurrences").reset_index()
    keys = keys.merge(counts, on=["pred_label", "gt_label"], how="left")
    in_train = sampled[sampled.split == "train"][["pred_label", "gt_label"]].drop_duplicates()
    train_ids = keys.merge(in_train, on=["pred_label", "gt_label"]).pair_id.tolist()
    rng = random.Random(seed + 1)
    chosen = set(rng.sample(train_ids, min(n_agreement, len(train_ids))))
    keys["agreement_subset"] = keys.pair_id.isin(chosen)
    return keys


def attach_pair_ids(sampled, label_pairs):
    return sampled.merge(label_pairs[["pair_id", "pred_label", "gt_label"]], on=["pred_label", "gt_label"], how="left")


# ---------------------------------------------------------------------------
# 4. Annotations
# ---------------------------------------------------------------------------

def annotation_path(folder, role):
    return os.path.join(folder, f"annotations_{role}.csv")


def load_annotations(folder, role):
    """pair_id -> bool (similar) for one annotator role; empty if not started."""
    path = annotation_path(folder, role)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df.pair_id, df.similar.astype(bool)))


def annotator_agreement(folder, label_pairs):
    """Cohen's kappa between the two annotators on the agreement subset."""
    a, b = load_annotations(folder, "primary"), load_annotations(folder, "second")
    ids = [p for p in label_pairs[label_pairs.agreement_subset].pair_id if p in a and p in b]
    if len(ids) < 2:
        return dict(n_pairs=len(ids), kappa=np.nan, raw_agreement=np.nan)
    ya, yb = [a[p] for p in ids], [b[p] for p in ids]
    return dict(n_pairs=len(ids), kappa=cohen_kappa_score(ya, yb),
                raw_agreement=float(np.mean(np.array(ya) == np.array(yb))))


# --- Excel coding sheets for additional coders --------------------------------

SHEET_COLUMNS = ["pair_id", "Detected label", "Ground-truth label", "Answer"]
ANSWERS = ("Similar", "Not similar")
_ANSWER_VALUES = {"similar": True, "not similar": False}


def write_coding_sheet(label_pairs, path):
    """
    Blank Excel coding sheet with every label pair (same random order as the app;
    no scores, pipelines or train/test side). 'Answer' only accepts Similar /
    Not similar from a drop-down. A second tab holds the guideline.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Coding"
    ws.append(SHEET_COLUMNS)
    for r in label_pairs.itertuples():
        ws.append([r.pair_id, r.pred_label, r.gt_label, None])
    n = len(label_pairs) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(ANSWERS) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid answer",
                        error="Choose Similar or Not similar from the list.")
    ws.add_data_validation(dv)
    dv.add(f"D2:D{n}")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    for col, width in zip("ABCD", (10, 36, 36, 16)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, max_col=3):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"

    guide = wb.create_sheet("Guideline")
    guide.column_dimensions["A"].width = 110
    lines = ["How to fill in this sheet",
             "In the 'Coding' tab, choose Similar or Not similar in the Answer column for every row. "
             "Save the file under your name (e.g. Rosa.xlsx) and send it back. Don't change the other columns.",
             ""] + [line.replace("**", "").replace("*", "") for line in GUIDELINE.strip().splitlines()]
    for k, line in enumerate(lines, start=1):
        guide.cell(row=k, column=1, value=line).alignment = Alignment(wrap_text=True, vertical="top")
    guide["A1"].font = Font(bold=True, size=13)
    wb.save(path)


def read_coding_sheets(folder, label_pairs):
    """
    Read every filled-in coding sheet (*.xlsx) in `folder`; the coder's name is the
    file name. Returns ({coder: {pair_id: bool}}, problems): a sheet whose pairs
    don't match the current label_pairs.csv (e.g. made from an earlier sample) is
    rejected; blank and invalid answers are left out and reported.
    """
    coders, problems = {}, []
    if not os.path.isdir(folder):
        return coders, problems
    expected = label_pairs.set_index("pair_id")[["pred_label", "gt_label"]]
    for name in sorted(f for f in os.listdir(folder) if f.lower().endswith(".xlsx") and not f.startswith("~$")):
        coder = os.path.splitext(name)[0]
        try:
            df = pd.read_excel(os.path.join(folder, name), sheet_name="Coding", dtype=str)
        except Exception as e:  # unreadable file or no "Coding" tab
            problems.append(f"{name}: could not be read ({type(e).__name__}: {e})")
            continue
        if not set(SHEET_COLUMNS) <= set(df.columns):
            problems.append(f"{name}: missing columns {sorted(set(SHEET_COLUMNS) - set(df.columns))}")
            continue
        df = df.dropna(subset=["pair_id"])
        unknown = sorted(set(df.pair_id) - set(expected.index))
        changed = [p for p, a, b in zip(df.pair_id, df["Detected label"], df["Ground-truth label"])
                   if p in expected.index and (str(a), str(b)) != (str(expected.loc[p, "pred_label"]),
                                                                   str(expected.loc[p, "gt_label"]))]
        if unknown or changed:
            problems.append(f"{name}: REJECTED - its pairs don't match the current sample "
                            f"({len(unknown)} unknown pair ids, {len(changed)} with different labels). "
                            f"It was probably made before the last 'sample'; send the coder a new sheet.")
            continue
        answers, blank, invalid = {}, 0, []
        for pid, ans in zip(df.pair_id, df["Answer"]):
            key = " ".join(str(ans).strip().lower().split()) if isinstance(ans, str) else ""
            if not key:
                blank += 1
            elif key in _ANSWER_VALUES:
                answers[pid] = _ANSWER_VALUES[key]
            else:
                invalid.append(f"{pid}={ans!r}")
        if blank:
            problems.append(f"{name}: {blank} pair(s) without an answer (left out)")
        if invalid:
            problems.append(f"{name}: {len(invalid)} invalid answer(s), left out: {', '.join(invalid[:5])}"
                            + (" ..." if len(invalid) > 5 else ""))
        coders[coder] = answers
    return coders, problems


def fleiss_kappa(ratings):
    """Fleiss' kappa for binary ratings: rows = items, columns = raters (bool), every
    item rated by every rater."""
    ratings = np.asarray(ratings, dtype=int)
    n_items, n_raters = ratings.shape
    if n_items == 0 or n_raters < 2:
        return np.nan
    counts = np.stack([ratings.sum(axis=1), n_raters - ratings.sum(axis=1)], axis=1)  # similar / not similar
    p_item = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()
    p_cat = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_cat ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else np.nan


def coder_agreement(coders, reference="primary"):
    """
    Agreement between all coders ({coder: {pair_id: bool}}):
      per_coder  each coder vs. the reference coder: pairs in common, Cohen's
                 kappa, raw agreement, share answered 'similar'
      pairwise   Cohen's kappa for every pair of coders
      overall    Fleiss' kappa over the pairs answered by every coder
    """
    names = list(coders)

    def compare(a, b):
        ids = sorted(set(coders[a]) & set(coders[b]))
        if len(ids) < 2:
            return len(ids), np.nan, np.nan
        ya, yb = [coders[a][p] for p in ids], [coders[b][p] for p in ids]
        same = float(np.mean(np.array(ya) == np.array(yb)))
        kappa = cohen_kappa_score(ya, yb) if len(set(ya) | set(yb)) > 1 else np.nan
        return len(ids), kappa, same

    per_coder = []
    for c in names:
        row = dict(coder=c, pairs_answered=len(coders[c]),
                   share_similar=float(np.mean(list(coders[c].values()))) if coders[c] else np.nan)
        if reference in coders and c != reference:
            n, k, same = compare(reference, c)
            row.update({f"pairs_in_common_with_{reference}": n, f"kappa_vs_{reference}": k,
                        f"raw_agreement_vs_{reference}": same})
        per_coder.append(row)

    pairwise = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n, k, same = compare(a, b)
            pairwise.append(dict(coder_a=a, coder_b=b, pairs_in_common=n, cohen_kappa=k, raw_agreement=same))

    common = sorted(set.intersection(*(set(coders[c]) for c in names))) if len(names) >= 2 else []
    overall = dict(coders=len(names), pairs_answered_by_all=len(common),
                   fleiss_kappa=fleiss_kappa([[coders[c][p] for c in names] for p in common]) if common else np.nan)
    return pd.DataFrame(per_coder), pd.DataFrame(pairwise), overall


# ---------------------------------------------------------------------------
# 5. Similarity scores
# ---------------------------------------------------------------------------

def sentence_transformer_scores(text1, text2, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    e1 = model.encode(list(text1), convert_to_tensor=True, normalize_embeddings=True)
    e2 = model.encode(list(text2), convert_to_tensor=True, normalize_embeddings=True)
    return (e1 * e2).sum(dim=1).cpu().numpy()


def clip_scores(text1, text2, arch="ViT-B-32", pretrained="openai"):
    """CLIP text-text cosine similarity, as in the Prediction app."""
    import open_clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(arch)

    def encode(texts):
        with torch.no_grad():
            emb = model.encode_text(tokenizer(list(texts)).to(device))
        return emb / emb.norm(dim=-1, keepdim=True)

    return (encode(text1) * encode(text2)).sum(dim=1).cpu().numpy()


SCORERS = {"sentence_transformer": sentence_transformer_scores, "clip": clip_scores}


def score_label_pairs(label_pairs, cache_path=None, scorers=None):
    """Similarity of every label pair under each model; cached to cache_path."""
    scorers = scorers or SCORERS
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if set(label_pairs.pair_id) <= set(cached.pair_id) and all(m in cached for m in scorers):
            return cached
    scores = label_pairs[["pair_id", "pred_label", "gt_label"]].copy()
    for name, fn in scorers.items():
        scores[name] = fn(scores.pred_label.astype(str), scores.gt_label.astype(str))
    if cache_path:
        scores.to_csv(cache_path, index=False)
    return scores


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------

def fbeta_optimum(y, s, beta):
    """F-beta optimal threshold on the precision-recall curve (as in s04)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() == 0:
        return np.nan
    prec, rec, thr = precision_recall_curve(y, s)
    f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-10)
    i = int(np.argmax(f))
    return float(thr[i]) if i < len(thr) else 1.0


def scores_at(y, s, threshold, beta):
    """Precision, recall and F-beta of 'similar if score >= threshold'."""
    y = np.asarray(y, bool)
    pred = np.asarray(s, float) >= threshold
    tp, fp, fn = int(np.sum(pred & y)), int(np.sum(pred & ~y)), int(np.sum(~pred & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0
    return dict(fbeta=f, precision=p, recall=r, tp=tp, fp=fp, fn=fn)


def _cluster_bootstrap(groups, n_boot, seed):
    """Yield row indices of resamples that draw whole images with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(n_boot):
        drawn = rng.choice(unique, len(unique), replace=True)
        yield np.concatenate([rows_of[g] for g in drawn])


def _interval(values):
    v = pd.Series(values).dropna()
    return (v.quantile(0.025), v.quantile(0.975)) if len(v) else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# 6. Train: compare models, choose threshold
# ---------------------------------------------------------------------------

def grouped_cv(y, scores, groups, beta, n_splits=5, n_repeats=20, seed=0):
    """
    Repeated stratified k-fold cross-validation with folds split by image.
    In every fold the threshold is chosen on the training part and applied to the
    held-out part; decisions are pooled per repeat. All models use the same folds.
    Returns (per-repeat DataFrame, {model: fold thresholds}, number of folds used).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y, int)
    n_splits = min(n_splits, int(y.sum()), int((1 - y).sum()), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Too few similar / dissimilar pairs or images for cross-validation")
    rows, fold_thresholds = [], {m: [] for m in scores}
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        decisions = {m: np.zeros(len(y), bool) for m in scores}
        for train, held_out in cv.split(np.zeros(len(y)), y, groups):
            for m, s in scores.items():
                t = fbeta_optimum(y[train], s[train], beta)
                fold_thresholds[m].append(t)
                decisions[m][held_out] = s[held_out] >= t
        for m in scores:
            yb, pred = y.astype(bool), decisions[m]
            tp, fp, fn = int(np.sum(pred & yb)), int(np.sum(pred & ~yb)), int(np.sum(~pred & yb))
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f = (1 + beta**2) * p * rc / (beta**2 * p + rc) if p + rc else 0.0
            rows.append(dict(repeat=r, model=m, fbeta=f, precision=p, recall=rc))
    return pd.DataFrame(rows), {m: np.asarray(v, float) for m, v in fold_thresholds.items()}, n_splits


def train_analysis(occ, y, scores, beta, n_splits=5, n_repeats=20, n_boot=2000, final_threshold="median", seed=0):
    """
    occ: train occurrences (one row per sampled pair, with image_group);
    y: annotation per row; scores: {model: similarity per row}.
    Returns a dict with the comparison table, CV repeats, fold thresholds, the
    chosen model and threshold (the 'lock'), and PR curves for the figure.
    """
    groups = occ.image_group.values
    cv_rows, fold_thr, k = grouped_cv(y, scores, groups, beta, n_splits, n_repeats, seed)
    models = list(scores)

    # threshold-free comparison: average precision (PR-AUC), with image bootstrap
    ap = {m: average_precision_score(y, scores[m]) for m in models}
    boot = {m: [] for m in models}
    for idx in _cluster_bootstrap(groups, n_boot, seed):
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        for m in models:
            boot[m].append(average_precision_score(y[idx], scores[m][idx]))

    table = []
    for m in models:
        cvm = cv_rows[cv_rows.model == m]
        thr = fold_thr[m]
        full = fbeta_optimum(y, scores[m], beta)
        chosen = float(np.nanmedian(thr)) if final_threshold == "median" else full
        table.append(dict(
            model=m, pr_auc=ap[m], pr_auc_ci_low=_interval(boot[m])[0], pr_auc_ci_high=_interval(boot[m])[1],
            cv_fbeta=cvm.fbeta.mean(), cv_fbeta_low=_interval(cvm.fbeta)[0], cv_fbeta_high=_interval(cvm.fbeta)[1],
            cv_precision=cvm.precision.mean(), cv_recall=cvm.recall.mean(),
            fold_threshold_median=float(np.nanmedian(thr)),
            fold_threshold_p25=float(np.nanpercentile(thr, 25)), fold_threshold_p75=float(np.nanpercentile(thr, 75)),
            full_train_optimum=full, final_threshold=chosen,
            **{f"train_{key}": v for key, v in scores_at(y, scores[m], chosen, beta).items()},
        ))
    table = pd.DataFrame(table)

    # paired differences between the first two models (same folds / same resamples)
    diff = None
    if len(models) >= 2:
        a, b = models[0], models[1]
        d_ap = np.array(boot[a]) - np.array(boot[b])
        d_cv = (cv_rows[cv_rows.model == a].fbeta.values - cv_rows[cv_rows.model == b].fbeta.values)
        diff = dict(models=f"{a} - {b}", pr_auc_diff=ap[a] - ap[b],
                    pr_auc_diff_low=_interval(d_ap)[0], pr_auc_diff_high=_interval(d_ap)[1],
                    cv_fbeta_diff=float(np.mean(d_cv)), cv_fbeta_diff_low=_interval(d_cv)[0],
                    cv_fbeta_diff_high=_interval(d_cv)[1])

    # decision rule fixed in advance: highest CV F-beta, ties broken by PR-AUC
    best = table.sort_values(["cv_fbeta", "pr_auc"], ascending=False).iloc[0]
    curves = {m: precision_recall_curve(y, scores[m]) for m in models}
    return dict(table=table, diff=diff, cv=cv_rows, fold_thresholds=fold_thr, n_splits=k,
                chosen_model=best.model, chosen_threshold=float(best.final_threshold), curves=curves)


# ---------------------------------------------------------------------------
# 7. Test: apply the locked threshold to held-out images
# ---------------------------------------------------------------------------

def test_analysis(occ, y, scores, thresholds, beta, n_boot=2000, seed=0):
    """Locked threshold per model applied to held-out pairs, with image-bootstrap
    95% intervals. thresholds: {model: locked threshold}."""
    groups = occ.image_group.values
    rows = []
    for m, t in thresholds.items():
        point = scores_at(y, scores[m], t, beta)
        boot = [scores_at(y[idx], scores[m][idx], t, beta) for idx in _cluster_bootstrap(groups, n_boot, seed)]
        row = dict(model=m, threshold=t, **point)
        for key in ("fbeta", "precision", "recall"):
            lo, hi = _interval([b[key] for b in boot])
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = lo, hi
        rows.append(row)
    per_pipeline = []
    for (pipeline, name), g in occ.assign(_i=np.arange(len(occ))).groupby(["pipeline", "pipeline_name"]):
        for m, t in thresholds.items():
            per_pipeline.append(dict(pipeline=name, model=m, n_pairs=len(g),
                                     **scores_at(y[g._i.values], scores[m][g._i.values], t, beta)))
    return pd.DataFrame(rows), pd.DataFrame(per_pipeline)


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

def write_lock(path, payload):
    """Write a new lock. If an earlier lock was already used on the test set, its
    record is kept under 'earlier_locks' so the history can't be lost by re-training."""
    earlier = []
    if os.path.exists(path):
        old = read_lock(path)
        earlier = old.pop("earlier_locks", [])
        if old.get("test_runs"):
            earlier.append(old)
    payload = dict(payload, locked_at=datetime.now().isoformat(timespec="seconds"), test_runs=[],
                   earlier_locks=earlier)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def read_lock(path):
    with open(path) as f:
        return json.load(f)


def record_test_run(path):
    lock = read_lock(path)
    lock["test_runs"].append(datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(lock, f, indent=2, default=float)
    return len(lock["test_runs"])
