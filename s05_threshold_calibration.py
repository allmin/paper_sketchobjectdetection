"""
s05: choose the semantic-similarity threshold on annotated pairs, and test it on
held-out images.

Run the steps in order, from the project folder:

  1. python s05_threshold_calibration.py sample
       Collects every prediction-ground-truth pair with IoU >= MIN_IOU from the
       pipelines below (only images all of them processed), splits the IMAGES into
       train / test, samples the same number of pairs per pipeline on each side and
       writes the label pairs to annotate.
  2. Annotate in the app:  streamlit run s04_app.py  ->  "Calibration annotation"
       role "primary" annotates all pairs; role "second" (another person) only the
       agreement subset, for Cohen's kappa. No scores or pipeline names are shown.
  3. python s05_threshold_calibration.py train
       Compares Sentence-Transformer and CLIP on the TRAIN images with repeated
       cross-validation (folds split by image), chooses the model and threshold by
       the rule below and writes them to threshold_lock.json.
  4. python s05_threshold_calibration.py test
       Applies the locked threshold to the held-out TEST images, once. Refuses to
       run without a lock and warns if the test set was evaluated before.

Extra coders, in Excel (any time after step 1):
  python s05_threshold_calibration.py sheet
       Writes coding_sheet.xlsx: all label pairs with an Answer column that only
       accepts Similar / Not similar, plus the guideline. Each coder fills in a copy;
       save it as <OUT_DIR>/coders/<coder name>.xlsx.
  python s05_threshold_calibration.py agreement
       Cohen's kappa of every coder against the primary annotations, between every
       pair of coders, and Fleiss' kappa over all of them -> agreement_report.xlsx.
       Can be run whenever a sheet comes in; it doesn't touch the train/test analysis.
       The same tables are added to the train and test reports. The primary
       annotations (app) stay the reference standard; extra coders only measure
       how reliable it is.

Everything is written to OUT_DIR. Decide the SETTINGS before step 1 and don't
change them afterwards: they are recorded in settings.json, and step 3 / 4 refuse
to run if they were changed.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from classes import threshold_calibration as tc
from classes.evaluation import report_corrupt_files

# ===========================================================================
# SETTINGS (fix these before sampling)
# ===========================================================================
DATA_ROOT = "01_synthetic_data"
DETECTIONS_ROOT = "02_detected_picture_elements"
OUT_DIR = "03_evaluation/threshold_calibration"

PIPELINES = {                                  # folder in 02_detected_picture_elements -> name in the paper
    "llm_gemma4-31b_icons_rgb_fullres": "GC_big",
    "llm_gemma4-latest_icons_rgb_fullres": "GC_small",
    "kde_scene_llm_gemma4-31b_icons_rgb_fullres": "SGC_big",
    "kde_scene_llm_gemma4-latest_icons_rgb_fullres": "SGC_small",
}
MIN_IOU = 0.3                                  # same as the matching in s03
TEST_FRACTION = 0.3                            # share of images held out
PAIRS_PER_PIPELINE = {"train": 100, "test": 50}
AGREEMENT_PAIRS = 100                          # train pairs for the second annotator
BETA = 2.0                                     # F-beta: > 1 favours recall
CV_FOLDS = 5
CV_REPEATS = 20
BOOTSTRAP_SAMPLES = 2000
# final threshold of the chosen model: "median" of the CV fold thresholds (steadier
# with few similar pairs) or "full" = F-beta optimum on all train pairs
FINAL_THRESHOLD = "median"
# model choice, fixed in advance: highest cross-validated F-beta, ties -> PR-AUC
SEED = 2026

SETTINGS = dict(DATA_ROOT=DATA_ROOT, DETECTIONS_ROOT=DETECTIONS_ROOT, PIPELINES=PIPELINES, MIN_IOU=MIN_IOU,
                TEST_FRACTION=TEST_FRACTION, PAIRS_PER_PIPELINE=PAIRS_PER_PIPELINE,
                AGREEMENT_PAIRS=AGREEMENT_PAIRS, BETA=BETA, CV_FOLDS=CV_FOLDS, CV_REPEATS=CV_REPEATS,
                FINAL_THRESHOLD=FINAL_THRESHOLD, SEED=SEED)


MODEL_NAMES = {"sentence_transformer": "Sentence-Transformer", "clip": "CLIP"}


def path(name):
    return os.path.join(OUT_DIR, name)


CODERS_DIR = os.path.join(OUT_DIR, "coders")  # filled-in Excel coding sheets, one per coder


# ===========================================================================
# Step 1: sample
# ===========================================================================

def diagnose(info):
    """Explain why no pairs were found."""
    print(f"\nDIAGNOSIS (run from: {os.getcwd()})")
    coverage = pd.DataFrame(info["coverage"])
    if coverage.empty:
        print(f"- No data folders with ground truth in {DATA_ROOT}/.")
    else:
        print(f"- Ground-truth images per data folder, and how many of them each pipeline has predictions for:")
        print(coverage.rename(columns={p: n for p, n in PIPELINES.items()}).to_string(index=False))
        missing = [n for p, n in PIPELINES.items() if coverage[p].sum() == 0]
        if missing:
            print(f"- No predictions at all for: {', '.join(missing)}. Only images that ALL pipelines processed "
                  f"are used, so one missing pipeline leaves nothing. Check the folder names in PIPELINES.")
    if os.path.isdir(DETECTIONS_ROOT):
        existing = sorted(d for d in os.listdir(DETECTIONS_ROOT) if os.path.isdir(os.path.join(DETECTIONS_ROOT, d)))
        print(f"- Pipeline folders that exist in {DETECTIONS_ROOT}/:")
        for d in existing:
            batches = sorted(os.listdir(os.path.join(DETECTIONS_ROOT, d)))
            print(f"    {d}   ({len(batches)} data folder(s): {', '.join(batches[:3])}{' ...' if len(batches) > 3 else ''})")
    else:
        print(f"- {DETECTIONS_ROOT}/ does not exist here.")
    if info["data_folders"]:
        print(f"- On the {sum(info['data_folders'].values())} images all pipelines processed, "
              f"{info['overlapping_pairs']} prediction/ground-truth pairs overlap at all; the highest IoU is "
              f"{info['max_iou']:.3f}. If that is far below {MIN_IOU}, the boxes are probably in different frames "
              f"(check the predictions' image_size / box format against s03).")


def cmd_sample(force=False):
    if os.path.exists(path("label_pairs.csv")) and not force:
        sys.exit(f"{path('label_pairs.csv')} exists already: sampling again would replace the pairs "
                 f"(and orphan any annotations). Use --force if that is really what you want.")
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs, info = tc.collect_pairs(DATA_ROOT, DETECTIONS_ROOT, PIPELINES, MIN_IOU)
    if pairs.empty:
        diagnose(info)
        report_corrupt_files(info["failed_files"], path("failed_json.csv"))
        sys.exit(f"No pairs with IoU >= {MIN_IOU} found - see the diagnosis above.")
    split = tc.split_images(pairs.image_group, TEST_FRACTION, SEED)
    pairs["split"] = pairs.image_group.map(split)
    sampled, shortfalls = tc.sample_pairs(pairs, split, PAIRS_PER_PIPELINE, SEED)
    label_pairs = tc.label_pairs_to_annotate(sampled, AGREEMENT_PAIRS, SEED)
    sampled = tc.attach_pair_ids(sampled, label_pairs)

    pd.Series(split, name="split").rename_axis("image_group").reset_index().to_csv(path("image_split.csv"), index=False)
    pairs.to_csv(path("all_pairs.csv"), index=False)
    sampled.to_csv(path("sampled_pairs.csv"), index=False)
    label_pairs.to_csv(path("label_pairs.csv"), index=False)
    with open(path("settings.json"), "w") as f:
        json.dump(SETTINGS, f, indent=2)

    print(f"Images per data folder (processed by all {len(PIPELINES)} pipelines): {info['data_folders']}"
          f"; skipped {info['images_missing_a_pipeline']} image(s) missing a pipeline"
          f" and {info['images_skipped_corrupt']} with an unreadable JSON file")
    print(f"Images: {sum(v == 'train' for v in split.values())} train, {sum(v == 'test' for v in split.values())} test")
    print(f"Pairs with IoU >= {MIN_IOU}: {len(pairs)} ({int(pairs.identical_labels.sum())} with identical labels, "
          f"left out of sampling)")
    print(sampled.groupby(["split", "pipeline_name"]).size().unstack(fill_value=0).to_string())
    for s in shortfalls:
        print("NOTE:", s)
    print(f"{len(label_pairs)} unique label pairs to annotate ({int(label_pairs.agreement_subset.sum())} also for the "
          f"second annotator). Next: annotate in the app, then run 'train'.")
    report_corrupt_files(info["failed_files"], path("failed_json.csv"))


# ===========================================================================
# Steps 3-4 helpers
# ===========================================================================

def check_settings():
    with open(path("settings.json")) as f:
        saved = json.load(f)
    if saved != json.loads(json.dumps(SETTINGS)):
        changed = [k for k in SETTINGS if saved.get(k) != json.loads(json.dumps(SETTINGS[k]))]
        sys.exit(f"SETTINGS differ from the ones used for sampling ({', '.join(changed)}). "
                 f"Restore them; results must use the settings fixed in advance.")


def annotated_side(side):
    """Occurrences of one side with their primary annotation and both models' scores."""
    sampled = pd.read_csv(path("sampled_pairs.csv"))
    label_pairs = pd.read_csv(path("label_pairs.csv"))
    labels = tc.load_annotations(OUT_DIR, "primary")
    occ = sampled[sampled.split == side].copy()
    missing = sorted(set(occ.pair_id) - set(labels))
    if missing:
        sys.exit(f"{len(missing)} of {occ.pair_id.nunique()} {side} label pairs are not annotated yet "
                 f"(role 'primary'). Finish annotating first.")
    scores = tc.score_label_pairs(label_pairs, cache_path=path("similarity_scores.csv"))
    occ = occ.merge(scores[["pair_id"] + list(tc.SCORERS)], on="pair_id", how="left")
    y = occ.pair_id.map(labels).astype(int).values
    return occ, y, {m: occ[m].values.astype(float) for m in tc.SCORERS}, label_pairs


# ===========================================================================
# Step 3: train
# ===========================================================================

def cmd_train():
    check_settings()
    occ, y, scores, label_pairs = annotated_side("train")
    res = tc.train_analysis(occ, y, scores, BETA, CV_FOLDS, CV_REPEATS, BOOTSTRAP_SAMPLES, FINAL_THRESHOLD, SEED)
    agreement = tc.annotator_agreement(OUT_DIR, label_pairs)
    per_coder, pairwise, overall, problems = agreement_tables(label_pairs, verbose=False)

    with pd.ExcelWriter(path("train_report.xlsx"), engine="openpyxl") as w:
        pd.DataFrame([dict(n_pairs=len(y), n_similar=int(y.sum()), n_images=occ.image_group.nunique(),
                           cv_folds_used=res["n_splits"], **{k: str(v) for k, v in SETTINGS.items()})]
                     ).to_excel(w, sheet_name="setup", index=False)
        res["table"].to_excel(w, sheet_name="model comparison", index=False)
        if res["diff"]:
            pd.DataFrame([res["diff"]]).to_excel(w, sheet_name="paired difference", index=False)
        res["cv"].to_excel(w, sheet_name="cv repeats", index=False)
        pd.DataFrame({m: pd.Series(v) for m, v in res["fold_thresholds"].items()}).to_excel(
            w, sheet_name="fold thresholds", index=False)
        pd.DataFrame([agreement]).to_excel(w, sheet_name="annotator agreement", index=False)
        write_agreement_sheets(w, per_coder, pairwise, overall, problems)
    plot_train(res, path("train_figure.png"))

    if os.path.exists(path("threshold_lock.json")):
        previous = tc.read_lock(path("threshold_lock.json"))
        if previous["test_runs"] or previous.get("earlier_locks"):
            print("WARNING: the test set has already been evaluated with an earlier lock; a new lock after seeing "
                  "the test results no longer gives an untouched test. This is recorded in threshold_lock.json.")
    tc.write_lock(path("threshold_lock.json"), dict(
        model=res["chosen_model"], threshold=res["chosen_threshold"], beta=BETA,
        thresholds_all_models=dict(zip(res["table"].model, res["table"].final_threshold)),
        rule="highest cross-validated F-beta (ties: PR-AUC); final threshold = " + FINAL_THRESHOLD,
    ))

    t = res["table"].set_index("model")
    pd.set_option("display.width", 160)
    print(t[["pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "cv_fbeta", "cv_fbeta_low", "cv_fbeta_high",
             "cv_precision", "cv_recall", "final_threshold"]].round(3).to_string())
    if res["diff"]:
        print("Paired difference:", {k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in res["diff"].items()})
    print(f"Annotator agreement (app roles, agreement subset): {agreement}")
    if overall["coders"] > 2:
        print(f"Fleiss' kappa over all {overall['coders']} coders: {overall['fleiss_kappa']:.3f} "
              f"(details: run 'agreement')")
    print(f"Chosen: {res['chosen_model']} at threshold {res['chosen_threshold']:.3f} -> locked in "
          f"{path('threshold_lock.json')}. Next: run 'test' once.")


def plot_train(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sentence_transformer": "#2f6690", "clip": "#d1495b"}
    labels = MODEL_NAMES
    table = res["table"].set_index("model")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for m, (prec, rec, thr) in res["curves"].items():
        c = colors.get(m, None)
        axs[0].plot(rec, prec, color=c, lw=1.6, label=f"{labels.get(m, m)} (PR-AUC {table.loc[m, 'pr_auc']:.2f})")
        axs[0].plot(table.loc[m, "train_recall"], table.loc[m, "train_precision"], "o", color=c, ms=7,
                    markeredgecolor="black")
    axs[0].set(title="(a) Precision-recall on train pairs\n(dot: final threshold)", xlabel="recall",
               ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02))
    axs[0].legend(fontsize=8, frameon=False, loc="lower left")

    models = list(res["curves"])
    data = [res["cv"][res["cv"].model == m].fbeta.values for m in models]
    bp = axs[1].boxplot(data, patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(colors.get(m, "#999999"))
        patch.set_alpha(0.6)
    axs[1].set_xticks(range(1, len(models) + 1), [labels.get(m, m) for m in models])
    axs[1].set(title=f"(b) Cross-validated F-{BETA:g}\n({res['n_splits']}-fold x {CV_REPEATS}, folds split by image)",
               ylabel=f"F-{BETA:g} on held-out folds")

    bins = np.linspace(0, 1, 51)  # same bins for every model
    for m in models:
        thr = res["fold_thresholds"][m]
        axs[2].hist(thr[~np.isnan(thr)], bins=bins, alpha=0.5, color=colors.get(m), label=labels.get(m, m))
        axs[2].axvline(table.loc[m, "final_threshold"], color=colors.get(m), ls="--", lw=1.5)
    axs[2].set(title="(c) Threshold chosen in each fold\n(dashed: final threshold)", xlabel="similarity threshold",
               ylabel="folds", xlim=(0, 1))
    axs[2].legend(fontsize=8, frameon=False)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===========================================================================
# Step 4: test
# ===========================================================================

def all_coders(label_pairs, verbose=True):
    """Every coder's answers: the app roles ('primary', 'second') and the Excel
    sheets in CODERS_DIR. Returns ({coder: {pair_id: bool}}, problems)."""
    coders = {}
    for role in tc.ROLES:
        answers = tc.load_annotations(OUT_DIR, role)
        if answers:
            coders[role] = answers
    sheets, problems = tc.read_coding_sheets(CODERS_DIR, label_pairs)
    for name, answers in sheets.items():
        key = name if name not in coders else f"{name} (sheet)"
        coders[key] = answers
    if verbose:
        for prob in problems:
            print("NOTE:", prob)
    return coders, problems


def agreement_tables(label_pairs, verbose=True):
    coders, problems = all_coders(label_pairs, verbose)
    per_coder, pairwise, overall = tc.coder_agreement(coders, reference="primary")
    return per_coder, pairwise, overall, problems


def write_agreement_sheets(writer, per_coder, pairwise, overall, problems):
    per_coder.to_excel(writer, sheet_name="coder agreement", index=False)
    pairwise.to_excel(writer, sheet_name="coder pairwise kappa", index=False)
    pd.DataFrame([overall]).to_excel(writer, sheet_name="coder overall", index=False)
    if problems:
        pd.DataFrame({"problem": problems}).to_excel(writer, sheet_name="coder sheet problems", index=False)


def cmd_sheet():
    if not os.path.exists(path("label_pairs.csv")):
        sys.exit("No label_pairs.csv: run 'sample' first.")
    label_pairs = pd.read_csv(path("label_pairs.csv"))
    os.makedirs(CODERS_DIR, exist_ok=True)
    tc.write_coding_sheet(label_pairs, path("coding_sheet.xlsx"))
    print(f"Wrote {path('coding_sheet.xlsx')} with {len(label_pairs)} label pairs. Give each coder a copy; "
          f"save the filled-in files as {CODERS_DIR}/<coder name>.xlsx, then run 'agreement'.")


def cmd_agreement():
    if not os.path.exists(path("label_pairs.csv")):
        sys.exit("No label_pairs.csv: run 'sample' first.")
    label_pairs = pd.read_csv(path("label_pairs.csv"))
    per_coder, pairwise, overall, problems = agreement_tables(label_pairs)
    if per_coder.empty:
        sys.exit(f"No annotations yet: annotate in the app (role 'primary') and/or put sheets in {CODERS_DIR}/")
    with pd.ExcelWriter(path("agreement_report.xlsx"), engine="openpyxl") as w:
        write_agreement_sheets(w, per_coder, pairwise, overall, problems)
    pd.set_option("display.width", 160)
    print(f"\nCoders ({len(label_pairs)} label pairs in the sample):")
    print(per_coder.round(3).to_string(index=False))
    if not pairwise.empty:
        print("\nCohen's kappa for every pair of coders:")
        print(pairwise.round(3).to_string(index=False))
    print(f"\nFleiss' kappa over all {overall['coders']} coders: {overall['fleiss_kappa']:.3f} "
          f"(on the {overall['pairs_answered_by_all']} pairs every coder answered)")
    print(f"Saved to {path('agreement_report.xlsx')}")


def cmd_test():
    check_settings()
    if not os.path.exists(path("threshold_lock.json")):
        sys.exit("No threshold_lock.json: run 'train' first. The test set must only see a locked threshold.")
    lock = tc.read_lock(path("threshold_lock.json"))
    if lock.get("earlier_locks"):
        print(f"WARNING: {len(lock['earlier_locks'])} earlier lock(s) were already evaluated on this test set "
              f"before this one was made, so it is no longer an untouched test (see threshold_lock.json).")
    if lock["test_runs"]:
        print(f"WARNING: the test set was already evaluated {len(lock['test_runs'])} time(s) with this lock "
              f"({', '.join(lock['test_runs'])}). The results below are the same, but don't change anything "
              f"based on them.")
    occ, y, scores, _ = annotated_side("test")
    thresholds = lock["thresholds_all_models"]
    overall, per_pipeline = tc.test_analysis(occ, y, scores, thresholds, BETA, BOOTSTRAP_SAMPLES, SEED)

    # robustness: test pairs whose label pair never occurs on the train side
    sampled = pd.read_csv(path("sampled_pairs.csv"))
    train_ids = set(sampled[sampled.split == "train"].pair_id)
    unseen = ~occ.pair_id.isin(train_ids).values
    unseen_res = None
    if unseen.sum() and y[unseen].sum():
        unseen_res, _ = tc.test_analysis(occ[unseen], y[unseen], {m: s[unseen] for m, s in scores.items()},
                                         thresholds, BETA, BOOTSTRAP_SAMPLES, SEED)
    overall["role"] = np.where(overall.model == lock["model"], "chosen (primary result)", "other model (secondary)")
    n_runs = tc.record_test_run(path("threshold_lock.json"))

    with pd.ExcelWriter(path("test_report.xlsx"), engine="openpyxl") as w:
        overall.to_excel(w, sheet_name="test result", index=False)
        per_pipeline.to_excel(w, sheet_name="per pipeline", index=False)
        if unseen_res is not None:
            unseen_res.to_excel(w, sheet_name="label pairs unseen in train", index=False)
        write_agreement_sheets(w, *agreement_tables(pd.read_csv(path("label_pairs.csv")), verbose=False))
        pd.DataFrame([dict(test_pairs=len(y), test_similar=int(y.sum()), test_images=occ.image_group.nunique(),
                           share_label_pairs_also_in_train=float(1 - unseen.mean()), evaluation_number=n_runs,
                           lock=json.dumps(lock))]).to_excel(w, sheet_name="setup", index=False)

    chosen = overall[overall.model == lock["model"]].iloc[0]
    train = pd.read_excel(path("train_report.xlsx"), sheet_name="model comparison").set_index("model").loc[lock["model"]]
    text = (
        f"Pairs for annotation were sampled at random from all prediction-ground-truth pairs with IoU >= {MIN_IOU:g} "
        f"from {', '.join(PIPELINES.values())}, irrespective of their semantic similarity, with equal allocation "
        f"across pipelines; pairs with identical labels were excluded. Images were split into a training "
        f"({1 - TEST_FRACTION:.0%}) and a held-out test set ({TEST_FRACTION:.0%}) before sampling. "
        f"On the training images, {MODEL_NAMES.get(lock['model'], lock['model'])} embeddings were selected by repeated "
        f"{CV_FOLDS}-fold cross-validation with folds split by image (cross-validated F-{BETA:g} {train.cv_fbeta:.2f}, "
        f"95% range {train.cv_fbeta_low:.2f}-{train.cv_fbeta_high:.2f}), giving a threshold of "
        f"{lock['threshold']:.3f}. On the held-out test images ({len(y)} pairs, {occ.image_group.nunique()} images), "
        f"this threshold achieved F-{BETA:g} {chosen.fbeta:.2f} (95% CI {chosen.fbeta_ci_low:.2f}-"
        f"{chosen.fbeta_ci_high:.2f}), precision {chosen.precision:.2f} ({chosen.precision_ci_low:.2f}-"
        f"{chosen.precision_ci_high:.2f}) and recall {chosen.recall:.2f} ({chosen.recall_ci_low:.2f}-"
        f"{chosen.recall_ci_high:.2f}); confidence intervals from {BOOTSTRAP_SAMPLES} bootstrap resamples of images."
    )
    with open(path("paper_text.md"), "w") as f:
        f.write(text + "\n")

    pd.set_option("display.width", 160)
    print(overall.drop(columns=["tp", "fp", "fn"]).round(3).to_string(index=False))
    print("\nPer pipeline:\n", per_pipeline.round(3).to_string(index=False))
    if unseen_res is not None:
        print(f"\nOnly label pairs unseen in train ({int(unseen.sum())} pairs):\n",
              unseen_res[["model", "fbeta", "precision", "recall"]].round(3).to_string(index=False))
    print("\nFor the paper:\n" + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sample", "train", "test", "sheet", "agreement"])
    ap.add_argument("--force", action="store_true", help="sample: replace an existing sample")
    a = ap.parse_args()
    {"sample": lambda: cmd_sample(a.force), "train": cmd_train, "test": cmd_test,
     "sheet": cmd_sheet, "agreement": cmd_agreement}[a.step]()


if __name__ == "__main__":
    main()
