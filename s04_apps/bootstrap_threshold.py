"""
Bootstrap analysis of the semantic-similarity threshold chosen by F-beta.

Question: does the particular set of annotated pairs (500 pairs sampled from
03_evaluation/similarity_cache.xlsx and annotated in the Annotation app) bias the
threshold and the F-beta score reported for the Sentence-Transformer model?

Three checks, all on the annotated pairs:

1. Sampling variability (non-parametric bootstrap).
   Resample the annotated pairs with replacement B times, pick the F-beta optimal
   threshold on each resample exactly as the Prediction app does, and report
   percentile confidence intervals for the threshold, F-beta, precision and recall.
   Bias of the threshold estimate = mean(bootstrap thresholds) - original threshold.

2. Optimism of the reported F-beta.
   The threshold is chosen and scored on the same pairs, so the reported F-beta is
   optimistic. For each resample, the threshold chosen on the resample is also
   scored on the pairs NOT drawn into it (out-of-bag). Efron's optimism
   estimate = mean(in-bag F-beta - F-beta of the same threshold on the original
   pairs); corrected F-beta = original F-beta - optimism. The out-of-bag F-beta is
   reported as a second, more pessimistic estimate.

3. Sample size.
   Repeatedly draw 50, 100, ... pairs (with replacement, i.e. an m-out-of-n
   bootstrap, so every size is comparable; subsamples without replacement would
   understate the spread as they approach the full set) and report how much the
   chosen threshold spreads. If the spread is still shrinking noticeably at the
   full sample size, more annotations would help.

4. Repeated stratified cross-validation (the honest F-beta to report).
   Split the pairs into k folds with the same share of similar pairs, choose the
   threshold on k-1 folds, apply it to the held-out fold, and pool the held-out
   decisions into one precision / recall / F-beta per repeat. Repeating with
   different splits gives a mean and a 95% range. The final threshold is still
   the one chosen on all pairs; the CV score estimates how well that way of
   choosing a threshold works on pairs it has not seen.

Optionally (5) checks that the annotated pairs are representative of the
similarity cache they were drawn from (Kolmogorov-Smirnov test on similarity).

Run from the project root:
    python apps/bootstrap_threshold.py
    python apps/bootstrap_threshold.py --n-boot 5000 --beta 2 --reported-threshold 0.375
Outputs go to 03_evaluation/threshold_bootstrap/ (summary.xlsx, bootstrap_threshold.png).
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

DEFAULT_ANNOTATIONS = "03_evaluation/similarity_cache_sampled_annotated.pickle"
DEFAULT_CACHE = "03_evaluation/similarity_cache.xlsx"
DEFAULT_OUT = "03_evaluation/threshold_bootstrap"


# ---------------------------------------------------------------------------
# Core metrics (same definitions as apps/prediction.py)
# ---------------------------------------------------------------------------

def fbeta_optimum(y_true, scores, beta=2.0):
    """F-beta optimal threshold, picked from the precision-recall curve exactly as
    the Prediction app does. Returns dict(threshold, fbeta, precision, recall)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:  # no similar pairs in this (re)sample: undefined
        return dict(threshold=np.nan, fbeta=np.nan, precision=np.nan, recall=np.nan)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-10)
    i = int(np.argmax(f))
    t = thr[i] if i < len(thr) else 1.0
    return dict(threshold=float(t), fbeta=float(f[i]), precision=float(prec[i]), recall=float(rec[i]))


def scores_at(y_true, scores, threshold, beta=2.0):
    """Precision, recall and F-beta of the rule 'similar if score >= threshold'."""
    y_true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(scores, dtype=float) >= threshold
    tp = int(np.sum(pred & y_true))
    fp = int(np.sum(pred & ~y_true))
    fn = int(np.sum(~pred & y_true))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) else 0.0
    return dict(fbeta=f, precision=p, recall=r)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def bootstrap(y_true, scores, beta=2.0, n_boot=2000, stratified=False, seed=0,
              reported_threshold=None):
    """
    Non-parametric bootstrap of the threshold selection.

    stratified=True resamples similar and dissimilar pairs separately, keeping
    their proportion fixed (use it if you consider the class balance fixed by
    design rather than a property of the sample).

    Returns (original, boot_df, summary_df).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    original = fbeta_optimum(y, s, beta)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    rows = []
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        else:
            idx = rng.integers(0, n, n)
        opt = fbeta_optimum(y[idx], s[idx], beta)
        row = {"b": b, **{f"boot_{k}": v for k, v in opt.items()}}
        if not np.isnan(opt["threshold"]):
            # same threshold scored on the original pairs (for Efron's optimism)
            on_orig = scores_at(y, s, opt["threshold"], beta)
            row["orig_fbeta_at_boot_threshold"] = on_orig["fbeta"]
            # ... and on the pairs that were not drawn (out-of-bag)
            oob = np.setdiff1d(np.arange(n), idx)
            if len(oob) and y[oob].sum() > 0:
                o = scores_at(y[oob], s[oob], opt["threshold"], beta)
                row.update({f"oob_{k}": v for k, v in o.items()})
        if reported_threshold is not None:
            fixed = scores_at(y[idx], s[idx], reported_threshold, beta)
            row.update({f"fixed_{k}": v for k, v in fixed.items()})
        rows.append(row)
    boot = pd.DataFrame(rows)

    def ci(col):
        v = boot[col].dropna()
        return v.mean(), v.std(), v.quantile(0.025), v.quantile(0.975)

    summary = []
    for metric in ["threshold", "fbeta", "precision", "recall"]:
        mean, sd, lo, hi = ci(f"boot_{metric}")
        summary.append(dict(
            quantity=f"chosen {metric}", original=original[metric], boot_mean=mean,
            boot_sd=sd, ci_low=lo, ci_high=hi, bias=mean - original[metric],
        ))
    optimism = (boot["boot_fbeta"] - boot["orig_fbeta_at_boot_threshold"]).dropna().mean()
    summary.append(dict(quantity="F-beta optimism (Efron)", original=np.nan, boot_mean=optimism,
                        boot_sd=np.nan, ci_low=np.nan, ci_high=np.nan, bias=np.nan))
    summary.append(dict(quantity="optimism-corrected F-beta", original=original["fbeta"] - optimism,
                        boot_mean=np.nan, boot_sd=np.nan, ci_low=np.nan, ci_high=np.nan, bias=np.nan))
    if "oob_fbeta" in boot:
        mean, sd, lo, hi = ci("oob_fbeta")
        summary.append(dict(quantity="out-of-bag F-beta", original=np.nan, boot_mean=mean,
                            boot_sd=sd, ci_low=lo, ci_high=hi, bias=np.nan))
    if reported_threshold is not None:
        for metric in ["fbeta", "precision", "recall"]:
            mean, sd, lo, hi = ci(f"fixed_{metric}")
            summary.append(dict(quantity=f"{metric} at reported threshold {reported_threshold:g}",
                                original=scores_at(y, s, reported_threshold, beta)[metric],
                                boot_mean=mean, boot_sd=sd, ci_low=lo, ci_high=hi, bias=np.nan))
        inside = (summary[0]["ci_low"] <= reported_threshold <= summary[0]["ci_high"])
        summary.append(dict(quantity=f"reported threshold {reported_threshold:g} inside 95% CI",
                            original=float(inside), boot_mean=np.nan, boot_sd=np.nan,
                            ci_low=np.nan, ci_high=np.nan, bias=np.nan))
    return original, boot, pd.DataFrame(summary)


def sample_size_curve(y_true, scores, beta=2.0, sizes=None, n_rep=500, seed=1):
    """Spread of the chosen threshold for resamples of different sizes (m-out-of-n bootstrap)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    if sizes is None:
        sizes = [k for k in (50, 100, 150, 200, 300, 400) if k < n] + [n]
    rows = []
    for k in sizes:
        for _ in range(n_rep):
            idx = rng.integers(0, n, k)
            opt = fbeta_optimum(y[idx], s[idx], beta)
            rows.append(dict(n=k, threshold=opt["threshold"], fbeta=opt["fbeta"]))
    df = pd.DataFrame(rows)
    out = df.groupby("n").agg(
        threshold_median=("threshold", "median"),
        threshold_sd=("threshold", "std"),
        threshold_p025=("threshold", lambda v: v.quantile(0.025)),
        threshold_p975=("threshold", lambda v: v.quantile(0.975)),
        fbeta_median=("fbeta", "median"),
        fbeta_sd=("fbeta", "std"),
    ).reset_index()
    return out


def repeated_cv(y_true, scores, beta=2.0, n_splits=5, n_repeats=50, seed=0):
    """
    Repeated stratified k-fold cross-validation of the threshold selection.

    Returns (summary_df, repeats_df, fold_thresholds):
      repeats_df       one row per repeat: pooled held-out F-beta, precision, recall
      fold_thresholds  the threshold chosen in every training split
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_splits = min(n_splits, int(y.sum()), int((1 - y).sum()))  # each fold needs both classes
    if n_splits < 2:
        raise ValueError("Cross-validation needs at least 2 similar and 2 dissimilar pairs")
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    decisions = np.zeros((n_repeats, len(y)), dtype=bool)
    fold_thresholds = []
    for k, (train, test) in enumerate(cv.split(s.reshape(-1, 1), y)):
        t = fbeta_optimum(y[train], s[train], beta)["threshold"]
        fold_thresholds.append(t)
        decisions[k // n_splits, test] = s[test] >= t

    rows = []
    yb = y.astype(bool)
    for r in range(n_repeats):
        pred = decisions[r]
        tp, fp, fn = int(np.sum(pred & yb)), int(np.sum(pred & ~yb)), int(np.sum(~pred & yb))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = (1 + beta**2) * p * rc / (beta**2 * p + rc) if (p + rc) else 0.0
        rows.append(dict(repeat=r, fbeta=f, precision=p, recall=rc))
    repeats = pd.DataFrame(rows)
    fold_thresholds = np.asarray(fold_thresholds, dtype=float)

    summary = [
        dict(quantity=f"CV {metric} ({n_splits}-fold x {n_repeats})", mean=repeats[metric].mean(),
             ci_low=repeats[metric].quantile(0.025), ci_high=repeats[metric].quantile(0.975))
        for metric in ["fbeta", "precision", "recall"]
    ]
    summary.append(dict(quantity="CV fold thresholds (median, 25th-75th pct)", mean=np.nanmedian(fold_thresholds),
                        ci_low=np.nanpercentile(fold_thresholds, 25), ci_high=np.nanpercentile(fold_thresholds, 75)))
    return pd.DataFrame(summary), repeats, fold_thresholds


def representativeness(sample_scores, cache_scores):
    """Two-sample KS test: do the annotated pairs follow the cache's similarity distribution?"""
    from scipy.stats import ks_2samp
    res = ks_2samp(sample_scores, cache_scores)
    return dict(ks_statistic=float(res.statistic), p_value=float(res.pvalue),
                sample_median=float(np.median(sample_scores)), cache_median=float(np.median(cache_scores)),
                n_sample=len(sample_scores), n_cache=len(cache_scores))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(original, boot, size_df, beta, reported_threshold=None, out_path=None, cv_fbeta=None):
    """cv_fbeta: optional (mean, low, high) of the cross-validated F-beta, drawn in panel (b)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    c_main, c_ref, c_ci = "#2f6690", "#d1495b", "#9db4c0"

    # (a) bootstrap distribution of the chosen threshold
    t = boot["boot_threshold"].dropna()
    lo, hi = t.quantile(0.025), t.quantile(0.975)
    axs[0].hist(t, bins=40, color=c_main, alpha=0.85)
    axs[0].axvspan(lo, hi, color=c_ci, alpha=0.3, label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    axs[0].axvline(original["threshold"], color="black", lw=1.5, label=f"chosen {original['threshold']:.3f}")
    if reported_threshold is not None:
        axs[0].axvline(reported_threshold, color=c_ref, ls="--", lw=1.5, label=f"reported {reported_threshold:g}")
    axs[0].set(title="(a) F-β optimal threshold", xlabel="similarity threshold", ylabel="bootstrap samples")
    axs[0].legend(fontsize=8, frameon=False)

    # (b) F-beta: in-bag (as reported) vs out-of-bag
    f_in = boot["boot_fbeta"].dropna()
    axs[1].hist(f_in, bins=40, color=c_main, alpha=0.7, label="threshold chosen and scored on same pairs")
    if "oob_fbeta" in boot:
        axs[1].hist(boot["oob_fbeta"].dropna(), bins=40, color=c_ref, alpha=0.55,
                    label="same threshold, unseen (out-of-bag) pairs")
    axs[1].axvline(original["fbeta"], color="black", lw=1.5, label=f"reported F-{beta:g} {original['fbeta']:.3f}")
    if cv_fbeta is not None:
        mean, lo, hi = cv_fbeta
        axs[1].axvspan(lo, hi, color="#f4a259", alpha=0.25)
        axs[1].axvline(mean, color="#e07a00", lw=1.5, ls="-.",
                       label=f"cross-validated F-{beta:g} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    axs[1].set(title=f"(b) F-{beta:g} score", xlabel=f"F-{beta:g}", ylabel="bootstrap samples")
    axs[1].legend(fontsize=8, frameon=False)

    # (c) threshold spread vs number of annotated pairs
    axs[2].fill_between(size_df["n"], size_df["threshold_p025"], size_df["threshold_p975"],
                        color=c_ci, alpha=0.5, label="95% range")
    axs[2].plot(size_df["n"], size_df["threshold_median"], "o-", color=c_main, label="median")
    if reported_threshold is not None:
        axs[2].axhline(reported_threshold, color=c_ref, ls="--", lw=1.2, label=f"reported {reported_threshold:g}")
    axs[2].set(title="(c) Threshold vs. number of annotated pairs", xlabel="annotated pairs used",
               ylabel="chosen threshold")
    axs[2].legend(fontsize=8, frameon=False)

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Data loading / CLI
# ---------------------------------------------------------------------------

def load_annotations(path=DEFAULT_ANNOTATIONS, score_col="similarity"):
    if not os.path.exists(path) and os.path.exists(os.path.basename(path)):
        path = os.path.basename(path)  # also accept the pickle in the project root
    df = pd.read_pickle(path)
    if score_col not in df.columns:
        raise KeyError(f"Column {score_col!r} not in {path}; available: {list(df.columns)}")
    df = df.dropna(subset=[score_col, "annotation"])
    return df["annotation"].astype(bool).values, df[score_col].astype(float).values, df


def run(annotations=DEFAULT_ANNOTATIONS, cache=DEFAULT_CACHE, out_dir=DEFAULT_OUT, score_col="similarity",
        beta=2.0, n_boot=2000, stratified=False, reported_threshold=0.375, seed=0, n_rep=500,
        cv_folds=5, cv_repeats=50):
    y, s, df = load_annotations(annotations, score_col)
    original, boot, summary = bootstrap(y, s, beta, n_boot, stratified, seed, reported_threshold)
    size_df = sample_size_curve(y, s, beta, n_rep=n_rep, seed=seed + 1)
    cv_summary, cv_repeats_df, fold_thresholds = repeated_cv(y, s, beta, cv_folds, cv_repeats, seed)

    rep = None
    if cache and os.path.exists(cache):
        cache_df = pd.read_excel(cache)
        cache_df = cache_df[cache_df["similarity"] > 0]  # same filter as the Annotation app
        rep = representativeness(df["similarity"].values, cache_df["similarity"].values)

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "bootstrap_threshold.png")
    cv_row = cv_summary.iloc[0]
    plot(original, boot, size_df, beta, reported_threshold, fig_path,
         cv_fbeta=(cv_row["mean"], cv_row["ci_low"], cv_row["ci_high"]))
    with pd.ExcelWriter(os.path.join(out_dir, "summary.xlsx"), engine="openpyxl") as w:
        pd.DataFrame([dict(n_pairs=len(y), n_similar=int(y.sum()), prevalence=float(y.mean()), beta=beta,
                           n_boot=n_boot, stratified=stratified, score_col=score_col,
                           cv_folds=cv_folds, cv_repeats=cv_repeats)]).to_excel(w, sheet_name="setup", index=False)
        summary.to_excel(w, sheet_name="bootstrap summary", index=False)
        cv_summary.to_excel(w, sheet_name="cross-validation", index=False)
        cv_repeats_df.to_excel(w, sheet_name="cv repeats", index=False)
        size_df.to_excel(w, sheet_name="sample size", index=False)
        if rep:
            pd.DataFrame([rep]).to_excel(w, sheet_name="representativeness", index=False)
        boot.to_excel(w, sheet_name="bootstrap samples", index=False)
    return dict(original=original, summary=summary, size_df=size_df, cv_summary=cv_summary,
                representativeness=rep, figure=fig_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="similarity cache the pairs were sampled from ('' to skip)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--score-col", default="similarity",
                    help="similarity column; 'similarity' is the Sentence-Transformer score cached by s04")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--stratified", action="store_true", help="keep the similar/dissimilar proportion fixed")
    ap.add_argument("--reported-threshold", type=float, default=0.375)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=50)
    a = ap.parse_args()
    res = run(a.annotations, a.cache, a.out_dir, a.score_col, a.beta, a.n_boot, a.stratified,
              a.reported_threshold, a.seed, cv_folds=a.cv_folds, cv_repeats=a.cv_repeats)
    pd.set_option("display.width", 160)
    print("Original:", {k: round(v, 4) for k, v in res["original"].items()})
    print(res["summary"].round(4).to_string(index=False))
    print("\nRepeated cross-validation:\n", res["cv_summary"].round(4).to_string(index=False))
    print("\nThreshold vs sample size:\n", res["size_df"].round(4).to_string(index=False))
    if res["representativeness"]:
        print("\nRepresentativeness vs cache:", res["representativeness"])
    print(f"\nSaved to {a.out_dir}/")


if __name__ == "__main__":
    main()
