"""
Figures and reports for the evaluation (used by s04 / classes.evaluation):
  draw_annotated_image()          GT / TP / FP / FN boxes drawn on an image
  plot_iou_semantic_mapping()     IoU vs. semantic similarity per prediction / GT
  threshold_sensitivity_report()  precision / recall / ranks across thresholds
"""

import os

import numpy as np
import pandas as pd

BOX_COLORS = {
    "GT": (255, 255, 0),   # yellow
    "TP": (0, 255, 0),     # green
    "FP": (255, 0, 0),     # red
    "FN": (255, 165, 0),   # orange
}


def draw_annotated_image(image, gt_df, tp_df, fp_df, fn_df, save_path):
    """Save a copy of the image with the ground truth (yellow), true positives
    (green), false positives (red) and false negatives (orange) drawn on it.
    TP/FP use the prediction's rescaled box; their labels are drawn slightly
    lower so they don't cover the ground-truth label."""
    from PIL import ImageDraw

    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    for tag, df in (("GT", gt_df), ("TP", tp_df), ("FP", fp_df), ("FN", fn_df)):
        color = BOX_COLORS[tag]
        offset = 10 if tag in ("TP", "FP") else 0
        for _, row in df.iterrows():
            box = row["reconst_bounding_box"] if "reconst_bounding_box" in row else row["bounding_box"]
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 + offset), f"{tag}: {row['represents']}", fill=color)
    img.save(save_path)


def plot_iou_semantic_mapping(map_df, semantic_threshold, iou_threshold=None, title="", out_path=None):
    """
    Scatter of IoU (x) against semantic similarity (y) for the rows of
    Evaluator.iou_semantic_mapping(), coloured by TP / FP / FN, with the two
    matching thresholds as dashed lines and marginal histograms.
    The four regions of the plot read as:
      top-right      label and box both good enough (TP; FP/FN here lost the assignment)
      top-left       right label, box misplaced
      bottom-right   box on the object, wrong label
      bottom-left    neither
    iou_threshold=None (centre-distance metrics) draws only the semantic line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "TP": dict(color="#2a9d8f", marker="o", label="true positive (matched pair)"),
        "FP": dict(color="#e76f51", marker="x", label="false positive → closest GT"),
        "FN": dict(color="#6c757d", marker="^", label="false negative → closest prediction"),
    }
    fig = plt.figure(figsize=(7.5, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(grid[1, 0])
    ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    bins = np.linspace(0, 1, 26)

    for role, st in styles.items():
        d = map_df[map_df.role == role] if len(map_df) else map_df
        n = len(d)
        if n == 0:
            continue
        ax.scatter(d.iou, d.semantic_similarity, s=22, alpha=0.6, color=st["color"], marker=st["marker"],
                   label=f"{st['label']} (n={n})", linewidths=1)
        ax_top.hist(d.iou, bins=bins, color=st["color"], alpha=0.5)
        ax_right.hist(d.semantic_similarity, bins=bins, color=st["color"], alpha=0.5, orientation="horizontal")

    ax.axhline(semantic_threshold, color="black", ls="--", lw=1)
    ax.text(1.0, semantic_threshold, f" similarity {semantic_threshold:g}", va="bottom", ha="right", fontsize=8)
    y_min = min(-0.02, map_df.semantic_similarity.min() - 0.02) if len(map_df) else -0.02
    if iou_threshold is not None:
        ax.axvline(iou_threshold, color="black", ls="--", lw=1)
        ax.text(iou_threshold, y_min, f" IoU {iou_threshold:g}", va="bottom", ha="left", fontsize=8)
        # TP / FP / FN counts in each of the four regions, in its top corner
        if len(map_df):
            for x_ok in (True, False):
                for y_ok in (True, False):
                    sel = map_df[((map_df.iou >= iou_threshold) == x_ok) &
                                 ((map_df.semantic_similarity >= semantic_threshold) == y_ok)]
                    counts = " · ".join(f"{r} {int((sel.role == r).sum())}" for r in styles)
                    ax.text(0.99 if x_ok else 0.01, 1.0 if y_ok else semantic_threshold - 0.015, counts,
                            ha="right" if x_ok else "left", va="top", fontsize=7.5, color="#333333",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(y_min, 1.02)
    ax.set_xlabel("IoU of bounding boxes")
    ax.set_ylabel("semantic similarity of labels")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.22), fontsize=8, frameon=False, ncol=1)
    ax_top.axis("off")
    ax_right.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    for a in (ax,):
        a.spines[["top", "right"]].set_visible(False)
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def threshold_sensitivity_report(super_summary, out_dir, pipeline_labels, metrics, reference, interval=None):
    """
    Turn a super summary (one row per pipeline|metric|threshold|matching[|folder])
    into a sensitivity report: precision / recall per pipeline at each semantic
    threshold, the pipeline ranks at each threshold, and how well each threshold's
    ranking agrees with the one at `reference` (Kendall's tau, same best pipeline).
    Writes sensitivity.xlsx and one figure per data folder to out_dir.

    pipeline_labels: {pipeline folder: name shown in the report}
    metrics:         {"precision": summary row name, "recall": summary row name}
    interval:        optional (low, high) threshold band to shade
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for _, r in super_summary.iterrows():
        parts = str(r["pipeline"]).split("|")
        pipe = parts[0].rsplit("_", 1)[0]  # strip "_<element type>"
        row = dict(pipeline=pipe, label=pipeline_labels.get(pipe, pipe), proximity=parts[1],
                   threshold=float(parts[2]), matching=parts[3] if len(parts) > 3 else "",
                   data=parts[4] if len(parts) > 4 else "")
        for name, col in metrics.items():
            v = pd.to_numeric(r.get(col), errors="coerce")
            row[name] = np.nan if pd.isna(v) or v < 0 else float(v)  # -1 = undefined
        p, rc = row.get("precision"), row.get("recall")
        row["f1"] = 2 * p * rc / (p + rc) if p is not None and rc is not None and (p + rc) > 0 else np.nan
        rows.append(row)
    long = pd.DataFrame(rows)
    if long.empty:
        print("Threshold sensitivity: nothing to report")
        return None

    # ranks per (data folder, proximity, matching, threshold): 1 = best
    keys = ["data", "proximity", "matching", "threshold"]
    for m in ["precision", "recall", "f1"]:
        long[f"rank_{m}"] = long.groupby(keys)[m].rank(ascending=False, method="min")

    agreement = []
    for (data, prox, match), g in long.groupby(["data", "proximity", "matching"]):
        ref = g[np.isclose(g.threshold, reference)].set_index("label")
        for t, gt in g.groupby("threshold"):
            gt = gt.set_index("label")
            common = ref.index.intersection(gt.index)
            row = dict(data=data, proximity=prox, matching=match, threshold=t, n_pipelines=len(common))
            for m in ["precision", "recall", "f1"]:
                a, b = ref.loc[common, m], gt.loc[common, m]
                ok = a.notna() & b.notna()
                tau = kendalltau(a[ok], b[ok])[0] if ok.sum() >= 2 else np.nan
                row[f"kendall_tau_{m}"] = tau
                row[f"best_{m}"] = gt[m].idxmax() if gt[m].notna().any() else ""
                row[f"same_best_{m}_as_{reference:g}"] = (
                    bool(ref[m].notna().any() and gt[m].notna().any() and gt[m].idxmax() == ref[m].idxmax())
                )
            agreement.append(row)
    agreement = pd.DataFrame(agreement)

    with pd.ExcelWriter(os.path.join(out_dir, "sensitivity.xlsx"), engine="openpyxl") as w:
        for m in ["precision", "recall", "f1"]:
            long.pivot_table(index=["data", "label"], columns="threshold", values=m).to_excel(w, sheet_name=m)
            long.pivot_table(index=["data", "label"], columns="threshold", values=f"rank_{m}").to_excel(
                w, sheet_name=f"rank {m}")
        agreement.to_excel(w, sheet_name=f"agreement with {reference:g}", index=False)
        long.to_excel(w, sheet_name="all", index=False)

    # one figure per data folder: precision and recall vs threshold, one line per pipeline
    order = [l for l in pipeline_labels.values() if l in set(long.label)] + \
            sorted(set(long.label) - set(pipeline_labels.values()))
    colors = dict(zip(order, plt.get_cmap("tab10").colors))
    fig_paths = []
    for data, g in long.groupby("data"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, m in zip(axs, ["precision", "recall"]):
            for label in order:
                d = g[g.label == label].sort_values("threshold")
                if len(d):
                    ax.plot(d.threshold, d[m], "o-", color=colors[label], label=label, lw=1.6, ms=4)
            if interval:
                ax.axvspan(*interval, color="#9db4c0", alpha=0.18, lw=0)
            ax.axvline(reference, color="black", ls="--", lw=1)
            ax.set(xlabel="semantic similarity threshold", ylabel=m, title=m.capitalize(), ylim=(0, 1.02))
            ax.spines[["top", "right"]].set_visible(False)
        axs[1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Threshold sensitivity{(' | ' + data) if data else ''} "
                     f"(dashed: reported {reference:g}" + (", band: bootstrap 95% CI)" if interval else ")"),
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"threshold_sensitivity{('_' + data) if data else ''}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(path)
    print(f"Saved threshold sensitivity report to {out_dir}")
    return long, agreement, fig_paths
