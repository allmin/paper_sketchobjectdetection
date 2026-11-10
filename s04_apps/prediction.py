import streamlit as st
import pandas as pd
import os
import numpy as np
import torch
import open_clip
from transformers import CLIPTokenizer, CLIPModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
sentf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)

def clip_text_similarity(t1, t2):
    tokens = tokenizer([t1, t2])
    tokens = tokens.to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)

    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
    sim = (emb[0] @ emb[1]).item()              # cosine similarity

    return sim

def sentf_text_similarity(t1, t2):
    emb1 = sentf_model.encode(t1, convert_to_tensor=True)
    emb2 = sentf_model.encode(t2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def load_data():
    path = '03_evaluation/similarity_cache_sampled_annotated.pickle'

    if os.path.exists(path):
        df_sampled = pd.read_pickle(path)
        df_sampled["clip_similarity"] = df_sampled.apply(lambda row: clip_text_similarity(row["text1"], row["text2"]), axis=1)
        df_sampled["sentf_similarity"] = df_sampled.apply(lambda row: sentf_text_similarity(row["text1"], row["text2"]), axis=1)
    else:
        st.write('Unable to find 03_evaluation/similarity_cache_sampled_annotated.pickle. Please run the Annotation module first')

    return df_sampled


import altair as alt

def show():
    st.title("App 2: Precision & Recall Dashboard")

    if 'df' not in st.session_state:
        st.session_state.df = load_data()

    df = st.session_state.df

    beta = st.slider(
        "F-beta weight (β)",
        min_value=0.25, max_value=4.0, value=2.0, step=0.25,
        help="β > 1 favors recall, β < 1 favors precision, β = 1 is F1"
    )

    thresholds = np.round(np.arange(0, 1.01, 0.01), 2)

    accuracy = {}
    precision = {}
    recall = {}
    auc_scores = {}

    y_true = df.annotation.astype(int)

    for similarity in ['sentf', 'clip']:
        df['similarity'] = df[f"{similarity}_similarity"]

        similar_similarity = df[df.annotation == True].similarity
        dissimilar_similarity = df[df.annotation == False].similarity
        percentile_similar = [similar_similarity.quantile(0.25), similar_similarity.quantile(0.75)]
        percentile_dissimilar = [dissimilar_similarity.quantile(0.25), dissimilar_similarity.quantile(0.75)]

        for t in thresholds:
            tp = len(df[(df.annotation == True) & (df.similarity >= t)])
            fp = len(df[(df.annotation == False) & (df.similarity >= t)])
            tn = len(df[(df.annotation == False) & (df.similarity < t)])
            fn = len(df[(df.annotation == True) & (df.similarity < t)])

            accuracy[t] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            precision[t] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[t] = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics_df = pd.DataFrame({
            "threshold": thresholds,
            "accuracy": [accuracy[t] for t in thresholds],
            "precision": [precision[t] for t in thresholds],
            "recall": [recall[t] for t in thresholds],
        })

        st.header(f"Method: {similarity}")

        # ---- Threshold-based metrics chart with labeled axes ----
        st.subheader("📈 Precision / Recall / Accuracy vs Threshold")

        metrics_long = metrics_df.melt(
            id_vars="threshold", var_name="metric", value_name="value"
        )
        threshold_chart = (
            alt.Chart(metrics_long)
            .mark_line()
            .encode(
                x=alt.X("threshold", title="Threshold"),
                y=alt.Y("value", title="Score"),
                color=alt.Color("metric", title="Metric"),
            )
            .properties(height=350)
        )
        st.altair_chart(threshold_chart, use_container_width=True)

        # ---- Precision-Recall curve + AUC ----
        prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, df.similarity)
        pr_auc = auc(rec_curve, prec_curve)
        ap_score = average_precision_score(y_true, df.similarity)
        auc_scores[similarity] = pr_auc

        st.subheader(f"📉 Precision-Recall Curve (AUC = {pr_auc:.3f})")

        pr_df = pd.DataFrame({"recall": rec_curve, "precision": prec_curve}).sort_values("recall")
        pr_chart = (
            alt.Chart(pr_df)
            .mark_line()
            .encode(
                x=alt.X("recall", title="Recall"),
                y=alt.Y("precision", title="Precision"),
            )
            .properties(height=350)
        )
        st.altair_chart(pr_chart, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric(label=f"{similarity} PR-AUC", value=f"{pr_auc:.3f}")
        col2.metric(label=f"{similarity} Average Precision", value=f"{ap_score:.3f}")

        # ---- Optimal threshold via F-beta ----
        fbeta_scores = (1 + beta**2) * (prec_curve * rec_curve) / (beta**2 * prec_curve + rec_curve + 1e-10)
        best_idx = np.argmax(fbeta_scores)
        best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 1.0
        best_fbeta = fbeta_scores[best_idx]
        best_precision = prec_curve[best_idx]
        best_recall = rec_curve[best_idx]

        st.subheader(f"🎯 Optimal Threshold (max F-{beta:g})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Threshold", f"{best_threshold:.3f}")
        c2.metric(f"F-{beta:g}", f"{best_fbeta:.3f}")
        c3.metric("Precision", f"{best_precision:.3f}")
        c4.metric("Recall", f"{best_recall:.3f}")

        # Dashboard-style metric cards
        st.subheader("🔍 Summary at Key Thresholds")
        key_thresholds = percentile_dissimilar + percentile_similar
        key_thresholds = [round(float(i), 2) for i in key_thresholds]
        cols = st.columns(len(key_thresholds))

        for col, t in zip(cols, key_thresholds):
            col.metric(
                label=f"Threshold {t:0.02f}",
                value=f"Prec {precision[t]:0.02f}",
                delta=f"Rec {recall[t]:0.02f}"
            )

        with st.expander("Show raw metric table"):
            st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

        with st.expander("Show similar table"):
            st.dataframe(df[df.annotation == True], use_container_width=True)

        st.divider()

    # ---- Final comparison of the two methods ----
    st.header("🏆 Comparison Summary")
    comparison_df = pd.DataFrame({
        "method": list(auc_scores.keys()),
        "PR-AUC": list(auc_scores.values()),
    })
    st.dataframe(comparison_df.style.format({"PR-AUC": "{:.3f}"}), use_container_width=True)

    st.divider()
    show_bootstrap(df, beta)


def show_bootstrap(df, beta):
    """Does the choice of annotated pairs bias the threshold / F-beta? (see bootstrap_threshold.py)"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bootstrap_threshold import bootstrap, sample_size_curve, plot, repeated_cv

    st.header("🎲 Bootstrap: is the threshold robust to the annotated sample?")
    st.write(
        "Resamples the annotated pairs with replacement, re-selects the F-β optimal threshold on each "
        "resample, and reports 95% intervals. *Optimism* is how much the F-β is inflated because the "
        "threshold is chosen and scored on the same pairs; the out-of-bag F-β scores each threshold on "
        "pairs it was not chosen on. *Cross-validation* chooses the threshold on part of the pairs and "
        "scores it on the rest, repeated over many splits: this is the F-β to report next to the threshold."
    )
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Similarity", ["sentf", "clip"], index=0)
    n_boot = c2.number_input("Bootstrap samples", 200, 20000, 2000, step=200)
    reported = c3.number_input("Reported threshold", 0.0, 1.0, 0.375, step=0.005, format="%.3f")
    stratified = c4.checkbox("Keep similar/dissimilar ratio fixed", value=False)
    c5, c6, _, _ = st.columns(4)
    cv_folds = c5.number_input("CV folds", 2, 20, 5, step=1)
    cv_repeats = c6.number_input("CV repeats", 5, 500, 50, step=5)

    if not st.button("Run bootstrap"):
        return
    y = df.annotation.astype(bool).values
    s = df[f"{method}_similarity"].astype(float).values
    with st.spinner("Bootstrapping and cross-validating..."):
        original, boot, summary = bootstrap(y, s, beta, int(n_boot), stratified, 0, reported)
        size_df = sample_size_curve(y, s, beta)
        cv_summary, cv_rep, _ = repeated_cv(y, s, beta, int(cv_folds), int(cv_repeats))
    cv_f = cv_summary.iloc[0]
    st.pyplot(plot(original, boot, size_df, beta, reported, cv_fbeta=(cv_f["mean"], cv_f.ci_low, cv_f.ci_high)))

    thr = summary.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Chosen threshold", f"{original['threshold']:.3f}",
              help=f"95% CI {thr.ci_low:.3f} - {thr.ci_high:.3f}")
    k1.caption(f"95% CI [{thr.ci_low:.3f}, {thr.ci_high:.3f}], bias {thr.bias:+.3f}")
    corrected = summary.loc[summary.quantity == "optimism-corrected F-beta", "original"].iloc[0]
    optimism = summary.loc[summary.quantity == "F-beta optimism (Efron)", "boot_mean"].iloc[0]
    k2.metric(f"F-{beta:g} (reported)", f"{original['fbeta']:.3f}")
    k2.caption(f"optimism {optimism:.3f} → corrected {corrected:.3f}")
    inside = reported and thr.ci_low <= reported <= thr.ci_high
    k3.metric(f"Reported {reported:.3f} in 95% CI?", "yes" if inside else "no")

    st.subheader("Cross-validated (threshold chosen and scored on different pairs)")
    m1, m2, m3 = st.columns(3)
    for col, (_, r), name in zip((m1, m2, m3), cv_summary.iloc[:3].iterrows(), (f"F-{beta:g}", "Precision", "Recall")):
        col.metric(f"CV {name}", f"{r['mean']:.3f}")
        col.caption(f"95% range [{r.ci_low:.3f}, {r.ci_high:.3f}]")
    st.dataframe(cv_summary.style.format(precision=3, na_rep=""), use_container_width=True)

    st.subheader("Bootstrap")
    st.dataframe(summary.style.format(precision=3, na_rep=""), use_container_width=True)
    with st.expander("Threshold vs. number of annotated pairs"):
        st.dataframe(size_df.style.format(precision=3), use_container_width=True)


if __name__ == "__main__":
    show()