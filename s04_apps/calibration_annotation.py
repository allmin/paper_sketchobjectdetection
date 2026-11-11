"""
Blind annotation of label pairs for threshold calibration (see s05_threshold_calibration.py).

Shows only the two labels: no similarity scores, no pipeline names, no train/test
side, and the second annotator never sees the first one's answers.
  role "primary": annotates every label pair
  role "second":  annotates only the agreement subset (for Cohen's kappa)
Answers are saved after every click to annotations_<role>.csv in the folder.
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

DEFAULT_FOLDER = "03_evaluation/threshold_calibration"

# Agree on this before annotating starts, and give both annotators the same text.
GUIDELINE = """
Decide whether the **detected label** names the same thing as the **ground-truth label**,
so that a coder would accept it as the granular code for that icon.

- **Similar:** synonyms and spelling variants (*sofa* / *couch*); the same object described in
  more or less detail (*red ball* / *ball*); a part or a closely related type with a similar
  appearance (*face* / *person*, *kitten* / *cat*).
- **Not similar:** a different object (*cat* / *dog*); a label so broad it would not count as the
  code (*thing*, *object*, *animal* for *cat*); a description of something else in the picture.
- Judge only the two labels. Don't try to guess which model produced them.
"""


def load_pairs(folder, role):
    pairs = pd.read_csv(os.path.join(folder, "label_pairs.csv"))
    if role == "second":
        pairs = pairs[pairs.agreement_subset].reset_index(drop=True)
    return pairs


def load_answers(folder, role):
    path = os.path.join(folder, f"annotations_{role}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {r.pair_id: dict(similar=bool(r.similar), annotator=r.annotator, annotated_at=r.annotated_at)
            for r in df.itertuples()}


def save_answers(folder, role, answers):
    rows = [dict(pair_id=pid, **a) for pid, a in answers.items()]
    pd.DataFrame(rows, columns=["pair_id", "similar", "annotator", "annotated_at"]).to_csv(
        os.path.join(folder, f"annotations_{role}.csv"), index=False)


def first_open(pairs, answers, start=0):
    """Index of the next unannotated pair from `start` (wrapping), or None."""
    n = len(pairs)
    for k in range(n):
        i = (start + k) % n
        if pairs.pair_id[i] not in answers:
            return i
    return None


def show():
    st.title("Calibration annotation")
    c1, c2, c3 = st.columns([2, 1, 1])
    folder = c1.text_input("Folder", DEFAULT_FOLDER)
    role = c2.selectbox("Role", ["primary", "second"],
                        help="primary: all pairs. second: a different person, only the agreement subset.")
    annotator = c3.text_input("Your name or initials")

    if not os.path.exists(os.path.join(folder, "label_pairs.csv")):
        st.warning(f"No label_pairs.csv in {folder}. Run `python s05_threshold_calibration.py sample` first.")
        return
    if not annotator.strip():
        st.info("Enter your name or initials to start.")
        return

    key = (folder, role)
    if st.session_state.get("cal_key") != key:
        st.session_state.cal_key = key
        st.session_state.cal_pairs = load_pairs(folder, role)
        st.session_state.cal_answers = load_answers(folder, role)
        st.session_state.cal_index = first_open(st.session_state.cal_pairs, st.session_state.cal_answers) or 0
    pairs, answers = st.session_state.cal_pairs, st.session_state.cal_answers

    done = sum(pid in answers for pid in pairs.pair_id)
    st.progress(done / len(pairs) if len(pairs) else 1.0, text=f"{done} of {len(pairs)} annotated")
    with st.expander("Guideline", expanded=done == 0):
        st.markdown(GUIDELINE)

    if done == len(pairs):
        st.success("All pairs are annotated. You can still go back and change answers.")
    i = st.session_state.cal_index % len(pairs)
    row = pairs.iloc[i]

    st.subheader(f"Pair {i + 1} of {len(pairs)}")
    a, b = st.columns(2)
    a.caption("Detected label")
    a.info(row.pred_label)
    b.caption("Ground-truth label")
    b.info(row.gt_label)
    if row.pair_id in answers:
        st.caption(f"Your answer: **{'similar' if answers[row.pair_id]['similar'] else 'not similar'}** "
                   f"(click again to change)")

    def answer(similar):
        answers[row.pair_id] = dict(similar=similar, annotator=annotator.strip(),
                                    annotated_at=datetime.now().isoformat(timespec="seconds"))
        save_answers(folder, role, answers)
        nxt = first_open(pairs, answers, i + 1)
        st.session_state.cal_index = nxt if nxt is not None else (i + 1) % len(pairs)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("✅ Similar", use_container_width=True):
        answer(True)
        st.rerun()
    if b2.button("❌ Not similar", use_container_width=True):
        answer(False)
        st.rerun()
    if b3.button("⬅️ Back", use_container_width=True):
        st.session_state.cal_index = (i - 1) % len(pairs)
        st.rerun()
    if b4.button("Skip ➡️", use_container_width=True):
        st.session_state.cal_index = (i + 1) % len(pairs)
        st.rerun()


show()
