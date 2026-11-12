
# paper_sketchobjectdetection

This project evaluates the performance of different techniques for **object detection in hand‑drawn images**.

The focus is on detecting **icon‑like objects**, with potential extension to text.

---

## Data

- Uses **synthetically generated compositions** of real hand‑drawn sketches  
- Source dataset: [TU Berlin Hand Sketch Image Dataset](https://www.kaggle.com/datasets/zara2099/tu-berlin-hand-sketch-image-dataset)
- Synthetic generation is used to enable controlled experiments

Generation parameters include:
- Number of types of icons
- Number of icons per type
- Amount of overlap
- Maximum scale-up factor of icons

These settings aim to mimic **rich pictures**.

Code:
- Run s01_synthesize_data.py. Set the appropriate parameters in the main function.

---

## Running order

Run the scripts in this order from the project folder (settings are at the top of each script):

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `s01_synthesize_data.py` | Synthetic rich pictures from TU Berlin sketches | `01_synthetic_data/` |
| 1b | `s01b_prepare_real_data.py` | Real rich pictures and their ground truth (instead of / next to step 1) | `01_real_data/` |
| 2 | `s02_detect_picture_elements.py` | Detect and label icons with each pipeline | `02_detected_picture_elements/` |
| 3 | `s03_threshold_calibration.py sample` | Sample prediction-ground-truth pairs for annotation, split images into train / test | `03_evaluation/threshold_calibration/` |
| 3b | `streamlit run s03b_annotation_app.py` | Annotate the pairs ("Calibration annotation"); `s03_threshold_calibration.py sheet` / `agreement` for extra coders in Excel | annotations |
| 3 | `s03_threshold_calibration.py train`, then `test` | Choose and lock the semantic-similarity threshold, then check it once on the held-out images | `threshold_lock.json` |
| 4 | `s04_evaluate_detections.py` | Precision / recall of every pipeline with the locked threshold | `03_evaluation/`, `03_evaluation_real/` |

Shared code is in `classes/`; the Streamlit pages are in `apps/`.

---

## Icon Detection Approaches

The following categories of detection methods are studied:

- Bottom‑up approaches  
 - Region up

- Middle‑level approaches
 - scene down approach

- Canvas‑down approaches  

Performance is analyzed across different synthetic generation settings.

---

## Goal

To compare how different detection techniques perform on hand‑drawn, icon‑based images under varying scene complexity.
