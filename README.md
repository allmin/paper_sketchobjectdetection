
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
- Run 01_datasynthesis.py. Set the appropriate parameters in the main function.

---

## Icon Detection Approaches

The following categories of detection methods are studied:

- Bottom‑up approaches  

- Middle‑level approaches  
- Top‑down approaches  

Performance is analyzed across different synthetic generation settings.

---

## Goal

To compare how different detection techniques perform on hand‑drawn, icon‑based images under varying scene complexity.
