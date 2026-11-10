"""
Optional COCO-style detection metrics (mAP, mAR, ...) via torchmetrics.

Only used when COMPUTE_DETECTION_METRICS is True in s03. Free-text labels are
first mapped to ground-truth class ids by semantic similarity; the similarity is
used as the detection score.
"""

import pandas as pd
import torch

IOU_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


class SemanticLabelMapper:
    """Maps a free-text label to the closest ground-truth class by embedding similarity."""

    def __init__(self, class_names, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.55):
        from sentence_transformers import SentenceTransformer

        self.class_names = list(class_names)
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.class_emb = self.model.encode(self.class_names, convert_to_tensor=True)

    def map(self, label):
        """(class id, similarity); (-1, 0.0) for a missing label or when the best
        similarity is below the threshold."""
        if not isinstance(label, str) or not label.strip():
            return (-1, 0.0)
        emb = self.model.encode(label, convert_to_tensor=True)
        sim = torch.nn.functional.cosine_similarity(emb, self.class_emb)
        best = torch.argmax(sim).item()
        score = sim[best].item()
        return (-1, 0.0) if score < self.threshold else (best, score)

    def map_batch(self, labels):
        return [self.map(label) for label in labels]


def _boxes(boxes):
    # shape (N, 4) also when N == 0 (an image whose predictions were all dropped)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def detection_metrics(gt_df, pred_df, semantic_threshold, model_name):
    """mAP / mAR etc. over all images (rows grouped by 'uid'), as a (metric, value) frame."""
    from torchmetrics.detection import MeanAveragePrecision

    mapper = SemanticLabelMapper(gt_df.represents.unique(), model_name=model_name, threshold=semantic_threshold)
    preds, targets = [], []
    for uid in gt_df.uid.unique():
        gt_rows = gt_df[gt_df.uid == uid]
        pred_rows = pred_df[pred_df.uid == uid]
        gt_ids = [cid for cid, _ in mapper.map_batch(gt_rows['represents'])]
        # keep only predictions whose label maps to a ground-truth class
        kept = [(cid, score, box) for (cid, score), box in
                zip(mapper.map_batch(pred_rows['represents']), pred_rows['reconst_bounding_box'].tolist())
                if cid is not None and cid >= 0]
        preds.append(dict(boxes=_boxes([box for _, _, box in kept]),
                          scores=torch.tensor([s for _, s, _ in kept], dtype=torch.float32),
                          labels=torch.tensor([c for c, _, _ in kept], dtype=torch.int64)))
        targets.append(dict(boxes=_boxes(gt_rows['bounding_box'].tolist()),
                            labels=torch.tensor(gt_ids, dtype=torch.int64)))
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", iou_thresholds=IOU_THRESHOLDS)
    metric.update(preds, targets)
    result = metric.compute()
    return pd.DataFrame({"metric": list(result.keys()),
                         "value": [t.item() if t.ndim == 0 else t.tolist() for t in result.values()]})
