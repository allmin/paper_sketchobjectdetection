import os
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
import open_clip

# =============================================================================
# Configuration
# =============================================================================

IMAGE_DIR = (
    "01_synthetic_data/"
    "05_clusters_08_classes_01_to_10_images_per_class_20_minimum_icons_"
    "00_to_00_percent_overlap_00_minimum_overlap_images_"
    "3.0_max_scaling_factor"
)

CANDIDATE_LABELS = [
    "car",
    "house",
    "tree",
    "person",
    "cat",
    "dog",
    "money",
    "telephone",
    "chair",
    "table",
    "hospital",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Scene Proposal
# =============================================================================


def sp_preprocess(img_np, blur_k=5, block_size=11, C=2):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )
    return thresh


def sp_get_components(binary, min_comp_area=10):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area >= min_comp_area:
            components.append(
                {"bbox": (x, y, w, h), "centroid": (cx, cy), "area": area}
            )
    return components


def sp_cluster_components(components, eps=50, min_samples=3):
    points = np.array([c["centroid"] for c in components])
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    clusters = {}
    for i, label in enumerate(labels):
        uid = f"noise_{i}" if label == -1 else label
        clusters.setdefault(uid, []).append(components[i])

    return list(clusters.values())


# =============================================================================
# Region Proposal (Edge / Contour Based)
# =============================================================================


def run_custom_contour(img_np, min_area=500, edge_min=50, edge_max=150):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_min, edge_max)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            boxes.append((x, y, x + w, y + h))

    return boxes


# =============================================================================
# CLIP Classifier
# =============================================================================


class CLIPClassifier:
    def __init__(self, device):
        self.device = device
        self.clip_models = {}

    def _load_clip(self, model_name="ViT-B-32"):
        if model_name not in self.clip_models:
            print(f"Loading CLIP {model_name}...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer(model_name)

            model.eval().to(self.device)

            self.clip_models[model_name] = {
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
            }

        return self.clip_models[model_name]

    def run_clip(self, image_crop, candidate_labels, model_name="ViT-B-32"):
        clip = self._load_clip(model_name)
        model = clip["model"]
        preprocess = clip["preprocess"]
        tokenizer = clip["tokenizer"]

        img = preprocess(image_crop).unsqueeze(0).to(self.device)
        text = tokenizer(candidate_labels).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            probs = (image_features @ text_features.T).softmax(dim=-1)

        idx = probs.argmax()
        confidence = float(probs[0, idx])
        return candidate_labels[idx], confidence


# =============================================================================
# Image Processing Pipeline
# =============================================================================


def process_image(image_path, classifier):
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    draw = ImageDraw.Draw(img_pil)

    detections = []

    # --- Scene Proposal ---
    binary = sp_preprocess(img_np)
    components = sp_get_components(binary)
    scenes = sp_cluster_components(components)

    for scene_idx, scene in enumerate(scenes, start=1):
        xs = [c["bbox"][0] for c in scene]
        ys = [c["bbox"][1] for c in scene]
        xe = [c["bbox"][0] + c["bbox"][2] for c in scene]
        ye = [c["bbox"][1] + c["bbox"][3] for c in scene]

        sx1, sy1, sx2, sy2 = min(xs), min(ys), max(xe), max(ye)

        # --- Draw Scene (YELLOW) ---
        draw.rectangle(
            (int(sx1), int(sy1), int(sx2), int(sy2)),
            outline="yellow",
            width=3,
        )
        draw.text(
            (int(sx1), max(0, int(sy1) - 14)),
            f"scene{scene_idx}",
            fill="yellow",
        )

        # --- Add Scene to JSON ---
        detections.append(
            {
                "element_type": "scene",
                "class": f"scene{scene_idx}",
                "bounding_box": [
                    int(sx1),
                    int(sy1),
                    int(sx2),
                    int(sy2),
                ],
            }
        )

        # --- Region Proposals Inside Scene ---
        scene_crop = img_np[sy1:sy2, sx1:sx2]
        if scene_crop.size == 0:
            continue

        region_boxes = run_custom_contour(scene_crop)

        for x1, y1, x2, y2 in region_boxes:
            gx1, gy1 = sx1 + x1, sy1 + y1
            gx2, gy2 = sx1 + x2, sy1 + y2

            crop = img_pil.crop((gx1, gy1, gx2, gy2))
            label, conf = classifier.run_clip(crop, CANDIDATE_LABELS)

            # --- Draw Object (RED) ---
            draw.rectangle(
                (int(gx1), int(gy1), int(gx2), int(gy2)),
                outline="red",
                width=2,
            )
            draw.text(
                (int(gx1), max(0, int(gy1) - 12)),
                f"{label} {conf:.2f}",
                fill="red",
            )

            # --- Add Object to JSON ---
            detections.append(
                {
                    "element_type": "icon",
                    "class": str(label),
                    "bounding_box": [
                        int(gx1),
                        int(gy1),
                        int(gx2),
                        int(gy2),
                    ],
                    "confidence": float(conf),
                }
            )

    return img_pil, detections


# =============================================================================
# Main
# =============================================================================


def main():
    global IMAGE_DIR, DEVICE
    classifier = CLIPClassifier(DEVICE)

    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        if "_detected." in fname or "data_summary" in fname:
            continue

        image_path = os.path.join(IMAGE_DIR, fname)
        print(f"Processing {image_path}")

        detected_img, detections = process_image(image_path, classifier)

        base, ext = os.path.splitext(fname)
        IMAGE_DIR_OUT = IMAGE_DIR.replace(
            "01_synthetic_data", "02_detected_picture_elements"
        )
        os.makedirs(IMAGE_DIR_OUT, exist_ok=True)
        out_img = os.path.join(IMAGE_DIR_OUT, f"{base}_detected{ext}")
        out_json = os.path.join(IMAGE_DIR_OUT, f"{base}_detected.json")

        detected_img.save(out_img)

        with open(out_json, "w") as f:
            json.dump(detections, f, indent=2)

    print("✅ Processing complete")


if __name__ == "__main__":
    main()
