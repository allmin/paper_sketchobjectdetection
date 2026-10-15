"""
DINOv3 + CLIP zero-shot object finder (Streamlit)
==================================================

What this actually does
------------------------
DINOv3 (facebookresearch/dinov3) is a self-supervised vision backbone. It has
no text encoder and no detection head, so it cannot take a list like
"transmission tower, road, tree" and return boxes on its own. Grounding DINO
*can* do that, but despite the name it is unrelated to DINOv3 (its "DINO"
comes from a different, closed-set DETR variant; its backbone is Swin
Transformer + BERT).

This app is an experimental hybrid pipeline built specifically because you
asked for DINOv3 in the loop:

  1. Run DINOv3 on the image and take its per-patch feature grid.
  2. Cluster those patch features spatially (KMeans + connected components)
     to propose candidate object regions. This step is class-agnostic --
     DINOv3 has no idea what "tree" means, it just groups visually similar,
     spatially contiguous patches.
  3. Crop each candidate region and score it with CLIP against your text
     list (CLIP is the only place text ever enters the pipeline).
  4. Keep regions whose best CLIP similarity clears a threshold, run a
     per-label NMS, and draw the surviving boxes.

Expect coarse, grid-aligned boxes, missed instances, and false positives --
this is not a substitute for a trained/text-conditioned detector such as
Grounding DINO or YOLO-World. See README.md for details and that alternative.

Setup
-----
    pip install -r requirements.txt
    huggingface-cli login   # DINOv3 weights are gated, see README.md
    streamlit run app.py
"""

import colorsys
import hashlib

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

st.set_page_config(page_title="DINOv3 + CLIP Zero-Shot Object Finder", layout="wide")

# Only the ViT variants -- the ConvNeXt DINOv3 checkpoints don't expose the
# CLS + register + patch token sequence this pipeline relies on.
DINOV3_MODELS = {
    "DINOv3 ViT-S/16 (fastest)": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "DINOv3 ViT-B/16 (balanced)": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "DINOv3 ViT-L/16 (best features, slow on CPU)": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading DINOv3 backbone...")
def load_dinov3(model_id: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(DEVICE).eval()
    return processor, model


@st.cache_resource(show_spinner="Loading CLIP (used only to score/label regions)...")
def load_clip():
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True)
    model.to(DEVICE).eval()
    return processor, model


def label_to_color(label: str):
    """Deterministic, visually distinct color per label string."""
    h = int(hashlib.md5(label.encode()).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def get_patch_grid_features(image: Image.Image, processor, model, input_size: int):
    """Run DINOv3 and return the patch feature grid plus the pixel size the model saw."""
    resized = image.convert("RGB").resize((input_size, input_size), Image.BICUBIC)
    try:
        inputs = processor(images=resized, do_resize=False, do_center_crop=False, return_tensors="pt")
    except TypeError:
        # Processor doesn't support these override kwargs -- fall back to its defaults.
        inputs = processor(images=resized, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pixel_h, pixel_w = inputs["pixel_values"].shape[-2:]
    patch_size = getattr(model.config, "patch_size", 16)
    grid_h, grid_w = pixel_h // patch_size, pixel_w // patch_size

    hidden = outputs.last_hidden_state[0]  # (num_tokens, hidden_dim)
    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
    num_prefix = 1 + num_register_tokens  # CLS token + register tokens
    patch_tokens = hidden[num_prefix : num_prefix + grid_h * grid_w]
    features = patch_tokens.reshape(grid_h, grid_w, -1).cpu().numpy()
    return features, grid_h, grid_w, pixel_h, pixel_w


def propose_regions(features, grid_h, grid_w, n_clusters, min_patches):
    """Cluster DINOv3 patch features into candidate object regions. No text is used here."""
    flat = features.reshape(-1, features.shape[-1])
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6)
    n_clusters = max(1, min(n_clusters, flat.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit(flat)
    cluster_map = kmeans.labels_.reshape(grid_h, grid_w)

    boxes_patch_coords = []
    for cluster_id in range(n_clusters):
        mask = cluster_map == cluster_id
        components = cc_label(mask, connectivity=2)
        for region in regionprops(components):
            if region.area < min_patches:
                continue
            min_row, min_col, max_row, max_col = region.bbox  # rows=y, cols=x, half-open
            boxes_patch_coords.append((min_col, min_row, max_col, max_row))
    return boxes_patch_coords


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def label_regions_with_clip(image, boxes_pixel_coords, object_names, clip_processor, clip_model, sim_threshold):
    """Crop each candidate region and score it against the free-text object list with CLIP."""
    text_inputs = clip_processor(text=object_names, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    results = []
    for box in boxes_pixel_coords:
        x1, y1, x2, y2 = box
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = image.crop((x1, y1, x2, y2))
        clip_inputs = clip_processor(images=crop, return_tensors="pt")
        clip_inputs = {k: v.to(DEVICE) for k, v in clip_inputs.items()}
        with torch.no_grad():
            image_embed = clip_model.get_image_features(**clip_inputs)
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        sims = (image_embed @ text_embeds.T).squeeze(0)
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score >= sim_threshold:
            results.append((box, object_names[best_idx], best_score))
    return results


def per_label_nms(results, iou_threshold=0.4):
    """Greedy NMS, applied only between boxes that share the same assigned label."""
    results = sorted(results, key=lambda r: r[2], reverse=True)
    kept = []
    for box, label, score in results:
        overlaps_kept = any(label == k_label and iou(box, k_box) > iou_threshold for k_box, k_label, _ in kept)
        if not overlaps_kept:
            kept.append((box, label, score))
    return kept


def draw_boxes(image, results):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for box, label, score in results:
        color = label_to_color(label)
        draw.rectangle(box, outline=color, width=3)
        caption = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((box[0], box[1]), caption, font=font)
        draw.rectangle([box[0], text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill=color)
        draw.text((box[0] + 2, box[1] - 2), caption, fill="black", font=font)
    return canvas


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("DINOv3 + CLIP zero-shot object finder")
st.caption(
    "DINOv3 has no text grounding of its own, so this app clusters DINOv3 patch features into "
    "candidate regions first, then uses CLIP purely to score/label each region against your text "
    "list. Experimental unsupervised pipeline, not a trained detector like Grounding DINO or "
    "YOLO-World -- expect coarse, grid-aligned boxes, and don't expect every named object to be found."
)

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("DINOv3 backbone", list(DINOV3_MODELS.keys()), index=0)
    input_size = st.select_slider("Input resolution (px, square)", options=[224, 448, 672, 896], value=448)
    n_clusters = st.slider("Feature clusters (candidate region groups)", 2, 12, 6)
    min_patches = st.slider("Minimum region size (patches)", 1, 30, 4)
    sim_threshold = st.slider("CLIP similarity threshold", 0.0, 0.4, 0.20, 0.01)
    nms_iou = st.slider("Same-label NMS IoU threshold", 0.1, 0.9, 0.4, 0.05)
    st.markdown("---")
    st.caption(
        "DINOv3 checkpoints on Hugging Face are gated: accept the license at "
        "huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and run `huggingface-cli login` "
        "(or set the HF_TOKEN environment variable) before starting this app."
    )

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
object_names_raw = st.text_input(
    "Object names to look for (comma-separated)",
    placeholder="transmission tower, road, tree, building",
)

run = st.button("Find objects", type="primary", disabled=not (uploaded_file and object_names_raw.strip()))

if run:
    image = Image.open(uploaded_file).convert("RGB")
    object_names = [n.strip() for n in object_names_raw.split(",") if n.strip()]

    dinov3_processor, dinov3_model = load_dinov3(DINOV3_MODELS[model_choice])
    clip_processor, clip_model = load_clip()

    with st.spinner("Extracting DINOv3 features and proposing regions..."):
        features, grid_h, grid_w, pixel_h, pixel_w = get_patch_grid_features(
            image, dinov3_processor, dinov3_model, input_size
        )
        boxes_patch = propose_regions(features, grid_h, grid_w, n_clusters, min_patches)

        scale_x = image.width / pixel_w
        scale_y = image.height / pixel_h
        patch_w = pixel_w / grid_w
        patch_h = pixel_h / grid_h
        boxes_pixel = [
            (
                int(x1 * patch_w * scale_x),
                int(y1 * patch_h * scale_y),
                int(x2 * patch_w * scale_x),
                int(y2 * patch_h * scale_y),
            )
            for x1, y1, x2, y2 in boxes_patch
        ]

    with st.spinner("Scoring regions against your object list with CLIP..."):
        labeled = label_regions_with_clip(
            image, boxes_pixel, object_names, clip_processor, clip_model, sim_threshold
        )
        labeled = per_label_nms(labeled, nms_iou)

    if not labeled:
        st.warning(
            "No regions matched your object list above the similarity threshold. "
            "Try lowering the threshold, or increasing clusters/resolution."
        )
        st.image(image, caption="Original image", use_container_width=True)
    else:
        annotated = draw_boxes(image, labeled)
        st.image(annotated, caption=f"Found {len(labeled)} region(s)", use_container_width=True)
        st.subheader("Detections")
        for box, label, score in sorted(labeled, key=lambda r: r[2], reverse=True):
            st.write(f"**{label}** -- similarity {score:.2f} -- box {box}")