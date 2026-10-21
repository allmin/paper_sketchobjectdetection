"""
File: helper_functions.py
Path: paper_sketchobjectdetection/classes/helper_functions.py

Purpose:
    class to perform region proposal on image.

Author: Allmin
Created: 2026-06-05
"""
# Path to your sketch
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BINARY_THRESHOLD = 220
CC_CONNECTIVITY = 8
COORD_NORM_SIZE = 1000

def find_clusters(points, method="dbscan", params = {}):
    if method == "dbscan":
        # print(f"performing dbscan with eps = {params['epsilon']}")
        clustering = DBSCAN(eps=params['epsilon'], min_samples=params['min_sample']).fit(points)
        labels = clustering.labels_
    else:
        labels = None
    return labels


def sketch_to_binary(img_np, kernel_size=3, use_fill = True):
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, sketch = cv2.threshold(img_np, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    sketch = sketch.astype(bool)
    binary = sketch

    if use_fill:

        # dilation/closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        sketch_closed = cv2.morphologyEx(
            sketch.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
        ).astype(bool)

        # Fill holes
        foreground = binary_fill_holes(sketch_closed)
        sketch = foreground.astype(np.uint8) * 255
        binary =  sketch
    return binary

def connected_components(binary_image):
    #run connected components and return the num_components, labels
    
    # Ensure input is uint8 with values 0 or 255
    img = (binary_image > 0).astype(np.uint8) * 255

    # Run connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8, ltype=cv2.CV_32S
    )
    return num_labels, labels, stats, centroids

def compute_kde_density(coords, xs, ys, bw, grid_res, visualize):
    kde = gaussian_kde(coords, bw_method=bw)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xyrange = [xmin, ymin, xmax, ymax]

    yy, xx = np.mgrid[ymin:ymax:grid_res*1j, xmin:xmax:grid_res*1j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    density = kde(grid).reshape(xx.shape)
    maybe_show(density, "KDE density", visualize)

    return density, xyrange

def maybe_show(img, title=None, visualize=False, cmap="hot"):
    if not visualize:
        return
    plt.figure()
    if title:
        plt.title(title)
    cmap = 'gray' if (hasattr(img, 'ndim') and img.ndim == 2) else None
    plt.imshow(img, origin="upper", cmap=cmap)
    plt.colorbar() if cmap == "hot" else None
    plt.show()


# ------------------------------------------------------------
# Updated helper functions with visualization hooks
# ------------------------------------------------------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


def bbox_center(box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return np.array([cx, cy]) 
    
def cc2bbox(stats, grid_res=1, xyrange=None):
    if xyrange:
        xmin, ymin, xmax, ymax = xyrange
        scaling = True
    else:
        scaling = False
    bboxes=[]
    for i in range(1, len(stats)):  # skip background
        x, y, w, h, area = stats[i]
        
        if scaling:
            # Map from KDE grid → original image coordinates
            scale_x = (xmax - xmin) / grid_res
            scale_y = (ymax - ymin) / grid_res

            x1 = xmin + x * scale_x
            y1 = ymin + y * scale_y
            x2 = xmin + (x + w) * scale_x
            y2 = ymin + (y + h) * scale_y
        else: 
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
        bboxes.append([x1,y1,x2,y2])
    return bboxes

def bbox_match(bbox_fg, bbox_scene):
    matches = []

    for fg in bbox_fg:
        best_iou = -1
        best_idx = -1

        # First pass: IoU matching
        for i, sc in enumerate(bbox_scene):
            score = iou(fg, sc)
            if score > best_iou:
                best_iou = score
                best_idx = i

        # If no overlap at all → fallback to closest center
        if best_iou == 0:
            fg_c = bbox_center(fg)
            min_dist = float("inf")
            closest_idx = -1

            for i, sc in enumerate(bbox_scene):
                sc_c = bbox_center(sc)
                dist = np.linalg.norm(fg_c - sc_c)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            best_idx = closest_idx

        matches.append(best_idx)

    return matches


def resize_to_thousand(img_pil):
    target_width = COORD_NORM_SIZE 
    w, h = img_pil.size
    new_height = int((target_width / w) * h)
    img_resized = img_pil.resize((target_width, new_height), Image.LANCZOS)
    return img_resized

def preprocess_image(pil_img, gaussian_blur_kernel, visualize):
    img = np.array(pil_img.convert("L"))
    maybe_show(img, "Original grayscale", visualize)

    img_blur = cv2.GaussianBlur(img, (gaussian_blur_kernel, gaussian_blur_kernel), 0)
    maybe_show(img_blur, "Blurred image", visualize)

    fg_mask = sketch_to_binary(img_blur)
    maybe_show(fg_mask, "Foreground mask", visualize, cmap="gray")

    ys, xs = np.where(fg_mask)
    coords = np.vstack([xs, ys])
    return img, fg_mask, coords, xs, ys





def threshold_density(density, kde_thresh, visualize):
    d_min, d_max = density.min(), density.max()
    if d_max - d_min == 0:
        return None

    density_norm = (density - d_min) / (d_max - d_min)
    mask = (density_norm > kde_thresh).astype(np.uint8) * 255

    maybe_show(mask, "KDE threshold mask", visualize, cmap="gray")
    return mask


def extract_scene_bboxes(mask, grid_res, xyrange, visualize):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=CC_CONNECTIVITY)

    maybe_show(labels, "Connected components (KDE mask)", visualize, cmap="nipy_spectral")

    scene_bboxes = cc2bbox(stats, grid_res, xyrange)
    return num_labels, labels, stats, scene_bboxes

def get_single_bbox(xyrange):
    xmin, ymin, xmax, ymax = xyrange
    wmin = xmax-xmin
    hmin = ymax-ymin
    num_labels = 1
    labels = None
    stats = [[xmin, ymin, wmin, hmin, int(wmin*hmin)]]
    return num_labels, labels, stats
def find_outer_bbox(bboxes):
    # bboxes: list of [x1, y1, x2, y2]
    if not bboxes:
        return None
    x1 = min(box[0] for box in bboxes)
    y1 = min(box[1] for box in bboxes)
    x2 = max(box[2] for box in bboxes)
    y2 = max(box[3] for box in bboxes)
    return [x1, y1, x2, y2]
    
def match_and_expand_bboxes(fg_mask, scene_bboxes, visualize, pil_img):
    fg_num_labels, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    fg_bboxes = cc2bbox(fg_stats)

    fg2scene = bbox_match(fg_bboxes, scene_bboxes)

    new_scene_bboxes = []

    for sc_id in range(len(scene_bboxes)):
        assigned = [fg_bboxes[i] for i, sid in enumerate(fg2scene) if sid == sc_id]
        new_scene_bboxes.append(find_outer_bbox(assigned))

    # Optional visualization
    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(np.array(pil_img), origin='upper')

        color_map = ["red","blue","green","orange","purple","cyan","magenta","yellow","brown","pink"]

        # Draw original scene boxes
        for i, (x1,y1,x2,y2) in enumerate(scene_bboxes):
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                           edgecolor=color_map[i%10], fill=False))

        # Draw expanded boxes
        for i, box in enumerate(new_scene_bboxes):
            if box:
                x1,y1,x2,y2 = box
                ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                               edgecolor=color_map[i%10],
                                               linestyle="dotted", fill=False))

        plt.title("Scene bounding boxes (original + expanded)")
        plt.show()
    
            # Generate scene-only images
    scene_only_images = []
    img_np = np.array(pil_img)

    for sc_id in range(len(scene_bboxes)):
        # Create an empty black image
        scene_img = img_np.copy()

        # Find all fg components belonging to this scene
        assigned_fg_ids = [i for i, sid in enumerate(fg2scene) if sid == sc_id]

       # Mask of all pixels NOT belonging to this scene
        mask_keep = np.zeros(fg_labels.shape, dtype=bool)
        for fg_id in assigned_fg_ids:
            mask_keep |= (fg_labels == (fg_id + 1))

        # Zero out everything except the assigned foreground
        scene_img[~mask_keep] = 255

        scene_only_images.append(Image.fromarray(scene_img))

        # Crop scene-only images to their bounding boxes
    cropped_scene_images = []

    for sc_id, img in enumerate(scene_only_images):
        bbox = new_scene_bboxes[sc_id]
        if bbox is None:
            cropped_scene_images.append(None)
            continue

        x1, y1, x2, y2 = bbox
        cropped = img.crop((x1, y1, x2, y2))
        cropped_scene_images.append(cropped)


    return fg_bboxes, fg2scene, new_scene_bboxes, scene_only_images, cropped_scene_images


# ------------------------------------------------------------
# Main function with visualization flag
# ------------------------------------------------------------

def run_kde(img,
            gaussian_blur_kernel=5,
            bw="scott",
            grid_res=20,
            kde_thresh='auto',
            visualize=False):



    density, xyrange = compute_kde_density(
        coords, xs, ys, bw, grid_res, visualize
    )

    
    mask = threshold_density(density, kde_thresh, visualize)
    
    if mask is None:
        num_labels, labels, stats = get_single_bbox(xyrange)
    
    else:
        num_labels, labels, stats, scene_bboxes = extract_scene_bboxes(
            mask, grid_res, xyrange, visualize
        )


    return scene_bboxes




def extract_centroid_points(stats, centroids):
    components = []
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if area >= min_comp_area:
            components.append(
                {"bbox": (x, y, x+w, y+h), "centroid": (cx, cy), "area": area}
            )
    points = np.array([c["centroid"] for c in components])
    return points




def run_dbscan(fg_mask, min_comp_area=10, eps=215, min_samples=3, visualize=False):

    # --- Step 1: Connected Components ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)
    points = extract_centroid_points(stats, centroids)

    # Visualization: Connected components + centroids
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(fg_mask, cmap="gray")
        for c in components:
            x1, y1, x2, y2 = c["bbox"]
            cx, cy = c["centroid"]
            plt.scatter(cx, cy, c="red")
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="yellow", linewidth=1.5)
            )
        plt.title("Connected Components + Centroids")
        plt.show()

    if len(points) == 0:
        return []

    # --- Step 2: DBSCAN clustering ---
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Visualization: DBSCAN clusters
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(fg_mask, cmap="gray")
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for lbl, color in zip(unique_labels, colors):
            cluster_points = points[labels == lbl]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f"Cluster {lbl}")

        plt.legend()
        plt.title("DBSCAN Clustering of Centroids")
        plt.show()

    # --- Step 3: Group components by cluster ---
    clusters = {}
    for i, label in enumerate(labels):
        uid = f"noise_{i}" if label == -1 else label
        clusters.setdefault(uid, []).append(components[i])

    # --- Step 4: Compute outer bounding boxes ---
    scene_bboxes = []
    for cluster in clusters.values():
        bboxes = [component['bbox'] for component in cluster]
        cluster_bbox = find_outer_bbox(bboxes)
        scene_bboxes.append(cluster_bbox)

    # Visualization: Final merged scene bounding boxes
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(fg_mask, cmap="gray")

        for (x1, y1, x2, y2) in scene_bboxes:
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="cyan", linewidth=2)
            )

        plt.title("Final Scene Bounding Boxes (Merged Clusters)")
        plt.show()

    return scene_bboxes

import cv2
import numpy as np

def run_superpixel_segmentation(pil_img,
                                fg_mask,
                                n_segments=200,
                                compactness=10.0,
                                visualize=False):

    img = np.array(pil_img.convert("RGB"))

    # --- Step 1: SLIC Superpixels (OpenCV) ---
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img,
        algorithm=cv2.ximgproc.SLICO,
        region_size=max(10, int(np.sqrt(img.shape[0] * img.shape[1] / n_segments))),
        ruler=compactness
    )
    slic.iterate(10)
    slic.enforceLabelConnectivity()

    labels = slic.getLabels()
    num_superpixels = slic.getNumberOfSuperpixels()

    # Visualization: Superpixel boundaries
    if visualize:
        mask = slic.getLabelContourMask()
        vis = img.copy()
        vis[mask == 255] = [0, 255, 0]
        plt.figure(figsize=(8, 8))
        plt.imshow(vis)
        plt.title("SLIC Superpixels (OpenCV)")
        plt.axis("off")
        plt.show()

    # --- Step 2: Build GrabCut mask from fg_mask ---
    gc_mask = np.where(fg_mask > 0, 3, 2).astype(np.uint8)

    # --- Step 3: GraphCut refinement ---
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel,
                iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    final_mask = np.where(
        (gc_mask == 1) | (gc_mask == 3),
        1, 0
    ).astype(np.uint8)

    # Visualization: GraphCut mask
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(final_mask, cmap="gray")
        plt.title("GraphCut Mask")
        plt.axis("off")
        plt.show()

    # --- Step 4: Group superpixels by final mask ---
    scene_bboxes = []
    for sp_id in range(num_superpixels):
        sp_mask = (labels == sp_id)
        if np.mean(final_mask[sp_mask]) < 0.5:
            continue

        ys, xs = np.where(sp_mask)
        if len(xs) == 0:
            continue

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        scene_bboxes.append([x1, y1, x2, y2])

    # --- Step 5: Merge overlapping boxes (simple passthrough for now) ---
    merged = scene_bboxes.copy()

    # Visualization: Final bounding boxes
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="upper")
        for (x1, y1, x2, y2) in merged:
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=2
                )
            )
        plt.title("Scene Bounding Boxes (Superpixel + GraphCut)")
        plt.axis("off")
        plt.show()

    return merged

import base64
import json
import re
import requests
from io import BytesIO

def run_vllm(pil_img, visualize=False, ollama_url="http://localhost:11434"):
    """
    Use Ollama (gemma3:27b) to detect scene bounding boxes in a sketch image.
    Returns a list of [x1, y1, x2, y2] boxes in pixel coordinates.
    """
    w, h = pil_img.size

    # --- Step 1: Encode image to base64 ---
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # --- Step 2: Prompt ---
    prompt = f"""You are analyzing a hand-drawn sketch image of size {w}x{h} pixels (width x height).

The image contains multiple distinct scene clusters or groups of drawn objects.
Your task is to identify each separate scene/cluster and return a bounding box for each.

Return ONLY a JSON array of bounding boxes. Each box must be:
  {{"x1": int, "y1": int, "x2": int, "y2": int}}

Where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, in pixel coordinates.

Rules:
- Return at least 1 bounding box.
- Do not include any explanation, markdown, or text outside the JSON array.
- Group spatially close objects into the same bounding box.
- Separate groups of objects that are clearly distinct and far apart.

Example output format:
[
  {{"x1": 10, "y1": 20, "x2": 300, "y2": 400}},
  {{"x1": 500, "y1": 50, "x2": 900, "y2": 600}}
]
"""

    # --- Step 3: Call Ollama ---
    payload = {
        "model": "gemma4:latest",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"[run_vllm] Ollama request failed: {e}")
        return [[0, 0, w, h]]  # fallback: full image

    # --- Step 4: Parse JSON from response ---
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

    # Extract JSON array
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if not match:
        print(f"[run_vllm] Could not find JSON array in response:\n{raw_text}")
        return [[0, 0, w, h]]  # fallback

    try:
        boxes_raw = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[run_vllm] JSON parse error: {e}\nRaw: {raw_text}")
        return [[0, 0, w, h]]

    # --- Step 5: Convert to [x1, y1, x2, y2] list, clamp to image bounds ---
    scene_bboxes = []
    for box in boxes_raw:
        try:
            x1 = max(0, int(box["x1"]))
            y1 = max(0, int(box["y1"]))
            x2 = min(w, int(box["x2"]))
            y2 = min(h, int(box["y2"]))
            if x2 > x1 and y2 > y1:
                scene_bboxes.append([x1, y1, x2, y2])
        except (KeyError, ValueError, TypeError) as e:
            print(f"[run_vllm] Skipping malformed box {box}: {e}")
            continue

    if not scene_bboxes:
        print("[run_vllm] No valid boxes parsed, falling back to full image.")
        return [[0, 0, w, h]]

    # --- Step 6: Optional visualization ---
    if visualize:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(pil_img)
        color_map = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
        for i, (x1, y1, x2, y2) in enumerate(scene_bboxes):
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor=color_map[i % len(color_map)],
                fill=False,
                linewidth=2
            ))
            ax.text(x1, y1 - 5, f"Scene {i}", color=color_map[i % len(color_map)], fontsize=9)
        ax.set_title(f"run_vllm: {len(scene_bboxes)} scene(s) detected")
        plt.axis("off")
        plt.show()

    return scene_bboxes

def full_image_detect_label_llm(pil_img, ollama_url="http://localhost:11434", visualize=True):
    """
    Detect and label individual objects in an image using Ollama gemma4:31b.
    Model returns bboxes in 1000x1000 scale → transformed to actual image dimensions.
    """
    import base64, json, re, requests
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    w, h = pil_img.size

    # --- Encode image ---
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # --- Prompt ---
    prompt = """Detect every individual object in this image and label it.

Return ONLY a JSON array. Each element must be:
  {"label": "object name", "x1": int, "y1": int, "x2": int, "y2": int}

Coordinates are in a 1000x1000 normalized space (0–1000 for both axes).
(x1, y1) = top-left, (x2, y2) = bottom-right.

Rules:
- One entry per distinct object instance.
- No explanation, markdown, or text outside the JSON array.
- Keep labels short (1-3 words).

Example:
[
  {"label": "cat", "x1": 120, "y1": 80, "x2": 400, "y2": 350},
  {"label": "dog", "x1": 500, "y1": 200, "x2": 900, "y2": 700}
]"""

    # --- Call Ollama ---
    payload = {
        "model": "gemma4:31b",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"[full_image_detect_label_llm] Request failed: {e}")
        return []

    # --- Parse JSON ---
    clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if not match:
        print(f"[full_image_detect_label_llm] No JSON array found:\n{raw_text}")
        return []

    try:
        detections_raw = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[full_image_detect_label_llm] JSON parse error: {e}\n{raw_text}")
        return []

    # --- Transform 1000x1000 → actual image dimensions ---
    detections = []
    for det in detections_raw:
        try:
            x1 = max(0, int(det["x1"] / 1000 * w))
            y1 = max(0, int(det["y1"] / 1000 * h))
            x2 = min(w, int(det["x2"] / 1000 * w))
            y2 = min(h, int(det["y2"] / 1000 * h))
            label = str(det.get("label", "?"))
            if x2 > x1 and y2 > y1:
                detections.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        except (KeyError, ValueError, TypeError) as e:
            print(f"[full_image_detect_label_llm] Skipping malformed detection {det}: {e}")

    print(f"[full_image_detect_label_llm] Detected {len(detections)} object(s).")

    # --- Visualize ---
    if visualize:
        color_cycle = [
            "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
            "#42d4f4","#f032e6","#bfef45","#fabed4","#469990"
        ]
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(pil_img)

        for i, det in enumerate(detections):
            color = color_cycle[i % len(color_cycle)]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            ))
            ax.text(
                x1, max(y1 - 6, 0), det["label"],
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none")
            )

        ax.set_title(f"{len(detections)} object(s) detected", fontsize=13)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return detections