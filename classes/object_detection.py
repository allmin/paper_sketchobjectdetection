"""
File: object_detection.py
Path: paper_sketchobjectdetection/classes/object_detection.py

Purpose:
    class to perform object detection image.

Author: Allmin
Created: 2026-06-05
"""

import torch
import open_clip
import numpy as np
import requests
import json, re, base64
from io import BytesIO
import os, shutil
from PIL import Image, ImageDraw, ImageFont
from tictoc import tic, toc

class ObjectDetector:
    """
    Generic object detection wrapper supporting:
    - CLIP classification on crops
    - LLM-based full-image object detection
    """
    # -------------------------------------------------------------------------
    # PARAMETER SPECIFICATIONS
    # -------------------------------------------------------------------------
    METHOD_SPECS = {
        "clip": {
            "predefined_labels": {
                "description": "List of object classes to classify",
                "min": None,
                "max": None,
                "recommended": ["car", "person", "hospital", "building"],
            },
            "clip_model": {
                "description": "CLIP model name",
                "min": None,
                "max": None,
                "recommended": "ViT-B-32",
            }
        },

        "llm": {
            "ollama_url": {
                "description": "URL of Ollama server",
                "min": None,
                "max": None,
                "recommended": "http://localhost:11434",
            },
            "model_name": {
                "description": "LLM model name",
                "min": None,
                "max": None,
                "recommended": "gemma4:latest",
            },
        }
    }

    def __init__(self, method, params={}, name=""):
        self.method = method.lower()
        self.params = self._assign_default_params(params)
        self.image = None
        self.clip_cache = {}  # store loaded CLIP models
        self.type = 'object_detection'
        self.name = name
        self.detections = []

    # PARAMETER HANDLING
    def _assign_default_params(self, user_params):
        """Fill missing parameters with recommended."""
        specs = self.METHOD_SPECS.get(self.method, {})
        final = {}

        for key, meta in specs.items():
            if key in user_params:
                final[key] = user_params[key]
            else:
                final[key] = meta["recommended"]
        return final

    def set_image(self, image, offset=(0,0)):
        self.image = image
        self.offset = offset


    
    def run(self):
        if self.image is None:
            raise ValueError("No image set. Call set_image(image) first.")
        print(f"Running object detection using {self.method}:{self.params}")
        self.detect_objects()
        self._get_modified_detections()
        return self.detections

    def detect_objects(self):
        if self.method == "clip":
            self._detect_clip()

        elif self.method == "llm":
            self._detect_llm()

        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    #CLIP
    def _load_clip(self, model_name):
        """Load CLIP model once and cache it."""
        if model_name not in self.clip_cache:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()

            self.clip_cache[model_name] = {
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
            }

        return self.clip_cache[model_name]

    def _detect_clip(self):
        """
        CLIP does NOT detect bounding boxes.
        We return a single object covering the whole image.
        """
        labels = self.params["predefined_labels"]
        model_name = self.params["clip_model"]

        clip = self._load_clip(model_name)
        model = clip["model"]
        preprocess = clip["preprocess"]
        tokenizer = clip["tokenizer"]

        img = preprocess(self.image).unsqueeze(0)
        text = tokenizer(labels)

        start_time = tic()
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            probs = (image_features @ text_features.T).softmax(dim=-1)

        idx = probs.argmax().item()
        step_time = toc(start_time)

        confidence = float(probs[0, idx])

        w, h = self.image.size

        self.detections = [{
            "label": labels[idx],
            "confidence": confidence,
            "x1": 0,
            "y1": 0,
            "x2": w,
            "y2": h,
            "step_time": step_time
        }]

    # LLM DETECTION
    def _detect_llm(self):
        pil_img = self.image
        w, h = pil_img.size

        # Encode image
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        prompt = """Detect every individual object in this hand drawn sketch and label it.

Return ONLY a JSON array. Each element must be:
  {"label": "object/text", "x1": int, "y1": int, "x2": int, "y2": int}

Coordinates are in a 1000x1000 normalized space.

Rules:
- One entry per distinct object instance.
- No explanation, markdown, or text outside the JSON array.
- Keep labels short.
- If the object is a text, label it with is text itself within quotes.
"""

        payload = {
            "model": self.params["model_name"],
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        try:
            time_start = tic()
            response = requests.post(
                f"{self.params['ollama_url']}/api/generate",
                json=payload,
                timeout=500
            )
            step_time = toc(time_start)
            response.raise_for_status()
            raw_text = response.json().get("response", "")
        except Exception as e:
            print(f"[LLM] Request failed: {e}")
            return []

        # Extract JSON
        clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match is None:
            print("[LLM] No JSON found")
            return []

        try:
            detections_raw = json.loads(match.group())
        except:
            print("[LLM] JSON parse error")
            return []

        # Convert 1000x1000 → actual image size
        detections = []
        for det in detections_raw:
            try:
                x1 = int(det["x1"] / 1000 * w)
                y1 = int(det["y1"] / 1000 * h)
                x2 = int(det["x2"] / 1000 * w)
                y2 = int(det["y2"] / 1000 * h)

                if x2 > x1 and y2 > y1:
                    detections.append({
                        "label": det["label"],
                        "confidence": "N/a",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "step_time": step_time
                    })
            except:
                continue
        self.detections = detections
        return detections
    


    def draw_detections(self, image=None, detections=None, scenes=None, file_path=None):
        """
        Draw bounding boxes + labels on a copy of the image.
        detections format:
        [
            {"label": "cat", "x1": 10, "y1": 20, "x2": 200, "y2": 300},
            ...
        ]
        """
        if image is None:
            image = self.image
        if detections is None:
            detections = self.detections

        image = image.convert("RGB")
        box_color="#00ff00"
        scene_color = "#ff0000"
        text_color="red"
        
        folder_path = os.path.dirname(file_path)    

        # Work on a copy so original stays untouched
        out = image.copy()
        draw = ImageDraw.Draw(out)

        # Optional: load a nicer font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        if scenes:
            for scene_bbox in scenes:
                if scene_bbox:
                    draw.rectangle(scene_bbox, outline=scene_color, width = 4)

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = det.get("label", "?")
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            # Draw label background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill=box_color
            )

            # Draw label text
            draw.text(
                (x1 + 2, y1 - text_h - 2),
                label,
                fill=text_color,
                font=font
            )
    
        if file_path:
            out.save(file_path)
        return out
    
    def _get_modified_detections(self):
        if self.offset[0] == 0 and self.offset[1] == 0:
            self.modified_detections = self.detections
        else:
            self.modified_detections = []
            for detection in self.detections:
                modified_detection = detection.copy()
                modified_detection['x1'] += self.offset[0]
                modified_detection['y1'] += self.offset[1]
                modified_detection['x2'] += self.offset[0]
                modified_detection['y2'] += self.offset[1]
                self.modified_detections.append(modified_detection)
                # print(detection, self.offset, modified_detection)
    
    def _resize_width(self, image, width):
        w, h = image.size
        new_w = width
        new_h = int(h * (new_w / w))
        out = image.resize((new_w, new_h), Image.BICUBIC)
        return out

            
if __name__ == "__main__":
    from PIL import Image

    # Path to test image
    img_path = "/home/P103236/projects/paper_sketchobjectdetection/test_data/realdata/SampleRP.png"
    # Choose method: "clip" or "llm"
    method = "llm"   # change to "clip" to test CLIP
    # Optional: override parameters
    params = {
        # For LLM:
        "model_name": "gemma4:latest",
        "ollama_url": "http://localhost:11434",

        # For CLIP:
        "predefined_labels": ["car", "person", "tree"],
        "clip_model": "ViT-B-32",
    }

    # Initialize detector
    detector = ObjectDetector(method=method, params=params)

    # Load image
    print(f"Loading test image: {img_path}")
    test_img = Image.open(img_path).convert("RGB")
    test_img = detector._resize_width(test_img,1000)
    
    # Set image
    detector.set_image(test_img)
    
    # Run detection
    print("Running object detection...")
    detector.run()
    detector.draw_detections(detector.image, detector.detections, None, f"temp/02_detections_{detector.method}.png")

    # Print results
    print("\n=== DETECTIONS ===")
    for det in detector.detections:
        print(det)
    
    
    


