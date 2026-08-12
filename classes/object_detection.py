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
import ollama
import easyocr
reader = easyocr.Reader(['en','nl'])
from PIL import Image

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

    def __init__(self, method, params={}, name="", bbox_refiner=None):
        self.method = method.lower()
        self.params = self._assign_default_params(params)
        self.image = None
        self.clip_cache = {}  # store loaded CLIP models
        self.type = 'object_detection'
        self.name = name
        self.detections = []
        self.bbox_refiner = bbox_refiner
        self._init_models()
        self.detected_labels = []
    
    def _init_models(self):
        if self.bbox_refiner is not None:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.gdino_processor = AutoProcessor.from_pretrained(model_id)
            self.refiner_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")


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
        self._refine_bbox()
        return self.detections

    def detect_objects(self):
        if self.method == "clip":
            self._detect_clip()

        elif self.method == "llm":
            self._detect_llm()

        else:
            raise ValueError(f"Unknown detection method: {self.method}")
        self.modified_detections = self._apply_offset(self.detections)

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

        self.detected_labels.append(labels[idx])

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
            self.detected_labels.append(det["label"])
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

    # def _detect_llm(self, mode="chat"):
    #     """
    #     mode = "chat"     → fastest, uses ollama.chat() with raw PNG bytes
    #     mode = "generate" → legacy, uses /api/generate with base64
    #     """

    #     pil_img = self.image
    #     w, h = pil_img.size

    #     # Encode image once
    #     buffer = BytesIO()
    #     pil_img.save(buffer, format="PNG")
    #     img_bytes = buffer.getvalue()

    #     # Base64 only needed for generate mode
    #     img_b64 = None
    #     if mode == "generate":
    #         import base64
    #         img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    #     prompt = """Detect every individual object in this hand drawn sketch and label it.

    #                 Return ONLY a JSON array. Each element must be:
    #                 {"label": "object/text", "x1": int, "y1": int, "x2": int, "y2": int}

    #                 Coordinates are in a 1000x1000 normalized space.

    #                 Rules:
    #                 - One entry per distinct object instance.
    #                 - No explanation, markdown, or text outside the JSON array.
    #                 - Keep labels short.
    #                 - If the object is a text, label it with the text itself.
    #                 """

    #     # -------------------------------
    #     # MODE 1: FAST → ollama.chat()
    #     # -------------------------------
    #     if mode == "chat":
    #         response = ollama.chat(
    #             model=self.params["model_name"],
    #             messages=[
    #                 {
    #                     "role": "user",
    #                     "content": prompt,
    #                     "images": [img_bytes]
    #                 }
    #             ],
    #             options={"temperature": 0.1}
    #         )
    #         raw_text = response["message"]["content"]

    #     # -------------------------------
    #     # MODE 2: LEGACY → /api/generate
    #     # -------------------------------
    #     elif mode == "generate":
    #         payload = {
    #             "model": self.params["model_name"],
    #             "prompt": prompt,
    #             "images": [img_b64],
    #             "stream": False,
    #             "options": {"temperature": 0.1}
    #         }

    #         try:
    #             response = requests.post(
    #                 f"{self.params['ollama_url']}/api/generate",
    #                 json=payload,
    #                 timeout=500
    #             )
    #             response.raise_for_status()
    #             raw_text = response.json().get("response", "")
    #         except Exception as e:
    #             print(f"[LLM] Request failed: {e}")
    #             return []

    #     else:
    #         raise ValueError(f"Unknown mode: {mode}")

    #     # -------------------------------
    #     # Extract JSON array
    #     # -------------------------------
    #     match = re.search(r"
    #         \[.*\]
    #         ", raw_text, re.DOTALL)
    #     if not match:
    #         print("[LLM] No JSON found")
    #         return []

    #     try:
    #         detections_raw = json.loads(match.group())
    #     except Exception as e:
    #         print(f"[LLM] JSON parse error: {e}")
    #         return []

    #     # -------------------------------
    #     # Convert normalized → pixel coords
    #     # -------------------------------
    #     detections = []
    #     for det in detections_raw:
    #         try:
    #             x1 = int(det["x1"] / 1000 * w)
    #             y1 = int(det["y1"] / 1000 * h)
    #             x2 = int(det["x2"] / 1000 * w)
    #             y2 = int(det["y2"] / 1000 * h)

    #             if x2 > x1 and y2 > y1:
    #                 detections.append({
    #                     "label": det["label"],
    #                     "confidence": "N/a",
    #                     "x1": x1,
    #                     "y1": y1,
    #                     "x2": x2,
    #                     "y2": y2
    #                 })
    #         except:
    #             continue

    #     self.detections = detections
    #     return detections
    


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
    
    def _apply_offset(self, detections):
        if self.offset[0] == 0 and self.offset[1] == 0:
            modified_detections = detections
        else:
            modified_detections = []
            for detection in detections:
                modified_detection = detection.copy()
                modified_detection['x1'] += self.offset[0]
                modified_detection['y1'] += self.offset[1]
                modified_detection['x2'] += self.offset[0]
                modified_detection['y2'] += self.offset[1]
                modified_detections.append(modified_detection)
        return modified_detections
    
    def _resize_width(self, image, width):
        w, h = image.size
        new_w = width
        new_h = int(h * (new_w / w))
        out = image.resize((new_w, new_h), Image.BICUBIC)
        return out
    
    def _refine_bbox(self, box_threshold=0.3, text_threshold=0.3):
        if self.bbox_refiner is None:
            return None
        
        image = self.image
        # GroundingDINO expects a list-of-lists for text prompts
        text_labels = set(self.detected_labels)
        text_input = [text_labels]
        text_input = [i for i in text_input if type(i)==str]
        print(text_input)

        refined_detections = []
        
        if self.bbox_refiner == "gdino":
            time_start = tic()
            inputs = self.gdino_processor(
                images=image,
                text=text_input,
                return_tensors="pt"
            ).to(self.refiner_model.device)

            with torch.no_grad():
                outputs = self.refiner_model(**inputs)

            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]
            )

            step_time = toc(time_start)

            
            result = results[0] 

            for box, score, label in zip(
                result["boxes"],
                result["scores"],
                result["text_labels"]  # safer than "labels"
            ):
                x1,y1,x2,y2 = box
                refined_detections.append({                
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "step_time":step_time,
                    "score": float(score.item()),
                })

            self.refined_detections = refined_detections
        self.modified_refined_detections = self._apply_offset(self.refined_detections)
        

class ObjectVerifier:
    def __init__(
        self,
        models=["qwen3.5:9b", "llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "gemma4:latest"],
        judge="llama3.1:70b",
        name="ensemble"
    ):
        self.models = models
        self.judge = judge
        self.name = name

    def _crop_to_bytes(self, image, bbox):
        crop = image.crop(bbox)
        buf = BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue(), crop

    def _ask_model(self, model, crop_bytes):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What object is shown in this image crop?",
                        "images": [crop_bytes]
                    }
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[ERROR from {model}: {e}]"

    def _judge(self, bbox, model_answers):
        judge_prompt = (
            "You are an ensemble judge. Multiple vision models were asked to identify "
            "the object inside a bounding box crop. Your job is to analyze all answers "
            "and produce a single, best final answer.\n\n"
            f"Bounding box: {bbox}\n\n"
            "Model responses:\n"
        )

        for m, ans in model_answers.items():
            judge_prompt += f"- {m}: {ans}\n"

        judge_prompt += "\nFinal answer (one short phrase):"

        response = ollama.chat(
            model=self.judge,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        return response["message"]["content"]

    def run(self, image, bboxes):
        results = []

        for bbox in bboxes:
            crop_bytes, crop_img = self._crop_to_bytes(image, bbox)
            text = self.extract_text_or_false(crop_img)

            model_answers = {}
            for model in self.models:
                ans = self._ask_model(model, crop_bytes)
                model_answers[model] = ans

            final_answer = self._judge(bbox, model_answers)

            results.append({
                "bbox": bbox,
                "text": text,
                "crop_img": crop_img,
                "model_answers": model_answers,
                "final_answer": final_answer
            })

        return results

    def save_crops(self, results, directory):
        """
        Saves crops to disk with filenames:
        model1 + model2 + ... = final_label.png
        """
        os.makedirs(directory, exist_ok=True)

        def clean(text):
            # Remove characters invalid in filenames
            text = re.sub(r"[^\w\s-]", "", text)
            return text.strip().replace(" ", "_")

        for item in results:
            answers = item["model_answers"]
            final = item["final_answer"]

            # Build filename
            parts = [clean(ans) for ans in answers.values()]
            final_clean = clean(final)

            filename = " + ".join(parts) + f" = {final_clean}.png"
            filepath = os.path.join(directory, filename)

            # Save crop
            item["crop_img"].save(filepath, format="PNG")
    
    
    def extract_text_or_false(self, img):
        if hasattr(img, "convert"):
            img = np.array(img)

        results = reader.readtext(img, detail=0)
        text = " ".join(results).strip()
        return text if text else False
    

if __name__ == "__main__":
    from PIL import Image

    # Path to test image
    img_path = "/home/P103236/projects/paper_sketchobjectdetection/test_data/realdata/SampleRP.png"
    # Choose method: "clip" or "llm"
    test_object_detection = False

    if test_object_detection:
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
        detector = ObjectDetector(method=method, params=params)#, bbox_refiner="gdino")

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

        {'label': 'house/building', 'confidence': 'N/a', 'x1': 500, 'y1': 453, 'x2': 750, 'y2': 558, 'step_time': 66.69296979904175}
        # print("\n=== REFINED DETECTIONS ===")
        # for det in detector.modified_refined_detections:
        #     print(det)
    

    detector = ObjectDetector(method='llm', params={})   #, bbox_refiner="gdino")
    detections = [{'label': 'people group', 'confidence': 'N/a', 'x1': 20, 'y1': 69, 'x2': 100, 'y2': 125, 'step_time': 50.60171842575073},
                    {'label': 'text FINANCIAL INSTITUTIONS', 'confidence': 'N/a', 'x1': 20, 'y1': 432, 'x2': 250, 'y2': 453, 'step_time': 50.60171842575073},
                    {'label': 'house drawing', 'confidence': 'N/a', 'x1': 500, 'y1': 453, 'x2': 700, 'y2': 558, 'step_time': 50.60171842575073},
                    {'label': 'car drawing', 'confidence': 'N/a', 'x1': 700, 'y1': 474, 'x2': 800, 'y2': 544, 'step_time': 50.60171842575073},
                    ]
    
    # Load image
    print(f"Loading test image: {img_path}")
    test_img = Image.open(img_path).convert("RGB")
    test_img = detector._resize_width(test_img,1000)
    verifier = ObjectVerifier(models=["llava:7b", "moondream:1.8b"],judge="llama3.2:latest")
    
    bboxes = [[det['x1'],det['y1'],det['x2'],det['y2']] for det in detections]
    results = verifier.run(test_img, bboxes)
    print(results)
    verifier.save_crops(results, "temp/verifier_results")
        
        


