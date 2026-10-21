"""
File: pipeline.py
Path: paper_sketchobjectdetection/classes/pipeline.py

Purpose:
    class to perform a pipeline of regionproposer and object detection on an image.

Author: Allmin
Created: 2026-06-05
"""
from classes.region_proposal import RegionProposer
from classes.object_detection import ObjectDetector, ObjectVerifier
import cv2, os
from PIL import Image
import traceback
import shutil
import json
import numpy as np
from tictoc import tic, toc

class Pipeline:
    def __init__(self, steps, name='', resize_width=None, color=True, proposal_width=1000):
        """
        resize_width:   None -> the models see the image at its original resolution.
                        An int (the old behaviour was 1000) resizes the image to that width first.
        color:          True -> the models see the colour image; CLAHE is applied to the
                        lightness channel only. False -> old behaviour: grayscale + CLAHE.
        proposal_width: region proposal (KDE/DBSCAN) runs, unchanged, on a grayscale+CLAHE
                        copy at this width, because its pixel parameters were tuned on
                        1000-px images. The pipeline cuts the matching regions out of the
                        full-resolution (colour) image for the detectors. None -> no scaling.
        Results of non-default preprocessing go to pipeline folders with a suffix
        (e.g. "_rgb_fullres") so they never mix with the old grayscale/1000-px results.
        """
        self.elapsed_time = {}
        self.steps = steps
        self.resize_width = resize_width
        self.color = color
        self.proposal_width = proposal_width
        self.preprocess_tag = self._preprocess_tag()
        steps_name = self._obtain_steps_name(self.steps)
        if name:
            self.name = name
        else:
            self.name = steps_name
        self.cropped_scene_images = []
        self.cropped_scene_locations= []
        self.scenes = None
        self.detections = []
    

        

        
    def run(self, image, skip_existing=False):
        self.cropped_scene_images = []
        self.cropped_scene_locations= []
        self.scenes = None
        if type(image) == str:
            self.image_path = image
            image = Image.open(image)
        else:
            self.image_path = "temp_image_folder/temp_image.png"
        self.file_name = os.path.basename(self.image_path)
        self.folder_name = os.path.dirname(self.image_path).rstrip('/')
        self.result_folder_path = f"02_detected_picture_elements/{self.name}/{self.folder_name.split('/')[-1]}"
        self.result_img_path = f"{self.result_folder_path}/{self.file_name}"
        if skip_existing and os.path.exists(self.result_img_path):
            return 0
        
        os.makedirs(self.result_folder_path, exist_ok=True)
        self.raw_image = image
        self.image = self._preprocess_image(self.raw_image)
        current_output = None
    
        min_enum_id = 0
        for i in range(len(self.steps)-1):
            filt_steps = self.steps[:-i-1]
            filt_steps_name = self._obtain_steps_name(filt_steps)
            print(i, filt_steps_name)
            filt_step_output = f"02_detected_picture_elements/{filt_steps_name}/{self.folder_name.split('/')[-1]}"
            filt_step_path = f"{filt_step_output}/{self.file_name}"
            last_object = self.steps[-i-2]
            current_object = self.steps[-i-1]

            print(filt_step_path, current_object.type, os.path.exists(filt_step_path))
            last_method = self.steps[-i-2].method
            if last_object.type == 'object_detection' and current_object.type == 'object_verification' and os.path.exists(filt_step_path):
                print(f"The results of pipeline {filt_steps_name} already exist. Loading it")
                self.load_json_detections(filt_step_path)
                self.load_json_elapsed_time(filt_step_path)
                print(self.detections)
                self.object_detector = last_object
                self.object_detector_model_name = self.object_detector.params['model_name']
                min_enum_id = len(self.steps)-1
                break



        for enumid, step_obj in enumerate(self.steps):
            if enumid < min_enum_id:
                print(f"{enumid}<{min_enum_id}")
                continue
            step_type = f"{step_obj.method}_{step_obj.name}"
            print(step_type)
            t1 = tic()
            if step_obj.type == 'region_proposal':
                self.region_proposer = step_obj
                self._run_region_proposal()
            elif step_obj.type == 'object_detection':
                self.object_detector = step_obj
                self._run_object_detection()
            elif step_obj.type == 'object_verification':
                self.object_verifier = step_obj
                self._run_object_verification()
            self.elapsed_time[step_type] = toc(t1)
        return 1
    
    def _preprocess_tag(self):
        """Folder-name suffix describing non-legacy preprocessing ('' = grayscale, 1000 px)."""
        tag = ""
        if self.color:
            tag += "_rgb"
        if self.resize_width is None:
            tag += "_fullres"
        elif self.resize_width != 1000:
            tag += f"_w{self.resize_width}"
        return tag

    def _preprocess_image(self, image):
        raw = image
        image = self._to_rgb_on_white(image) if self.color else image
        if self.resize_width:
            image = self._resize_width(image, width=self.resize_width)
        image = self._apply_clahe_color(image) if self.color else self._apply_clahe(image)

        # Region proposal works on its own copy, made exactly as before
        # (grayscale + CLAHE at proposal_width), because its pixel parameters were
        # tuned on that. Its regions are mapped back onto `image` (see _full_crop).
        work_width = self.proposal_width or image.size[0]
        if self.color or work_width != image.size[0]:
            work = self._to_rgb_on_white(raw)
            self.proposal_image = self._apply_clahe(self._resize_width(work, width=work_width))
        else:
            self.proposal_image = image
        self.proposal_scale = self.proposal_image.size[0] / image.size[0]
        return image

    @staticmethod
    def _to_rgb_on_white(img):
        """RGB image; transparent areas (e.g. RGBA PNGs from plt.imsave) become white."""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, img).convert("RGB")
        return img.convert("RGB")

    CLIP_LIMIT = 2.0
    TILE_GRID_SIZE = (8, 8)

    def _apply_clahe(self,img):
        if img.mode != "L":
            img = img.convert("L")

        # PIL → NumPy
        np_img = np.array(img)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.CLIP_LIMIT, tileGridSize=self.TILE_GRID_SIZE)

        # Apply CLAHE directly
        cl = clahe.apply(np_img)

        # NumPy → PIL
        return Image.fromarray(cl)

    def _apply_clahe_color(self, img):
        """CLAHE on the lightness channel (CIELAB L) only; colours (a, b) are untouched."""
        lab = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.CLIP_LIMIT, tileGridSize=self.TILE_GRID_SIZE)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    
    def save_json(self, file_path):
        detections_list = []
        
        for det in self.detections:    
            
            detection_dict = {}
            detection_dict["element_type"]= "Icon"
            detection_dict["represents"]= det["label"]
            detection_dict["bounding_box"] = [int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])]
            # Size of the image the boxes refer to (the preprocessed image), so the
            # evaluator can map boxes onto the ground-truth image at any resolution.
            detection_dict["image_size"] = list(self.image.size)
            
            if "step_time" in det:
                detection_dict["step_time"] = det["step_time"]
            if "model_answers" in det:
                detection_dict["model_answers"] = det["model_answers"]
            if "text" in det:
                detection_dict["text"] = det["text"]
            detections_list.append(detection_dict)

        if self.scenes:
            ind = 0
            for scene in self.scenes:
                if scene:
                    ind+=1
                    detection_dict = {}
                    detection_dict["element_type"] = "Scene"
                    detection_dict["represents"] =  f"cluster{ind}"
                    detection_dict["bounding_box"] = [int(i) for i in scene]
                    detection_dict["cluster_center"] = self._get_bbox_center(scene)
                    detections_list.append(detection_dict)
        
        with open(file_path.replace('.json', '_detected.json'), "w") as f:
            print(detections_list)
            json.dump(detections_list, f, indent=4)
        
        with open(file_path.replace('.json', '_time.json'),"w") as f:
            json.dump(self.elapsed_time, f, indent=4)


    def load_json_detections(self, file_path):
        """
        Loads the *_detected.json file produced by save_json().
        Returns a list of detection dictionaries with proper types.
        """

        detected_path = file_path.replace('.png', '_detected.json')

        with open(detected_path, "r") as f:
            data = json.load(f)

        detections = []
        scenes = []

        for item in data:
            print(item)
            det = {}

            # Required fields
            det["element_type"] = item.get("element_type")
            det["represents"] = item.get("represents")
            det["label"] = det["represents"]

            # Bounding box (ensure ints)
            bbox = item.get("bounding_box", [])
            det["bounding_box"] = [int(v) for v in bbox]
            det['x1'] = det["bounding_box"][0]
            det['y1'] = det["bounding_box"][1]
            det['x2'] = det["bounding_box"][2]
            det['y2'] = det["bounding_box"][3]

            # Optional fields
            if "step_time" in item:
                det["step_time"] = item["step_time"]

            if "model_answers" in item:
                det["model_answers"] = item["model_answers"]

            if "text" in item:
                det["text"] = item["text"]

            if "cluster_center" in item:
                det["cluster_center"] = item["cluster_center"]

            if det['element_type'] == 'Icon':
                detections.append(det)
            elif det['element_type'] == 'Scene':
                scenes.append(det)

        self.scenes = [scene['bounding_box'] for scene in scenes]
        self.detections = detections

    def load_json_elapsed_time(self, file_path):
        """
        Loads the *_detected.json file produced by save_json().
        Returns a list of detection dictionaries with proper types.
        """

        detected_path = file_path.replace('.png', '_time.json')

        with open(detected_path, "r") as f:
            data = json.load(f)
        times = data
        self.elapsed_time = data


    def _full_crop(self, proposer, sc_id, work_offset):
        """
        Cut region `sc_id` found by `proposer` (which ran on the proposal copy) out of
        the full-resolution model image self.image, keeping only that region's
        foreground (everything else white, as RegionProposer does).
        Returns (crop, bbox_in_self_image) or (None, None).
        """
        bbox = proposer.cropped_scene_locations[sc_id]
        if bbox is None or proposer.cropped_scene_images[sc_id] is None:
            return None, None
        s = self.proposal_scale
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        ids = [i + 1 for i, sid in enumerate(proposer.fg_scene_mapping) if sid == sc_id]
        mask = np.isin(proposer.fg_labels[y1:y2, x1:x2], ids).astype(np.uint8)
        # Region in proposal-copy coordinates of the whole page -> self.image coordinates
        gx1, gy1, gx2, gy2 = x1 + work_offset[0], y1 + work_offset[1], x2 + work_offset[0], y2 + work_offset[1]
        W, H = self.image.size
        fx1, fy1 = max(0, int(np.floor(gx1 / s))), max(0, int(np.floor(gy1 / s)))
        fx2, fy2 = min(W, int(np.ceil(gx2 / s))), min(H, int(np.ceil(gy2 / s)))
        if fx2 <= fx1 or fy2 <= fy1 or mask.size == 0:
            return None, None
        patch = np.array(self.image.crop((fx1, fy1, fx2, fy2)))
        mask = cv2.resize(mask, (fx2 - fx1, fy2 - fy1), interpolation=cv2.INTER_NEAREST)
        if s < 1:  # slightly grow the upscaled mask so anti-aliased stroke edges are kept
            k = 2 * int(np.ceil(1.0 / s)) + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        patch[~mask.astype(bool)] = 255
        return Image.fromarray(patch), [fx1, fy1, fx2, fy2]

    def _collect_regions(self, proposer, work_offset):
        """Proposer results -> (work crops, work locations, full crops, full locations)."""
        out = ([], [], [], [])
        for sc_id, (wimg, wbox) in enumerate(zip(proposer.cropped_scene_images, proposer.cropped_scene_locations)):
            if not (wimg and wbox):
                continue
            full_img, full_box = self._full_crop(proposer, sc_id, work_offset)
            if full_img is None:
                continue
            out[0].append(wimg)
            out[1].append(self._offset_bbox(work_offset, wbox))
            out[2].append(full_img)
            out[3].append(full_box)
        return out

    def _run_region_proposal(self):
        # The proposer always runs on the proposal copy (self.proposal_image and crops of
        # it, kept in self._work_*); the detectors get the matching full-resolution crops
        # of self.image (self.cropped_scene_images / self.cropped_scene_locations).
        if len(self.cropped_scene_images)==0:
            print("****Performing first level of region proposal")
            self.region_proposer.set_image(self.proposal_image)
            self.region_proposer.run()
            (self._work_images, self._work_locations,
             self.cropped_scene_images, self.cropped_scene_locations) = self._collect_regions(self.region_proposer, (0, 0))
            self.scenes = self.cropped_scene_locations.copy()
        else:
            print("****Performing further level of region proposal")
            new = ([], [], [], [])
            for img_id, (img, offset) in enumerate(zip(self._work_images, self._work_locations)):

                if img and offset:
                    self.region_proposer.set_image(img)
                    try:
                        self.region_proposer.run()
                        for lst, items in zip(new, self._collect_regions(self.region_proposer, offset)):
                            lst.extend(items)
                        assert (len(new[2]) == len(new[3]))
                    except Exception as e:
                        # print(f"Skipping Image of size {img.size} Due to error: {e}")
                        # traceback.print_exc()
                        pass
            (self._work_images, self._work_locations,
             self.cropped_scene_images, self.cropped_scene_locations) = new
        for img_id, (img, offset) in enumerate(zip(self.cropped_scene_images, self.cropped_scene_locations)):
            if img:
                os.makedirs(f"{self.result_folder_path}/tmp/{self.file_name}_{self.region_proposer.method}_{self.region_proposer.name}", exist_ok=True)
                img.save(f"{self.result_folder_path}/tmp/{self.file_name}_{self.region_proposer.method}_{self.region_proposer.name}/region_{img_id}_{offset[0]}-{offset[1]}.png")
    
    def _run_object_detection(self):
        if hasattr(self,  'region_proposer'):
            self.detections = []
            for img_id, (img, offset) in enumerate(zip(self.cropped_scene_images, self.cropped_scene_locations)):
                if img:
                    self.object_detector.set_image(img, offset=offset)
                    self.object_detector.run()
                    self.detections.extend(self.object_detector.modified_detections)
        elif self.object_detector.method == 'llm':
            self.object_detector.set_image(self.image)
            self.object_detector.run()
            if self.object_detector.detections:
                self.detections = self.object_detector.detections
        else:
            raise ModuleNotFoundError("No Region Proposer found. please add it to the pipeline before the object detection or use an llm object detector")
        labels = "-".join([i['label'] for i in self.detections])
    
    
    def _run_object_verification(self):
        if hasattr(self,  'object_detector'):
            self.verified_detections = []
            if self.detections:
                tol = 5
                bboxes = [[det['x1'],det['y1'],det['x2'],det['y2']] for det in self.detections]
                # The detector's model name and the 'represents' key are only set when
                # detections were loaded from an earlier run; derive them for fresh detections too.
                model_name = getattr(self, 'object_detector_model_name', None) or \
                             self.object_detector.params.get('model_name', self.object_detector.method)
                known_labels = [{model_name: det.get('represents', det.get('label'))} for det in self.detections]
                verified_objects = self.object_verifier.run(self.image, bboxes, tol, known_labels)
                for det, veridit in zip(self.detections, verified_objects):
                    det['text'] = veridit['text']
                    det['model_answers'] = veridit['model_answers']
                    det['detector_label'] = det['label']
                    det['label'] = veridit['final_label']

    def draw_result(self):
        img_path = f"{self.result_folder_path}/{self.file_name}"
        json_path = f"{self.result_folder_path}/{self.file_name.split('.')[0]+'.json'}"
        if self.detections:
            self.object_detector.draw_detections(image=self.image, detections=self.detections, scenes = self.scenes, file_path=img_path)
            self.save_json(file_path = json_path)
            print(f"results saved in {img_path}")
        else:
            raise ValueError("Detection not found. Run the pipelinefirst.")
    
    def _obtain_steps_name(self, steps):
        get_model_name = lambda x: x.params["model_name"].replace(':','-')
        steps_name = '_'.join([f"{i.method}_{get_model_name(i)}_{i.name}" if (i.method=='llm') else f"{i.method}_{i.name}" for i in steps])
        return steps_name + self.preprocess_tag
    
    def _offset_bbox(self, offset, bbox):
        return [bbox[0]+offset[0], bbox[1]+offset[1], bbox[2]+offset[0], bbox[3]+offset[1]]

    def _resize_width(self, image, width):
        w, h = image.size
        new_w = width
        new_h = int(h * (new_w / w))
        out = image.resize((new_w, new_h), Image.BICUBIC)
        return out
    
    def _get_bbox_center(self, bbox):
        print(bbox)
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2, (y1+y2)/2)

if __name__ == "__main__":
    PREDEFINED_LABELS = ["person","tree","car","calender", "cat", "pounds","house", "church"]
    scene_proposal_kde = RegionProposer(method="kde", params={"kde_thresh":0.3}, name='scene')   
    icon_proposal = RegionProposer(method="dbscan", params={"epsilon":1, "min_sample":1}, name='object')   
    object_detection_clip = ObjectDetector(method="Clip", params={"predefined_labels":PREDEFINED_LABELS}, name='label')
    object_detection_llm_big = ObjectDetector(method="llm", params={"model_name":"gemma4:31b"}, name='icons')
    object_verification_ensemble_small = ObjectVerifier(models=["qwen3.5:9b","llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "gemma4:latest"], judge = "llama3.1:70b", method="ensemble", name="verify", use_known_label = True)
    # pipeline = Pipeline([scene_proposal, object_detection_llm_big, object_verifier])
    pipeline_scene_big_verify = Pipeline([scene_proposal_kde, object_detection_llm_big, object_verification_ensemble_small])

    
    # pipeline.run("test_data/synthdata/image_000.png")
    pipeline_scene_big_verify.run("01_synthetic_data/02_to_05_clusters_05_to_20_classes_03_to_05_images_per_class_20_minimum_icons_00_to_00_percent_overlap_00_minimum_overlap_images_3.0_max_scaling_factor/image_092.png")
    pipeline_scene_big_verify.draw_result()
