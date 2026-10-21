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
    def __init__(self, steps, name=''):
        self.elapsed_time = {}
        self.steps = steps
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
    
    def _preprocess_image(self, image):
        image = self._resize_width(image, width=1000)
        image = self._apply_clahe(image)
        return image
    
    def _apply_clahe(self,img):
        CLIP_LIMIT = 2.0
        TILE_GRID_SIZE = (8, 8)
        if img.mode != "L":
            img = img.convert("L")

        # PIL → NumPy
        np_img = np.array(img)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

        # Apply CLAHE directly
        cl = clahe.apply(np_img)

        # NumPy → PIL
        return Image.fromarray(cl)
    
    def save_json(self, file_path):
        detections_list = []
        
        for det in self.detections:    
            
            detection_dict = {}
            detection_dict["element_type"]= "Icon"
            detection_dict["represents"]= det["label"]
            detection_dict["bounding_box"] = [int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])]
            
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


    def _run_region_proposal(self):
        if len(self.cropped_scene_images)==0:
            print("****Performing first level of region proposal")
            self.region_proposer.set_image(self.image)
            self.region_proposer.run()
            self.cropped_scene_images.extend(self.region_proposer.cropped_scene_images)
            self.cropped_scene_locations.extend(self.region_proposer.cropped_scene_locations)
            self.scenes = self.cropped_scene_locations.copy()
        else:
            print("****Performing further level of region proposal")
            new_cropped_scene_images = []
            new_cropped_scene_locations = []
            for img_id, (img, offset) in enumerate(zip(self.cropped_scene_images, self.cropped_scene_locations)):

                if img and offset:
                    self.region_proposer.set_image(img)
                    try:
                        self.region_proposer.run()
                        for cimage, bbox in zip(self.region_proposer.cropped_scene_images, self.region_proposer.cropped_scene_locations):
                            if cimage and bbox:
                                new_cropped_scene_images.append(cimage)
                                new_cropped_scene_locations.append(self._offset_bbox(offset, bbox))
                        assert (len(new_cropped_scene_images) == len(new_cropped_scene_locations))
                    except Exception as e:
                        # print(f"Skipping Image of size {img.size} Due to error: {e}")
                        # traceback.print_exc()
                        pass
            self.cropped_scene_images = new_cropped_scene_images
            self.cropped_scene_locations = new_cropped_scene_locations
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
        return steps_name
    
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
