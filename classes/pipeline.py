"""
File: pipeline.py
Path: paper_sketchobjectdetection/classes/pipeline.py

Purpose:
    class to perform a pipeline of regionproposer and object detection on an image.

Author: Allmin
Created: 2026-06-05
"""
from classes.region_proposal import RegionProposer
from classes.object_detection import ObjectDetector
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
        self._obtain_step_name()
        if name:
            self.name = name
        else:
            self.name = self.steps_name
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
        

        
        # if os.path.exists(self.result_folder_path):
        #     shutil.rmtree(self.result_folder_path)
        os.makedirs(self.result_folder_path, exist_ok=True)
        self.raw_image = image
        self.image = self._preprocess_image(self.raw_image)
        current_output = None
        for enumid, step_obj in enumerate(self.steps):
            step_type = f"{step_obj.method}_{step_obj.name}"
            t1 = tic()
            if step_obj.type == 'region_proposal':
                self.region_proposer = step_obj
                self._run_region_proposal()
            elif step_obj.type == 'object_detection':
                self.object_detector = step_obj
                self._run_object_detection()
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
            detection_dict["step_time"] = det["step_time"]
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
        # os.makedirs(f"{self.result_folder_path}/02_{self.object_detector.method}_{self.object_detector.name}", exist_ok=True)
        # self.object_detector.image.save(f"{self.result_folder_path}/02_{self.object_detector.method}_{self.object_detector.name}/object_{img_id}_{offset[0]}-{offset[1]}_{labels}.png")
    
    def draw_result(self):
        img_path = f"{self.result_folder_path}/{self.file_name}"
        json_path = f"{self.result_folder_path}/{self.file_name.split('.')[0]+'.json'}"
        if self.detections:
            self.object_detector.draw_detections(image=self.image, detections=self.detections, scenes = self.scenes, file_path=img_path)
            self.save_json(file_path = json_path)
            print(f"results saved in {img_path}")
        else:
            raise ValueError("Detection not found. Run the pipelinefirst.")
    
    def _obtain_step_name(self):
        get_model_name = lambda x: x.params["model_name"].replace(':','-')
        self.steps_name = '_'.join([f"{i.method}_{get_model_name(i)}_{i.name}" if (i.method=='llm') else f"{i.method}_{i.name}" for i in self.steps])
    
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
    scene_proposal = RegionProposer(method="dbscan", params={"epsilon":15, "min_sample":1}, name='scene')   
    icon_proposal = RegionProposer(method="dbscan", params={"epsilon":1, "min_sample":1}, name='object')   
    object_detection_clip = ObjectDetector(method="Clip", params={"predefined_labels":PREDEFINED_LABELS}, name='label')
    pipeline = Pipeline([scene_proposal, icon_proposal, object_detection_clip])
    pipeline.run("test_data/synthdata/image_000.png")
    pipeline.draw_result()
