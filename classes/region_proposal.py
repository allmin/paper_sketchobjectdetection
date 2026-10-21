"""
File: region_proposal.py
Path: paper_sketchobjectdetection/classes/region_proposal.py

Purpose:
    class to perform region proposal on image.

Author: Allmin
Created: 2026-06-05
"""

import shutil
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from classes.helper_functions import bbox_match, sketch_to_binary, connected_components, find_clusters, cc2bbox, find_outer_bbox
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
import datetime;




# App defaults: KDE density slicing followed by region refinement.
DEFAULT_METHOD = "kde"
DEFAULT_REFINE = True
DEFAULT_REFINE_OVERLAP = 0.8


def overlap_ratio(a, b):
    """Share of the smaller box that lies inside the other box:
    intersection area / area of the smaller box (1.0 = fully contained)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    if smaller <= 0:
        return 0.0
    return iw * ih / smaller


def refine_regions(boxes, overlap_threshold):
    """Region refinement: whenever the overlap ratio of two boxes is larger than
    `overlap_threshold` they are replaced by their union; the scan restarts from
    the beginning until no pair can be merged any more."""
    boxes = [list(map(float, b)) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if overlap_ratio(boxes[i], boxes[j]) > overlap_threshold:
                    a, b = boxes[i], boxes[j]
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


class RegionProposer:
    def __init__(self, method=DEFAULT_METHOD, params=None, name="",
                 refine=DEFAULT_REFINE, refine_overlap=DEFAULT_REFINE_OVERLAP):
        self.method = method
        self.params = dict(params) if params else {}   # copy: never mutate caller's dict
        self.refine = refine
        self.refine_overlap = refine_overlap
        self.expected_input = np.ndarray
        self.output_type = dict
        self.output = {}
        self.name = name
        self.type = f"region_proposal"
        
    
    def set_image(self,image):
        #if input is a PIL Image, convert to numpy array:
        # print(f"input image type = {type(image)}")
        if type(image) != np.ndarray: 
            # print("converting image to array")
            image = np.array(image)
        self.image = image
    
    def run(self):
        print(f"Running region proposal using {self.method}: {self.params} on Image of size:{self.image.shape}")
        if self.image is None:
            raise ValueError("No image set. Call set_image(image) first.")
        self.generate_regions()
        # self._save_img_array(self.cropped_scene_images, tag='cropped_regions')

    def generate_regions(self):
        if self.method.lower() == "dbscan":
            self.param_specs = {"min_sample":{"description":"minimum number of sample to assign a cluster", 
                                                    "min":1, "max":1000, "recommended":3},
                                "epsilon":{"description":"", 
                                                    "min":1, "max":1000, "recommended":20},
                                    }
            self._apply_default_parameters()
            self.binary_image = sketch_to_binary(self.image, use_fill=True)
            _,self.fg_labels,self.fg_stats,self.centroids = connected_components(self.binary_image)
            # self.scene_bboxes = self._centroid_to_box(self.centroids)
            # self.clusters = find_clusters(self.centroids, method="dbscan", params=self.params)
            # self._match_and_expand_bboxes()
            self.fg_points = np.array(np.where(self.binary_image)).T
            print(self.fg_points)

            self.clusters = find_clusters(self.fg_points, method="dbscan", params=self.params)
            print(len(self.clusters))
            self.scene_bboxes = self._scene_bboxes_from_clusters()   # ← was _centroid_to_box
            self._refine_scene_bboxes()
            self._match_and_expand_bboxes()
            
            print("Completed region proposal")
            
        elif self.method.lower() == "kde":
            self.param_specs = {"kde_thresh":{"description":"the level at which the normalised kde maps need to be sliced",
                                                "min":0, "max":1, "recommended":0.3},
                                "bandwidth_method":{"description":"The method used to calculate the estimator bandwidth.",
                                                "min":"scott", "max":"scott", "recommended":"scott"},
                                "grid_resolution":{"description":"resolution of the KDE. higher=slower but better spatial resolution.",
                                                "min":1, "max":100, "recommended":50}}
            self._apply_default_parameters()
            self.binary_image = sketch_to_binary(self.image)
            _,self.fg_labels,self.fg_stats,self.centroids = connected_components(self.binary_image)
            self.coords = self._get_coords_binary(self.binary_image)
            self._compute_kde_density()
            self._threshold_density()
            self.scene_bboxes = self._extract_scene_bboxes()
            self._refine_scene_bboxes()
            self._match_and_expand_bboxes()
            print("finish")
    
    def _scene_bboxes_from_clusters(self):
        """Convert DBSCAN cluster labels into one bbox per cluster."""
        unique_labels = set(self.clusters)
        scene_bboxes = []
        # centroids shape: (N, 2), clusters shape: (N,)
        for label in unique_labels:
            # cluster_centroids = self.centroids[mask]  # points in this cluster
            # xs, ys = cluster_centroids[:, 0], cluster_centroids[:, 1]
            relevant_index = np.where(self.clusters == label)
            # print(label, relevant_index)
            relevant_points = np.array([self.fg_points[i] for i in relevant_index[0]])
            xs, ys = relevant_points[:,0], relevant_points[:,1]
            scene_bboxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        return scene_bboxes

    def _refine_scene_bboxes(self):
        """Region refinement stage (on by default; disable with refine=False)."""
        if self.refine and self.scene_bboxes:
            self.scene_bboxes = refine_regions(self.scene_bboxes, self.refine_overlap)

    def _apply_default_parameters(self):
        for (param,info) in self.param_specs.items():
            if param not in self.params:
                print(f"Parameter {param} not specified, using default value of {info['recommended']}")
                self.params[param] = info["recommended"]
                    
    
    def _merge_component_bboxes(self):
        self.new_scene_bboxes = []
        for sc_id in range(len(self.scene_bboxes)):
            assigned = [self.fg_bboxes[i] for i, sid in enumerate(self.fg_scene_mapping) if sid == sc_id]
            bbox = find_outer_bbox(assigned)
            self.new_scene_bboxes.append(bbox)
        

    def get_scene_images(self):
        self.scene_only_images = []
        for sc_id in range(len(self.scene_bboxes)):
            # Create an empty black image
            scene_img = self.image.copy()
            # Find all fg components belonging to this scene
            assigned_fg_ids = [i for i, sid in enumerate(self.fg_scene_mapping) if sid == sc_id]
            # Mask of all pixels NOT belonging to this scene
            mask_keep = np.zeros(self.fg_labels.shape, dtype=bool)
            for fg_id in assigned_fg_ids:
                mask_keep |= (self.fg_labels == (fg_id + 1))
            # Zero out everything except the assigned foreground
            scene_img[~mask_keep] = 255
            self.scene_only_images.append(Image.fromarray(scene_img))
    
    def crop_scene_images(self):
        self.cropped_scene_images = []
        self.cropped_scene_locations = []
        for sc_id, img in enumerate(self.scene_only_images):
            bbox = self.new_scene_bboxes[sc_id]
            if bbox is None:
                self.cropped_scene_images.append(None)
                self.cropped_scene_locations.append(None)
                continue
            x1, y1, x2, y2 = bbox
            cropped = img.crop((x1, y1, x2, y2))
            self.cropped_scene_images.append(cropped)
            self.cropped_scene_locations.append(bbox)

    def _match_and_expand_bboxes(self):
        self.fg_bboxes = cc2bbox(self.fg_stats)
        self.fg_scene_mapping = bbox_match(self.fg_bboxes, self.scene_bboxes)
        self._merge_component_bboxes()
        self.get_scene_images()
        self.crop_scene_images()
    
    def _save_img_array(self, image_array, tag = ''):
        # image_array is a list of PIL Images
        path_folder = f"results/02_{self.name}-{self.type}_using_{self.method}_{tag}"
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.makedirs(path_folder, exist_ok=True)
        for idx, img in enumerate(image_array):
            if not img:
                continue
            img_path = os.path.join(path_folder, f"img_{idx}.png")
            img.save(img_path)
            print(f"saved in {img_path}")
    
    def _centroid_to_box(self,clusters):
        return [[a,b,a,b] for (a,b) in clusters.tolist()]

    def _compute_kde_density(self):
        coords= self.coords
        kde = gaussian_kde(coords, bw_method=self.params["bandwidth_method"])
        grid_res = self.params["grid_resolution"]
        xmin, xmax = coords[0].min(), coords[0].max()
        ymin, ymax = coords[1].min(), coords[1].max()
        self.xyrange = [xmin, ymin, xmax, ymax]
        yy, xx = np.mgrid[ymin:ymax:grid_res*1j, xmin:xmax:grid_res*1j]
        grid = np.vstack([xx.ravel(), yy.ravel()])
        self.density = kde(grid).reshape(xx.shape)
    
    def _threshold_density(self):
        kde_thresh = self.params["kde_thresh"]
        d_min, d_max = self.density.min(), self.density.max()
        if d_max - d_min == 0:
            self.kde_mask = None
        else:
            density_norm = (self.density - d_min) / (d_max - d_min)
            mask = (density_norm > kde_thresh).astype(np.uint8) * 255
            self.kde_mask = mask
    
    def _extract_scene_bboxes(self):
        grid_res = self.params["grid_resolution"]
        if self.kde_mask is None:   # flat density: one scene covering all ink
            return [list(self.xyrange)]
        num_labels, labels, stats, _ = connected_components(self.kde_mask)
        scene_bboxes = cc2bbox(stats, grid_res, self.xyrange)
        return scene_bboxes
        
    def _get_coords_binary(self,binary_image):
        ys, xs = np.where(binary_image)
        return np.vstack([xs, ys])


if __name__ == "__main__":
    image = Image.open("test_data/realdata/SampleRP.png")
    rp = RegionProposer()   # KDE (kde_thresh 0.3) + region refinement (overlap 0.8)
    rp.set_image(image)
    rp.run()
    rp._save_img_array(rp.cropped_scene_images ,tag='edgebased')





