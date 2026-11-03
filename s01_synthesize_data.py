import json
import math
import os
import random
from dataclasses import dataclass
from itertools import combinations, product

import cv2
import kagglehub
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

# setting random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Separate seed for choosing overlap levels, so adding/removing levels does not
# change which icons, classes and positions the main random stream produces.
OVERLAP_LEVEL_SEED = 2024


def _fmt_pct(value):
    """Format an overlap percentage for labels/folder names: 5 -> '05', 2.5 -> '2.5'."""
    return f"{int(value):02d}" if float(value).is_integer() else f"{value:g}"


def make_overlap_levels(overlap_ranges, min_num_overlap_images):
    """
    Turn a list of (min %, max %) overlap ranges into level dicts used during placement.

    Each level carries its own min_num_overlap_images; it is forced to 0 for a
    zero-overlap level (as the original single-level code did).
    """
    if len(overlap_ranges) == 0:
        raise ValueError("At least one overlap range is required")
    levels = []
    for low, high in overlap_ranges:
        if high < low:
            raise ValueError(f"Overlap range ({low}, {high}): max is smaller than min")
        levels.append(
            {
                "min": low,
                "max": high,
                "min_num": 0 if high == 0 else min_num_overlap_images,
                "label": f"{_fmt_pct(low)}-{_fmt_pct(high)}",
            }
        )
    return levels


def balanced_level_assignment(n, levels, rng):
    """
    Assign n items to levels so each level is used equally often (+-1), in random order.

    The levels are shuffled before cycling, so when n < len(levels) (e.g. an image
    with 2 clusters and 4 levels) a random subset is used rather than always the
    first levels.
    """
    order = list(levels)
    rng.shuffle(order)
    assigned = [order[i % len(order)] for i in range(n)]
    rng.shuffle(assigned)
    return assigned


def get_dataset_path():
    # Download the TU Berlin dataset if it is not already available locally or is outdated.
    path = kagglehub.dataset_download("zara2099/tu-berlin-hand-sketch-image-dataset")
    path = os.path.join(path, "TUBerlin", "png_ready")
    print("Path to dataset files:", path)
    return path


class MultiIconDataset:
    def __init__(self, datapath, overlap_logic="bounding_box", max_attempts=1000):
        self.path = datapath
        self.MAX_ATTEMPTS = max_attempts
        self.overlap_logic = overlap_logic
        self.canvas_width = 4950
        self.canvas_height = 3510
        self.empty_canvas = 255 * np.ones(
            (self.canvas_height, self.canvas_width, 3), dtype=np.uint8
        )
        self.image_cache = {}

    # ------------------------------------------------------------------
    # Overlap as COVERED AREA
    #
    # The overlap of icon A is the share of A that is covered by at least one
    # other icon of the same cluster:
    #
    #     overlap_p(A) = |A  ∩  union(other icons in A's cluster)| / |A| * 100
    #
    # A region covered by two or more other icons is counted once, so the value
    # is always within 0..100. "A" is the icon's foreground mask
    # (overlap_logic="pixel") or its whole bounding box ("bounding_box").
    #
    # Each placed icon keeps a boolean "covered" array in its own frame. When a
    # new icon is accepted, the covered arrays (and overlap_p) of the icons it
    # touches are updated, so later placements are checked against current values.
    # ------------------------------------------------------------------

    def _occupancy(self, icon_meta):
        """Boolean array (icon frame) of the pixels that count as the icon's area."""
        if self.overlap_logic == "pixel":
            return icon_meta["foreground_mask"].astype(bool)
        if self.overlap_logic == "bounding_box":
            return np.ones(icon_meta["image"].shape[:2], dtype=bool)
        raise ValueError(f"Unknown overlap logic: {self.overlap_logic}")

    def _prepare_overlap_state(self, icon_meta):
        """Cache occupancy and area on the icon (once) before it is placed."""
        if "occupancy" not in icon_meta:
            icon_meta["occupancy"] = self._occupancy(icon_meta)
            icon_meta["icon_area"] = int(icon_meta["occupancy"].sum())

    @staticmethod
    def _intersection(a, b):
        """Canvas-coordinate intersection (x0, y0, x1, y1) of two boxes, or None."""
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x0 >= x1 or y0 >= y1:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _local_slices(bbox, region):
        """Slices selecting a canvas region inside an icon's own frame."""
        x0, y0, x1, y1 = region
        return (slice(y0 - bbox[1], y1 - bbox[1]), slice(x0 - bbox[0], x1 - bbox[0]))

    @staticmethod
    def _pct(covered_px, area):
        return covered_px / area * 100 if area > 0 else 0.0

    # def load_image(self, path):
    #     if path not in self.image_cache.keys():
    #         img = mpimg.imread(path)
    #         if img.max() <= 1.0:
    #             img = (img * 255).astype(np.uint8)
    #         foreground_mask = self.sketch_to_binary(img)
    #         self.image_cache[path] = (img, foreground_mask)
    #     return self.image_cache[path]
    
    def load_image(self, path):
        if path not in self.image_cache:
            img = mpimg.imread(path)

            # Normalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            # Binary mask
            foreground_mask = self.sketch_to_binary(img)

            # --- Compute bounding box of foreground ---
            ys, xs = np.where(foreground_mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                # No foreground → keep original
                cropped_img = img
                cropped_mask = foreground_mask
            else:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                # --- Crop image and mask ---
                cropped_img = img[y_min:y_max+1, x_min:x_max+1]
                cropped_mask = foreground_mask[y_min:y_max+1, x_min:x_max+1]

            self.image_cache[path] = (cropped_img, cropped_mask)

        return self.image_cache[path]


    def _display_classes(self):
        classes = os.listdir(self.path)
        print("Number of classes in the dataset:", len(classes))
        print("Classes in the dataset:", classes[:10])

    def _list_few_classes(self, num_classes=5):
        classes = os.listdir(self.path)
        print("Number of classes in the dataset:", len(classes))
        print("Classes in the dataset:", classes[:num_classes])

    def _display_few_images(self, num_classes=5, num_images_per_class=5):
        classes = os.listdir(self.path)
        print("Number of classes in the dataset:", len(classes))
        print("Classes in the dataset:", classes[:num_classes])
        for class_name in classes[
            :num_classes
        ]:  # Display images from the first few classes
            class_path = os.path.join(self.path, class_name)
            image_files = os.listdir(class_path)
            for image_file in image_files[
                :num_images_per_class
            ]:  # Display the first few images of each class
                image_path = os.path.join(class_path, image_file)
                img, foreground_mask = self.load_image(image_path)
                image_shape = img.shape
                print(
                    f"Class: {class_name}, Image: {image_file}, Shape_CV: {image_shape}"
                )
                plt.imshow(img)
                plt.title(f"Class: {class_name}")
                plt.axis("off")
                plt.show()

    def _create_dataset_dict(self):
        classes = os.listdir(self.path)
        # a dictionary of class names and their corresponding image file paths
        dataset = {}
        for class_name in classes:
            class_path = os.path.join(self.path, class_name)
            image_files = os.listdir(class_path)
            dataset[class_name] = [
                os.path.join(class_path, image_file) for image_file in image_files
            ]
        print("Dataset dictionary created with class names and image file paths.")
        self.dataset = dataset
        # Summary of classnames and number of images in each class
        for class_name, image_paths in self.dataset.items():
            print(f"Class: {class_name}, Number of images: {len(image_paths)}")

        print("Dataset is ready for use.")

    def sketch_to_binary(self, mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # --- Threshold robustly ---
        _, sketch = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)
        sketch = sketch.astype(bool)

        # --- STRONG dilation to remove gaps ---
        kernel = np.ones((11, 11), np.uint8)
        sketch_closed = cv2.morphologyEx(
            sketch.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
        ).astype(bool)

        inside = binary_fill_holes(sketch_closed)

        # make a mega side-by-side image of the input sketch and the inside binary mask and save it as a binary image with random file name in temp folder
        side_by_side = np.hstack(
            (sketch.astype(np.uint8) * 255, inside.astype(np.uint8) * 255)
        )
        os.makedirs("temp/01_temp_binary_foreground_images", exist_ok=True)
        filename = f"temp/01_temp_binary_foreground_images/fill_{random.randint(1000,9999)}.png"
        cv2.imwrite(filename, side_by_side)

        return inside.astype(np.uint8)

    def calculate_overlap_percentage(self, icon_bbox, icon_meta, existing_bbox_list):
        """
        Covered-area overlap of a candidate placement with icons of the same cluster.

        Returns (overlap_percentage, covered, overlap_info):
          overlap_percentage  % of the candidate icon covered by existing icons
          covered             boolean array (icon frame) of those covered pixels
          overlap_info        one entry per existing icon the candidate touches:
                              (existing_meta, slices, newly_covered, new_percent)
                              where new_percent is that icon's covered-area overlap
                              AFTER the candidate is added. Pass the whole result to
                              commit_overlap() once the placement is accepted.
        """
        self._prepare_overlap_state(icon_meta)
        occ = icon_meta["occupancy"]
        covered = None
        overlap_info = []

        for existing_bbox, existing_meta in existing_bbox_list:
            if existing_meta["cluster"] != icon_meta["cluster"]:
                continue  # only icons of the same cluster count
            region = self._intersection(icon_bbox, existing_bbox)
            if region is None:
                continue
            new_sl = self._local_slices(icon_bbox, region)
            ex_sl = self._local_slices(existing_bbox, region)
            new_occ = occ[new_sl]
            ex_occ = existing_meta["occupancy"][ex_sl]

            # candidate pixels covered by this existing icon (union over all of them)
            hit = new_occ & ex_occ
            if hit.any():
                if covered is None:
                    covered = np.zeros_like(occ)
                covered[new_sl] |= hit

            # existing-icon pixels that the candidate would newly cover
            newly = hit & ~existing_meta["covered"][ex_sl]
            n_new = int(newly.sum())
            if n_new > 0:
                new_pct = self._pct(
                    existing_meta["covered_px"] + n_new, existing_meta["icon_area"]
                )
                overlap_info.append((existing_meta, ex_sl, newly, new_pct))

        if covered is None:
            covered = np.zeros_like(occ)
        overlap_percentage = self._pct(int(covered.sum()), icon_meta["icon_area"])
        return overlap_percentage, covered, overlap_info

    def commit_overlap(self, icon_meta, overlap_percentage, covered, overlap_info):
        """Record an accepted placement: set the new icon's overlap state and
        update the overlap of every existing icon it covers."""
        icon_meta["covered"] = covered
        icon_meta["covered_px"] = int(covered.sum())
        icon_meta["overlap_p"] = overlap_percentage
        for existing_meta, ex_sl, newly, new_pct in overlap_info:
            existing_meta["covered"][ex_sl] |= newly
            existing_meta["covered_px"] += int(newly.sum())
            existing_meta["overlap_p"] = new_pct

    def recompute_final_overlaps(self, placed):
        """
        Recompute every placed icon's covered-area overlap from the finished
        layout (independent of the incremental bookkeeping) and store it in
        overlap_p. Returns the largest difference to the incremental value,
        which should be 0.
        """
        max_diff = 0.0
        for meta in placed:
            bbox = meta["bounding_box"]
            covered = np.zeros_like(meta["occupancy"])
            for other in placed:
                if other is meta or other["cluster"] != meta["cluster"]:
                    continue
                region = self._intersection(bbox, other["bounding_box"])
                if region is None:
                    continue
                sl = self._local_slices(bbox, region)
                other_sl = self._local_slices(other["bounding_box"], region)
                covered[sl] |= meta["occupancy"][sl] & other["occupancy"][other_sl]
            final = self._pct(int(covered.sum()), meta["icon_area"])
            max_diff = max(max_diff, abs(final - meta["overlap_p"]))
            meta["overlap_p"] = final
        return max_diff

    def insert_icon_on_canvas(self, canvas, icon, x, y):

        icon_height, icon_width = icon.shape[:2]
        # print(f"inserting icon: {x}, {y}, {icon_height}, {icon_width}")
        canvas[y : y + icon_height, x : x + icon_width, :] = np.minimum(
            canvas[y : y + icon_height, x : x + icon_width, :], icon
        )
        return canvas

    def _sample_classes(self, num_classes):
        return random.sample(list(self.dataset.keys()), num_classes)

    def specify_synthetic_data_subsets(self, Config):
        data_synth = {}
        for synth_num in range(Config.num_synthetic):
            num_class = random.randint(Config.num_class[0], Config.num_class[1])
            selected_classes = self._sample_classes(num_class)
            print("Selected classes for the new image:", selected_classes)
            data_synth[synth_num] = []
            while len(data_synth[synth_num]) < Config.minimum_icons:
                for class_name in selected_classes:
                    image_paths = self.dataset[class_name]
                    num_images = random.randint(*Config.num_images_per_class)
                    selected_images = random.sample(image_paths, num_images)
                    for selected_image in selected_images:
                        scale_factor = random.uniform(1, Config.max_scale_factor)
                        icon, foreground_mask = self.load_image(selected_image)
                        # scale icon pixels to be between 0 and 255
                        if icon.max() <= 1.0:
                            icon = (icon * 255).astype(np.uint8)
                        icon_height, icon_width = icon.shape[:2]
                        scaled_size = (
                            int(icon_width * scale_factor),   # width
                            int(icon_height * scale_factor),  # height
                        )
                        icon_scaled = cv2.resize(icon, scaled_size)
                        foreground_mask_scaled = cv2.resize(
                            foreground_mask, scaled_size, interpolation=cv2.INTER_NEAREST
                        )
                        data_synth[synth_num].append(
                            {
                                "represents": class_name,
                                "image_path": selected_image,
                                "image": icon_scaled,
                                "foreground_mask": foreground_mask_scaled,
                                "scale_factor": scale_factor,
                                "scaled_size": scaled_size,  # already (width, height)
                            }
                        )
        print(
            "Data formulation complete. Created",
            Config.num_synthetic,
            "new data subsets with",
            Config.num_class,
            "classes each.",
        )
        return data_synth

    def assign_clusters(self, data_synth):
        cluster_center_dict = {}
        candidate_cluster_centers = []
        for i in range(self.num_clusters * 5):
            random_cluster_center = (
                random.randint(
                    0 + self.minimum_intercluster_distance,
                    self.canvas_width - self.minimum_intercluster_distance,
                ),
                random.randint(
                    0 + self.minimum_intercluster_distance,
                    self.canvas_height - self.minimum_intercluster_distance,
                ),
            )
            candidate_cluster_centers.append(random_cluster_center)

        # matrix measuring eucledian distance between each candidate cluster center
        distance_matrix = np.zeros(
            (len(candidate_cluster_centers), len(candidate_cluster_centers))
        )
        for i in range(len(candidate_cluster_centers)):
            for j in range(i + 1, len(candidate_cluster_centers)):
                distance = np.sqrt(
                    (candidate_cluster_centers[i][0] - candidate_cluster_centers[j][0])
                    ** 2
                    + (
                        candidate_cluster_centers[i][1]
                        - candidate_cluster_centers[j][1]
                    )
                    ** 2
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        selected_cluster_centers = []
        end_coordinates = [
            [0, 0],
            [0, self.canvas_height],
            [self.canvas_width, 0],
            [self.canvas_width, self.canvas_height],
        ]
        for i in range(len(candidate_cluster_centers)):
            if len(selected_cluster_centers) >= self.num_clusters:
                break
            if all(
                distance_matrix[i, j] >= self.minimum_intercluster_distance
                for j in range(len(candidate_cluster_centers))
                if j != i and j in (selected_cluster_centers + end_coordinates)
            ):
                selected_cluster_centers.append(i)
                cluster_center_dict[len(selected_cluster_centers) - 1] = (
                    candidate_cluster_centers[i]
                )
        print(
            "Total candidate cluster centers generated:", len(candidate_cluster_centers)
        )

        for synth_num, icons in data_synth.items():
            for icon_meta in icons:
                icon_meta["cluster"] = random.choice(list(cluster_center_dict.keys()))
                icon_meta["cluster_center"] = cluster_center_dict[icon_meta["cluster"]]
        print(f"Assigned cluster labels to each icon in the synthetic image.")
        return data_synth

    def is_icon_within_canvas(self, icon_bbox):
        if (
            icon_bbox[0] >= 0
            and icon_bbox[1] >= 0
            and icon_bbox[2] <= self.canvas_width
            and icon_bbox[3] <= self.canvas_height
        ):
            return True
        else:
            return False

    def get_paste_location(
        self,
        icon_meta,
        existing_bbox_list,
        level,
    ):
        """
        Find a paste location for one icon.

        `level` is the overlap level (see make_overlap_levels) that applies to this
        icon's image or cluster: min/max overlap in percent and min_num, the number
        of icons already in the cluster before the max-overlap limit is enforced.
        Overlap is covered area (see calculate_overlap_percentage), only measured
        against icons in the same cluster; all icons of a cluster share one level.

        A location is accepted when the icon lies on the canvas and
          * its own overlap is <= max, and
          * every existing icon it touches stays <= max after the addition
            (both max checks are relaxed while the cluster holds <= min_num icons),
          * its own overlap is >= min (if the cluster already has icons).
        Overlap only grows as icons are added, so min stays satisfied, and the
        neighbour check keeps every icon <= max in the final picture.

        Returns (p_x, p_y, bbox, overlap_percent, skip_icon, overlap_state), where
        overlap_state is (covered, overlap_info) for commit_overlap().
        """
        eps = 1e-9  # tolerance for float comparisons of percentages
        min_overlap = level["min"]
        max_overlap = level["max"]
        min_num_overlap_images = level["min_num"]
        icon_height, icon_width = icon_meta["image"].shape[:2]
        icon_cluster_center = icon_meta["cluster_center"]
        n_in_cluster = sum(
            1 for _, meta in existing_bbox_list if meta["cluster"] == icon_meta["cluster"]
        )
        enforce_max = n_in_cluster > min_num_overlap_images
        enforce_min = n_in_cluster > 0
        skip_icon = False
        attempt = 0
        p_x = p_y = icon_bounding_box = None
        overlap_percent, overlap_state = None, None
        while True:
            attempt += 1
            if attempt > self.MAX_ATTEMPTS:
                skip_icon = True
                break

            d_r = random.randint(0, self.cluster_radius)
            # random angle (in radians)
            theta = random.uniform(0, 2 * math.pi)

            # random location around icon_cluster_center at distance d_r
            p_x = int(icon_cluster_center[0] + d_r * math.cos(theta))
            p_y = int(icon_cluster_center[1] + d_r * math.sin(theta))
            if p_x < 0 or p_y < 0:
                continue

            icon_bounding_box = (
                p_x,
                p_y,
                p_x + icon_width,
                p_y + icon_height,
            )
            if not self.is_icon_within_canvas(icon_bounding_box):
                continue
            # covered-area overlap of this icon, and of the neighbours after adding it
            overlap_percent, covered, overlap_info = self.calculate_overlap_percentage(
                icon_bounding_box, icon_meta, existing_bbox_list
            )
            if enforce_max:
                if overlap_percent > max_overlap + eps:
                    continue
                if any(new_pct > max_overlap + eps for (_, _, _, new_pct) in overlap_info):
                    continue
            if enforce_min and overlap_percent < min_overlap - eps:
                continue
            overlap_state = (covered, overlap_info)
            break

        return p_x, p_y, icon_bounding_box, overlap_percent, skip_icon, overlap_state

    def form_image_and_save(self, data_synth, synth_num, output_folder):
        image = self.empty_canvas.copy()
        gt_list = []  # List to store ground truth labels and bounding boxes
        for icon_meta in data_synth[synth_num]:
            if icon_meta["skip_icon"]:
                continue
            icon = icon_meta["image"]
            paste_x, paste_y = icon_meta["paste_location"]
            icon_bounding_box = (
                paste_x,
                paste_y,
                paste_x + icon_meta["scaled_size"][0],
                paste_y + icon_meta["scaled_size"][1],
            )
            image = self.insert_icon_on_canvas(image, icon, paste_x, paste_y)
            # Remove the image data and foreground mask from the metadata to save space in the .json file
            icon_meta["element_type"] = "Icon"
            icon_meta["bounding_box"] = icon_bounding_box
            del icon_meta["image"]
            del icon_meta["skip_icon"]
            del icon_meta["foreground_mask"]
            # working arrays used for the overlap bookkeeping
            for key in ("occupancy", "covered", "covered_px"):
                icon_meta.pop(key, None)

            gt_list.append(icon_meta)
        # Save the synthetic image and its ground truth labels as a .json file
        synthetic_image_path = f"{output_folder}/image_{synth_num:03d}.png"
        plt.imsave(synthetic_image_path, image)
        scene_dict = {}
        for icon in gt_list:
            cluster = icon["cluster"]
            x1, y1, x2, y2 = icon["bounding_box"]

            if cluster not in scene_dict:
                # Initialize with first bounding box
                scene_dict[cluster] = [x1, y1, x2, y2]
            else:
                # Expand bounding box
                scene_dict[cluster][0] = min(scene_dict[cluster][0], x1)  # min x
                scene_dict[cluster][1] = min(scene_dict[cluster][1], y1)  # min y
                scene_dict[cluster][2] = max(scene_dict[cluster][2], x2)  # max x
                scene_dict[cluster][3] = max(scene_dict[cluster][3], y2)  # max y

        cluster_level = {icon["cluster"]: icon["overlap_level"] for icon in gt_list}
        scene_list = []
        for scene, (cluster, bbox) in enumerate(scene_dict.items()):
            scene_list.append(
                {
                    "element_type": "Scene",
                    "represents": f"cluster{scene}",
                    "bounding_box": bbox,
                    "overlap_level": cluster_level[cluster],
                }
            )

        with open(f"{output_folder}/image_{synth_num:03d}_gt.json", "w") as gt_file:
            json.dump(scene_list + gt_list, gt_file)
        return gt_list

    def edge_distance(self, x, y, w, h):
        return min(x, y, w - x, h - y)

    def farthest_point_sampling(self, n_points, width, height, rng, n_candidates=3000, margin_frac=0.05):
        margin = margin_frac * min(width, height)
        points = [(
            rng.uniform(margin, width - margin),
            rng.uniform(margin, height - margin),
        )]

        for _ in range(n_points - 1):
            best_candidate, best_score = None, -1
            for _ in range(n_candidates):
                cx = rng.uniform(margin, width - margin)
                cy = rng.uniform(margin, height - margin)
                min_dist_to_points = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in points)
                dist_to_edge = self.edge_distance(cx, cy, width, height)
                score = min(min_dist_to_points, dist_to_edge)
                if score > best_score:
                    best_score, best_candidate = score, (cx, cy)
            points.append(best_candidate)

        min_dist = math.inf
        closest_pair = None

        if len(points) >= 2:
            # Compare every pair of distinct points
            for pa, pb in combinations(points, 2):
                d = math.dist(pa, pb)  # Euclidean distance
                if d < min_dist:
                    min_dist = d
                    closest_pair = (pa, pb)

        return points, min_dist



    @staticmethod
    def overlap_folder_tag(Config):
        """
        Folder-name part describing the overlap setting. A single level keeps the
        original name (e.g. '00_to_00_percent_overlap_00_minimum_overlap_images'),
        so existing s02/s03 outputs still line up. Several levels give e.g.
        'mixed_overlap_00-00_00-10_10-20_per_image_02_minimum_overlap_images'.
        """
        levels = Config.overlap_levels
        if len(levels) == 1:
            lvl = levels[0]
            return (
                f"{_fmt_pct(lvl['min'])}_to_{_fmt_pct(lvl['max'])}_percent_overlap_"
                f"{lvl['min_num']:02d}_minimum_overlap_images"
            )
        labels = "_".join(lvl["label"] for lvl in levels)
        return (
            f"mixed_overlap_{labels}_{Config.overlap_assignment}_"
            f"{Config.min_num_overlap_images:02d}_minimum_overlap_images"
        )

    def generate_synthetic_data(self, Config):
        global DF_LIST
        if Config.overlap_assignment not in ("per_image", "per_cluster"):
            raise ValueError(
                f"overlap_assignment must be 'per_image' or 'per_cluster', got {Config.overlap_assignment!r}"
            )
        # create a 4961 x 3508 pixel white canvas to place the selected images on it.
        # choice is based on 300DPI of A3 image
        output_folder = (
            f"01_synthetic_data/"
            f"{Config.num_clusters[0]:02d}_to_{Config.num_clusters[1]:02d}_clusters_"
            f"{Config.num_class[0]:02d}_to_{Config.num_class[1]:02d}_classes_"
            f"{Config.num_images_per_class[0]:02d}_to_{Config.num_images_per_class[1]:02d}_images_per_class_"
            f"{Config.minimum_icons:02d}_minimum_icons_"
            f"{self.overlap_folder_tag(Config)}_"
            f"{Config.max_scale_factor:.1f}_max_scaling_factor"
        )
        level_rng = random.Random(OVERLAP_LEVEL_SEED)
        # per_image: every image gets one level, levels spread evenly over the batch
        image_levels = balanced_level_assignment(
            Config.num_synthetic, Config.overlap_levels, level_rng
        )

        os.makedirs(output_folder, exist_ok=True)

        # Create synthetic data by randomly selecting images from the dataset and placing them on a canvas to create new images.
        # The number of synthetic images to create, the number of classes to select for each synthetic image, and the number of images to select from each class can be specified as parameters.
        data_synth = self.specify_synthetic_data_subsets(Config)

        # The canvas will be used to create a synthetic image by placing the selected images from the dataset on it at random locations.
        for synth_num in range(Config.num_synthetic):
            self.num_clusters = random.randint(Config.num_clusters[0], Config.num_clusters[1])
            self.num_icons_per_cluster = Config.num_icon_per_cluster
            self.cluster_points, self.minimum_intercluster_distance = self.farthest_point_sampling(self.num_clusters, width=4961, height=3508, rng=random.Random(123 + synth_num), n_candidates=3000, margin_frac=0.05 )
            self.cluster_radius = int(0.4 * self.minimum_intercluster_distance)
            self.num_icon_per_cluster_list = random.choices([i for i in range(self.num_icons_per_cluster[0], self.num_icons_per_cluster[1]+1)], k=self.num_clusters)
            existing_bbox_list = []  # List to store bounding boxes of placed images
            icon_per_cluster_dict = {i:0 for i in range(self.num_clusters)}
            # per_cluster: clusters of one image get different levels (overlap is only
            # measured within a cluster, so each cluster can follow its own level)
            if Config.overlap_assignment == "per_cluster":
                cluster_levels = balanced_level_assignment(
                    self.num_clusters, Config.overlap_levels, level_rng
                )
            else:
                cluster_levels = [image_levels[synth_num]] * self.num_clusters
            for icon_meta in data_synth[synth_num]:
                icon_meta['cluster'], icon_meta['cluster_center'] = random.choice(list(enumerate(self.cluster_points)))
                max_num_icons = self.num_icon_per_cluster_list[icon_meta['cluster']]
                level = cluster_levels[icon_meta['cluster']]
                icon_meta["overlap_level"] = level["label"]
                icon_meta["overlap_min"] = level["min"]
                icon_meta["overlap_max"] = level["max"]
                icon_meta["overlap_min_num_images"] = level["min_num"]

                if icon_per_cluster_dict[icon_meta['cluster']] >= max_num_icons:
                    icon_meta["paste_location"] = (None, None)
                    icon_meta["overlap_p"] = None
                    icon_meta["bounding_box"] = None
                    icon_meta["skip_icon"] = True
                    continue
                
                p_x, p_y, icon_bounding_box, overlap_percentage, skip_icon, overlap_state = (
                    self.get_paste_location(
                        icon_meta,
                        existing_bbox_list,
                        level,
                    )
                )
                icon_meta["paste_location"] = (p_x, p_y)
                icon_meta["bounding_box"] = icon_bounding_box
                icon_meta["skip_icon"] = skip_icon

                if skip_icon:
                    icon_meta["overlap_p"] = None
                else:
                    # updates this icon AND the icons it overlaps
                    covered, overlap_info = overlap_state
                    self.commit_overlap(icon_meta, overlap_percentage, covered, overlap_info)
                    icon_per_cluster_dict[icon_meta['cluster']] += 1
                    existing_bbox_list.append((icon_bounding_box, icon_meta))

            # Final overlaps from the finished layout (sanity check of the bookkeeping)
            placed = [meta for _, meta in existing_bbox_list]
            max_diff = self.recompute_final_overlaps(placed)
            if max_diff > 1e-6:
                print(f"WARNING image {synth_num}: incremental overlap differed from final by {max_diff:.4f}%")
            over_max = [m for m in placed if m["overlap_p"] > m["overlap_max"] + 1e-9]
            if over_max:
                print(f"NOTE image {synth_num}: {len(over_max)} icon(s) above their max overlap "
                      f"(placed while the cluster was below min_num_overlap_images)")

        # pasting the selected images on the canvas and saving the synthetic image along with its ground truth labels and bounding boxes in a .json file. The ground truth labels will include the class name, bounding box coordinates, overlap percentage, and scale factor for each pasted image.
        gt_list_list = []
        
        for synth_num in range(Config.num_synthetic):
            gt_list = self.form_image_and_save(data_synth, synth_num, output_folder)
            gt_list_list.append(gt_list)

        # Summarise once per batch. (Previously this ran inside the loop above, so the
        # cumulative table was appended to DF_LIST again after every image and the
        # overall data_summary.xlsx contained many duplicate rows.)
        outpath, DF = self.summarize_synthetic_data(
            gt_list_list, out_path=os.path.join(output_folder, "data_summary.png")
        )
        DF["output_folder"] = output_folder
        if len(DF_LIST) == 0:
            DF_LIST = DF
        else:
            DF_LIST = pd.concat([DF_LIST, DF], axis=0)
        DF_LIST.to_excel("01_synthetic_data/data_summary.xlsx", index=False)
        return DF

    def summarize_synthetic_data(
        self, listobject, out_path="01_synthetic_data/datasummary.png"
    ):
        """
        Summarizes a list of images where each image contains multiple icon elements.
        Produces a single figure with multiple subplots and saves it to disk.
        """

        records = []

        # Flatten listobject while keeping image index
        for image_id, image_elems in enumerate(listobject):
            for elem in image_elems:
                row = elem.copy()
                row["image_id"] = image_id
                records.append(row)

        df = pd.DataFrame(records)

        # Count unique clusters per image
        clusters_per_image = df.groupby("image_id")["cluster"].nunique()

        # Create output directory
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # ---- Plotting ----
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Histogram of icon classes
        df["represents"].value_counts().plot(kind="bar", ax=axs[0, 0])
        axs[0, 0].set_title("Histogram of Icon Classes")
        axs[0, 0].set_xlabel("Class")
        axs[0, 0].set_ylabel("Count")

        # 2. Boxplot of scale factor

        scale_per_image = [
            df.loc[df["image_id"] == img_id, "scale_factor"].values
            for img_id in sorted(df["image_id"].unique())
        ]

        # Add overall scale factors
        scale_per_image.append(df["scale_factor"].values)

        # Labels
        labels = [f"Img {i}" for i in sorted(df["image_id"].unique())]
        labels.append("Overall")

        axs[0, 1].boxplot(scale_per_image, tick_labels=labels)
        axs[0, 1].set_title("Scale Factor Distribution per Image + Overall")
        axs[0, 1].set_ylabel("Scale Factor")
        axs[0, 1].tick_params(axis="x", rotation=45)

        # 3. Boxplot of icon area
        axs[0, 2].boxplot(df["icon_area"])
        axs[0, 2].set_title("Icon Area Distribution")

        # 4. Boxplot of unique clusters per image
        axs[1, 0].boxplot(clusters_per_image)
        axs[1, 0].set_title("Unique Cluster IDs per Image")

        # 5. Boxplot of overlap area
        axs[1, 1].boxplot(df["overlap_p"])
        axs[1, 1].set_title("Overlap Area Distribution")

        # 6. Achieved overlap per requested overlap level
        level_names = sorted(df["overlap_level"].unique())
        axs[1, 2].boxplot(
            [df.loc[df["overlap_level"] == lvl, "overlap_p"].values for lvl in level_names],
            tick_labels=[
                f"{lvl}\n(n={int((df['overlap_level'] == lvl).sum())})" for lvl in level_names
            ],
        )
        axs[1, 2].set_title("Achieved Overlap % per Overlap Level")
        axs[1, 2].set_xlabel("Requested overlap range (%)")
        axs[1, 2].set_ylabel("Overlap %")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        df.to_excel(out_path.replace(".png", ".xlsx"), index=False)

        return out_path, df


if __name__ == "__main__":
    DF_LIST = []
    @dataclass
    class Config:
        num_synthetic: int
        num_class: int
        num_images_per_class: tuple[int, int]
        minimum_icons: int
        overlap_levels: list  # built by make_overlap_levels()
        overlap_assignment: str  # "per_image" or "per_cluster"
        min_num_overlap_images: int
        max_scale_factor: float

    num_synthetic_list = (100,)
    num_class_list = ((5, 20),)
    num_images_per_class_list = ((3, 5),)
    minimum_icons_list = (20,)
    # Each entry is ONE batch (one output folder). A batch may hold several
    # (min %, max %) overlap ranges; its images (or clusters) are spread evenly
    # over them. A single range reproduces the old fixed-overlap behaviour and
    # folder name, e.g. ((0, 0),).
    overlap_level_sets = (
        ((0, 10)),
    )
    # "per_image":   every image uses one level, levels vary across the batch
    # "per_cluster": clusters within the same image use different levels
    overlap_assignment_list = ("per_image",)
    min_num_overlap_images_list = (0,)
    max_scale_factor_list = (3.0,)
    num_clusters_list = ((2, 5),)
    num_icons_per_cluster = ((3,6),)

    dataset_path = get_dataset_path()
    dataset_obj = MultiIconDataset(
        dataset_path, overlap_logic="pixel", max_attempts=1000
    )
    dataset_obj._display_classes()
    dataset_obj._list_few_classes(num_classes=5)
    dataset_obj._display_few_images(num_classes=5, num_images_per_class=5)
    dataset_obj._create_dataset_dict()
    for (
        num_synthetic,
        num_class,
        num_images_per_class,
        minimum_icons,
        overlap_ranges,
        overlap_assignment,
        min_num_overlap_images,
        max_scale_factor,
        num_clusters,
        num_icon_per_cluster
    ) in product(
        num_synthetic_list,
        num_class_list,
        num_images_per_class_list,
        minimum_icons_list,
        overlap_level_sets,
        overlap_assignment_list,
        min_num_overlap_images_list,
        max_scale_factor_list,
        num_clusters_list,
        num_icons_per_cluster
    ):
        # With one level, per_image and per_cluster are identical: run it once.
        if len(overlap_ranges) == 1 and overlap_assignment != overlap_assignment_list[0]:
            continue

        Config.num_synthetic = num_synthetic
        Config.num_class = num_class
        Config.num_images_per_class = num_images_per_class
        Config.minimum_icons = minimum_icons
        Config.overlap_levels = make_overlap_levels(overlap_ranges, min_num_overlap_images)
        Config.overlap_assignment = overlap_assignment
        Config.min_num_overlap_images = min_num_overlap_images
        Config.max_scale_factor = max_scale_factor
        Config.num_clusters = num_clusters
        Config.num_icon_per_cluster = num_icon_per_cluster

        DF = dataset_obj.generate_synthetic_data(Config)
