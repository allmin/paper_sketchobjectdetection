from ast import main

import cv2
import kagglehub
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from torch import classes
from itertools import product
from scipy.ndimage import binary_fill_holes
from dataclasses import dataclass, field
import math
import os
import pandas as pd
import matplotlib.pyplot as plt

# setting random seed for reproducibility
random.seed(42)
np.random.seed(42)


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

    def _icon_area(self, icon_meta):
        if self.overlap_logic == "bounding_box":
            icon_height, icon_width = icon_meta["image"].shape[:2]
            area = max(0, icon_width) * max(0, icon_height)
        elif self.overlap_logic == "pixel":
            area = int(np.sum(np.sum(icon_meta["foreground_mask"])))
        return area

    def _bbox_overlap_area(self, a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        overlap_w = max(0, min(ax1, bx1) - max(ax0, bx0))
        overlap_h = max(0, min(ay1, by1) - max(ay0, by0))

        return overlap_w * overlap_h

    def _pixel_overlap(self, icon_bbox, icon_mask, existing_bbox, existing_mask):
        # Step 1: find overlap region in canvas coords
        x0 = max(icon_bbox[0], existing_bbox[0])
        y0 = max(icon_bbox[1], existing_bbox[1])
        x1 = min(icon_bbox[2], existing_bbox[2])
        y1 = min(icon_bbox[3], existing_bbox[3])

        if x0 >= x1 or y0 >= y1:
            return 0  # no overlap

        # Step 2: convert to local coordinates
        icon_slice = self._extract_region(icon_bbox, icon_mask, x0, y0, x1, y1)
        existing_slice = self._extract_region(
            existing_bbox, existing_mask, x0, y0, x1, y1
        )

        # Step 3: align shapes safely
        h = min(icon_slice.shape[0], existing_slice.shape[0])
        w = min(icon_slice.shape[1], existing_slice.shape[1])

        icon_slice = icon_slice[:h, :w]
        existing_slice = existing_slice[:h, :w]

        # Step 4: compute overlap
        return int(np.sum(np.logical_and(icon_slice, existing_slice)))

    def _extract_region(self, bbox, mask, x0, y0, x1, y1):
        bx0, by0, _, _ = bbox
        local_x0 = x0 - bx0
        local_y0 = y0 - by0
        local_x1 = x1 - bx0
        local_y1 = y1 - by0
        h, w = mask.shape[:2]
        local_x1 = min(local_x1, w)
        local_y1 = min(local_y1, h)
        return mask[local_y0:local_y1, local_x0:local_x1]

    def load_image(self, path):
        if path not in self.image_cache.keys():
            img = mpimg.imread(path)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            foreground_mask = self.sketch_to_binary(img)
            self.image_cache[path] = (img, foreground_mask)
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
        Compute how much the given icon overlaps (in %) with existing icons
        from the same cluster.
        """

        icon_mask = icon_meta["foreground_mask"]
        icon_area = self._icon_area(icon_meta)
        icon_meta["icon_area"] = icon_area

        # Only compare with same-cluster icons
        same_cluster = [
            (ind, bbox, meta)
            for ind, (bbox, meta) in enumerate(existing_bbox_list)
            if meta["cluster"] == icon_meta["cluster"]
        ]

        total_overlap = 0
        overlap_info = []

        for ind, existing_bbox, existing_meta in same_cluster:
            if self.overlap_logic == "bounding_box":
                overlap = self._bbox_overlap_area(icon_bbox, existing_bbox)

            elif self.overlap_logic == "pixel":
                overlap = self._pixel_overlap(
                    icon_bbox,
                    icon_mask,
                    existing_bbox,
                    existing_meta["foreground_mask"],
                )

            else:
                raise ValueError(f"Unknown overlap logic: {self.overlap_logic}")

            if overlap > 0:
                total_overlap += overlap
                # The updated overlap percent of existing icons
                after_overlap_existing_percent = (
                    existing_meta["overlap_p"]
                    + (overlap / existing_meta["icon_area"]) * 100
                )
                overlap_info.append(
                    (ind, existing_meta, overlap, after_overlap_existing_percent)
                )
        overlap_percentage = (total_overlap / icon_area) * 100 if icon_area > 0 else 0
        return (overlap_percentage, overlap_info)

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
            selected_classes = self._sample_classes(Config.num_class)
            print("Selected classes for the new data subset:", selected_classes)
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
                            int(icon_height * scale_factor),
                            int(icon_width * scale_factor),
                        )
                        icon_scaled = cv2.resize(icon, scaled_size)
                        foreground_mask_scaled = cv2.resize(
                            foreground_mask,
                            scaled_size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        data_synth[synth_num].append(
                            {
                                "class": class_name,
                                "image_path": selected_image,
                                "image": icon_scaled,
                                "foreground_mask": foreground_mask_scaled,
                                "scale_factor": scale_factor,
                                "scaled_size": scaled_size[::-1],  # (width, height)
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
        print(f"Assigned cluster labels to each icon in the synthetic data subsets.")
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
        Config,
    ):
        icon_height, icon_width = icon_meta["image"].shape[:2]
        icon_cluster_center = icon_meta["cluster_center"]
        skip_icon = False
        max_attempts = self.MAX_ATTEMPTS
        attempt = 0
        while True:
            attempt += 1

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
            icon_bbox_begin_x = icon_bounding_box[0]
            icon_bbox_begin_y = icon_bounding_box[1]
            icon_bbox_end_x = icon_bounding_box[2]
            icon_bbox_end_y = icon_bounding_box[3]
            # calculate percentage of overlap with existing images on the canvas
            overlap_percent, overlap_info = self.calculate_overlap_percentage(
                icon_bounding_box, icon_meta, existing_bbox_list
            )
            existing_bbox_list_sel = [
                i for i in existing_bbox_list if i[1]["cluster"] == icon_meta["cluster"]
            ]
            if (
                self.is_icon_within_canvas(icon_bounding_box)
                and (
                    overlap_percent <= Config.max_overlap_of_images
                    if len(existing_bbox_list_sel) > Config.min_num_overlap_images
                    else True
                )
                and (
                    all(
                        [
                            after_merge_overlap < Config.max_overlap_of_images
                            for (_, _, _, after_merge_overlap) in overlap_info
                        ]
                    )
                )
            ):
                if (
                    overlap_percent >= Config.min_overlap_of_images
                    if (len(existing_bbox_list_sel) > 0)
                    else True
                ):
                    break

            if attempt > max_attempts:
                skip_icon = True
                break

        return p_x, p_y, icon_bounding_box, overlap_percent, skip_icon

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
            icon_meta["element_type"] = "icon"
            icon_meta["bounding_box"] = icon_bounding_box
            del icon_meta["image"]
            del icon_meta["skip_icon"]
            del icon_meta["foreground_mask"]

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

        scene_list = []
        for scene, bbox in enumerate(scene_dict.items):
            scene_list.append(
                {
                    "element_type": "scene",
                    "class": f"cluster{scene}",
                    "bounding_box": bbox,
                }
            )

        with open(f"{output_folder}/image_{synth_num:03d}_gt.json", "w") as gt_file:
            json.dump(scene_list + gt_list, gt_file)
        return gt_list

    def generate_synthetic_data(self, Config):
        self.num_clusters = Config.num_clusters
        self.minimum_intercluster_distance = 512
        self.cluster_radius = int(self.minimum_intercluster_distance / 2)

        # create a 4961 x 3508 pixel white canvas to place the selected images on it.
        # choice is based on 300DPI of A3 image
        output_folder = (
            f"01_synthetic_data/"
            f"{Config.num_clusters:02d}_clusters_"
            f"{Config.num_class:02d}_classes_"
            f"{Config.num_images_per_class[0]:02d}_to_{Config.num_images_per_class[1]:02d}_images_per_class_"
            f"{Config.minimum_icons:02d}_minimum_icons_"
            f"{Config.min_overlap_of_images:02d}_to_{Config.max_overlap_of_images:02d}_percent_overlap_"
            f"{Config.min_num_overlap_images:02d}_minimum_overlap_images_"
            f"{Config.max_scale_factor:.1f}_max_scaling_factor"
        )

        os.makedirs(output_folder, exist_ok=True)

        # Create synthetic data by randomly selecting images from the dataset and placing them on a canvas to create new images.
        # The number of synthetic images to create, the number of classes to select for each synthetic image, and the number of images to select from each class can be specified as parameters.
        data_synth = self.specify_synthetic_data_subsets(Config)
        data_synth = self.assign_clusters(data_synth)

        # The canvas will be used to create a synthetic image by placing the selected images from the dataset on it at random locations.
        for synth_num in range(Config.num_synthetic):
            existing_bbox_list = []  # List to store bounding boxes of placed images
            for icon_meta in data_synth[synth_num]:
                p_x, p_y, icon_bounding_box, overlap_percentage, skip_icon = (
                    self.get_paste_location(
                        icon_meta,
                        existing_bbox_list,
                        Config,
                    )
                )
                icon_meta["paste_location"] = (p_x, p_y)
                icon_meta["overlap_p"] = overlap_percentage
                icon_meta["bounding_box"] = icon_bounding_box
                icon_meta["skip_icon"] = skip_icon
                existing_bbox_list.append((icon_bounding_box, icon_meta))

        # pasting the selected images on the canvas and saving the synthetic image along with its ground truth labels and bounding boxes in a .json file. The ground truth labels will include the class name, bounding box coordinates, overlap percentage, and scale factor for each pasted image.
        gt_list_list = []
        DF_LIST = []
        for synth_num in range(Config.num_synthetic):
            gt_list = self.form_image_and_save(data_synth, synth_num, output_folder)
            gt_list_list.append(gt_list)
            DF, outpath = self.summarize_synthetic_data(
                gt_list_list, out_path=os.path.join(output_folder, "data_summary.png")
            )
            DF["output_folder"] = output_folder
            if len(DF_LIST) == 0:
                DF_LIST = DF
            else:
                DF_LIST = pd.concat(DF_LIST, DF, axis=1)
        DF_LIST.to_excel("01_synthetic_data/data_summary.xlsx", index=False)

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
        df["class"].value_counts().plot(kind="bar", ax=axs[0, 0])
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

        axs[0, 1].boxplot(scale_per_image, labels=labels)
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

        # Empty subplot (for layout symmetry)
        axs[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        df.to_excel(out_path.replace(".png", ".xlsx"), index=False)

        return out_path, df


if __name__ == "__main__":

    @dataclass
    class Config:
        num_synthetic: int
        num_class: int
        num_images_per_class: tuple[int, int]
        minimum_icons: int
        min_overlap_of_images: float
        max_overlap_of_images: float
        min_num_overlap_images: int
        max_scale_factor: float

    num_synthetic_list = (3,)
    num_class_list = (random.randint(5, 20),)
    num_images_per_class_list = ((1, 10),)
    minimum_icons_list = (20,)
    overlap_of_images_list = (
        (0, 0),
        (0, 5),
        (5, 10),
    )
    min_num_overlap_images_list = (3,)
    max_scale_factor_list = (3.0,)
    num_clusters_list = (random.randint(5, 10),)

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
        overlap_of_images,
        min_num_overlap_images,
        max_scale_factor,
        num_clusters,
    ) in product(
        num_synthetic_list,
        num_class_list,
        num_images_per_class_list,
        minimum_icons_list,
        overlap_of_images_list,
        min_num_overlap_images_list,
        max_scale_factor_list,
        num_clusters_list,
    ):
        min_overlap_of_images, max_overlap_of_images = overlap_of_images
        if max_overlap_of_images == 0:
            min_num_overlap_images = 0

        Config.num_synthetic = num_synthetic
        Config.num_class = num_class
        Config.num_images_per_class = num_images_per_class
        Config.minimum_icons = minimum_icons
        Config.max_overlap_of_images = max_overlap_of_images
        Config.min_overlap_of_images = min_overlap_of_images
        Config.min_num_overlap_images = min_num_overlap_images
        Config.max_scale_factor = max_scale_factor
        Config.num_clusters = num_clusters

        if max_overlap_of_images - min_overlap_of_images >= 0:
            DF = dataset_obj.generate_synthetic_data(
                Config,
            )
