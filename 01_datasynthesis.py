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

# setting random seed for reproducibility
random.seed(42)
np.random.seed(42)


def get_dataset_path():
    # Download the TU Berlin dataset if it is not already available locally or is outdated.
    path = kagglehub.dataset_download("zara2099/tu-berlin-hand-sketch-image-dataset")
    path = os.path.join(path, "TUBerlin", "png_ready")
    print("Path to dataset files:", path)
    return path


class dataset:
    def __init__(self, datapath):
        self.path = datapath
        self.canvas_width = 4950
        self.canvas_height = 3510
        self.empty_canvas = 255 * np.ones(
            (self.canvas_height, self.canvas_width, 3), dtype=np.uint8
        )
        self.image_cache = {}

    def load_image(self, path):
        if path not in self.image_cache:
            img = mpimg.imread(path)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            self.image_cache[path] = img
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
                img = self.load_image(image_path)
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

    def calculate_overlap_percentage(self, icon_bounding_box, existing_bbox_list):
        overlap_area = 0
        total_overlap_percentage = 0
        icon_col, icon_row, icon_col_end, icon_row_end = icon_bounding_box
        icon_size = (icon_col_end - icon_col, icon_row_end - icon_row)
        icon_area = icon_size[0] * icon_size[1]
        for existing_bbox in existing_bbox_list:
            ext_col, ext_row, ext_col_end, ext_row_end = existing_bbox
            x_overlap = max(
                0,
                min(icon_col_end, ext_col_end) - max(icon_col, ext_col),
            )
            y_overlap = max(
                0,
                min(icon_row_end, ext_row_end) - max(icon_row, ext_row),
            )
            overlap_area_current = x_overlap * y_overlap
            overlap_area += overlap_area_current
            total_overlap_percentage = overlap_area / icon_area * 100
        return total_overlap_percentage

    def insert_icon_on_canvas(self, canvas, icon, x, y):
        icon_height, icon_width = icon.shape[:2]
        canvas[x : x + icon_width, y : y + icon_height, :] = np.minimum(
            canvas[x : x + icon_width, y : y + icon_height, :], icon
        )
        return canvas

    def instantiate_synthetic_data_subsets(
        self,
        choice_num_synthetic,
        choice_num_class,
        choice_num_images_per_class,
        choice_max_scale_factor,
    ):
        data_synth = {}
        for synth_num in range(choice_num_synthetic):
            selected_classes = random.sample(
                list(self.dataset.keys()), choice_num_class
            )
            print("Selected classes for the new data subset:", selected_classes)
            data_synth[synth_num] = []
            for class_name in selected_classes:
                image_paths = self.dataset[class_name]
                selected_images = random.sample(
                    image_paths, choice_num_images_per_class
                )
                for selected_image in selected_images:
                    scale_factor = random.uniform(1, choice_max_scale_factor)
                    icon = self.load_image(selected_image)
                    # scale icon pixels to be between 0 and 255
                    if icon.max() <= 1.0:
                        icon = (icon * 255).astype(np.uint8)
                    icon_height, icon_width = icon.shape[:2]
                    scaled_size = (
                        int(icon_height * scale_factor),
                        int(icon_width * scale_factor),
                    )
                    icon_scaled = cv2.resize(icon, scaled_size)
                    data_synth[synth_num].append(
                        {
                            "class": class_name,
                            "image_path": selected_image,
                            "image": icon_scaled,
                            "scale_factor": scale_factor,
                            "scaled_size": scaled_size[::-1],  # (width, height)
                        }
                    )
        print(
            "Data synthesis complete. Created",
            choice_num_synthetic,
            "new data subsets with",
            choice_num_class,
            "classes each.",
        )
        return data_synth

    def get_paste_location(
        self,
        icon_width,
        icon_height,
        rand_bbox_list,
        choice_max_overlap_of_images,
        choice_min_overlap_of_images,
    ):
        while True:
            rand_x = random.randint(0, self.canvas_width)
            rand_y = random.randint(0, self.canvas_height)
            icon_bounding_box = (
                rand_x,
                rand_y,
                rand_x + icon_width,
                rand_y + icon_height,
            )
            # calculate percentage of overlap with existing images on the canvas
            total_overlap = self.calculate_overlap_percentage(
                icon_bounding_box, rand_bbox_list
            )

            if (
                total_overlap <= choice_max_overlap_of_images
                and (
                    total_overlap >= choice_min_overlap_of_images
                    if len(rand_bbox_list) > 0
                    else True
                )
                and icon_bounding_box[2] <= self.canvas_width
                and icon_bounding_box[3] <= self.canvas_height
            ):
                break
        return rand_x, rand_y, icon_bounding_box, total_overlap

    def form_image_and_save(self, data_synth, synth_num, output_folder):
        image = self.empty_canvas.copy()
        gt_list = []  # List to store ground truth labels and bounding boxes
        for icon_meta in data_synth[synth_num]:
            icon = icon_meta["image"]
            paste_x, paste_y = icon_meta["paste_location"]
            icon_bounding_box = (
                paste_x,
                paste_y,
                paste_x + icon_meta["scaled_size"][0],
                paste_y + icon_meta["scaled_size"][1],
            )
            image = self.insert_icon_on_canvas(image, icon, paste_x, paste_y)
            # Add the bounding box of the pasted image to the list
            del icon_meta[
                "image"
            ]  # Remove the image data from the metadata to save space in the .json file
            gt_list.append(icon_meta)
        # Save the synthetic image and its ground truth labels as a .json file
        synthetic_image_path = f"{output_folder}/image_{synth_num:03d}.png"
        plt.imsave(synthetic_image_path, image)
        with open(f"{output_folder}/image_{synth_num:03d}_gt.json", "w") as gt_file:
            json.dump(gt_list, gt_file)

    def generate_synthetic_data(
        self,
        choice_num_synthetic=1000,
        choice_num_class=10,
        choice_num_images_per_class=1,
        choice_max_overlap_of_images=0.5,
        choice_min_overlap_of_images=0.0,
        choice_max_scale_factor=3.0,
        output_folder="01_synthetic_data",
    ):
        # create a 4961 x 3508 pixel white canvas to place the selected images on it.
        # choice is based on 300DPI of A3 image

        os.makedirs(output_folder, exist_ok=True)

        # Create synthetic data by randomly selecting images from the dataset and placing them on a canvas to create new images. The number of synthetic images to create, the number of classes to select for each synthetic image, and the number of images to select from each class can be specified as parameters.
        ## Important:
        # randomly pick choice_num_synthetic times choice_num_class classes from the dataset and randomly select choice_num_images_per_class image from each class to create a new data subset for each synthetic image.
        # This will ensure that the synthetic images are created with a diverse set of classes and images from the dataset, and that the same class or image is not repeated multiple times in the same synthetic image.
        data_synth = self.instantiate_synthetic_data_subsets(
            choice_num_synthetic,
            choice_num_class,
            choice_num_images_per_class,
            choice_max_scale_factor,
        )

        # The canvas will be used to create a synthetic image by placing the selected images from the dataset on it on random locations.
        for synth_num in range(choice_num_synthetic):
            rand_bbox_list = []  # List to store bounding boxes of placed images
            for icon_meta in data_synth[synth_num]:
                icon_height, icon_width = icon_meta["image"].shape[:2]
                rand_x, rand_y, icon_bounding_box, total_overlap = (
                    self.get_paste_location(
                        icon_meta,
                        rand_bbox_list,
                        choice_max_overlap_of_images,
                        choice_min_overlap_of_images,
                    )
                )
                rand_bbox_list.append((icon_bounding_box, icon_meta))
                icon_meta["paste_location"] = (rand_x, rand_y)
                icon_meta["overlap"] = total_overlap
                icon_meta["icon_bounding_box"] = icon_bounding_box

        # pasting the selected images on the canvas and saving the synthetic image along with its ground truth labels and bounding boxes in a .json file. The ground truth labels will include the class name, bounding box coordinates, overlap percentage, and scale factor for each pasted image.
        for synth_num in range(choice_num_synthetic):
            self.form_image_and_save(data_synth, synth_num, output_folder)


if __name__ == "__main__":
    choice_num_synthetic_list = (1,)
    choice_num_class_list = (10,)
    choice_num_images_per_class_list = (3,)
    choice_overlap_of_images_list = (
        (0, 0),
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
    )
    choice_max_scale_factor_list = (3.0,)

    dataset_path = get_dataset_path()
    dataset_obj = dataset(dataset_path)
    dataset_obj._display_classes()
    dataset_obj._list_few_classes(num_classes=5)
    dataset_obj._display_few_images(num_classes=5, num_images_per_class=5)
    dataset_obj._create_dataset_dict()
    for (
        choice_num_synthetic,
        choice_num_class,
        choice_num_images_per_class,
        choice_overlap_of_images,
        choice_max_scale_factor,
    ) in product(
        choice_num_synthetic_list,
        choice_num_class_list,
        choice_num_images_per_class_list,
        choice_overlap_of_images_list,
        choice_max_scale_factor_list,
    ):
        choice_min_overlap_of_images, choice_max_overlap_of_images = (
            choice_overlap_of_images
        )
        if choice_max_overlap_of_images - choice_min_overlap_of_images >= 0:
            dataset_obj.generate_synthetic_data(
                choice_num_synthetic=choice_num_synthetic,
                choice_num_class=choice_num_class,
                choice_num_images_per_class=choice_num_images_per_class,
                choice_max_overlap_of_images=choice_max_overlap_of_images,
                choice_min_overlap_of_images=choice_min_overlap_of_images,
                choice_max_scale_factor=choice_max_scale_factor,
                output_folder=(
                    f"01_synthetic_data/"
                    f"{choice_num_class:02d}_classes_"
                    f"{choice_num_images_per_class:02d}_images_per_class_"
                    f"{choice_min_overlap_of_images:02d}_to_{choice_max_overlap_of_images:02d}_percent_overlap_"
                    f"upto_{choice_max_scale_factor:.1f}_scaling_factor"
                ),
            )
