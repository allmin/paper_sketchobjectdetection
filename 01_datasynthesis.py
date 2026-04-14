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
                img = mpimg.imread(image_path)
                image_shape = img.shape
                print(f"Class: {class_name}, Image: {image_file}, Shape: {image_shape}")
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
        overlap_percentage = 0
        max_overlap = 0
        icon_col, icon_row, icon_col_end, icon_row_end = icon_bounding_box
        icon_size = (icon_col_end - icon_col, icon_row_end - icon_row)
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
            overlap_area = x_overlap * y_overlap
            icon_area = icon_size[0] * icon_size[1]
            if icon_area > 0:
                overlap_percentage = overlap_area / icon_area * 100
            max_overlap = max(max_overlap, overlap_percentage)
        return max_overlap

    def generate_synthetic_data(
        self,
        choice_num_synthetic=1000,
        choice_num_class=10,
        choice_num_images_per_class=1,
        choice_allowed_overlap_of_images=0.5,
        choice_max_scale_factor=3.0,
        output_folder="01_synthetic_data",
    ):
        # Create synthetic data by randomly selecting images from the dataset and placing them on a canvas to create new images. The number of synthetic images to create, the number of classes to select for each synthetic image, and the number of images to select from each class can be specified as parameters.
        ## Important:
        # randomly pick choice_num_synthetic times choice_num_class classes from the dataset and randomly select choice_num_images_per_class image from each class to create a new data subset
        os.makedirs(output_folder, exist_ok=True)
        data_synth = {}
        for synth_num in range(choice_num_synthetic):
            selected_classes = random.sample(
                list(self.dataset.keys()), choice_num_class
            )
            print("Selected classes for the new data subset:", selected_classes)
            data_synth[synth_num] = {}
            for class_name in selected_classes:
                image_paths = self.dataset[class_name]
                selected_images = random.sample(
                    image_paths, choice_num_images_per_class
                )
                data_synth[synth_num][class_name] = selected_images

        print(
            "Data synthesis complete. Created",
            choice_num_synthetic,
            "new data subsets with",
            choice_num_class,
            "classes each.",
        )

        # create a 4961 x 3508 pixel white canvas to place the selected images on it. The canvas will be used to create a synthetic image by placing the selected images from the dataset on it on random locations.
        canvas_width = 4961
        canvas_height = 3508
        empty_canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

        for synth_num in range(choice_num_synthetic):
            gt_list = (
                []
            )  # List to store ground truth labels and bounding boxes for the synthetic image
            image = empty_canvas.copy()
            print(f"Data subset {synth_num}:")
            rand_bbox_list = []  # List to store bounding boxes of placed images
            for class_name, image_paths in data_synth[synth_num].items():
                for image_path in image_paths:
                    icon = mpimg.imread(image_path)
                    # scale icon pixels to be between 0 and 255
                    if icon.max() <= 1.0:
                        icon = (icon * 255).astype(np.uint8)
                    # scale icon to be between 100 and 500 pixels in width and height
                    scale_factor = random.uniform(1, choice_max_scale_factor)
                    icon = cv2.resize(
                        icon,
                        (
                            int(icon.shape[1] * scale_factor),
                            int(icon.shape[0] * scale_factor),
                        ),
                    )
                    icon_height, icon_width = icon.shape[:2]

                    while True:
                        rand_x = random.randint(0, canvas_width)
                        rand_y = random.randint(0, canvas_height)
                        icon_bounding_box = (
                            rand_x,
                            rand_y,
                            rand_x + icon_width,
                            rand_y + icon_height,
                        )
                        # calculate percentage of overlap with existing images on the canvas

                        max_overlap = self.calculate_overlap_percentage(
                            icon_bounding_box, rand_bbox_list
                        )

                        if (
                            max_overlap <= choice_allowed_overlap_of_images
                            and icon_bounding_box[2] <= canvas_width
                            and icon_bounding_box[3] <= canvas_height
                        ):
                            break
                    # insert the icon on the canvas at the random location
                    print(
                        f"\t\t Placing a {class_name} image on the canvas with bounding box: {icon_bounding_box} and overlap percentage: {max_overlap:.2f}%. Icon scaling factor: {scale_factor:.2f}."
                    )
                    image[
                        rand_y : rand_y + icon_height,
                        rand_x : rand_x + icon_width,
                        :,
                    ] = np.minimum(
                        image[
                            rand_y : rand_y + icon_height,
                            rand_x : rand_x + icon_width,
                            :,
                        ],
                        icon,
                    )
                    # Add the bounding box of the placed image to the list
                    rand_bbox_list.append(icon_bounding_box)
                    gt_list.append(
                        {
                            "class": class_name,
                            "bbox": icon_bounding_box,
                            "overlap": max_overlap,
                            "scale": scale_factor,
                        }
                    )
            # Save the synthetic image and its ground truth labels as a .json file
            synthetic_image_path = f"{output_folder}/generated_image_{synth_num}.png"
            plt.imsave(synthetic_image_path, image)
            with open(
                f"{output_folder}/generated_image_{synth_num}_gt.json", "w"
            ) as gt_file:
                json.dump(gt_list, gt_file)


if __name__ == "__main__":
    choice_num_synthetic_list = (20,)
    choice_num_class_list = (10,)
    choice_num_images_per_class_list = (3,)
    choice_allowed_overlap_of_images_list = (0, 50)
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
        choice_allowed_overlap_of_images,
        choice_max_scale_factor,
    ) in product(
        choice_num_synthetic_list,
        choice_num_class_list,
        choice_num_images_per_class_list,
        choice_allowed_overlap_of_images_list,
        choice_max_scale_factor_list,
    ):
        dataset_obj.generate_synthetic_data(
            choice_num_synthetic=choice_num_synthetic,
            choice_num_class=choice_num_class,
            choice_num_images_per_class=choice_num_images_per_class,
            choice_allowed_overlap_of_images=choice_allowed_overlap_of_images,
            choice_max_scale_factor=choice_max_scale_factor,
            output_folder=(
                f"01_synthetic_data_"
                f"{choice_num_class:02d}_classes_"
                f"{choice_num_images_per_class:02d}_images_per_class_"
                f"maxoverlap_{choice_allowed_overlap_of_images:02d}_percent_"
                f"scale_{choice_max_scale_factor:.1f}"
            ),
        )
