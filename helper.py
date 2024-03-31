import os
import shutil
import random
from typing import Tuple
from PIL import Image
import torch


def format_output_single_element(cls, conf, data):
    # Find the index of the maximum confidence score
    max_conf_index = torch.argmax(conf)

    # Extract corresponding elements for class, confidence, and data
    cls_max = cls[max_conf_index].unsqueeze(0)  # Class ID as single-element tensor
    conf_max = conf[max_conf_index].unsqueeze(
        0
    )  # Max confidence as single-element tensor
    data_max = data[max_conf_index].unsqueeze(0)  # Data row as single-row tensor

    # Get the number of elements in the 'conf' tensor
    num_elements = conf.numel()

    return cls_max, conf_max, num_elements, data_max


def print_first_n_elements(input_dict, n):
    # Iterate over both keys and values in the dictionary
    for i, (key, value) in enumerate(input_dict.items()):
        if i < n:
            print(f"{key}: {value}")
        else:
            break  # Stop after printing the Nth element


def numerical_sort(filename):
    basename = os.path.splitext(filename)[0]  # Remove the extension
    try:
        number = int(basename)  # Convert to integer for comparison
        return (0, number)  # Leading 0 indicates a successful int conversion
    except ValueError:
        # In case the filename doesn't start with a number, use lexicographic sort
        return (1, basename)  # Leading 1 indicates fall back to lexicographic


def read_yolo_labels_ordered_modified(labels_dir):
    labels_dict = {}

    # Iterate over files in the labels directory, sorted alphabetically
    for label_file in sorted(os.listdir(labels_dir)):
        # Ensure we're only reading .txt files
        if label_file.endswith(".txt"):
            # Extract the filename without the extension for use as a key and append .jpeg
            file_key = os.path.splitext(label_file)[0] + ".jpeg"
            # Initialize list to store object details
            labels_for_file = []

            # Read lines from the label file
            with open(
                os.path.join(labels_dir, label_file), "r", encoding="utf-8"
            ) as file:
                lines = file.readlines()

                # Check if file is empty
                if not lines:
                    labels_for_file.append((0, None, None, None, None))
                else:
                    for line in lines:
                        parts = line.strip().split()
                        class_id, x_center, y_center, width, height = map(float, parts)
                        class_id = int(class_id)

                        # Append object details to the list as a tuple
                        labels_for_file.append(
                            (class_id, x_center, y_center, width, height)
                        )

            # Map list of tuples to the filename key in the dictionary
            labels_dict[file_key] = labels_for_file

    return labels_dict


def prepare_labels_for_sklearn(labels_dict):
    y_true = []

    for _, object_list in labels_dict.items():
        # Only consider the first object for this example
        obj = object_list[0]
        # Directly append the class ID; since `(0, None, None, None, None)` should also count as a valid label
        y_true.append(obj[0])

    return y_true


def list_of_images(dirpath):
    # Get a list of all files in the directory
    all_files = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]

    # Filter out non-image files (assuming images have extensions like .jpg, .jpeg, .png, etc.)
    image_files = [f for f in all_files if f.lower().endswith((".jpeg"))]
    return sorted(image_files)


def assign_labels(pos_dir, neg_dir, test_dir):
    labels = []
    for file in sorted(os.listdir(test_dir)):
        pos_path = os.path.join(pos_dir, file)
        neg_path = os.path.join(neg_dir, file)

        if os.path.exists(pos_path):
            labels.append(1)
        elif os.path.exists(neg_path):
            labels.append(0)
        else:
            raise FileNotFoundError(
                f"No duplicate found for {file} in either positive or negative directories."
            )

    return labels


def rename_files_with_suffix(directory_path, suffix="_p"):
    """
    This function renames all files in the given directory by adding a specified suffix to their names,
    before the file extension. E.g., 'filename.jpeg' -> 'filename_p.jpeg'

    :param directory_path: Path to the directory containing files to be renamed.
    :param suffix: Suffix to add to the filenames (default is '_p').
    """
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Get the file name without extension and the file extension separately
        name, extension = os.path.splitext(filename)
        # Construct the new filename with the suffix added before the file extension
        new_filename = f"{name}{suffix}{extension}"
        # Rename the file
        os.rename(
            os.path.join(directory_path, filename),
            os.path.join(directory_path, new_filename),
        )
        print(f"Renamed {filename} to {new_filename}")


def check_filenames_match(images_dir, labels_dir):
    # List all files in each directory
    image_files = os.listdir(images_dir)
    label_files = os.listdir(labels_dir)

    # Extract filenames without the specific extension
    image_filenames = set([file[:-5] for file in image_files if file.endswith(".jpeg")])
    label_filenames = set([file[:-4] for file in label_files if file.endswith(".txt")])

    # Check if the sets of filenames match
    match = image_filenames == label_filenames

    # Difference (if needed for debugging or info)
    missing_in_images = label_filenames - image_filenames
    missing_in_labels = image_filenames - label_filenames

    return match, missing_in_images, missing_in_labels


def find_files_with_same_name(dir1: str, dir2: str) -> list:
    # List files in dir1
    files_dir1 = {
        file for file in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, file))
    }

    # List files in dir2
    files_dir2 = {
        file for file in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, file))
    }

    # Find common files
    common_files = list(files_dir1.intersection(files_dir2))
    common_files = [filename for filename in common_files]

    return common_files


def rename_and_copy_files(common_files: list, dir1: str, target_dir: str):
    for filename in common_files:
        # Generate new filename by appending '_01' before the file extension
        root, ext = os.path.splitext(filename)
        new_filename = f"{root}_01{ext}"

        # Copy the file from dir1 with the new filename to the target directory
        shutil.copy(
            os.path.join(dir1, filename), os.path.join(target_dir, new_filename)
        )
        # Delete the original file from dir1
        os.remove(os.path.join(dir1, filename))


def clean_destination_dirs(base_path: str, dirs: Tuple[str, ...]):
    """
    Removes specified directories and their contents, then recreates the directories.
    """
    for dir_name in dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)
    print("Destination directories cleaned and recreated.")


def split_and_copy_dataset(
    base_path: str,
    subdir: str,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    assert sum(splits) == 1, "The splits must sum to 1"

    # Set the seed for reproducibility
    # random.seed(seed)

    # Paths for the images and labels inside your positive dataset
    source_images_path = os.path.join(base_path, subdir, "images")
    source_labels_path = os.path.join(base_path, subdir, "labels")

    # Get all image filenames
    image_files = os.listdir(source_images_path)
    # Correctly pair image files with their corresponding label files (.jpg with .txt)
    paired_files = [
        (img, img.replace(".jpeg", ".txt"))
        for img in image_files
        if img.endswith(".jpeg")
    ]

    # Shuffle the file list for random split
    random.shuffle(paired_files)

    # Calculate indices for train, val, and test splits
    total_files = len(paired_files)
    train_end = int(total_files * splits[0])
    val_end = train_end + int(total_files * splits[1])

    # Split the dataset
    train_files = paired_files[:train_end]
    val_files = paired_files[train_end:val_end]
    test_files = paired_files[val_end:]

    # Function to copy the paired files to their new destination
    def copy_files(file_pairs, split_type):
        images_dest_path = os.path.join(base_path, split_type, "images")
        labels_dest_path = os.path.join(base_path, split_type, "labels")

        # Ensure the destination directories exist
        os.makedirs(images_dest_path, exist_ok=True)
        os.makedirs(labels_dest_path, exist_ok=True)

        for img_file, label_file in file_pairs:
            src_img_path = os.path.join(source_images_path, img_file)
            src_label_path = os.path.join(source_labels_path, label_file)

            dest_img_path = os.path.join(images_dest_path, img_file)
            dest_label_path = os.path.join(labels_dest_path, label_file)

            # Copy the image and label files to their new destinations
            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_label_path, dest_label_path)

    # Execute copying for splits
    copy_files(train_files, "train")
    copy_files(val_files, "validation")
    copy_files(test_files, "test")

    print(
        f"Dataset split and copied: {len(train_files)} for training, {len(val_files)} for validation, and {len(test_files)} for testing."
    )


def convert_images_to_jpeg(directory_path):

    for filename in os.listdir(directory_path):
        # Create full file path
        file_path = os.path.join(directory_path, filename)
        # Ignore directories and hidden files
        if os.path.isdir(file_path) or filename.startswith("."):
            continue

        name, extension = os.path.splitext(filename)

        # If the file is already a JPEG, skip it
        if extension.lower() in [".jpeg", ".jpg"]:
            print(f"{filename} is already a JPEG. Skipping conversion.")
            continue

        # Convert to JPEG format
        try:
            with Image.open(file_path) as img:
                # Convert image to RGB if necessary (JPEG does not support alpha channel)
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                new_filename = f"{name}.jpeg"
                new_file_path = os.path.join(directory_path, new_filename)
                img.save(new_file_path, "JPEG")
                os.remove(file_path)  # Remove the original file after conversion
                print(f"Converted and saved {new_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":

    base_path = "./yolo_dataset"
    subdir = "positive"
    splits = (0.8, 0.1, 0.1)

    clean_destination_dirs(base_path, ("train", "validation", "test"))

    split_and_copy_dataset(base_path, subdir, splits)
    subdir = "negative"
    split_and_copy_dataset(base_path, subdir, splits)
    subdir = "no-area"
    split_and_copy_dataset(base_path, subdir, splits)

    # Checks if dir have same images and labels filename
    _types = ["train", "validation", "test"]

    for _type in _types:
        labels_dir = f"./yolo_dataset/{_type}/labels"
        images_dir = f"./yolo_dataset/{_type}/images"
        match, missing_in_images, missing_in_labels = check_filenames_match(
            images_dir, labels_dir
        )
        print(f"Filenames match for {_type} dir is {match}")
        if not match:
            print(f"Missing in images: {missing_in_images}")
            print(f"Missing in labels: {missing_in_labels}")
