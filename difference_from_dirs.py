import os
from typing import Tuple
import shutil
from pathlib import Path


def clean_destination_dirs(base_path: str, dirs: Tuple[str]):
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


def files_difference(dir1, dir2):
    # List all files in each directory
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    # Find files that are in dir1 but not in dir2
    diff = files_dir1.difference(files_dir2)

    # Convert the set to a list
    difference = sorted(list(diff))

    return difference


def copy_files(file_list, dir1, dir2):
    # Ensure the target directory exists; if not, create it
    os.makedirs(dir2, exist_ok=True)

    for file_name in file_list:
        src_file_path = os.path.join(dir1, file_name)
        dest_file_path = os.path.join(dir2, file_name)

        # Copy each file from dir1 to dir2
        shutil.copy(src_file_path, dest_file_path)


def create_empty_files(file_list, target_dir):
    # Ensure the target directory exists; if not, create it
    os.makedirs(target_dir, exist_ok=True)

    for file_name in file_list:
        # Extract the filename without the extension
        stem = Path(file_name).stem
        # Construct the new filename with .txt extension
        new_file_name = f"{stem}.txt"
        new_file_path = os.path.join(target_dir, new_file_name)
        # Create an empty file
        with open(new_file_path, "w", encoding="utf-8") as f:
            pass  # File is created and closed, remaining empty


def find_common_files(dir1, dir2):
    # List all files in each directory
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    # Find files that are common to both directories
    common_files = files_dir1.intersection(files_dir2)

    # Convert the set to a list
    common_files_list = list(common_files)

    return sorted(common_files_list)


# clean_destination_dirs("yolo_dataset/no-area", ("",))

# file_list = files_difference("converted/negative", "yolo_dataset/negative/images")
# # # print(file_list)
# print(len(file_list))

# copy_files(file_list, "converted/negative", "yolo_dataset/no-area/images")
# create_empty_files(file_list, "yolo_dataset/no-area/labels")

# file_list2 = find_common_files("dir1a/images", "./yolo_dataset/negative/images")
# print(file_list2)
# print(len(file_list2))

# file_list3 = find_common_files("dir1a", "./yolo_dataset/positive/images")
# print(file_list3)
# print(len(file_list3))


# file_list = files_difference("yolo_dataset/negative/images", "converted/negative")
# print(file_list)
# print(len(file_list))

# file_list = files_difference("converted/negative", "yolo_dataset/negative/images")
# print(len(file_list))
