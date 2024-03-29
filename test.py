import os
import re


# Function to escape special characters for regex matching
def escape_special_characters(name):
    # This will escape special characters like '.', '(', ')', etc.
    return re.escape(name)


images_dir = "./converted/positive"

image_files = os.listdir(images_dir)
raw_name = "s3://mcnv-bucket/positive/5.jpeg"

json_image_name = raw_name.split("/")[-1]  # "19b64e56-zuccolellaf022.jpeg"
# image_name = json_image_name.split("-", 1)[-1]  # "zuccolellaf022.jpeg"
# Escaping special characters in the image_name
escaped_image_name = escape_special_characters(json_image_name)
pattern = re.compile(
    f".*{escaped_image_name}$"
)  # Regex pattern to match the actual image name

matching_images = [img for img in image_files if pattern.match(img)]
print(pattern)
matched_image_name = matching_images[0]
print(matched_image_name)

if not matching_images:
    # If no matching image is found, skip this JSON file
    print("No matching")


def list_diff_files(dir1, dir2):
    # List files in the two directories
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    # Find differences and intersections
    unique_to_dir1 = files_dir1.difference(files_dir2)
    unique_to_dir2 = files_dir2.difference(files_dir1)
    common_files = files_dir1.intersection(files_dir2)

    return unique_to_dir1, unique_to_dir2


# Example usage (You should replace 'path_to_dir1' and 'path_to_dir2' with the actual paths)
dir1 = "./converted/positive"
dir2 = "./yolo_dataset/positive/images"
unique_to_dir1, unique_to_dir2 = list_diff_files(dir1, dir2)
print("Unique to dir1:", unique_to_dir1)
print("Unique to dir2:", unique_to_dir2)
