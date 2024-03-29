import os
from pathlib import Path
import json
import re
import shutil
import asyncio
from pprint import pprint
import aioboto3


case = "no-area"

BUCKET_NAME = "mcnv-output"
SUBDIR_NAME = f"{case}/"
IMAGES_DIR = "./converted/negative"
LOCAL_DIRECTORY = f"./from_s3/{case}"
YOLO_DIR = f"./yolo_dataset/{case}"
BATCH_SIZE = 100

label_dict = {"stabilized mcnv": 0, "mcnv region": 1, "no-area": 2}


def clear_files_in_directory(dirname):
    for root, dirs, files in os.walk(dirname):
        for file in files:
            file_path = Path(root) / file
            file_path.unlink()  # This deletes the file


def convert_bboxes_to_yolo_format(bbox, img_width, img_height):

    pixel_x = bbox["x"] / 100.0 * img_width
    pixel_y = bbox["y"] / 100.0 * img_height
    pixel_width = bbox["width"] / 100.0 * img_width
    pixel_height = bbox["height"] / 100.0 * img_height

    x_center = (pixel_x + pixel_width / 2) / img_width
    y_center = (pixel_y + pixel_height / 2) / img_height
    width_norm = pixel_width / img_width
    height_norm = pixel_height / img_height

    return {
        "x_norm": x_center,
        "y_norm": y_center,
        "width_norm": width_norm,
        "height_norm": height_norm,
    }


async def download_s3_object(s3_client, bucket_name, object_key, local_json_path):
    # Get the object from S3 asynchronously
    response = await s3_client.get_object(Bucket=bucket_name, Key=object_key)

    # Read the content of the object
    async with response["Body"] as stream:
        content = await stream.read()

    content = content.decode("utf-8")

    # ... rest of the content processing and file-saving logic remains the same
    # Check if the content is empty (likely a directory placeholder)
    if not content.strip():
        print(f"Skipping '{object_key}': No content (possible directory placeholder)")
        return  # Skip this object and move to the next one

    # Parse the content into a Python dict
    try:
        data_dict = json.loads(content)

        # Extract the desired parts

        image_path = data_dict["task"]["data"]["image"]
        try:
            result_data = data_dict["result"][0]
        except Exception as e:
            pprint(data_dict, indent=2)
            raise ValueError(f"Error: {e}") from e

        # Selecting only the specified fields
        selected_data = {
            "original_width": result_data["original_width"],
            "original_height": result_data["original_height"],
            "image_rotation": result_data["image_rotation"],
            "value": result_data["value"],
        }

        # Create a new dict with the extracted parts
        new_data_dict = {"image": image_path, "result": selected_data}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for object {object_key}: {e}")
        return  # Skip this object and move to the next one

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_json_path), exist_ok=True)

    # Save the new dict as a JSON file locally
    with open(local_json_path, "w", encoding="utf-8") as json_file:
        json.dump(new_data_dict, json_file, indent=4)


async def download_all_objects(
    bucket_name, local_directory, subdir_name, batch_size=BATCH_SIZE
):
    session = aioboto3.Session()

    async with session.client("s3") as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name, Prefix=subdir_name):
            if "Contents" in page:
                for i in range(0, len(page["Contents"]), batch_size):
                    # This slices the items to process in batches of size batch_size
                    batch = page["Contents"][i : i + batch_size]
                    tasks = []
                    for item in batch:
                        object_key = item["Key"]
                        local_json_path = os.path.join(
                            local_directory, os.path.relpath(object_key, subdir_name)
                        )
                        local_json_path += ".json"
                        task = asyncio.create_task(
                            download_s3_object(
                                s3_client, bucket_name, object_key, local_json_path
                            )
                        )
                        tasks.append(task)
                    # Wait until all tasks in the current batch are done before moving on to the next batch
                    await asyncio.gather(*tasks)
            else:
                print(
                    f"No contents found in prefix {subdir_name} in bucket {bucket_name}."
                )


# Function to escape special characters for regex matching
def escape_special_characters(name):
    # This will escape special characters like '.', '(', ')', etc.
    return re.escape(name)


def process_annotations(json_dir, images_dir, label_dict, dataset_base_dir):
    # Define paths for the YOLO dataset
    images_dst_dir = os.path.join(dataset_base_dir, "images")
    labels_dst_dir = os.path.join(dataset_base_dir, "labels")

    # Create the destination directories if they do not exist
    os.makedirs(images_dst_dir, exist_ok=True)
    os.makedirs(labels_dst_dir, exist_ok=True)

    # List all JSON and image files
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    image_files = os.listdir(images_dir)

    for json_file in json_files:
        # Construct full path to JSON file
        json_path = os.path.join(json_dir, json_file)

        # Read and parse JSON content
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract relevant information
        json_image_name = data["image"].split("/")[-1]  # "19b64e56-zuccolellaf022.jpeg"
        # image_name = json_image_name.split("-", 1)[-1]  # "zuccolellaf022.jpeg"
        escaped_image_name = escape_special_characters(json_image_name)
        pattern = re.compile(
            f"^{escaped_image_name}$"
        )  # Regex pattern to match the actual image name

        matching_images = [img for img in image_files if pattern.match(img)]

        if not matching_images:
            # If no matching image is found, skip this JSON file
            print(f"No matching for {json_file}")

        matched_image_name = matching_images[0]  # Assuming there's only one match
        annotations = data["result"]["value"]
        label_name = annotations["rectanglelabels"][0]
        label_index = label_dict.get(
            label_name, -1
        )  # Defaults to -1 if label is not found

        abs_reference = {
            "x": annotations["x"],
            "y": annotations["y"],
            "width": annotations["width"],
            "height": annotations["height"],
        }

        yolo_reference = convert_bboxes_to_yolo_format(
            abs_reference, annotations["width"], annotations["height"]
        )

        x_center = yolo_reference["x_norm"]
        y_center = yolo_reference["y_norm"]
        w = yolo_reference["width_norm"]
        h = yolo_reference["height_norm"]

        # Continue with matched_image_name which is the image filename found in images_dir
        image_src_path = os.path.join(images_dir, matched_image_name)

        # Copy the image to the YOLO dataset images directory
        image_dst_path = os.path.join(images_dst_dir, matched_image_name)
        shutil.copy(image_src_path, image_dst_path)

        # Prepare and write the annotation file in the YOLO dataset labels directory
        annotation_content = f"{label_index} {x_center} {y_center} {w} {h}\n"
        annotation_filename = os.path.splitext(matched_image_name)[0] + ".txt"
        annotation_file_path = os.path.join(labels_dst_dir, annotation_filename)
        with open(annotation_file_path, "w", encoding="utf-8") as af:
            af.write(annotation_content)


if __name__ == "__main__":

    clear_files_in_directory(LOCAL_DIRECTORY)
    clear_files_in_directory(YOLO_DIR)
    # Run the download_all_objects coroutine
    asyncio.run(download_all_objects(BUCKET_NAME, LOCAL_DIRECTORY, SUBDIR_NAME))
    process_annotations(LOCAL_DIRECTORY, IMAGES_DIR, label_dict, YOLO_DIR)
