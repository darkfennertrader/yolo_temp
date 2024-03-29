import os
import json
import boto3

BUCKET_NAME = "mcnv-output"
SUBDIR_NAME = "positive/"
LOCAL_DIRECTORY = "./from_s3"


def download_s3_objects(bucket_name, local_directory, subdir_name):
    # Create an S3 client
    s3 = boto3.client("s3")
    # Initialize the paginator
    paginator = s3.get_paginator("list_objects_v2")
    # Create a page iterator from the paginator
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=subdir_name)

    # Iterate over each page of objects
    for page in page_iterator:
        if "Contents" in page:
            for item in page["Contents"]:
                object_key = item["Key"]
                # Define the local JSON file path
                local_json_path = os.path.join(
                    local_directory, os.path.relpath(object_key, subdir_name) + ".json"
                )

                # Get the object from S3
                obj = s3.get_object(Bucket=bucket_name, Key=object_key)

                # Read the content of the object
                content = obj["Body"].read().decode("utf-8")

                # Check if the content is empty (likely a directory placeholder)
                if not content.strip():
                    print(
                        f"Skipping '{object_key}': No content (possible directory placeholder)"
                    )
                    continue  # Skip this object and move to the next one

                # Parse the content into a Python dict
                try:
                    data_dict = json.loads(content)

                    # Extract the desired parts
                    image_path = data_dict["task"]["data"]["image"]
                    result_data = data_dict["result"][0]
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
                    continue  # Skip this object and move to the next one

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(local_json_path), exist_ok=True)
                # Save the new dict as a JSON file locally
                with open(local_json_path, "w", encoding="utf-8") as json_file:
                    json.dump(new_data_dict, json_file, indent=4)

                print(
                    f"Processed '{object_key}' from S3 bucket '{bucket_name}' and saved data to '{local_json_path}'"
                )
        else:
            print(f"No contents found in prefix {subdir_name} in bucket {bucket_name}.")


if __name__ == "__main__":
    download_s3_objects(BUCKET_NAME, LOCAL_DIRECTORY, SUBDIR_NAME)
