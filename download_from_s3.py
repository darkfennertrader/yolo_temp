import os
import boto3


def download_s3_folder(bucket_name, s3_folder, local_dir):
    """
    Download the contents of a folder directory in an S3 bucket.

    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Folder path in the S3 bucket. This is essentially a prefix for the S3 objects.
    :param local_dir: Local directory to which the files will be downloaded.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Construct the full local file path
            local_file_path = os.path.join(local_dir, key[len(s3_folder) :])
            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            if not key.endswith(
                "/"
            ):  # Check if the key is a file and not a directory placeholder
                # Download the file
                s3.download_file(bucket_name, key, local_file_path)
                print(f"Downloaded {key} to {local_file_path}")


# Example usage
bucket_name = "vvip-yolo-bucket"
s3_folder = "validation/"  # Make sure it ends with a '/'
local_dir = "yolo_dataset/validation"  # The target local directory

download_s3_folder(bucket_name, s3_folder, local_dir)
