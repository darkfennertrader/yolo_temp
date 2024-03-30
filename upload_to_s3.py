import asyncio
import os
import aioboto3


async def clean_s3_directory(session, bucket, prefix):
    """
    Delete all objects under a specific prefix in the specified S3 bucket.
    """
    async with session.client("s3") as s3:
        paginator = s3.get_paginator("list_objects_v2")
        deletion_list = []

        # Use Paginator for handling large datasets
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                # Create a list of objects to delete, excluding the prefix itself
                deletion_list.extend(
                    [
                        {"Key": obj["Key"]}
                        for obj in page["Contents"]
                        if obj["Key"] != prefix  # Exclude the prefix itself
                    ]
                )

        # Delete the objects if list is not empty and well-formed
        if deletion_list and all(
            "Key" in obj and isinstance(obj["Key"], str) for obj in deletion_list
        ):
            # Ensure the deletion list does not exceed AWS limits
            for i in range(0, len(deletion_list), 1000):  # AWS limit per request
                chunk = deletion_list[i : i + 1000]
                await s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
            print(f"Cleaned {len(deletion_list)} objects from s3://{bucket}/{prefix}")
        else:
            print(f"No objects to delete at s3://{bucket}/{prefix}")


async def upload_file_to_s3(s3_client, bucket_name, file_path, s3_path):
    """
    Asynchronously upload a file to an S3 bucket.

    :param s3_client: The aioboto3 S3 client
    :param bucket_name: The name of the bucket
    :param file_path: The path to the file to upload
    :param s3_path: The S3 key under which to store the file
    """
    await s3_client.upload_file(file_path, bucket_name, s3_path)
    # print(f"Uploaded: {file_path} to {s3_path}")


async def upload_dir_to_s3(
    session, bucket_name, source_dir, s3_prefix="", allowed_extensions=None
):
    """
    Recursively upload a directory and its subdirectories to S3 asynchronously.

    :param bucket_name: The name of the bucket
    :param source_dir: The local directory to upload from
    :param s3_prefix: The prefix to add to S3 keys (used for maintaining directory structure)
    :param allowed_extensions: Optional list of allowed file extensions to upload
    """
    async with session.client("s3") as s3_client:
        tasks = []
        files_uploaded = 0
        for root, _, files in os.walk(source_dir):
            # Calculate the relative path to maintain the directory structure
            relative_path = os.path.relpath(root, source_dir)
            if relative_path == ".":
                relative_path = ""
            for file in files:
                if allowed_extensions and not any(
                    file.endswith(ext) for ext in allowed_extensions
                ):
                    continue
                files_uploaded += 1
                file_path = os.path.join(root, file)
                s3_path = os.path.join(s3_prefix, relative_path, file).replace(
                    "\\", "/"
                )  # Convert Windows paths to S3 format
                task = upload_file_to_s3(s3_client, bucket_name, file_path, s3_path)
                tasks.append(task)
        await asyncio.gather(*tasks)

        return files_uploaded


async def main():
    dir_name1 = "negative"
    dir_name2 = "positive"
    dir_name3 = "test"

    TRAIN_DIRECTORY = f"yolo_dataset/mar24/{dir_name1}"
    VALIDATION_DIRECTORY = f"yolo_dataset/mar24/{dir_name2}"
    TEST_DIRECTORY = f"yolo_dataset/mar24/{dir_name3}"
    BUCKET = "vvip-yolo-bucket2"
    PREFIXES = ["negative/", "positive/", "test/"]
    session = aioboto3.Session()

    print("\nUploading directories and ist structures.....")

    for prefix in PREFIXES:
        await clean_s3_directory(session, BUCKET, prefix)

    files_uploaded = await upload_dir_to_s3(
        session, BUCKET, TRAIN_DIRECTORY, s3_prefix=dir_name1
    )
    print(f"\n{dir_name1} files uploaded: {files_uploaded}")

    files_uploaded = await upload_dir_to_s3(
        session, BUCKET, VALIDATION_DIRECTORY, s3_prefix=dir_name2
    )
    print(f"\n{dir_name2} files uploaded: {files_uploaded}")

    files_uploaded = await upload_dir_to_s3(
        session, BUCKET, TEST_DIRECTORY, s3_prefix=dir_name3
    )
    print(f"\n{dir_name3} files uploaded: {files_uploaded}")


if __name__ == "__main__":
    asyncio.run(main())
