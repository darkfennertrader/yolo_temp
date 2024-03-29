import os
import random
import numpy as np
import onnxruntime
from PIL import Image
from ultralytics import YOLO

model = YOLO("models/best.pt").export(
    format="onnx", opset=9, data="yolo_dataset/dataset.yaml"
)

onnx_model = YOLO("models/best.onnx")


def sample_random_image_from_dir(directory_path, seed=42, valid_extensions=("jpeg")):
    """
    Select a random image file from a specified directory.

    Parameters:
    - directory_path: The path to the directory containing image files.
    - valid_extensions: A tuple of valid image file extensions.

    Returns:
    The full path to a randomly selected image file.
    """
    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter for files with valid image extensions
    image_files = [
        file for file in all_files if file.lower().endswith(valid_extensions)
    ]

    if not image_files:
        raise ValueError("No image files found in the specified directory")

    # Select a random image file
    random.seed(seed)
    selected_image = random.choice(image_files)

    # Return the full path to the selected image
    return os.path.join(directory_path, selected_image)


def inspect_yolo_inference_onnx(onnx_model_path, image_path, input_size=640):
    """
    Inspect the structure of the output from the YOLO model exported in ONNX format.

    Parameters:
    - onnx_model_path: The path to the ONNX model file.
    - image_path: The path to the image file.
    - input_size: The size to which the image is resized before passing to the model.
    """
    # Assume previous imports and image preprocessing steps before this comment

    session = onnxruntime.InferenceSession(onnx_model_path)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((input_size, input_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, [2, 0, 1])
    image_np = np.expand_dims(image_np, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_np})

    # For diagnostic purposes, let's print the output structure
    print("Output Structure:", type(output))
    if isinstance(output, list):
        print("Number of elements in output list:", len(output))
        for i, out in enumerate(output):
            print(
                f"Element {i}, Type: {type(out)}, Shape: {out.shape if hasattr(out, 'shape') else 'N/A'}"
            )


# Call the function with the correct paths to your model and an image
# inspect_yolo_inference_onnx(onnx_model_path, image_path)


def yolo_inference_onnx_v3(
    onnx_model_path, image_path, input_size=640, conf_thres=0.25
):
    """
    Perform inference using a version adapted to handle binary classification and confidence scores correctly.

    The function assumes class IDs should be within [0, 1] and confidence scores are already in the correct format.
    """

    # Load the model and prepare the image
    session = onnxruntime.InferenceSession(onnx_model_path)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((input_size, input_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, [2, 0, 1])
    image_np = np.expand_dims(image_np, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_np})[0]  # Assuming single output.

    # Process the output
    detections = []
    for det in output[0]:  # Loop through each detection in the batch
        class_id = int(det[1])
        # Ensure class ID is within expected range
        if class_id not in {0, 1}:
            continue

        score = det[2]
        # Skip detections below the confidence threshold
        if score < conf_thres:
            continue

        bbox = det[3:7]  # Extract bounding box coordinates
        detections.append(
            {
                "class": class_id,
                "probability": score,  # Interpret this value as given; adjust based on your model's training
                "bbox": bbox.tolist(),
            }
        )

    return detections


def yolo_inference_onnx(onnx_model_path, image_path, input_size=640, conf_thres=0.25):
    """
    Perform inference using a corrected version to accommodate the model's actual output format.

    Parameters:
    - onnx_model_path: Path to the ONNX model file.
    - image_path: Path to the image file.
    - input_size: Resize dimension for the input image.
    - conf_thres: Confidence threshold for filtering detections.
    """

    # Load the model and prepare the image
    session = onnxruntime.InferenceSession(onnx_model_path)
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((input_size, input_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, [2, 0, 1])
    image_np = np.expand_dims(image_np, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_np})[0]  # Assuming single output.

    # Process the output
    detections = []
    for det in output[0]:  # Loop through each detection in the batch
        score = det[2]
        if score < conf_thres:
            continue
        class_id = int(det[1])
        bbox = det[3:7]  # Extract bounding box coordinates
        detections.append(
            {"class": class_id, "probability": score, "bbox": bbox.tolist()}
        )

    return detections


# Given paths to the ONNX model and an image, you'd call this function as follows:
# detections = yolo_inference_onnx_v2(onnx_model_path, image_path)
# print(detections)


if __name__ == "__main__":
    ONNX_MODEL_PATH = "models/best.onnx"
    IMAGE_DIR = "yolo_dataset/test/images"

    image_path = sample_random_image_from_dir(IMAGE_DIR)

    detections = yolo_inference_onnx(ONNX_MODEL_PATH, image_path)
    print(detections)

    # inspect_yolo_inference_onnx(ONNX_MODEL_PATH, image_path)
    onnx_model.predict(image_path)
