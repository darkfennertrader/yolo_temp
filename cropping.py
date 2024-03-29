import os
import random
from ultralytics import YOLO


def select_random_images(dir_path, N):
    # Get a list of all files in the directory
    all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    # Filter out non-image files (assuming images have extensions like .jpg, .jpeg, .png, etc.)
    image_files = [
        f
        for f in all_files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    # Select N random images
    selected_images = random.sample(image_files, min(N, len(image_files)))

    return selected_images


def crop_yolo_bbox(image, bbox, image_size=None):
    """
    Crop the image to the bounding box specified by the YOLO model.

    Args:
    - image (numpy.ndarray): The input image.
    - bbox (list or tuple): The bounding box in YOLO format [x_center, y_center, width, height], values normalized to [0, 1].
    - image_size (tuple, optional): The size (width, height) of the image. If None, it will be extracted from the image.

    Returns:
    - cropped_image (numpy.ndarray): The cropped image.
    """
    if image_size is None:
        height, width = image.shape[:2]
    else:
        width, height = image_size

    # Convert from YOLO format to absolute coordinates
    x_center, y_center, w, h = bbox
    x_center *= width
    y_center *= height
    w *= width
    h *= height

    # Calculate the top-left and bottom-right coordinates
    x1 = int(round(x_center - w / 2))
    y1 = int(round(y_center - h / 2))
    x2 = int(round(x_center + w / 2))
    y2 = int(round(y_center + h / 2))

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


pos_dir = "/home/solidsnake/mcnv_raw_data/yolo_dataset/positive/images"
treated_dir = "/home/solidsnake/mcnv_raw_data/yolo_dataset/negative/images"
neg_dir = "/home/solidsnake/mcnv_raw_data/yolo_dataset/no-area/images"

model = YOLO("models/fine_tuned.pt")
# model.export(format="onnx", dynamic=True)
# model.export(format="tfjs")

pos_images = select_random_images(pos_dir, 3)
neg_images = select_random_images(neg_dir, 3)
treat_images = select_random_images(treated_dir, 1)
print(treat_images)

# Run batched inference on a list of images
results = model.predict(treat_images, device=0)


# Process results list
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # print(boxes)
    probs = result.probs  # Probs object for classification outputs
    print(probs)
    result.show()  # display to screen
    result.save(filename="tmp/result.jpg")  # save to disk
