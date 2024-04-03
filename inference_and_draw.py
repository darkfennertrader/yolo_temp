import os
import random
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from helper import (
    sample_random_image_and_label_from_dir,
    get_sorted_images_and_labels_from_dir,
)


palette = {"negative": [0, 255, 0], "positive": [0, 0, 255]}
label_text = {0: "negative", 1: "positive"}


def draw_bounding_box_yolo(image_path, bbox_yolo, prediction, label, thickness=2):
    """
    Draw a bounding box on an image from a YOLO format bounding box and display prediction and label below it.

    Parameters:
    - image_path: Path to the image file.
    - bbox_yolo: The YOLO format coordinates of the bounding box (x_center, y_center, width, height).
    - prediction: The prediction class or score.
    - label: The label of the bounding box.
    - thickness: The thickness of the bounding box lines.
    - palette: Dictionary mapping labels to colors (B, G, R).
    """

    # Default color is red if label not found in palette.
    pred_color = palette.get(prediction, [0, 0, 255])  # Get prediction color
    label_color = palette.get(label, [0, 0, 255])  # Get label color
    legend_color = (255, 255, 255)  # Black color for the legend

    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Convert from YOLO format to xmin, ymin, xmax, ymax format
    x_center, y_center, width, height = bbox_yolo
    xmin = int((x_center - width / 2) * w)
    xmax = int((x_center + width / 2) * w)
    ymin = int((y_center - height / 2) * h)
    ymax = int((y_center + height / 2) * h)

    # Draw the bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), pred_color, thickness)

    # Display prediction and label at the top left of the image
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    line_space = 30  # Increase line space for better separation

    # Place the label on top, followed by the prediction below it with increased separation
    prediction_position = (margin, margin + 15)
    label_position = (margin, prediction_position[1] + line_space)

    # Legend text in black color

    cv2.putText(
        image,
        "Prediction:",
        prediction_position,
        font,
        font_scale,
        legend_color,
        thickness,
    )

    cv2.putText(
        image,
        str(prediction),
        (prediction_position[0] + 200, prediction_position[1]),
        font,
        font_scale,
        pred_color,
        thickness,
    )
    cv2.putText(
        image,
        "Label:",
        label_position,
        font,
        font_scale,
        legend_color,
        thickness,
    )

    # Colorful value text
    cv2.putText(
        image,
        label,
        (label_position[0] + 200, label_position[1]),
        font,
        font_scale,
        label_color,
        thickness,
    )

    # Show the image with bounding box and texts
    plt.figure(figsize=(24, 20))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axis


def yolo_inference(model, threshold, iou, image, ground_truth):
    result = model.predict(
        image, conf=threshold, iou=iou, device=0, augment=False, verbose=False
    )[0].boxes

    if result.cls.numel() == 0:
        pred = 0
        bbox = [0, 0, 0, 0]
    else:
        pred = result.cls.item()
        bbox = result.xywhn.cpu().numpy().tolist()[0]

    # print("*" * 20)
    # print(result)
    # print()
    # print(result.xywh)
    # print(result.cls.numel())
    # print("*" * 20)

    true_label = label_text[ground_truth]

    draw_bounding_box_yolo(image, bbox, label_text[pred], true_label)


test_dir = "yolo_dataset/mar24/test2"
modelpath = "models/fine_tuned_Mar24.pt"
threshold = 0.535
iou = 0.7


model = YOLO(modelpath)
# test_image, label = sample_random_image_and_label_from_dir(test_dir)
image_list, label_list = get_sorted_images_and_labels_from_dir(test_dir)

for img, lab in zip(image_list, label_list):
    yolo_inference(model, threshold, iou, img, lab)

plt.show()
