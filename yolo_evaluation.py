import os
import numpy as np
import torch
from ultralytics import YOLO
from metrics import MetricsCalculator
from helper import rename_files_with_suffix, convert_images_to_jpeg


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


if __name__ == "__main__":

    modelpath = "models/fine_tuned.pt"
    model = YOLO(modelpath)

    # convert_images_to_jpeg("yolo_dataset/mar24/positive")

    # rename_files_with_suffix("yolo_dataset/mar24/positive")
    ############################################################################
    # Pre-Processing step: convert test directory to a suitable format for sklearn metrics
    # labelsdir = "yolo_dataset/test/labels"
    # testdir = "yolo_dataset/test/images"
    # res = read_yolo_labels_ordered_modified(labelsdir)
    # # print_first_n_elements(res, 20)

    # ground_truth_labels = prepare_labels_for_sklearn(res)
    # print(ground_truth_labels[:20])
    ################################################################
    ####    NEW TEST SET MARCH 2024   ######
    pos_dir = "yolo_dataset/mar24/positive"
    neg_dir = "yolo_dataset/mar24/negative"
    testdir = "yolo_dataset/mar24/test"

    ground_truth_labels = assign_labels(pos_dir, neg_dir, testdir)
    # print(ground_truth_labels[:20])
    ##############################################################

    test_images = list_of_images(testdir)
    # print(test_images)
    predictions = []
    pred_probabilities = []
    detection_counts = []
    for test_image in test_images:
        result = model.predict(test_image, device=0)[0]
        # print(result)
        boxes = result.boxes

        # print("\n", "*" * 80)
        # print(boxes)
        if boxes.cls.numel() == 0:
            predictions.append(0)
            pred_probabilities.append(1)
        else:
            _class, _conf, num_elements, _ = format_output_single_element(
                boxes.cls, boxes.conf, boxes.data
            )
            predictions.append(int(_class.item()))
            pred_probabilities.append(_conf.item())
            detection_counts.append(num_elements)
        #     print(int(_class.item()))
        #     print(_conf.item())
        # print("*" * 80)

    # result.show()

    metrics = MetricsCalculator(
        ground_truth_labels, predictions, pred_probabilities, detection_counts
    )

    print(metrics.compute_f1())
    print(metrics.compute_precision())
    print(metrics.compute_recall())
    metrics.save_and_show_plots("metrics_test_new.jpeg", show=True)
