import os
import numpy as np
import torch
from ultralytics import YOLO
from metrics import MetricsCalculator
from helper import (
    read_yolo_labels_ordered_modified,
    prepare_labels_for_sklearn,
    list_of_images,
    format_output_single_element,
)


if __name__ == "__main__":

    modelpath = "models/fine_tuned.pt"
    model = YOLO(modelpath)

    ############################################################################
    # Pre-Processing step: convert test directory to a suitable format for sklearn metrics
    valdir = "yolo_dataset/validation/images"
    labelsdir = "yolo_dataset/validation/labels"

    res = read_yolo_labels_ordered_modified(labelsdir)
    # print_first_n_elements(res, 20)

    ground_truth_labels = prepare_labels_for_sklearn(res)
    # print(ground_truth_labels[:20])
    ##############################################################

    test_images = list_of_images(valdir)
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
