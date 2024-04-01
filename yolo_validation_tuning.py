import os
import datetime
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from metrics import MetricsCalculator
from helper import (
    read_yolo_labels_ordered_modified,
    prepare_labels_for_sklearn,
    list_of_images,
    format_output_single_element,
    generate_unique_filename,
)


def generate_linspace(start, stop, step):
    # Calculate the number of samples needed to approximately achieve the step size
    # Adding 1 to include both endpoints
    num_samples = int(np.floor((stop - start) / step)) + 1

    # Generate linspace and convert to list
    linspace_values = np.linspace(start, stop, num_samples).tolist()

    return linspace_values


if __name__ == "__main__":

    modelpath = "models/fine_tuned.pt"
    valdir = "yolo_dataset/validation/images"
    labelsdir = "yolo_dataset/validation/labels"
    focus_type = "focus6"

    os.makedirs(f"tuning_results/{focus_type}", exist_ok=False)

    model = YOLO(modelpath)

    res = read_yolo_labels_ordered_modified(labelsdir)
    # print_first_n_elements(res, 20)
    ground_truth_labels = prepare_labels_for_sklearn(res)
    # print(ground_truth_labels[:20])
    test_images = list_of_images(valdir)

    # List to hold each iteration's metrics
    metrics_data = []
    for _threshold in generate_linspace(0.01, 0.8, 0.01):
        filename = generate_unique_filename(f"tuning_results/{focus_type}/report")
        predictions = []
        pred_probabilities = []
        detection_counts = []
        for test_image in test_images:
            result = model.predict(
                test_image, conf=_threshold, device=0, augment=True, verbose=False
            )[0]
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

        metrics = MetricsCalculator(
            ground_truth_labels, predictions, pred_probabilities, detection_counts
        )

        print(f"\nF1 for conf={_threshold}: {metrics.compute_f1()}")
        print(f"Precision for conf={_threshold}: {metrics.compute_precision()}")
        print(f"Recall for conf={_threshold}: {metrics.compute_recall()}")
        metrics.save_and_show_plots(filename)

        # Collecting the metrics
        iter_metrics = {
            "Threshold": _threshold,
            "F1": metrics.compute_f1(),
            "Precision": metrics.compute_precision(),
            "Recall": metrics.compute_recall(),
            "graphs": filename,
        }
        # Append the iteration's metrics to the list
        metrics_data.append(iter_metrics)

    # Convert the list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f"tuning_results/{focus_type}/metrics_{focus_type}.csv")

    print("\n", "*" * 80)
    df_results = pd.read_csv(
        f"tuning_results/{focus_type}/metrics_{focus_type}.csv", index_col=[0]
    )
    df_results.set_index("Threshold", inplace=True)
    print(df_results)
    print("\n", "*" * 80)
