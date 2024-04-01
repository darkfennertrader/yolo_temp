import os
import random
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics.utils import yaml_load
import matplotlib.pyplot as plt
from helper import convert_images_to_jpeg

# model = YOLO("models/fine_tuned.pt").export(
#     format="onnx", opset=9, data="yolo_dataset/dataset.yaml"
# )

ONNX_MODEL = "./models/fine_tuned.onnx"
CONFIDENCE_THRES = 0.45
IOU_THRES = 0.7
TEST_DIR = "yolo_dataset/mar24/test2"


class YOLOv9:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(
        self,
        onnx_model=ONNX_MODEL,
        confidence_thres=CONFIDENCE_THRES,
        iou_thres=IOU_THRES,
    ):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """

        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.img = None
        self.img_height = None
        self.img_width = None

        # Load the class names from the COCO dataset
        # self.classes = yaml_load(check_yaml("yolo_dataset/dataset.yaml"))["names"]
        self.classes = yaml_load("yolo_dataset/dataset.yaml")["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.color_palette[0] = [0, 255, 0]  # Set the color to green
        self.color_palette[1] = [0, 0, 255]  # BGR Set the color to red

        # Create an inference session using the ONNX model and specify execution providers
        # self.session = ort.InferenceSession(
        #     self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        # )
        self.session = ort.InferenceSession(
            self.onnx_model, providers=["CPUExecutionProvider"]
        )
        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)  # type: ignore

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,  # type: ignore
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Read the input image using OpenCV
        self.img = cv2.imread(input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        score_candidates = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            print(score)
            class_id = class_ids[i]
            print(class_id)
            score_candidates.append(score)

        # draw the rectangle that corresponds to the max predicted prob. if any
        if len(score_candidates) > 0:
            max_value_index = score_candidates.index(max(score_candidates))
            box = boxes[max_value_index]
            score = scores[max_value_index]
            class_id = class_ids[max_value_index]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def predict(self, input_image):
        img_data = self.preprocess(input_image)
        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image

    @staticmethod
    def sample_random_image_and_label_from_dir(directory, seed=None):
        # Get all image files from the directory that ends with .jpeg
        image_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.endswith(".jpeg")
        ]

        # Seed the random generator if a seed is provided
        if seed:
            random.seed(seed)

        # Choose a random image file
        selected_image = random.choice(image_files)

        # Prepare paths to check in positive and negative directories
        base_dir = os.path.dirname(directory)
        positive_path = os.path.join(
            base_dir, "positive", os.path.basename(selected_image)
        )
        negative_path = os.path.join(
            base_dir, "negative", os.path.basename(selected_image)
        )

        # Determine the label based on the existence of the file in positive/negative directories
        if os.path.exists(positive_path):
            label = "positive"
        elif os.path.exists(negative_path):
            label = "negative"
        else:
            label = "unknown"  # In case the image doesn't exist in either directory for some reason

        return selected_image, label

    @staticmethod
    def get_sorted_images_and_labels_from_dir(directory):
        # Get all image files from the directory
        image_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        # Prepare base directory for positive and negative path checks
        base_dir = os.path.dirname(directory)

        # Initialize a list to hold tuples of (selected_image, label)
        images_and_labels = []

        # Loop through all image files to determine their labels
        for image_file in image_files:
            positive_path = os.path.join(
                base_dir, "positive", os.path.basename(image_file)
            )
            negative_path = os.path.join(
                base_dir, "negative", os.path.basename(image_file)
            )

            # Determine the label based on the existence of the file in positive/negative directories
            if os.path.exists(positive_path):
                label = 1  # Label for positive
            elif os.path.exists(negative_path):
                label = 0  # Label for negative
            else:
                continue  # Skip images that don't exist in either directory

            # Append the selected image and its label as a tuple
            images_and_labels.append((image_file, label))

        # Sort the list of tuples based on the image files
        images_and_labels.sort(key=lambda x: x[0])

        # Unpack the sorted list of tuples into two lists
        selected_images, labels = (
            zip(*images_and_labels) if images_and_labels else ([], [])
        )

        return list(selected_images), list(labels)


def worker_function(image_path, ground_truth):
    # Create a new instance inside each worker
    yolo_instance = YOLOv9()
    prediction = yolo_instance.predict(image_path)
    return (prediction, ground_truth, image_path)


def perform_predictions_in_parallel(image_paths, ground_truths):
    # Context manager for the multiprocessing pool
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.starmap(worker_function, zip(image_paths, ground_truths))

    return results


def menu():
    print("\nMenu:")
    print("1. Predict a random image")
    print("2. Predict all images")
    print("3. Exit")
    return input("Enter your choice (1-3): ")


if __name__ == "__main__":

    print("Welcome to YOLOv9 Object Detection for MCNV disease")
    detection = YOLOv9()  # Create an instance of the YOLOv9 class
    convert_images_to_jpeg(TEST_DIR)

    while True:
        choice = menu()

        if choice == "1":
            rand_image, label = YOLOv9.sample_random_image_and_label_from_dir(TEST_DIR)
            output_image = detection.predict(rand_image)

            # Determine the color based on the label
            if (
                "positive" in label.lower()
            ):  # Assuming "positive" in the label indicates a positive case
                color = (0, 0, 255)  # Red color in BGR for positive cases
            else:
                color = (0, 255, 0)  # Green color in BGR for negative cases

            # Calculate the position to place the label above the image
            label_height = 35  # Adjust label height as needed
            output_image_with_label = np.zeros(
                (output_image.shape[0] + label_height, output_image.shape[1], 3),
                dtype=np.uint8,
            )
            output_image_with_label[label_height:, :] = output_image

            # Add label text above the output image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, label_height - 10)  # Adjust org to place the text appropriately
            fontScale = 1  # Font scale
            thickness = 2  # Line thickness in px
            output_image_with_label = cv2.putText(
                output_image_with_label,
                f"Label: {label}",
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # Convert BGR image to RGB
            output_image_rgb = cv2.cvtColor(output_image_with_label, cv2.COLOR_BGR2RGB)

            # Use Matplotlib to display the image along with the label
            plt.figure(figsize=(24, 16))  # You can adjust the figure size as needed
            plt.imshow(output_image_rgb)
            plt.axis("off")  # Don't show axes for images
            plt.show()

        elif choice == "2":
            images_list, labels_list = YOLOv9.get_sorted_images_and_labels_from_dir(
                TEST_DIR
            )
            print("\nPerforming predictions in parallel. Please wait...")
            results = perform_predictions_in_parallel(images_list, labels_list)
            print("\n", results)

        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2 or 3.")
