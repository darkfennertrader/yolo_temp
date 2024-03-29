def convert_bboxes_to_yolo_format(bboxes, img_width, img_height):

    yolo_bboxes = []
    for bbox in bboxes:

        pixel_x = bbox["x"] / 100.0 * img_width
        pixel_y = bbox["y"] / 100.0 * img_height
        pixel_width = bbox["width"] / 100.0 * img_width
        pixel_height = bbox["height"] / 100.0 * img_height

        x_center = (pixel_x + pixel_width / 2) / img_width
        y_center = (pixel_y + pixel_height / 2) / img_height
        width_norm = pixel_width / img_width
        height_norm = pixel_height / img_height

        yolo_bboxes.append(
            {
                "class_id": bbox["class_id"],
                "x_center_norm": x_center,
                "y_center_norm": y_center,
                "width_norm": width_norm,
                "height_norm": height_norm,
            }
        )

    return yolo_bboxes


# Example usage:
bboxes_example = [
    {"x": 75, "y": 0, "width": 25, "height": 34, "class_id": 0},
    {"x": 150, "y": 50, "width": 100, "height": 150, "class_id": 1},
    {
        "x": 64.10835214446952,
        "y": 35.12354750253761,
        "width": 12.942061700526708,
        "height": 28.88784522853637,
        "class_id": 0,
    },
]
bboxes_example = [
    {
        "x": 59.488151309979386,
        "y": 12.209302325581394,
        "width": 10.691418899028562,
        "height": 43.8953488372093,
        "class_id": 1,
    },
]

img_width_example = 1008
img_height_example = 596

yolo_bboxes_example = convert_bboxes_to_yolo_format(
    bboxes_example, img_width_example, img_height_example
)
print(yolo_bboxes_example)


# Filename for the text file
filename = "ZammarrelliA2.txt"

# Writing the formatted contents to a file
with open(filename, "w", encoding="utf-8") as f:
    for det in yolo_bboxes_example:
        f.write(
            f"{det['class_id']} {det['x_center_norm']} {det['y_center_norm']} "
            f"{det['width_norm']} {det['height_norm']}\n"
        )
