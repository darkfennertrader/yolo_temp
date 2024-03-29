import os
from PIL import Image
import matplotlib.pyplot as plt


def convert_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert to JPEG if necessary
        if img.format != "JPEG":
            img = img.convert("RGB")

        # Save the cropped image
        img.save(output_path, "JPEG")


# Function to convert and crop a dir of images
def convert_images(image_dir, output_dir):
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")
        ):
            input_path = os.path.join(image_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".jpeg"
            output_path = os.path.join(output_dir, output_filename)

            convert_image(input_path, output_path)

    # Display some of the cropped images using matplotlib
    cropped_images = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    num_images_to_display = 5  # Change this to display more or fewer images

    plt.figure(figsize=(15, 10))

    for i, img_path in enumerate(cropped_images[:num_images_to_display]):
        img = Image.open(img_path)
        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    pass
    # image_dir = "./raw_data/positive"
    # output_dir = "./converted/positive"
    # convert_images(image_dir, output_dir)

    # image_dir = "./raw_data/negative"
    # output_dir = "./converted/negative"

    # convert_images(image_dir, output_dir)
