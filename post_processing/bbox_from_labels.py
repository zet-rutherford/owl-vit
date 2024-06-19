import os
from PIL import Image, ImageDraw

def draw_bounding_boxes(image_dir, label_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

            # Load the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            width, height = image.size

            # Load the YOLO format labels
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()

                for label in labels:
                    label_data = label.strip().split()
                    label_index = int(label_data[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, label_data[1:])

                    # Convert YOLO format to pixel coordinates
                    x_min = int((x_center - bbox_width / 2) * width)
                    y_min = int((y_center - bbox_height / 2) * height)
                    x_max = int((x_center + bbox_width / 2) * width)
                    y_max = int((y_center + bbox_height / 2) * height)

                    # Draw the bounding box
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='red', width=2)

                    # Draw the label index (optional)
                    label_text = str(label_index)
                    draw.text((x_min, y_min - 10), label_text, fill='red')

            # Save the image with bounding boxes
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"Saved image with bounding boxes: {output_path}")

# Specify the image directory, label directory, and output directory
image_dir = './test_package'
label_dir = './labels'
output_dir = './results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Call the function to draw bounding boxes
draw_bounding_boxes(image_dir, label_dir, output_dir)