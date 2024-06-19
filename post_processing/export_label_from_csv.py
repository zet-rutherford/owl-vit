import ast
import os
import pandas as pd

def export_yolo_format(input_csv, output_dir):
    data = pd.read_csv(input_csv)

    for _, row in data.iterrows():
        image_id = row['id']
        width = int(row['width'])
        height = int(row['height'])
        results = ast.literal_eval(row['result'])

        output_file = os.path.join(output_dir, f"{os.path.splitext(image_id)[0]}.txt")

        with open(output_file, 'w') as f:
            for result in results:
                label = result['Label']
                confidence = result['Confidence']
                box = result['Box']

                # Convert bounding box coordinates to YOLO format
                x_center = (box[0] + box[2]) / (2 * width)
                y_center = (box[1] + box[3]) / (2 * height)
                bbox_width = (box[2] - box[0]) / width
                bbox_height = (box[3] - box[1]) / height

                # Write the YOLO format line
                yolo_line = f"{label_to_index[label]} {x_center} {y_center} {bbox_width} {bbox_height}"
                f.write(yolo_line + '\n')

        print(f"Exported YOLO format for {image_id}")

# Assuming you have a dictionary mapping labels to indices
label_to_index = {
    'p': 0,
    # Add more labels and their corresponding indices
}

# Specify the input CSV file and the output directory
input_csv = './dataset/1K_sample_package/annotations.csv'
output_dir = './dataset/1K_sample_package/labels'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Call the function to export the YOLO format
export_yolo_format(input_csv, output_dir)