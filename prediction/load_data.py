import os
from PIL import Image

def load_images_from_dir(input_dir: str):
    print("Loading images from directory...")
    images = []
    image_paths = []

    # Get a list of image file paths in the input directory
    image_file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.bmp'))]

    for image_path in image_file_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            image_paths.append(os.path.basename(image_path))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    print(f"Loaded {len(images)} images from {input_dir}")
    return images, image_paths

if __name__ == "__main__":
    images, input_prompt = load_images_from_dir("./test_package", 'package')