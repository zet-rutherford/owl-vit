from load_data import load_images_from_dir
from predict import ObjectDetectionModel
import sys
import time

if __name__ == "__main__":
    input_dir = sys.argv[1]
    input_prompt_str = sys.argv[2]
    print('-----------------------------START-------------------------------')

    images, image_paths = load_images_from_dir(input_dir)
    model = ObjectDetectionModel(device=0)
    print('-----------------------------Init model success-------------------------------')

    model.predict(images=images, image_paths=image_paths, prompt=input_prompt_str)
    print('-----------------------------Success-------------------------------')
    