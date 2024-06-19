# from groundingdino.util.inference import load_model, load_image, predict, annotate
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
from Owlv2 import ObjectDetectionModel_Owl
# from Grdino import ObjectDetectionModel_Grdino
import torch
import pandas as pd
import ast
# from load_data import download_image_with_retry
from config import app_config as cfg
import os


def convert_to_list_of_lists(input_string):
    # Splitting the string by ', ' and converting each item into a list
    input_list = ast.literal_eval(input_string)
    return [ [item] for item in input_list ]

class ObjectDetectionModel():
    def __init__(self, device: int):
        super().__init__()
        if device < 0:
            self.device = torch.device('cpu')
            print("[Device] cpu",)

        else:
            self.device = torch.device(f'cuda:{device}')
            print("[Device] ", f'cuda: {device}')
        
        #   model Grounding Dino
        #self.model_groundingdino = ObjectDetectionModel_Grdino(device=self.device, box_threshold = 0.2, text_threshold = 0.2)
        #   model OWL2
        self.model_owl = ObjectDetectionModel_Owl(device = device,
                                                threshold = cfg.MODEL_BASE_CONF,
                                                threshold_nms = cfg.MODEL_BASE_IOU,
                                                input_size= cfg.MODEL_BASE_INPUTSIZE)

        print("Init ObjectDetectionModel_Owl Success")

    def predict(self, images, image_paths, prompt: str):
        print(image_paths)
        print('[input prompt] ', prompt)
        image_count = 1  # Initialize the image counter
        output_file = "src/open-vocabulary-object-detection/prediction/tmp/annotations.csv"

        # Check if the output file exists, if not, create it with the header row
        if not os.path.isfile(output_file):
            with open(output_file, 'w', newline='') as file:
                file.write('id,width,height,result\n')

        for image, image_path in zip(images, image_paths):
            try:
                prediction = self.model_owl.predict(image, prompt)
                w, h = image.size
                data = {
                    'id': os.path.basename(image_path),
                    'width': str(w),
                    'height': str(h),
                    'result': str(prediction)
                }
                print(f"[{image_count}/{len(images)}] [predict success] {image_path}, {prediction}")

                # Append the data to the CSV file
                with open(output_file, 'a', newline='') as file:
                    pd.DataFrame([data]).to_csv(file, header=False, index=False)

                image_count += 1
            except Exception as e:
                print(f"[{image_count}/{len(images)}] [predict failed] {image_path}, {e}")
                image_count += 1

        print(f'[Total data] {image_count - 1}')
        print(f"Saved predictions to {output_file}")


