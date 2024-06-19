from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
import requests
from torchvision.ops import nms


class ObjectDetectionModel_Owl:
    def __init__(self, device, threshold = 0.18, threshold_nms = 0.3, input_size = 960):

        self.device = device
        self.threshold = threshold
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        self.input_size = input_size
        self.threshold_nms = threshold_nms

    def predict(self, image, texts):
        # image = Image.open(requests.get(image_path, stream=True).raw)
        # texts = [[text]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)

        # target_sizes = torch.Tensor([image.size[::-1]])
        target_sizes = torch.Tensor([[self.input_size, self.input_size]])

        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
        ratio =  max(image.size) / self.input_size

        predictions = []
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        # print("scores", scores)
        boxes_nms = []
  
        for box, score, label in zip(boxes, scores, labels):
            box = [int(i * ratio) for i in box.tolist()]
            box[0] = max(0, box[0])  # left
            box[1] = max(0, box[1])  # top
            box[2] = min(image.size[0], box[2])  # right
            box[3] = min(image.size[1], box[3])  # bottom
            boxes_nms.append(box)
        if(len(boxes_nms) > 0):
            boxes_tensor = torch.from_numpy(np.array(boxes_nms)).float().to(self.device)
            indices = nms(boxes_tensor, scores, iou_threshold=0.3)

            scores = scores[indices]
            boxes_tensor = boxes_tensor[indices]
            labels = labels[indices]

            for box, score, label in zip(boxes_tensor, scores, labels):
                box = box.tolist()
                predictions.append({
                    "Label": texts[label][0],         
                    "Confidence": round(score.item(), 2),
                    "Box": box,
                    "Custom": 0
                })

        return predictions
    
#   self test
# model = ObjectDetectionModel_Owl(device=0)
# url = 'https://cache.giaohangtietkiem.vn/d/e37ad1f7375bb49c2c3ead55a6deeba3.jpg'
# response = requests.get(url)
# image = Image.open(BytesIO(response.content)).convert("RGB")
# model.predict(image, [['barcode']])