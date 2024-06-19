from PIL import Image
import torch
import requests
from io import BytesIO
import logging
from groundingdino.util.inference import load_model, load_image, predict, annotate, convert_bbox
import cv2
import time
import numpy as np

class ObjectDetectionModel_Grdino():
    def __init__(self, device: int, box_threshold = 0.2, text_threshold = 0.2):
        super().__init__()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/gdinot-1.8m-odvg.pth", device = self.device)

    def predict(self, image, texts):
        # 'image' is now expected to be a URL

        image_source, image  =  load_image(image)

        predictions = []
        for text in texts:
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text[0],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device = self.device
            )
            boxes = convert_bbox(image_source, boxes)
            for box, score, label in zip(boxes, logits, phrases):
                box = [round(i, 2) for i in box.tolist()]
                predictions.append({
                    "Label": label,
                    "Confidence": round(score.item(), 3),
                    "Box": box
                })
        return predictions





