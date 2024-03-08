from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from label_studio_converter import brush
import os
import cv2
import numpy as np
import random
import string
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from google.cloud import storage
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

class SegModel(LabelStudioMLBase):
    
    def _get_image_url(self, task, value='image'):
        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('gs://'):
            # Generate signed URL for GCS
            bucket_name, object_name = image_url.replace('gs://', '').split('/', 1)
            storage_client = storage.Client.from_service_account_json('/secrets/service_account_key.json')
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            try:
                image_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(hours=1),  # Adjust expiration time as needed
                    method="GET",
                )
            except Exception as exc:
                logger.warning(f'Can\'t generate signed URL for {image_url}. Reason: {exc}')
        return image_url
        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        ################################
        # Health check & init
        ################################
        task = tasks[0]
        print('###########################')
        task_id = task['id']
        print(f'Task ID: {task_id}')
        print(f'task: {task}')
        print(f'context: {context}')
        print('###########################')
        
        from_name, to_name, type_ = 'tag', 'image', "brushlabels"
        
        ################################################################
        # Load image
        ################################################################
        img_path = task['data']['image']
        if img_path.startswith('gs://'):
            image_path = self._get_image_url(task)
        else:
            image_path = get_image_local_path(
                img_path,
                label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                label_studio_host=LABEL_STUDIO_HOST
            )
        if image_path.startswith('https://') or image_path.startswith('http://'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        image_width, image_height = image.size
        
        ################################################################
        # Running segmentation model
        # https://huggingface.co/mattmdjaga/segformer_b2_clothes
        ################################################################
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        ################################################################
        # Format prediction results
        ################################################################
        prediction_results = []
        # original label dict from the model
        # labels = {0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}
        # modified label dictionary
        labels_dict = {1: "accessory", 
                       3: "accessory", 
                       4: "upper-clothes", 
                       5: "skirt", 
                       6: "pants", 
                       7: "dress", 
                       8: "accessory", 
                       9: "footwear", 
                       10: "footwear", 
                       16: "accessory", 
                       17: "accessory"}
        unique_labels = np.unique(pred_seg.numpy())
        # merge left and right shoe
        if 10 in unique_labels:
            pred_seg[pred_seg == 10] = 9
        # merge all accessories
        elif 3 in unique_labels:
            pred_seg[pred_seg == 10] = 1
        elif 8 in unique_labels:
            pred_seg[pred_seg == 10] = 1
        elif 16 in unique_labels:
            pred_seg[pred_seg == 10] = 1
        elif 17 in unique_labels:
            pred_seg[pred_seg == 10] = 1
        unique_labels = np.unique(pred_seg.numpy())
        
        # divide masks
        for label in unique_labels:
            # ignore background, hair, face, and body
            if label not in labels_dict.keys():
                continue
            text_label = labels_dict[label]
            mask = ((pred_seg == label).int() * 255).numpy()
            mask = mask.astype(int)
            mask = brush.mask2rle(mask)
            prediction_results.append({
                "original_width": image_width,
                "original_height": image_height,
                "value": {
                    "format": "rle",
                    "rle": mask,
                    "brushlabels": [text_label]
                },
                "id": ''.join(
                    random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                    for _ in range(10)),
                "from_name": from_name,
                "to_name": to_name,
                "type": type_
            },)

        predictions = [
            {'result': prediction_results}
        ]
        
        return predictions