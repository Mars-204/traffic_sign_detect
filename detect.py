from typing import Any, Tuple
import torch
import supervision as sv
import transformers
import pytorch_lightning
import os
import torchvision
from pathlib import Path
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

dataset = 'C:\\MyData\\Codes\\traffic_sign_detection\\traffic_sign_detect\\Hackathon.v17i.coco'

ANNOTATION = "_annotations.coco.json"
TRAIN_DIR = os.path.join(dataset, "train")
VAL_DIR = os.path.join(dataset, "valid")
TEST_DIR = os.path.join(dataset, "test")

class TrafficDetect(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        data_folder: str,
        image_processor,
        train: bool = True
    ):
       annotation_path = os.path.join(data_folder, ANNOTATION)
       super(TrafficDetect, self).__init__(data_folder, annotation_path)
       self.processor = image_processor

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        images, annotations = super(TrafficDetect, self).__getitem__(index)
        image_id = self.ids[index]
        annotations = {'image_id':image_id, 'annotations': annotations}
        encoding = self.processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squueze()
        target = encoding["labels"][0]
        return pixel_values, target

TRAIN_DATASET = TrafficDetect(data_folder=TRAIN_DIR, image_processor=processor, train=True)
VAL_DATASET = TrafficDetect(data_folder=VAL_DIR, image_processor=processor, train=False)
TEST_DATASET = TrafficDetect(data_folder=TEST_DIR, image_processor=processor, train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))



def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)