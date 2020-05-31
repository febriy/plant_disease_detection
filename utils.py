from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist

import errno
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from config import PLANTDOC_CATEGORY_NAMES

class PlantDiseaseDataset(torch.utils.data.Dataset):
    """PlantDiseaseDataset."""

    def __init__(self, root_dir: str, coco_annotation_file: str, transform=None):
        """Read annotation file

        :param root_dir: Directory with all the images
        :type root_dir: str
        :param coco_annotation_file: JSON file containing annotations in COCO format. 
        :type coco_annotation_file: str
        :param transform: Optional transform to be applied on a sample, defaults to None
        :type transform: callable, optional
        """
        self.root_dir = root_dir
        self.transform = transform
        self.coco_annotations = COCO(coco_annotation_file)
        # print("self.coco_annotations.imgs.keys()",self.coco_annotations.imgs)
        self.image_ids_list = list(sorted(self.coco_annotations.imgs.keys()))

    def __len__(self):
        return len(self.image_ids_list)

    def __getitem__(self, index):

        coco = self.coco_annotations
        image_id = self.image_ids_list[index]

        annotation_id = coco.getAnnIds(imgIds=image_id)
        coco_annotation = coco.loadAnns(annotation_id)
        
        image_path = coco.loadImgs(image_id)[0]["file_name"]
        image = Image.open(os.path.join(self.root_dir, image_path))
        n_objects_in_image = len(coco_annotation)

        if n_objects_in_image > 0:

            # Bounding boxes
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, bbox input = [xmin, ymin, xmax, ymax]
            bounding_boxes = []
            for i in range(n_objects_in_image):
                xmin = coco_annotation[i]["bbox"][0]
                ymin = coco_annotation[i]["bbox"][1]
                xmax = xmin + coco_annotation[i]["bbox"][2]
                ymax = ymin + coco_annotation[i]["bbox"][3]
                bounding_boxes.append([xmin, ymin, xmax, ymax])
            bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
            
            category_ids = []
            for i in range(n_objects_in_image):
                category_ids.append(coco_annotation[0]["category_id"])
            labels = torch.as_tensor(category_ids, dtype=torch.int64)

            image_id = torch.tensor([image_id])
            
            areas = []
            for i in range(n_objects_in_image):
                areas.append(coco_annotation[i]["area"])
            areas = torch.as_tensor(areas, dtype=torch.float32)

            # is it a group of objects. In this case, all not in group.
            iscrowd = torch.zeros((n_objects_in_image,), dtype=torch.int64)  

            combined_annotations = {}
            combined_annotations["boxes"] = bounding_boxes
            combined_annotations["labels"] = labels
            combined_annotations["image_id"] = image_id
            combined_annotations["area"] = areas
            combined_annotations["iscrowd"] = iscrowd

            if self.transform is not None:
                image = self.transform(image)

            return image, combined_annotations

        elif n_objects_in_image == 0:

            combined_annotations = {}
            combined_annotations["boxes"] = torch.as_tensor([[1,1,416.0000,416.0000]], dtype=torch.float32)
            combined_annotations["labels"] = torch.zeros((1,), dtype=torch.int64)
            combined_annotations["image_id"] = torch.tensor([image_id])
            combined_annotations["area"] = torch.tensor([173056.000], dtype=torch.float32)
            combined_annotations["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
           
            if self.transform is not None:
                image = self.transform(image)

            return image, combined_annotations



def get_transform():
    transforms = [] # in case you need more than 1 transforms in the future
    transforms.append(torchvision.transforms.ToTensor()) 
    return torchvision.transforms.Compose(transforms)


def collate_fn(batch): # to create batch
    return tuple(zip(*batch))


def get_model_object_detection(num_classes):
    """ Finetuning from a pretrained model - load pretrained model and replace the pre-trained head """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

label_map = {k: v + 1 for v, k in enumerate(PLANTDOC_CATEGORY_NAMES)}


# https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
