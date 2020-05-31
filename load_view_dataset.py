from torch.utils import data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from config import *
from utils import (
    get_model_object_detection,
    collate_fn,
    get_transform,
    PlantDiseaseDataset,
)


print("Torch version:", torch.__version__)

plant_disease_dataset = PlantDiseaseDataset(
    root_dir=train_data_dir, coco_annotation_file=train_coco, transform=get_transform()
)

plant_data_iterable = torch.utils.data.DataLoader(
    plant_disease_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def show_image_and_bbox(data_loader):
    for images, annotations in data_loader:
        # image shape is [batch_size, 3 (due to RGB), height, width]
        img = transforms.ToPILImage()(images[0])

        fig,ax = plt.subplots(1)
        ax.imshow(img)

        bbox_numpy = annotations[0]["boxes"].numpy()
        for bbox in bbox_numpy:
            # minimum coordinate
            # https://github.com/matplotlib/matplotlib/issues/15401
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            rect = patches.Rectangle((x_min,y_min),width,height,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        print ("------------------")
        print (annotations)
        plt.show()

show_image_and_bbox(plant_data_iterable)