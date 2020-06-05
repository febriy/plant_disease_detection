import os
from torch.utils import data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2

from config import *
from utils import (
    get_model_object_detection,
    collate_fn,
    get_transform,
    PlantDiseaseDataset,
)

print("Torch version:", torch.__version__)



def load_checkpoint(checkpoint_path:str, model, optimizer):
    """Load checkpoint

    :param checkpoint_path: path to save checkpoint
    :type checkpoint_path: str
    :param model: model that we want to load checkpoint parameters into
    :param optimizer: optimizer we defined in previous training
    :return: model, optimizer, checkpoint['epoch'], valid_loss_min.item()
    """    
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['losses']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def get_prediction(img_path:str, inference_threshold: float):
    """Run object detection model on image and return the class and bounding box coordinates

    :param img_path: path to the image being inferred
    :type img_path: str
    :param inference_threshold: minimum value for prediction score, above which the bounding box will be drawn, defaults to 0.5
    :type inference_threshold: float, optional
    :return: class, box coordinates for prediction score > threshold
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)

    pred = model([img])
    pred_class = [PLANTDOC_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>inference_threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

def detect_object_draw_bbox(img_path: str, inference_threshold: float =0.5, line_thickness: float =1, text_size:float =0.5, text_thickness:float=1):
    """Run object detection model on image and draw bounding box around inferred targets. 

    :param img_path: path to the image being inferred
    :type img_path: str
    :param inference_threshold: minimum value for prediction score, above which the bounding box will be drawn, defaults to 0.5
    :type inference_threshold: float, optional
    :param line_thickness: thickness of the lines on the bounding box, defaults to 1
    :type line_thickness: float, optional
    :param text_size: size of the class label text, defaults to 1
    :type text_size: float, optional
    :param text_thickness: thickness of class lable text, defaults to 1
    :type text_thickness: float, optional
    """ 
    boxes_list, pred_class = get_prediction(img_path, inference_threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in range(len(boxes_list)):
        cv2.rectangle(img, boxes_list[box][0], boxes_list[box][1],color=(0, 255, 0), thickness=line_thickness)
        cv2.putText(img,pred_class[box], boxes_list[box][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_thickness)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialise model
    model = get_model_object_detection(num_classes)
    model.to(device)

    # define optimzer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # load checkpoint
    model, optimizer, start_epoch, valid_loss_min = load_checkpoint(checkpoint_path, model, optimizer)

    print("model = ", model)
    print("optimizer = ", optimizer)
    print("start_epoch = ", start_epoch)
    #print("valid_loss_min = ", valid_loss_min)
    #print("valid_loss_min = {:.6f}".format(valid_loss_min))

    # configure model to inference mode
    model.eval()

    # Run inference
    detect_object_draw_bbox("./PlantDoc.v1-resize-416x416.coco/test/730-grape-leaf-2560x1600-nature-wallpaper_jpg.rf.821004acd395400ae6663476f21d12d5.jpg", inference_threshold=0.3)
    detect_object_draw_bbox("./PlantDoc.v1-resize-416x416.coco/test/1b321015-6e33-4f18-aade-888f4383fe92_jpeg_jpg.rf.8e50dcca0004491260c481995288b250.jpg", inference_threshold=0.3)
