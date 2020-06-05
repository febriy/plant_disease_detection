import os
from torch.utils import data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

import torch
from config import *
from utils import (
    get_model_object_detection,
    collate_fn,
    get_transform,
    PlantDiseaseDataset,
)

print("Torch version:", torch.__version__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset set-up
train_dataset = PlantDiseaseDataset(
    root_dir=train_data_dir, coco_annotation_file=train_coco, transform=get_transform()
)

test_dataset = PlantDiseaseDataset(
    root_dir=test_data_dir, coco_annotation_file=test_coco, transform=get_transform()
)

plant_train_iterable = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

plant_test_iterable = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=train_batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

# Model set-up
model = get_model_object_detection(num_classes)
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=lr, momentum=momentum, weight_decay=weight_decay
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)








def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def train_one_epoch(model,optimizer, data_loader, device, epoch):
    len_dataloader = len(data_loader)
    losses_one_epoch = []
    model.train()
    i = 0
    for images, targets in data_loader:
        i += 1

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        losses_one_epoch.append(losses)

        optimizer.zero_grad() # clear the gradients of all optimized variables
        losses.backward() # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step() # perform a single optimization step (parameter update)

        print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")

    return losses_one_epoch
        
# Training
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}/{num_epochs}")

    losses_one_epoch = train_one_epoch(model,optimizer, plant_train_iterable, device, epoch)
    lr_scheduler.step()

    checkpoint = {
        'epoch': epoch + 1,
        'losses': losses_one_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    save_ckp(checkpoint, False, 'checkpoint_model.pt', 'best_model.pt')


# torch.save(model.state_dict(), 'model.pt')
# https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
# https://github.com/pytorch/vision/tree/master/references/detection
