# Importing all needed libraries

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from utils import view
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from model import TwoStageDetector, RegionProposalNetwork

# Defining hyperparamaters

EPOCH = 25
LR = 0.05
BATCH_SIZE = 32

# Defining required variables

image_loc = "datasets/train2017"
annotation_loc = "datasets/annotations/instances_train2017.json"

# Setting up COCO_Dataset into train and test datasets

COCO_Dataset = datasets.CocoDetection(
    root=image_loc,
    annFile=annotation_loc,
    transform=transforms.Compose(
        [transforms.ToTensor()],
    ),
)

train_size = int(0.8 * COCO_Dataset.__len__())
test_size = len(COCO_Dataset) - train_size

COCO_train, COCO_test = torch.utils.data.random_split(
    COCO_Dataset, [train_size, test_size]
)

# Obtaining appropriate loaders for training and testing images in batches

COCO_train_loader = DataLoader(
    COCO_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
)

COCO_test_loader = DataLoader(
    COCO_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
)

model = TwoStageDetector()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Prepping model to train!
model.train()

loss_list = []

for epoch in range(EPOCH):
    total_loss = 0
    for img_batch, gt_bboxes, gt_classes_batch in COCO_train_loader:
        # Calculating loss per batch
        loss = model(img_batch, gt_bboxes, gt_classes_batch)

        # Backpropogating values

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss

    loss_list.append(total_loss)
