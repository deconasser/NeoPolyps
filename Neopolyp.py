import os
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from torch import Tensor
from UNetDataCLass import UNetDataClass
from SegDataClass import SegDataClass
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from collections import OrderedDict
from torchsummary import summary
from torchgeometry.losses import one_hot
from albumentations import (
    RandomRotate90,
    Flip,
    Transpose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomBrightnessContrast,
    HorizontalFlip,
    VerticalFlip,
    RandomGamma,
    RGBShift,
)
torch.set_printoptions(profile="default")

device = torch.device("cuda" if torch.cuda.is_available () else "cpu")
print(device)

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

# Hyper params

num_classes = 3
epochs = 50
learning_rate = 1e-4
batch_size = 8
display_step = 50

# Model path
checkpoint_path = '/teamspace/studios/this_studio/NeoPolyps/checkpoints'
pretrained_path = "/teamspace/studios/this_studio/NeoPolyps/checkpoints"

images_path = "/teamspace/studios/this_studio/NeoPolyps/data/train/"
masks_path =  "/teamspace/studios/this_studio/NeoPolyps/data/train_gt/"

# Initialize lists to keep track of loss and accuracy
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

transform = Compose([Resize((512, 512), interpolation=InterpolationMode.BILINEAR), PILToTensor()])

unet_dataset = UNetDataClass(images_path, masks_path, transform)

train_size = 0.9
valid_size = 0.1
torch.manual_seed(42)
train_set, valid_set = random_split(unet_dataset, [int(train_size * len(unet_dataset)), int(valid_size * len(unet_dataset))])

augmentation = A.Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
    RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
])

# transform = transforms.ToTensor()
aug_dataset = SegDataClass(images_path, masks_path, transform=transform, augmentation=augmentation)
x, y = aug_dataset.__getitem__(20)
print(x, y)