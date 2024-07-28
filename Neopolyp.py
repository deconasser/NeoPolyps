from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from collections import OrderedDict
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

# Initialize lists to keep track of loss and accuracy
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []