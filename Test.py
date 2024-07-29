import torch
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from UNetTestDataClass import UNetTestDataClass
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '/teamspace/studios/this_studio/NeoPolyps/data/test/'
transform = Compose([Resize((512, 512), interpolation=InterpolationMode.BILINEAR), PILToTensor()])
unet_test_dataset = UNetTestDataClass(path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=8, shuffle=True)
# Model path
pretrained_path = '/teamspace/studios/this_studio/NeoPolyps/checkpoints'
num_classes = 3

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

try:
    checkpoint = torch.load(pretrained_path)

    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model)
    model.to(device)
except:
    model.apply(weights_init)
    model = nn.DataParallel(model)
    model.to(device)


for i, (data, path, h, w) in enumerate(test_dataloader):
    img = data
    break


model.eval()
with torch.no_grad():
    predict = model(img)

for i in range(5):
    # Convert the input image tensor to a PIL image and save it
    image_array = img[i].permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    image.save(f'image_{i}.png')
    
    # Convert the prediction to a one-hot encoded image and save it
    prediction = torch.argmax(predict[i], 0)
    one_hot_prediction = F.one_hot(prediction, num_classes=3).float().cpu().numpy()
    
    # Ensure one-hot image is compatible with PIL
    one_hot_image = Image.fromarray((one_hot_prediction * 255).astype(np.uint8))
    one_hot_image.save(f'predict_{i}.png')