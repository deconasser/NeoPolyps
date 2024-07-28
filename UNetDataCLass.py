import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform):
        super(UNetDataClass, self).__init__()

        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)

        images_list = [images_path + image_name for image_name in images_list]
        masks_list = [masks_path + mask_name for mask_name in masks_list]

        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]

        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)



        # Normalize
        data = self.transform(data) / 255
        label = self.transform(label) / 255


        label = torch.where(label>0.65, 1.0, 0.0)

        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)

        return data, label

    def __len__(self):
        return len(self.images_list)