import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataloader(Dataset):
    def __init__(self, hr_dir, lr_dir, size=32):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.size = size
        self.hr_images = os.listdir(hr_dir)
        self.lr_images = os.listdir(lr_dir)
        self.transform_lr = transforms.Compose([transforms.CenterCrop(self.size),
                                               transforms.ToTensor()])
        self.transform_hr = transforms.Compose([transforms.CenterCrop(self.size*4),
                                               transforms.ToTensor()])

    def __len__(self):
        return min(len(self.hr_images), len(self.lr_images))

    def __getitem__(self, idx):
        hr_img_name = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_img_name = os.path.join(self.lr_dir, self.hr_images[idx])
        # print("HR:",hr_img_name)
        # print("LR:",lr_img_name, "\n")
        hr_image = Image.open(hr_img_name).convert("RGB")
        lr_image = Image.open(lr_img_name).convert("RGB")

        hr_image = self.transform_hr(hr_image)
        lr_image = self.transform_lr(lr_image)

        return hr_image, lr_image