import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataloader(Dataset):
    def __init__(self, hr_dir, size=32):
        self.hr_dir = hr_dir
        self.size = size
        self.hr_images = os.listdir(hr_dir)
        self.transform_lr = transforms.Compose([transforms.Resize((self.size*4, self.size*4)),
                                                transforms.CenterCrop(self.size*4),
                                                transforms.Resize(self.size),
                                               transforms.ToTensor()])
        self.transform_hr = transforms.Compose([transforms.CenterCrop(self.size*4),
                                               transforms.ToTensor()])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img_name = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_img = Image.open(hr_img_name)#.convert("RGB")
        # print(hr_img.size)
        hr_image = self.transform_hr(hr_img)
        lr_image = self.transform_lr(hr_img)
        return hr_image, lr_image