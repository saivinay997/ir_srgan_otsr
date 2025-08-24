import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

class HITUAVDataset(Dataset):
    """
    Dataset loader for HIT-UAV drone images for SRGAN training.
    Creates LR-HR pairs by downsampling the original images.
    """
    def __init__(self, hr_dir, val=False, size=128, scale=4, augment=True):
        self.hr_dir = hr_dir
        self.size = size
        self.scale = scale
        self.augment = augment
        
        # Get all image files
        self.hr_images = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if val:
            # Use 20% of images for validation
            self.hr_images = self.hr_images[:len(self.hr_images)//5]
        else:
            # Use 80% of images for training
            self.hr_images = self.hr_images[len(self.hr_images)//5:]
        
        # Transforms for HR images
        self.transform_hr = transforms.Compose([
            transforms.RandomCrop(self.size) if augment else transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Transforms for LR images (downsample by scale factor)
        self.transform_lr = transforms.Compose([
            transforms.RandomCrop(self.size) if augment else transforms.CenterCrop(self.size),
            transforms.Resize((self.size // self.scale, self.size // self.scale), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img_name = os.path.join(self.hr_dir, self.hr_images[idx])
        
        try:
            hr_img = Image.open(hr_img_name).convert('RGB')
            
            # Apply transforms
            hr_image = self.transform_hr(hr_img)
            lr_image = self.transform_lr(hr_img)
            
            return lr_image, hr_image
            
        except Exception as e:
            print(f"Error loading image {hr_img_name}: {e}")
            # Return a dummy image if there's an error
            dummy_img = torch.zeros(3, self.size, self.size)
            dummy_lr = torch.zeros(3, self.size // self.scale, self.size // self.scale)
            return dummy_lr, dummy_img

def create_hit_uav_dataloaders(train_dir, val_dir, batch_size=16, num_workers=4, 
                              hr_size=128, scale=4, augment=True):
    """
    Create training and validation dataloaders for HIT-UAV dataset.
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        hr_size: Size of HR patches
        scale: Super-resolution scale factor
        augment: Whether to apply data augmentation
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    
    # Create datasets
    train_dataset = HITUAVDataset(train_dir, val=False, size=hr_size, 
                                 scale=scale, augment=augment)
    val_dataset = HITUAVDataset(val_dir, val=True, size=hr_size, 
                               scale=scale, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader
