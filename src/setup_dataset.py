import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm
import random

def create_lr_hr_pairs(hr_dir, lr_dir, scale=4, lr_size=32):
    """
    Create low-resolution and high-resolution image pairs
    
    Args:
        hr_dir: Directory containing high-resolution images
        lr_dir: Directory to save low-resolution images
        scale: Super-resolution scale factor
        lr_size: Size of low-resolution images
    """
    
    # Create LR directory if it doesn't exist
    os.makedirs(lr_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    hr_files = []
    
    for ext in image_extensions:
        hr_files.extend([f for f in os.listdir(hr_dir) if f.lower().endswith(ext)])
    
    if not hr_files:
        print("No image files found in HR directory")
        return
    
    print(f"Found {len(hr_files)} images to process")
    
    # Process each image
    for hr_file in tqdm(hr_files, desc="Creating LR-HR pairs"):
        hr_path = os.path.join(hr_dir, hr_file)
        lr_path = os.path.join(lr_dir, hr_file)
        
        try:
            # Load image
            img = Image.open(hr_path).convert('RGB')
            
            # Resize to create LR version
            lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)
            
            # Save LR image
            lr_img.save(lr_path)
            
        except Exception as e:
            print(f"Error processing {hr_file}: {e}")

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data_dir: Directory containing all images
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(data_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print("No image files found in data directory")
        return
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Create directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move files
    print("Moving files to train directory...")
    for file in tqdm(train_files, desc="Train"):
        src = os.path.join(data_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.move(src, dst)
    
    print("Moving files to validation directory...")
    for file in tqdm(val_files, desc="Validation"):
        src = os.path.join(data_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.move(src, dst)
    
    print("Moving files to test directory...")
    for file in tqdm(test_files, desc="Test"):
        src = os.path.join(data_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.move(src, dst)
    
    print(f"Dataset split complete:")
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")

def organize_dataset_structure(base_dir):
    """
    Organize dataset into the required structure for training
    
    Args:
        base_dir: Base directory containing the dataset
    """
    
    # Create required directories
    train_hr_dir = os.path.join(base_dir, 'train', 'HR')
    train_lr_dir = os.path.join(base_dir, 'train', 'LR')
    val_hr_dir = os.path.join(base_dir, 'val', 'HR')
    val_lr_dir = os.path.join(base_dir, 'val', 'LR')
    test_hr_dir = os.path.join(base_dir, 'test', 'HR')
    test_lr_dir = os.path.join(base_dir, 'test', 'LR')
    
    os.makedirs(train_hr_dir, exist_ok=True)
    os.makedirs(train_lr_dir, exist_ok=True)
    os.makedirs(val_hr_dir, exist_ok=True)
    os.makedirs(val_lr_dir, exist_ok=True)
    os.makedirs(test_hr_dir, exist_ok=True)
    os.makedirs(test_lr_dir, exist_ok=True)
    
    print("Dataset structure created:")
    print(f"Train HR: {train_hr_dir}")
    print(f"Train LR: {train_lr_dir}")
    print(f"Val HR: {val_hr_dir}")
    print(f"Val LR: {val_lr_dir}")
    print(f"Test HR: {test_hr_dir}")
    print(f"Test LR: {test_lr_dir}")

def validate_dataset(data_dir):
    """
    Validate that the dataset is properly organized
    
    Args:
        data_dir: Directory containing the dataset
    """
    
    required_dirs = ['train/HR', 'train/LR', 'val/HR', 'val/LR']
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing directory: {full_path}")
            return False
        
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(full_path) if f.lower().endswith(ext)])
        
        print(f"{dir_path}: {len(image_files)} images")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Dataset setup for SRGAN training')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['create_pairs', 'split', 'organize', 'validate'],
                       help='Mode of operation')
    parser.add_argument('--hr_dir', type=str, 
                       help='Directory containing high-resolution images')
    parser.add_argument('--lr_dir', type=str, 
                       help='Directory to save low-resolution images')
    parser.add_argument('--data_dir', type=str, 
                       help='Dataset directory')
    parser.add_argument('--scale', type=int, default=4,
                       help='Super-resolution scale factor')
    parser.add_argument('--lr_size', type=int, default=32,
                       help='Size of low-resolution images')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio of test data')
    
    args = parser.parse_args()
    
    if args.mode == 'create_pairs':
        if not args.hr_dir or not args.lr_dir:
            print("Error: --hr_dir and --lr_dir are required for create_pairs mode")
            return
        create_lr_hr_pairs(args.hr_dir, args.lr_dir, args.scale, args.lr_size)
    
    elif args.mode == 'split':
        if not args.data_dir:
            print("Error: --data_dir is required for split mode")
            return
        split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    
    elif args.mode == 'organize':
        if not args.data_dir:
            print("Error: --data_dir is required for organize mode")
            return
        organize_dataset_structure(args.data_dir)
    
    elif args.mode == 'validate':
        if not args.data_dir:
            print("Error: --data_dir is required for validate mode")
            return
        validate_dataset(args.data_dir)

if __name__ == '__main__':
    main()
