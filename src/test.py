import torch
import yaml
import os
import argparse
from SRGAN_model import SRGANModel
from create_dataset import ImageDataloader
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
import numpy as np

def test_model(model_path, test_data_path, output_path, opt_path='opt.yml'):
    """
    Test the trained SRGAN model on test images
    
    Args:
        model_path: Path to the trained generator model
        test_data_path: Path to test images directory
        output_path: Path to save super-resolved images
        opt_path: Path to configuration file
    """
    
    # Load configuration
    with open(opt_path) as f:
        opt = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SRGANModel(opt, train=False)
    
    # Load trained generator
    if os.path.exists(model_path):
        model.load_network(model_path, model.netG)
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create test dataset
    test_dataset = ImageDataloader(test_data_path, val=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Test the model
    model.netG.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0
    
    with torch.no_grad():
        for idx, (hr_img, lr_img) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move to device
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)
            
            # Generate super-resolution image
            model.feed_data(hr_img, lr_img)
            model.test()
            
            # Get results
            visuals = model.get_current_visuals()
            sr_img = utils.tensor2img(visuals['SR'])
            gt_img = utils.tensor2img(visuals['GT'])
            
            # Save images
            img_name = f"test_{idx:04d}"
            sr_path = os.path.join(output_path, f"{img_name}_SR.png")
            gt_path = os.path.join(output_path, f"{img_name}_GT.png")
            
            utils.save_img(sr_img, sr_path)
            utils.save_img(gt_img, gt_path)
            
            # Calculate metrics
            crop_size = opt["scale"]
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
            
            psnr = utils.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = utils.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            
            total_psnr += psnr
            total_ssim += ssim
            num_images += 1
            
            if idx % 10 == 0:
                print(f"Image {idx}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
    
    # Calculate average metrics
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Tested on {num_images} images")
    print(f"Results saved to: {output_path}")
    
    return avg_psnr, avg_ssim

def main():
    parser = argparse.ArgumentParser(description='Test SRGAN model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained generator model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--output_path', type=str, default='test_results',
                       help='Path to save test results')
    parser.add_argument('--opt_path', type=str, default='opt.yml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.test_data, args.output_path, args.opt_path)

if __name__ == '__main__':
    main()
