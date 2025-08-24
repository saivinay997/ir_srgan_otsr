import torch
import yaml
import os
import argparse
from PIL import Image
import numpy as np
from SRGAN_model import SRGANModel
import utils
from torchvision import transforms

def load_image(image_path, size=128):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((size*4, size*4)),
        transforms.CenterCrop(size*4),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, output_path):
    """Save tensor as image"""
    img = utils.tensor2img(tensor)
    utils.save_img(img, output_path)

def super_resolve_image(model_path, input_path, output_path, opt_path='opt.yml'):
    """
    Perform super-resolution on a single image
    
    Args:
        model_path: Path to the trained generator model
        input_path: Path to input low-resolution image
        output_path: Path to save super-resolved image
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
    
    # Load input image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")
    
    # Load and preprocess image
    hr_img = load_image(input_path)
    lr_img = load_image(input_path, size=32)  # Create LR version
    
    # Move to device
    hr_img = hr_img.to(device)
    lr_img = lr_img.to(device)
    
    # Generate super-resolution image
    model.netG.eval()
    with torch.no_grad():
        model.feed_data(hr_img, lr_img)
        model.test()
        
        # Get results
        visuals = model.get_current_visuals()
        sr_img = visuals['SR']
    
    # Save result
    save_image(sr_img, output_path)
    print(f"Super-resolved image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Single image super-resolution')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained generator model')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input low-resolution image')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save super-resolved image')
    parser.add_argument('--opt_path', type=str, default='opt.yml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    super_resolve_image(args.model_path, args.input_path, args.output_path, args.opt_path)

if __name__ == '__main__':
    main()
