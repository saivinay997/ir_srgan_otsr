import torch
import yaml
import os
import argparse
from PIL import Image
import numpy as np
from SRGAN_model import SRGANModel
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def load_and_preprocess_image(image_path, target_size=128):
    """Load and preprocess image for the model"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create transforms
    transform_hr = transforms.Compose([
        transforms.Resize((target_size*4, target_size*4)),
        transforms.CenterCrop(target_size*4),
        transforms.ToTensor()
    ])
    
    transform_lr = transforms.Compose([
        transforms.Resize((target_size*4, target_size*4)),
        transforms.CenterCrop(target_size*4),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    # Apply transforms
    hr_tensor = transform_hr(image).unsqueeze(0)
    lr_tensor = transform_lr(image).unsqueeze(0)
    
    return hr_tensor, lr_tensor

def visualize_results(lr_img, sr_img, gt_img, save_path=None):
    """Visualize low-resolution, super-resolved, and ground truth images"""
    # Convert tensors to numpy arrays
    lr_np = utils.tensor2img(lr_img)
    sr_np = utils.tensor2img(sr_img)
    gt_np = utils.tensor2img(gt_img)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Low Resolution (32x32)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(sr_np, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Super Resolution (128x128)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(gt_np, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Ground Truth (128x128)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def calculate_metrics(sr_img, gt_img):
    """Calculate PSNR and SSIM metrics"""
    sr_np = utils.tensor2img(sr_img)
    gt_np = utils.tensor2img(gt_img)
    
    # Normalize to [0, 1]
    sr_np = sr_np / 255.0
    gt_np = gt_np / 255.0
    
    # Crop borders (remove 4 pixels from each side)
    crop_size = 4
    sr_cropped = sr_np[crop_size:-crop_size, crop_size:-crop_size]
    gt_cropped = gt_np[crop_size:-crop_size, crop_size:-crop_size]
    
    # Calculate metrics
    psnr = utils.calculate_psnr(sr_cropped * 255, gt_cropped * 255)
    ssim = utils.calculate_ssim(sr_cropped * 255, gt_cropped * 255)
    
    return psnr, ssim

def demo_super_resolution(model_path, input_path, output_dir, opt_path='opt.yml'):
    """
    Demo function to showcase SRGAN super-resolution capabilities
    
    Args:
        model_path: Path to the trained generator model
        input_path: Path to input image
        output_dir: Directory to save results
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    hr_img, lr_img = load_and_preprocess_image(input_path)
    
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
    
    # Calculate metrics
    psnr, ssim = calculate_metrics(sr_img, hr_img)
    
    # Save individual images
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    lr_path = os.path.join(output_dir, f"{base_name}_LR.png")
    sr_path = os.path.join(output_dir, f"{base_name}_SR.png")
    gt_path = os.path.join(output_dir, f"{base_name}_GT.png")
    vis_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    
    utils.save_img(utils.tensor2img(lr_img), lr_path)
    utils.save_img(utils.tensor2img(sr_img), sr_path)
    utils.save_img(utils.tensor2img(hr_img), gt_path)
    
    # Create visualization
    visualize_results(lr_img, sr_img, hr_img, vis_path)
    
    # Print results
    print(f"\nSuper-Resolution Results:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return psnr, ssim

def batch_demo(model_path, input_dir, output_dir, opt_path='opt.yml'):
    """Run demo on multiple images"""
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print("No image files found in input directory")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file}")
        
        input_path = os.path.join(input_dir, image_file)
        image_output_dir = os.path.join(output_dir, f"result_{i+1}")
        
        try:
            psnr, ssim = demo_super_resolution(model_path, input_path, image_output_dir, opt_path)
            results.append({
                'image': image_file,
                'psnr': psnr,
                'ssim': ssim
            })
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Print summary
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        
        print(f"\nBatch Processing Summary:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Processed {len(results)} images successfully")

def main():
    parser = argparse.ArgumentParser(description='SRGAN Demo - Super-Resolution Showcase')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained generator model')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                       help='Directory to save demo results')
    parser.add_argument('--opt_path', type=str, default='opt.yml',
                       help='Path to configuration file')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_demo(args.model_path, args.input_path, args.output_dir, args.opt_path)
    else:
        demo_super_resolution(args.model_path, args.input_path, args.output_dir, args.opt_path)

if __name__ == '__main__':
    main()
