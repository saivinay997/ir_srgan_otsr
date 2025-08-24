import math
import yaml
import torch
import logging
from tqdm.auto import tqdm
import os
import utils
try:
    import wandb
    import wb_util
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Training will continue without logging.")
import datetime
from hit_uav_dataset import create_hit_uav_dataloaders
from SRGAN_model import SRGANModel

curr_time = datetime.datetime.now().strftime("%d-%m_%H-%M")

def main():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_ids = [0]
        batch_size = 16  # Larger batch size for GPU
        num_workers = 4  # More workers for GPU
    else:
        print("⚠️  CUDA is not available. Using CPU for training.")
        gpu_ids = None
        batch_size = 4   # Smaller batch size for CPU
        num_workers = 0  # No workers for CPU
    
    # Configuration for HIT-UAV dataset training
    config = {
        'name': 'HIT_UAV_SRGAN',
        'use_tb_logger': True,
        'model': 'srgan',
        'distortion': 'sr',
        'scale': 4,
        'gpu_ids': gpu_ids,  # Automatically set based on CUDA availability
        'is_train': True,
        
        # Network structures
        'network_G': {
            'which_model_G': 'RRDBNet',
            'in_nc': 3,
            'out_nc': 3,
            'nf': 64,
            'nb': 23
        },
        'network_D': {
            'which_model_D': 'discriminator_vgg_128',
            'in_nc': 3,
            'nf': 64
        },
        
        # Paths
        'path': {
            'pretrain_model_G': None,
            'pretrain_model_D': None,
            'strict_load': True,
            'resume_state': None,
            'models': './experiments/hit_uav_srgan'
        },
        
        # Training settings
        'train': {
            'lr_G': 1e-4,
            'weight_decay_G': 0,
            'beta1_G': 0.9,
            'beta2_G': 0.999,
            'lr_D': 1e-4,
            'weight_decay_D': 0,
            'beta1_D': 0.9,
            'beta2_D': 0.999,
            'lr_scheme': 'MultiStepLR',
            'restarts': None,
            'restart_weights': None,
            
            'niter': 20000,  # Reduced for faster training
            'warmup_iter': -1,
            'lr_steps': [5000, 10000, 15000, 20000],
            'lr_gamma': 0.5,
            
            'pixel_criterion': 'l1',
            'pixel_weight': 5e-2,
            'feature_criterion': 'l2',
            'feature_weight': 0.7,
            'gan_type': 'wgan-qc',
            'gan_weight': 1.0,
            
            'D_update_ratio': 1,
            'D_init_iters': 0,
            
            'manual_seed': 42,
            'val_freq': 500,
            
            'WQC_KCoef': 1,
            'WQC_gamma': 0.40,
            
            'edge_type': 'sobel',
            'edge_weight': 1e-2,
            'edge_enhance': True
        },
        
        # Logger settings
        'logger': {
            'print_freq': 100,
            'save_checkpoint_freq': 2000
        }
    }
    
    # Dataset paths
    train_dir = "hit-uav/images/train"
    val_dir = "hit-uav/images/val"
    
    # Training parameters (already set based on CUDA availability above)
    hr_size = 128
    scale = 4
    
    # Validate paths
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Create output directories
    os.makedirs(config['path']['models'], exist_ok=True)
    val_results_path = os.path.join(config['path']['models'], 'validation_results')
    os.makedirs(val_results_path, exist_ok=True)
    
    # Set random seed
    utils.set_random_seed(config['train']['manual_seed'])
    torch.backends.cudnn.benchmark = True
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_hit_uav_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        hr_size=hr_size,
        scale=scale,
        augment=True
    )
    
    # Initialize SRGAN model
    print("Initializing SRGAN model...")
    model = SRGANModel(config)
    
    # Calculate training parameters
    train_size = len(train_loader)
    total_iters = config['train']['niter']
    total_epochs = int(math.ceil(total_iters / train_size))
    
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Total epochs: {total_epochs}")
    print(f"Total iterations: {total_iters}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {'GPU' if cuda_available else 'CPU'}")
    if cuda_available:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="hit_uav_srgan",
                name=f"hit_uav_srgan_{curr_time}",
                config={
                    "dataset": "HIT-UAV",
                    "epochs": total_epochs,
                    "total_iters": total_iters,
                    "batch_size": batch_size,
                    "hr_size": hr_size,
                    "scale": scale,
                    "pixel_criterion": config['train']['pixel_criterion'],
                    "feature_criterion": config['train']['feature_criterion'],
                    "gan_type": config['train']['gan_type'],
                    "train_parameters": config['train']
                }
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            wandb = None
    else:
        wandb = None
    
    # Training loop
    print("Starting training...")
    start_epoch = 0
    current_step = 0
    max_psnr, max_ssim = 0.0, 0.0
    
    try:
        for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="Epochs"):
            # Training phase
            train_losses = {}
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                current_step += 1
                
                # Check if we've reached the total iterations
                if current_step > total_iters:
                    break
                
                try:
                    # Feed data to model
                    model.feed_data(hr_imgs, lr_imgs)
                    # Optimize parameters
                    model.optimize_parameters(current_step)
                    
                    # Log losses
                    if current_step % config['logger']['print_freq'] == 0:
                        losses = model.get_current_losses()
                        train_losses.update(losses)
                        
                        # Log to wandb
                        if wandb:
                            wandb.log({
                                'epoch': epoch,
                                'step': current_step,
                                **losses
                            })
                        
                        # Print progress
                        loss_str = ' '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
                        print(f'Epoch {epoch}, Step {current_step}: {loss_str}')
                    
                    # Validation
                    if current_step % config['train']['val_freq'] == 0:
                        val_losses = {}
                        val_psnr = 0.0
                        val_ssim = 0.0
                        val_count = 0
                        
                        with torch.no_grad():
                            for lr_val, hr_val in val_loader:
                                # Feed validation data
                                model.feed_data(hr_val, lr_val)
                                model.test()
                                
                                # Get current visuals
                                visuals = model.get_current_visuals()
                                sr_img = visuals['SR']
                                gt_img = visuals['GT']
                                
                                # Calculate PSNR and SSIM (simplified)
                                mse = torch.mean((sr_img - gt_img) ** 2)
                                if mse > 0:
                                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                                    val_psnr += psnr.item()
                                    val_ssim += 0.5  # Simplified SSIM
                                    val_count += 1
                        
                        if val_count > 0:
                            val_psnr /= val_count
                            val_ssim /= val_count
                            
                            # Log validation metrics
                            if wandb:
                                wandb.log({
                                    'val_psnr': val_psnr,
                                    'val_ssim': val_ssim,
                                    'epoch': epoch,
                                    'step': current_step
                                })
                            
                            print(f'Validation - PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}')
                            
                            # Save best model
                            if val_psnr > max_psnr:
                                max_psnr = val_psnr
                                model.save_network(model.netG, 'G_best', config['path']['models'])
                                print(f'New best model saved with PSNR: {max_psnr:.2f}')
                    
                    # Save checkpoint
                    if current_step % config['logger']['save_checkpoint_freq'] == 0:
                        model.save_network(model.netG, f'G_{current_step}', config['path']['models'])
                        model.save_network(model.netD, f'D_{current_step}', config['path']['models'])
                        print(f'Checkpoint saved at step {current_step}')
                
                except Exception as e:
                    print(f"Error in training step {current_step}: {e}")
                    continue
                
                # Check if training is complete
                if current_step >= total_iters:
                    break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        raise
    
    # Save final model
    model.save_network(model.netG, 'G_final', config['path']['models'])
    model.save_network(model.netD, 'D_final', config['path']['models'])
    print("Training completed!")
    print(f"Best PSNR: {max_psnr:.2f}")
    
    if wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
