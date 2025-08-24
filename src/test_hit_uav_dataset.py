import torch
import matplotlib.pyplot as plt
import numpy as np
from hit_uav_dataset import create_hit_uav_dataloaders
import os

def visualize_samples(train_loader, num_samples=4):
    """Visualize some training samples to verify dataset loading."""
    
    # Get a batch
    lr_batch, hr_batch = next(iter(train_loader))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    fig.suptitle('HIT-UAV Dataset Samples (LR -> HR)', fontsize=16)
    
    for i in range(num_samples):
        # Convert tensors to numpy arrays for visualization
        lr_img = lr_batch[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
        hr_img = hr_batch[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize (assuming normalization to [-1, 1])
        lr_img = (lr_img + 1) / 2
        hr_img = (hr_img + 1) / 2
        
        # Clip to [0, 1]
        lr_img = np.clip(lr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # Display images
        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title(f'Low Resolution ({lr_img.shape[1]}x{lr_img.shape[0]})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(hr_img)
        axes[i, 1].set_title(f'High Resolution ({hr_img.shape[1]}x{hr_img.shape[0]})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hit_uav_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Sample visualization saved as 'hit_uav_samples.png'")
    print(f"LR image shape: {lr_batch[0].shape}")
    print(f"HR image shape: {hr_batch[0].shape}")

def test_dataset():
    """Test the HIT-UAV dataset loading."""
    
    # Dataset paths
    train_dir = "hit-uav/images/train"
    val_dir = "hit-uav/images/val"
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        return False
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        return False
    
    print("Creating dataloaders...")
    
    try:
        # Create dataloaders
        train_loader, val_loader = create_hit_uav_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=4,
            num_workers=2,
            hr_size=128,
            scale=4,
            augment=True
        )
        
        print("✓ Dataloaders created successfully!")
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Validation batches per epoch: {len(val_loader)}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        lr_batch, hr_batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully!")
        print(f"LR batch shape: {lr_batch.shape}")
        print(f"HR batch shape: {hr_batch.shape}")
        print(f"LR value range: [{lr_batch.min():.3f}, {lr_batch.max():.3f}]")
        print(f"HR value range: [{hr_batch.min():.3f}, {hr_batch.max():.3f}]")
        
        # Visualize samples
        print("\nVisualizing samples...")
        visualize_samples(train_loader)
        
        return True
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n✓ Dataset test completed successfully!")
        print("You can now proceed with training using train_hit_uav.py")
    else:
        print("\n✗ Dataset test failed. Please check the error messages above.")
