try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import torch




def log_image_table(lr_imgs, hr_imgs, sr_imgs, psnr, ssim, step):
    "Log a wandb.Table with (lr_imgs, hr_imgs, sr_imgs, step)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["img_id","lr_imgs", "hr_imgs", "sr_imgs","psnr", "ssim", "step"])
    img_id = f"img_at_{step}"
    table.add_data(img_id, wandb.Image(lr_imgs), wandb.Image(hr_imgs), wandb.Image(sr_imgs), psnr, ssim, step)
    wandb.log({"predictions_table":table})