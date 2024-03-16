import wandb
import torch




def log_image_table(lr_imgs, hr_imgs, sr_imgs, step):
    "Log a wandb.Table with (lr_imgs, hr_imgs, sr_imgs, step)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["lr_imgs", "hr_imgs", "sr_imgs", "step"])
    table.add_data(wandb.Image(lr_imgs), wandb.Image(hr_imgs), wandb.Image(sr_imgs), step)
    wandb.log({"predictions_table":table}, commit=False)