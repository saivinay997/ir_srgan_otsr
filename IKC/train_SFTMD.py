import yaml
import utils as util
import torch
import numpy as np
from create_dataset import ImageDataloader
from torch.utils.data import DataLoader
import wandb
import math
import datetime
from create_model import create_model
from tqdm.auto import tqdm
import os
import wb_util


curr_time = datetime.datetime.now().strftime("%d-%m_%H-%M")

def main(ymlpath, HR_train, HR_val, note, val_results_path, trained_model_path):
    with open(ymlpath) as f:
        opt_F = yaml.safe_load(f)

    seed = opt_F["train"]["manual_seed"]

    # create PCA matrix of enough kernel and save it, to ensure kernels have same corresponding kernel maps
    batch_ker = util.random_batch_kernel(batch=30000, l=opt_F['kernel_size'], sig_min=opt_F['sig_min'], sig_max=opt_F['sig_max'], rate_iso=1.0, scaling=3, tensor=False)
    print('batch kernel shape: {}'.format(batch_ker.shape))
    b = np.size(batch_ker, 0)
    batch_ker = batch_ker.reshape((b, -1))
    matrix = batch_ker
    print('matrix shape: {}'.format(matrix.shape))
    torch.save(matrix, './matrix.pth')
    print('Save matrix at: ./matrix.pth')

    torch.backends.cudnn.benchmark = True

    ## training and validation dataloader
    dataset_ratio = 200   # enlarge the size of each epoch
    for phase, dataset_opt in opt_F['datasets'].items():

        dataset = ImageDataloader(HR_train, val=False)
        train_loader = DataLoader(dataset, batch_size=opt_F["datasets"]["train"]["batch_size"],  num_workers=4)

        val_dataset = ImageDataloader(HR_val, val=True)
        val_loader = DataLoader(val_dataset, batch_size=opt_F["datasets"]["train"]["batch_size"], num_workers=4)

        ## setup traing hyperparameters
    start_epoch = 0
    current_step = 0
    total_epochs = 0
    print(len(dataset))
    train_size = int(math.ceil(len(dataset) / opt_F['datasets']['train']['batch_size']))
    total_iters = int(opt_F["train"]['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))

    wandb.init(
        # Set the project where this run will be logged
        project="IKC_OTSR_SRGAN", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"ir_experiment_{curr_time}", 
        # Track hyperparameters and run metadata
        config={
            "Note":note,
        "epochs": total_epochs,
        "total_iters": total_iters,
        "batch_size": opt_F["datasets"]["train"]["batch_size"],
        "pixel_criterion": opt_F["train"]["pixel_criterion"],
        "feature_criterion": opt_F["train"]["feature_criterion"],
        "gan_type": opt_F["train"]["gan_type"],
        "train_parameters": opt_F["train"]
        # "learning_rate": opt["train"]["lr"]
        })

    print(f"Staring the training from epoch: {start_epoch}. Total epoch: {total_epochs}")

    #### create model
    model = create_model(opt_F)

    for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="Epochs: "):
        for _, (hr_imgs, lr_imgs) in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### preprocessing for LR_img and kernel map
            prepro = util.SRMDPreprocessing(opt_F['scale'], matrix, random=True, para_input=opt_F['code_length'],
                                            kernel=opt_F['kernel_size'], noise=False, cuda=True, sig=opt_F['sig'],
                                            sig_min=opt_F['sig_min'], sig_max=opt_F['sig_max'], rate_iso=1.0, scaling=3,
                                            rate_cln=0.2, noise_high=0.0)
            LR_img, ker_map = prepro(hr_imgs)

            #### training
            model.feed_data(hr_imgs, LR_img, ker_map)
            model.optimize_parameters(current_step)

            if current_step % 50 == 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = avg_ssim = 0.0
                _lr_img = _hr_img = _sr_img = None
                idx = 0
                for val_hr, val_lr in val_loader:
                    if idx > 4:
                        break 
                    idx += 1
                    img_name = f"img_{idx}"
                    img_dir = os.path.join(val_results_path, img_name)
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    
                    #### preprocessing for LR_img and kernel map
                    prepro = util.SRMDPreprocessing(opt_F['scale'], matrix, random=True, para_input=opt_F['code_length'],
                                                    kernel=opt_F['kernel_size'], noise=False, cuda=True, sig=opt_F['sig'],
                                                    sig_min=opt_F['sig_min'], sig_max=opt_F['sig_max'], rate_iso=1.0, scaling=3,
                                                    rate_cln=0.2, noise_high=0.0)
                    LR_img, ker_map = prepro(val_hr)

                    model.feed_data(val_hr, LR_img, ker_map)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img  = util.tensor2img(visuals['SR'])
                    gt_img  =  util.tensor2img(visuals["GT"])
                    # print(sr_img.shape, gt_img.shape, visuals["LQ"].shape)
                    # log the images to wandb
                    _lr_img = visuals['LQ'].to("cpu").permute(1, 2, 0).numpy()
                    _hr_img = visuals['GT'].to("cpu").permute(1, 2, 0).numpy()
                    _sr_img = visuals['SR'].to("cpu").permute(1, 2, 0).numpy()
                    
                    # print("Starting to save image.")
                    if current_step%100 == 0:
                        save_img_path = os.path.join(img_dir, f"{current_step}_{idx}.png")
                        util.save_img(sr_img, save_img_path)

                    crop_size = opt_F["scale"]
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                if current_step % 100 == 0:
                    # max_psnr = avg_psnr
                    wb_util.log_image_table(lr_imgs=_lr_img, hr_imgs=_hr_img, sr_imgs=_sr_img, 
                                            psnr = avg_psnr, ssim = avg_ssim, step=current_step)
                if avg_psnr > max_psnr:
                    max_psnr = avg_psnr
                    wandb.log({"max_psnr":max_psnr})
                if avg_ssim > max_ssim:
                    max_ssim = avg_ssim
                    wandb.log({"max_ssim":max_ssim})
                    
                # logger.info(f"Epoch: {epoch} | iters: {current_step} | PSNR: {avg_psnr}")
                # logger.info(f"# Validation # PSNR: {avg_psnr}")
                # logger.info(f"# Validation # SSIM: {avg_ssim}")

                metrics = {"PSNR": avg_psnr, 
                           "SSIM": avg_ssim,
                           "epoch": epoch,
                           "iters": current_step}
                wandb.log(metrics)

        if epoch % 50 == 0 and epoch!= 0:
            print(f'Saving models and training states at epoch {epoch}')
            model.save(current_step, trained_model_path)
            # model.save_training_state(epoch, current_step)

    print(f'Saving models and training states at epoch {epoch}')
    model.save(current_step, trained_model_path)
    wandb.finish()