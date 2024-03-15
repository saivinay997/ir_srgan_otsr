import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty
from model import SRGAN_d, SRGAN_g
from imageTransform import ImageDataloader

def train(hr_dir, lr_dir):
    # device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1
    IMAGE_SIZE = 32
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 100
    FEATURES_CRITIC = 16
    FEATURES_GEN = 16
    CRITIC_ITERATIONS = 2
    LAMBDA_GP = 10

    transform_lr = transforms.Compose([transforms.CenterCrop(IMAGE_SIZE),
                                                transforms.ToTensor()])
    transform_hr = transforms.Compose([transforms.CenterCrop(IMAGE_SIZE*4),
                                                transforms.ToTensor()])

    

    dataset = ImageDataloader(hr_dir, lr_dir, IMAGE_SIZE)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    netG = SRGAN_g(in_nc=CHANNELS_IMG, out_nc=CHANNELS_IMG).to(device)

    netD = SRGAN_d(in_nc=CHANNELS_IMG).to(device)

    optG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.999))

    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    netG.train()
    netD.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (hr_img, lr_img) in enumerate(tqdm(data_loader)):
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)

            # Train Discriminator: max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATIONS):
                fake = netG(lr_img)
                critic_real = netD(hr_img).reshape(-1)
                critic_fake = netD(fake).reshape(-1)
                gp = gradient_penalty(netD, hr_img, fake, device=device)
                lossD = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp

                netD.zero_grad()
                lossD.backward(retain_graph=True)
                optD.step()

            # Train Generator: min -E[critic(gen_fake)]
            netG.zero_grad()
            critic_fake = netD(fake).reshape(-1)
            lossG = -torch.mean(critic_fake)
            lossG.backward()
            optG.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(data_loader)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = netG(lr_img)
                    # take out (up to) 8 examples
                    img_grid_real = torchvision.utils.make_grid(
                        hr_img[:8], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:8], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1