import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class SRGAN_g(nn.Module):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    def __init__(self, in_nc, out_nc):
        super(SRGAN_g, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_nc, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.residual_block = self.make_layer(16)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.subpixelconv1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.subpixelconv2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=out_nc, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def make_layer(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def forward(self, x):
        temp = self.conv1(x)
        x = self.residual_block(temp)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = nn.ReLU()(self.conv3(x))
        x = self.subpixelconv1(x)
        x = nn.ReLU()(self.conv4(x))
        x = self.subpixelconv2(x)
        x = nn.Tanh()(self.conv5(x))
        return x


class SRGAN_d(nn.Module):
    def __init__(self,in_nc, dim=64):
        super(SRGAN_d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_nc, out_channels=dim, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv3 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=dim * 4)
        self.conv4 = nn.Conv2d(
            in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=dim * 8)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 8, out_channels=dim * 16, kernel_size=4, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm2d(num_features=dim * 16)
        self.conv6 = nn.Conv2d(
            in_channels=dim * 16, out_channels=dim * 32, kernel_size=4, stride=2, padding=1
        )
        self.bn5 = nn.BatchNorm2d(num_features=dim * 32)
        self.conv7 = nn.Conv2d(
            in_channels=dim * 32, out_channels=dim * 16, kernel_size=1, stride=1
        )
        self.bn6 = nn.BatchNorm2d(num_features=dim * 16)
        self.conv8 = nn.Conv2d(
            in_channels=dim * 16, out_channels=dim * 8, kernel_size=1, stride=1
        )
        self.bn7 = nn.BatchNorm2d(num_features=dim * 8)
        self.conv9 = nn.Conv2d(
            in_channels=dim * 8, out_channels=dim * 2, kernel_size=1, stride=1
        )
        self.bn8 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv10 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, padding=1
        )
        self.bn9 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv11 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 8, kernel_size=3, stride=1, padding=1
        )
        self.bn10 = nn.BatchNorm2d(num_features=dim * 8)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.add = nn.Sequential(nn.ReLU(), nn.Conv2d(dim * 8, dim * 8, kernel_size=1, stride=1))
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=dim * 8 * 4, out_features=1)  # Calculate the correct number of features

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.bn1(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.bn2(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.leakyrelu(self.bn3(x))
        x = self.leakyrelu(self.conv5(x))
        x = self.leakyrelu(self.bn4(x))
        x = self.leakyrelu(self.conv6(x))
        x = self.leakyrelu(self.bn5(x))
        x = self.leakyrelu(self.conv7(x))
        x = self.leakyrelu(self.bn6(x))
        x = self.leakyrelu(self.conv8(x))
        x = self.leakyrelu(self.bn7(x))
        temp = x
        x = self.leakyrelu(self.conv9(x))
        x = self.leakyrelu(self.bn8(x))
        x = self.leakyrelu(self.conv10(x))
        x = self.leakyrelu(self.bn9(x))
        x = self.leakyrelu(self.conv11(x))
        x = self.bn10(x)
        # print("at layer 11:", x.shape)
        x = self.add(temp + x)
        # print("at layer add:", x.shape)
        x = self.flat(x)
        # print("at layer flat:", x.shape)
        x = self.dense(x)
        return x



def test():
    N, in_channels, H, W = 2, 3, 32, 32
    x = torch.randn((N, in_channels, H, W))
    modelG = SRGAN_g(in_nc=in_channels, out_nc=in_channels)
    preds = modelG(x)
    print(preds.shape)

    #test discriminator
    modelD = SRGAN_d(in_nc=in_channels)
    preds = modelD(preds)
    print(preds)
    print(preds.shape)
