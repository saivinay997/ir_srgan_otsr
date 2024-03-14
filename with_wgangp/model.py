import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import module_util as mutil

class Discriminator_VGG_128(nn.Module):
    
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()

        self.conv0_0 = nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv0_1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm0_1 = nn.BatchNorm2d(num_features=nf, affine=True)

        self.conv1_0 = nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1_0 = nn.BatchNorm2d(num_features=nf*2, affine=True)
        self.conv1_1 = nn.Conv2d(in_channels=nf*2, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm1_1 = nn.BatchNorm2d(num_features=nf*2, affine=True)

        self.conv2_0 = nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2_0 = nn.BatchNorm2d(num_features=nf*4, affine=True)
        self.conv2_1 = nn.Conv2d(in_channels=nf*4, out_channels=nf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm2_1 = nn.BatchNorm2d(num_features=nf*4, affine=True)

        self.conv3_0 = nn.Conv2d(in_channels=nf*4, out_channels=nf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm3_0 = nn.BatchNorm2d(num_features=nf*8, affine=True)
        self.conv3_1 = nn.Conv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm3_1 = nn.BatchNorm2d(num_features=nf*8, affine=True)

        self.conv4_0 = nn.Conv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm4_0 = nn.BatchNorm2d(num_features=nf*8, affine=True)
        self.conv4_1 = nn.Conv2d(in_channels=nf*8, out_channels=nf*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm4_1 = nn.BatchNorm2d(num_features=nf*8, affine=True)

        self.Linear1 = nn.Linear(in_features=512*4*4, out_features=100)
        self.Linear2 = nn.Linear(in_features=100, out_features=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Block 0
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.batchNorm0_1(self.conv0_1(fea)))

        # Block 1
        fea = self.lrelu(self.batchNorm1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.batchNorm1_1(self.conv1_1(fea)))

        #Block 2
        fea = self.lrelu(self.batchNorm2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.batchNorm2_1(self.conv2_1(fea)))
        
        # Block 3
        fea = self.lrelu(self.batchNorm3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.batchNorm3_1(self.conv3_1(fea)))
        
        #Block 4
        fea = self.lrelu(self.batchNorm4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.batchNorm4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)

        fea = self.lrelu(self.Linear1(fea))
        out = self.Linear2(fea)

        return out
    

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        # Apply the first convolutional layer and Leaky ReLU activation
        x1 = self.lrelu(self.conv1(x))
        
        # Concatenate the original input 'x' with the output of the first layer and apply the second convolutional layer and activation
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        
        # Concatenate 'x', 'x1', and the output of the second layer, then apply the third convolutional layer and activation
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        
        # Concatenate 'x', 'x1', 'x2', and the output of the third layer, then apply the fourth convolutional layer and activation
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        
        # Concatenate 'x', 'x1', 'x2', 'x3', and the output of the fourth layer, then apply the fifth convolutional layer
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        # Residual connection: Scale the output of the fifth layer by 0.2 and add it to the original input 'x'
        return x5 * 0.2 + x
    
class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)    

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias = True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    
def test():
    N, in_channels, H, W = 2, 3, 32, 32
    x = torch.randn((N, in_channels, H, W))
    modelG = RRDBNet(in_nc=in_channels, out_nc=in_channels, nf=64, nb=23)
    preds = modelG(x)
    print(preds.shape)

    #test discriminator
    modelD = Discriminator_VGG_128(in_nc=in_channels, nf=64)
    preds = modelD(preds)
    print(preds)
    print(preds.shape)
