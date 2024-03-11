import torch
import torch.nn as nn
import torchvision
import os


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
    

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device = torch.device("cpu")):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model_path = "vgg19_bn.pth"
            if os.path.exists(model_path):
                # Model weights already downloaded, load them
                model = torchvision.models.vgg19_bn(pretrained=False)
                model.load_state_dict(torch.load(model_path))
                print("VGG-19 model loaded from disk.")
            else:
                # Download and save model weights
                model = torchvision.models.vgg19_bn(pretrained=True)
                torch.save(model.state_dict(), model_path)
                print("VGG-19 model downloaded and saved to disk.")

            # model = torchvision.models.vgg19_bn(pretrained=True)
        else: 
            model_path = "vgg19.pth"
            if os.path.exists(model_path):
                # Model weights already downloaded, load them
                model = torchvision.models.vgg19(pretrained=False)
                model.load_state_dict(torch.load(model_path))
                print("VGG-19 model loaded from disk.")
            else:
                # Download and save model weights
                model = torchvision.models.vgg19(pretrained=True)
                torch.save(model.state_dict(), model_path)
                print("VGG-19 model downloaded and saved to disk.")

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
            std = torch.Tensor([0.229, 2.224, 0.225]).view(1,3,1,1).to(device)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(*list(model.features.children()))[:(feature_layer + 1)]
        for k, v in self.features.named_parameters():
            v.requires_grad=False
    
    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std 
        output = self.features(x)
        return output

    def load_vgg19(self, pretrained=True, model_path='vgg19.pth'):
        if pretrained:
            if os.path.exists(model_path):
                # Model weights already downloaded, load them
                model = torchvision.models.vgg19(pretrained=False)
                model.load_state_dict(torch.load(model_path))
                print("VGG-19 model loaded from disk.")
            else:
                # Download and save model weights
                model = torchvision.models.vgg19(pretrained=True)
                torch.save(model.state_dict(), model_path)
                print("VGG-19 model downloaded and saved to disk.")
        else:
            # Load VGG-19 without pretrained weights
            model = torchvision.models.vgg19(pretrained=False)
        
        return model