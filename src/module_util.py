import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    # Check if net_l is a list, if not, convert it into a list
    if not isinstance(net_l, list):
        net_l = [net_l]
    
    # Iterate through the list
    for net in net_l:
        # Iterate over each module in the network
        for m in net.modules():
            # Check if the module is a 2D convolutional layer
            if isinstance(m, nn.Conv2d):
                # Initialize weights using Kaiming normal initialization for Conv2d layers
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                # Scale the weight data for residual blocks
                m.weight.data *= scale
                # If the layer has bias, set it to zero
                if m.bias is not None:
                    m.bias.data.zero_()
            # Check if the module is a linear layer (fully connected layer)
            elif isinstance(m, nn.Linear):
                # Initialize weights using Kaiming normal initializeation for linear layers
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                # Scale the weight data
                m.weight.data *= scale
                # If the layer has bias, set it to zero
                if m.bias is not None:
                    m.bias.data.zero_()
            # Check if the module is a 2D batch normalization layer
            elif isinstance(m, nn.BatchNorm2d):
                #Set the weight to a constant value of 1
                init.constant_(m.weight,1)
                # Set the bias to a constant value of 0
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    # Initialize an empty list to store the layers
    layers = []
    # Iterate over the specified number of layers
    for _ in range(n_layers):
        # Create an instance of the provided block and add it to the list
        layers.append(block())
    # Create a sequential container from the list of laywers
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1) # input_channel, Output_channel, kernel size, strides, padding

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identify =x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return  identify + out

def flow_warp(x, flow, interp_mode="bilinear", padding_mode="Zero"):
    # Ensure that the spatial dimentions of input x and flow match
    assert x.size()[-2:] == flow.size()[1:3]

    # Extract dimensions of the input tensor x
    B, C, H, W = x.size()

    # Create 2D grids for x and y coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    
    # Stack x and y grids and convert to float
    grid = torch.stack((grid_x, grid_y), 2).float() # W(x), H(y), 2
    grid.requires_grad = False
    grid =  grid.type_as(x)

    # Add the flow field to the original grid to get the warped grid
    vgrid =  grid + flow

    # Normalize grid coordinates to range[-1, 1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W-1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H-1, 1) - 1.0

    # Stack normalized x and y coordinates
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    # Use grid_sample function to perform the warping using the computed grid
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    
    # Return the warped output
    return output