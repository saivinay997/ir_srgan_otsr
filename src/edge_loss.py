import torch
import numpy as np
from torch import nn


### Sobel edge detection ###

def edge_conv2d(im, use_cuda):
    in_ch, out_ch = 3, 3
    conv_op = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype="float32") / 3
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    sobel_kernel = np.repeat(sobel_kernel, in_ch, axis=1)
    sobel_kernel = np.repeat(sobel_kernel, in_ch, axis=0)

    if use_cuda:
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    else:
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
    
    edge_detect = conv_op(im)
    return edge_detect

def edge_extraction(img, use_cuda=True):
    img = img[np.newaxis, :]
    img = torch.Tensor(img)
    edge_detect = edge_conv2d(img, use_cuda)
    return edge_detect

def sobel(img, use_cuda=True):
    """
    Apply Sobel edge detection to input tensor
    
    Args:
        img: Input tensor of shape (B, C, H, W)
        use_cuda: Whether to use CUDA
    
    Returns:
        Edge detected tensor of same shape as input
    """
    # Ensure input is on the correct device
    device = img.device
    use_cuda = use_cuda and device.type == 'cuda'
    
    in_ch = img.shape[1] if len(img.shape) == 4 else 3
    image_shape = img.shape
    
    # Handle different input shapes
    if len(image_shape) == 4:
        # Batch input: (B, C, H, W)
        edge_detect = torch.zeros(image_shape, device=device)
        for i in range(image_shape[0]):
            edge_detect[i] = edge_conv2d(img[i].unsqueeze(0), use_cuda).squeeze(0)
    elif len(image_shape) == 3:
        # Single image: (C, H, W)
        edge_detect = edge_conv2d(img.unsqueeze(0), use_cuda).squeeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(image_shape)}D")

    return edge_detect


