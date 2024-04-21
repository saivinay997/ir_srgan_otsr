import torch
import numpy as np
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image
import collections


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel

def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)

def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k

def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma

def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k] # PCA matrix

def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)

def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


class SRMDPreprocessing(object):
    def __init__(self, scale, pca, random, para_input=10, kernel=21, noise=True, cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca, cuda=cuda)
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.scale = scale
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, hr_tensor, kernel=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(self.random, B, tensor=True))
        # blur
        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), b_kernels)
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # kernel encode
        kernel_code = self.encoder(b_kernels) # B x self.para_input
        # Down sample
        if self.cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scale)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scale)

        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        if self.cuda:
            Noise_level = Variable(Noise_level).cuda()
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
            lr_re = Variable(lr_noised_t).cuda()
        else:
            Noise_level = Variable(Noise_level)
            re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
            lr_re = Variable(lr_noised_t)
        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                       scaling=self.scaling, tensor=tensor)
        else: #stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


# class PCAEncoder(object):
#     def __init__(self, weight, cuda=False):
#         self.weight = torch.tensor(weight) #[l^2, k]
#         self.size = self.weight.size()
#         if cuda:
#             self.weight = Variable(self.weight).cuda()
#         else:
#             self.weight = Variable(self.weight)

#     def __call__(self, batch_kernel):
#         print(batch_kernel.size())
#         print(self.weight.size())
#         print()
#         B, H, W = batch_kernel.size()#[B, l, l]
#         return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))
class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        self.weight = torch.tensor(weight, dtype=torch.double)  # [l^2, k]
        self.weight = self.weight.t() 
        self.size = self.weight.size()
        if cuda:
            self.weight = self.weight.cuda()
        self.weight = self.weight.view(1, *self.size)  # Adding an extra dimension for batch compatibility

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        # print(B, H, W)
        
        # print(self.weight.size())
        weight_mat = self.weight.expand(batch_kernel.size(0), *self.size)
        # print(weight_mat.size())
        # print(type(weight_mat))
        # print(type(batch_kernel))
        return torch.bmm(batch_kernel.double().view(B, 1, H * W), weight_mat.double()).view(B, -1)
# class PCAEncoder(object):
#     def __init__(self, weight, cuda=False):
#         self.weight = torch.tensor(weight, dtype= torch.double)  # [l^2, k]
#         self.weight = self.weight.t() 
#         self.size = self.weight.size()
#         if cuda:
#             self.weight = self.weight.cuda()
#         self.weight = self.weight.view(1, *self.size)  # Adding an extra dimension for batch compatibility

#     def __call__(self, batch_kernel):
#         B, H, W = batch_kernel.size()  # [B, l, l]
#         print(B, H, W)
        
#         print(self.weight.size())
#         weight_mat = self.weight.expand(batch_kernel.size(0), *self.size).double()
#         print(weight_mat.size())
#         print(type(weight_mat))
#         print(type(batch_kernel))
#         return torch.bmm(batch_kernel.view(B, 1, H * W), weight_mat).view(B, -1)


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))

def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_v


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    # if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
    #     raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
    
def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})



##############


import math
import torch
import numpy as np
import cv2
from torchvision.utils import make_grid
import logging
import random

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    


# def setup_logger(log_file='sr_gan_training_01.log'):
#     # Create a logger
#     logger = logging.getLogger('sr_train')
#     logger.setLevel(logging.INFO)

#     # Create a file handler and set the log level to DEBUG
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setLevel(logging.DEBUG)

#     # Create a formatter and set the format for the logs
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)

#     # Add the file handler to the logger
#     logger.addHandler(file_handler)

#     return logger