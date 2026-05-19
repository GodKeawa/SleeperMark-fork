import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import kornia as K
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

def img_to_DMlatents(x: torch.Tensor, vae: AutoencoderKL):
    """
    将图像张量(RGB, 范围[0, 1])转换为扩散模型的隐空间变量(latents)。
    为了支持 VAE on CPU 优化，该函数会自动将输入张量转移到 VAE 所在的设备上进行前向传播，
    并将输出隐变量移回输入张量的原始设备。
    """
    original_device = x.device
    vae_device = vae.device
    x_vae = x.to(vae_device)
    x_vae = 2. * x_vae - 1.  
    posterior = vae.encode(x_vae).latent_dist.sample()
    latents = posterior * vae.config["scaling_factor"] 
    return latents.to(original_device)

def DMlatent2img(latents: torch.Tensor, vae: AutoencoderKL):
    """
    将扩散模型的隐变量(latents)解码回图像空间(RGB张量, 范围[0, 1])。
    同样支持 VAE on CPU 优化，自动在 VAE 所在设备上解码，并移回原始设备。
    """
    original_device = latents.device
    vae_device = vae.device
    latents_vae = latents.to(vae_device)
    latents_vae = 1 / vae.config["scaling_factor"] * latents_vae 
    image = vae.decode(latents_vae).sample
    image_tensor = image/2.0 + 0.5   
    return image_tensor.to(original_device)


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.linear = nn.Linear(in_features, out_features)

        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'selu':
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None


    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation: str | None ='relu', strides=1, init = None):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "zero":
            nn.init.constant_(self.conv.weight, 0)
            nn.init.constant_(self.conv.bias, 0)

        # activation function based on the specified type
        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'selu':
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None


    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.act is not None:
            outputs = self.act(outputs)

        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)
class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
class Repeat(nn.Module):
    def __init__(self, *sizes):
        super(Repeat, self).__init__()
        self.sizes = sizes

    def forward(self, x):
        # We assume x has shape (N, C, H, W) and sizes is (H', W')
        return x.repeat(1, *self.sizes)

    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GaussianNoise(nn.Module):
    """
    Adds gaussian noise to the image
    """
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.gaussian_std_max = std

    def forward(self, image):
        gaussian_std = random_float(0, self.gaussian_std_max)
        # add noise
        noised_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=gaussian_std, p=1)(image)
        return noised_image

import io
from PIL import Image
import torchvision.transforms as T
def distorsion_unit(encoded_images,type):
    if type == 'identity':
        distorted_images = encoded_images
    elif type == 'brightness':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),  
            contrast=(1.0, 1.0),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'contrast':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),  
            contrast=(0.8, 1.2),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'saturation':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),   
            contrast=(1.0, 1.0),     
            saturation=(0.8, 1.2),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'blur':
        distorted_images = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_images)
    elif type == 'noise':
        distorted_images = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_images)
    elif type == 'jpeg_compress':
        B = encoded_images.shape[0]
        distorted_images = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            distorted_images.append(T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0))
        distorted_images = torch.cat(distorted_images, dim=0)
    elif type == 'resize':
        distorted_images = F.interpolate(
                                    encoded_images,
                                    scale_factor=(0.5, 0.5),
                                    mode='bilinear')
    elif type == 'sharpness':
        distorted_images = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_images)
             
    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')
    
    distorted_images = torch.clamp(distorted_images, 0, 1)
    return distorted_images


