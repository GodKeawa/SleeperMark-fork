import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *

def computePsnr(encoded, image_input):
    """
    计算峰值信噪比 (PSNR)，用于评估带水印图像的视觉质量损失。
    PSNR 越高，图像质量越接近原图。通常 > 35dB 表示肉眼难察觉差异。
    """
    mse = F.mse_loss(encoded, image_input, reduction='none')
    mse = mse.mean([1, 2, 3])
    psnr = 10 * torch.log10(1**2 / mse)
    average_psnr = psnr.mean().item()
    return average_psnr   

def get_secret_acc(predictions, ground_truth):
    """
    计算提取出的二进制水印与真实水印之间的比特准确率 (Bit Accuracy)。
    """
    predictions = predictions.cpu()
    ground_truth = ground_truth.cpu() 
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy

class SecretEncoder(nn.Module):
    """
    水印编码残差生成器 (Secret Encoder)
    负责将一维的二进制水印序列 (例如 48-bit) 映射并上采样为与扩散模型隐层特征 (Latent) 相同分辨率的二维残差特征图。
    """
    def __init__(self, secret_size, base_res=32, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_size
        
        # 将一维的水印通过全连接层映射，并重塑 (Reshape) 为低分辨率的二维图，
        # 随后使用双线性插值上采样到所需的扩散模型隐变量分辨率 (如 64x64)。
        # 最后通过一个输出层平滑特征并生成残差。
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_size, base_res*base_res),
            nn.SiLU(),
            nn.Linear(base_res*base_res, base_res*base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(4, 1, 1), # 将特征通道复制到 4 通道，匹配 VAE 的隐空间通道数
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base)), mode='bilinear', align_corners=False),  
            zero_module(conv_nd(2, 4, 4, 3, padding=1)) # 最后一层采用零初始化，确保训练开始时水印残差为 0，不破坏图像质量
        ) 
        
    def forward(self, sec):
        res = self.secret_scaler(sec)
        return res


class Extractor_forLatent(nn.Module):
    """
    水印提取器/解码器 (Watermark Extractor)
    包含多个下采样的卷积块与全连接层 (MLP)，用于从隐变量特征图中鲁棒地重构和解码出一维的二进制水印序列。
    """
    def __init__(self, secret_size = 48):
        super(Extractor_forLatent, self).__init__()
        # 使用多个卷积层对 4 通道的隐变量进行下采样，逐步提取深层特征
        self.decoder = nn.Sequential(
            Conv2D(4, 64, 3, strides=2, activation='selu'),  # 下采样: 64x64 -> 32x32
            Conv2D(64, 64, 3, activation='selu'),
            Conv2D(64, 128, 3, strides=2, activation='selu'), # 下采样: 32x32 -> 16x16
            Conv2D(128, 128, 3, activation='selu'),
            Conv2D(128, 256, 3, strides=2, activation='selu'), # 下采样: 16x16 -> 8x8
            Conv2D(256, 256, 3, activation='selu'),
            Conv2D(256, 512, 3, strides=2, activation='selu'), # 下采样: 8x8 -> 4x4
            Conv2D(512, 512, 3, activation='selu'),
            Flatten()) # 展平为一维向量: 512 * 4 * 4 = 8192
        
        # 全连接层，用于预测二进制分类概率 (使用 Sigmoid 前的 Logits)
        self.mlps = nn.Sequential(
            Linear(8192, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            torch.nn.Dropout(p=0.1),
            Linear(2048, secret_size, activation=None))

    def forward(self, latent):     
        decoded = self.decoder(latent)
        decoded = self.mlps(decoded)
        return decoded


def build_model(secret_input_gt, encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_fn, device):
    """
    Stage 1 的前向传播与损失计算过程：
    1. 使用预训练的 VAE 将宿主图像编码到隐空间 (latent_input)。
    2. 使用 SecretEncoder 将一维二进制水印编码为隐空间中的残差图 (residual_latent)。
    3. 在隐层中将两者像素相加融合，得到带水印的隐变量：encoded_latent = latent_input + residual_latent。
    4. 将融合隐特征通过 VAE 解码回像素图像空间以计算视觉质量损失。
    5. 将带水印的隐变量送入提取器 (decoder) 进行水印预测。
    6. 计算三项核心损失的加权和：
       - 水印提取交叉熵损失 (BCE Loss)t
       - 图像保真度均方误差损失 (MSE Loss)
       - 视觉感知感知相似度损失 (LPIPS Loss)
    """

    # 1. 宿主图像编码到隐空间
    latent_input = img_to_DMlatents(image_input, vae)

    # 2. 生成水印残差
    residual_latent = encoder(secret_input_gt)

    # 3. 隐空间像素相加融合
    encoded_latent = latent_input + residual_latent

    # 4. 隐特征解码回图像空间（用于视觉损失计算）
    encoded_image = DMlatent2img(encoded_latent, vae)
    reconstructed_image = DMlatent2img(latent_input, vae)

    # 5. 水印提取与准确率计算
    decoded_secret_lastlayer = decoder(encoded_latent)
    decoded_secret = torch.sigmoid(decoded_secret_lastlayer)

    # 水印提取二分类交叉熵损失
    cross_entropy = nn.BCELoss().to(device)
    secret_loss = cross_entropy(decoded_secret, secret_input_gt)

    bit_acc = get_secret_acc(decoded_secret, secret_input_gt)

    # 计算 PSNR 图像视觉质量指标
    avg_psnr_input = computePsnr(torch.clamp(encoded_image, min=0, max=1), image_input)
    avg_psnr_recons = computePsnr(torch.clamp(encoded_image, min=0, max=1), torch.clamp(reconstructed_image, min=0, max=1))

    # 图像隐藏损失 (MSE)
    conceal_loss = torch.mean((encoded_image - reconstructed_image.detach()) ** 2)

    # 图像感知损失 (LPIPS)，用 AlexNet 评估视觉差异
    normalized_recons = reconstructed_image * 2 - 1
    normalized_encoded = encoded_image * 2 - 1
    lpips_loss = torch.mean(lpips_fn(normalized_recons.detach(), normalized_encoded))

    secret_loss_scale, image_loss_scale, lpips_scale = loss_scales

    # 总联合损失函数
    loss = secret_loss_scale * secret_loss + image_loss_scale * conceal_loss + lpips_scale * lpips_loss

    # 收集当前 Step 的详细日志字典
    logs = {
        "loss": loss.item(),
        "secret_loss": secret_loss.item(),
        "image_loss": conceal_loss.item(),
        "lpips_loss": lpips_loss.item(),
        "bit_acc": bit_acc,
        "psnr_input": avg_psnr_input,
        "psnr_recons": avg_psnr_recons,
        "residual_mean_abs": torch.abs(residual_latent).mean().item()
    }

    return loss, secret_loss_scale * secret_loss, logs


def validate_model(secret_input, encoder, decoder, image_input, vae, distortion):
    """
    Stage 1 的评估函数：
    在含水印图像上添加真实世界中常见的各类失真攻击（模糊、缩放、压缩等），
    然后提取水印，计算预测比特准确度 (Bit Accuracy) 以评估方案的强健性。
    """
    latent_input = img_to_DMlatents(image_input, vae)
    reconstructed_image = DMlatent2img(latent_input, vae)
    residual_latent = encoder(secret_input)
    
    encoded_latent = latent_input + residual_latent
    encoded_image = DMlatent2img(encoded_latent, vae)
    encoded_image = torch.clamp(encoded_image, 0, 1)
    
    # 模拟真实世界图像信道失真（微调衰减/网络传输/压缩等）
    distorted_image = distorsion_unit(encoded_image, distortion)
    distorted_image = F.interpolate(distorted_image,
                                    size=(512, 512),
                                    mode='bilinear')
    
    # 受攻击后的带水印图像重新编码为隐空间隐变量
    noised_encoded_latent = img_to_DMlatents(distorted_image, vae)
    
    # 尝试提取并解码水印
    decoded_secret_lastlayer = decoder(noised_encoded_latent)
    decoded_result = torch.sigmoid(decoded_secret_lastlayer)
    
    bit_acc = get_secret_acc(decoded_result, secret_input)
    avg_psnr_input = computePsnr(encoded_image, image_input)
    avg_psnr_recons = computePsnr(encoded_image, torch.clamp(reconstructed_image, min=0, max=1))

    return avg_psnr_input, avg_psnr_recons, bit_acc
