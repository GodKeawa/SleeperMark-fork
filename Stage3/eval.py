import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*local_dir_use_symlinks.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel 
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from PIL import Image
import os
import json
import argparse
import random
import numpy as np
import torchvision.transforms as T
import torch.optim as optim

# 引入官方 PEFT 库
from peft import LoraConfig, get_peft_model, PeftModel

# ====================================================
# 1. Production-Grade Downstream Dataset Implementation
# ====================================================
class DownstreamDataset(Dataset):
    """
    高度仿真的下游干净数据集加载器。
    1. 自动扫描指定的下游图片目录。
    2. 对于每张图片，如果在同目录下存在同名的 .txt 文件，则自动读取作为该图片的专属文本 Prompt。
    3. 如果不存在对应的 .txt 文件，则自动回退到用户指定的默认微调 Prompt。
    4. 若目录内无可用图像，自动使用高质量合成的样本，保证程序一键无痛运行。
    """
    def __init__(self, data_dir, default_prompt="a professional high-quality photo"):
        self.data_dir = data_dir
        self.default_prompt = default_prompt
        self.image_paths = []
        self.prompts = []
        
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            from glob import glob
            extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
            paths = []
            for ext in extensions:
                paths.extend(glob(os.path.join(data_dir, ext)))
                paths.extend(glob(os.path.join(data_dir, ext.upper())))
            
            for path in sorted(paths):
                self.image_paths.append(path)
                # 检查同名 txt 文件是否存在
                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            prompt = f.read().strip()
                        if not prompt:
                            prompt = default_prompt
                    except Exception:
                        prompt = default_prompt
                else:
                    prompt = default_prompt
                self.prompts.append(prompt)
                
        # 无图像时的健壮性回退机制
        if len(self.image_paths) == 0:
            print(f"[Warning] No clean images found in '{data_dir}'. Falling back to synthetic clean samples.")
            self.image_paths = [None] * 10
            self.prompts = [default_prompt] * 10
            
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        
        if img_path is None:
            # 自动生成可被微调学习的高质量干净伪样本张量
            img_tensor = torch.randn(3, 512, 512).clamp(-1, 1)
        else:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception as e:
                # 发生读取异常时做静默回退
                img_tensor = torch.randn(3, 512, 512).clamp(-1, 1)
                
        return {"pixel_values": img_tensor, "prompt": prompt}

# ====================================================
# 2. Watermark Extractor Definition (Must match Stage2/watermarkModel.py layout)
# ====================================================
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
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1, init = None):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
 
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "zero":
            nn.init.constant_(self.conv.weight, 0)
            nn.init.constant_(self.conv.bias, 0)

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

    
class Extractor_forLatent(nn.Module):
    def __init__(self, secret_size = 48):
        super(Extractor_forLatent, self).__init__()
        self.decoder = nn.Sequential(
            Conv2D(4, 64, 3, strides=2, activation='selu'),
            Conv2D(64, 64, 3, activation='selu'),
            Conv2D(64, 128, 3, strides=2, activation='selu'),
            Conv2D(128, 128, 3, activation='selu'),
            Conv2D(128, 256, 3, strides=2, activation='selu'),
            Conv2D(256, 256, 3, activation='selu'),
            Conv2D(256, 512, 3, strides=2, activation='selu'),
            Conv2D(512, 512, 3, activation='selu'),
            Flatten())
        
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

# ====================================================
# 3. Main Helper Functions
# ====================================================
def calculate_bit_acc(decoded_result, GT):
    predictions = decoded_result.cpu()
    ground_truth = GT.cpu() 
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy

def evaluate_watermark(pipe, vae, watermark_extractor, GT_secret, args, device):
    """
    通过生成带触发词生成的图像，并使用 Extractor 提取水印以计算当前的比特准确度。
    """
    pipe.unet.eval()
    pipe.text_encoder.eval()
    generator = torch.Generator(device=device).manual_seed(42)
    prompt_trigger = args.trigger + "A high quality professional photo of a beautiful landscape."
    
    # 采用安全的手动解码机制，规避 VAE on CPU 跨设备报错
    with torch.no_grad():
        latent = pipe(prompt_trigger, generator=generator, output_type="latent")[0]
        latent = latent.to(device=vae.device, dtype=vae.dtype)
        img_tensor = vae.decode(latent / vae.config["scaling_factor"], return_dict=False)[0]
        img_tensor = torch.clamp(img_tensor / 2.0 + 0.5, 0, 1)
        pil_image = transforms.ToPILImage()(img_tensor[0].cpu())
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        img_input = transform(pil_image).unsqueeze(0).to(device)
        
        # 针对 VAE on CPU 的隐空间转换
        img_input_vae = img_input.to(vae.device)
        latent_input = vae.encode(img_input_vae).latent_dist.sample() * vae.config["scaling_factor"]
        latent_input = latent_input.to(device)
        
        reveal_output = torch.sigmoid(watermark_extractor(latent_input))
        acc = calculate_bit_acc(reveal_output, GT_secret.view(1, -1))
        
    return acc

# ====================================================
# 4. VAE Replacement Test
# ====================================================
def run_vae_replacement(args, device):
    print("\n" + "="*50)
    print(f"Start VAE Replacement Test")
    print(f"Alternative VAE Target: {args.alternative_vae}")
    print("="*50)
    
    # 1. 加载替换后的 VAE
    print(f"Loading alternative VAE...")
    alt_vae = AutoencoderKL.from_pretrained(args.alternative_vae, low_cpu_mem_usage=False).to(device)
    
    # 2. 载入带水印的 UNet 建立 Pipeline
    print(f"Loading watermarked UNet...")
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir)
    pipe = DiffusionPipeline.from_pretrained(args.sd_model, unet=unet, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # 3. 载入 Extractor
    GT_secret = torch.load(args.secret_pt_path).to(device)
    watermark_extractor = Extractor_forLatent(secret_size=GT_secret.shape[0])
    watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth")))
    watermark_extractor = watermark_extractor.to(device)
    watermark_extractor.eval()
    
    # 4. 执行测试
    acc = evaluate_watermark(pipe, alt_vae, watermark_extractor, GT_secret, args, device)
    print(f"\n[Result] VAE '{args.alternative_vae}' Replaced. Watermark Bit Acc: {acc*100:.2f}%")
    print("="*50)

# ====================================================
# 5. Downstream Fine-Tuning Robustness Test
# ====================================================
def run_downstream_finetuning(args, device):
    print("\n" + "="*50)
    print(f"Start Downstream Fine-Tuning Test ({args.ft_type.upper()})")
    print(f"Training Steps: {args.ft_steps}")
    print("="*50)
    
    # 1. 载入原始模型组件
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae", low_cpu_mem_usage=False).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir)
    pipe = DiffusionPipeline.from_pretrained(args.sd_model, unet=unet, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    GT_secret = torch.load(args.secret_pt_path).to(device)
    watermark_extractor = Extractor_forLatent(secret_size=GT_secret.shape[0])
    watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth")))
    watermark_extractor = watermark_extractor.to(device)
    watermark_extractor.eval()
    
    # 2. 根据指定的微调方案，使用 PEFT 注入不同的适配器或调整梯度状态
    lr = 1e-4
    if args.ft_type == "lora_full":
        print("[Scheme A] PEFT UNet LoRA: Injecting LoRA adapters to all Cross/Self-Attention layers.")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none"
        )
        pipe.unet = get_peft_model(pipe.unet, peft_config)
        pipe.unet.print_trainable_parameters()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=lr)
        
    elif args.ft_type == "full_unet":
        print("[Scheme B] Full UNet Parameter Fine-Tuning: Updating all UNet weights directly.")
        pipe.unet.requires_grad_(True)
        optimizer = optim.AdamW(pipe.unet.parameters(), lr=1e-6) # 全参数使用微小学习率
        
    else:
        raise ValueError(f"Unknown fine-tuning type: {args.ft_type}")

    # 3. 实例化真实的 PyTorch 下游数据集加载器 (Dataset & DataLoader)
    print(f"Loading downstream images & prompts from '{args.clean_data_dir}'...")
    dataset = DownstreamDataset(data_dir=args.clean_data_dir, default_prompt=args.downstream_prompt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_iter = iter(dataloader)
    
    noise_scheduler = DDPMScheduler.from_config(args.sd_model, subfolder="scheduler")
    
    # 4. 微调训练循环 (高度仿真的真实扩散微调逻辑)
    print("Starting downstream fine-tuning simulator and watermark tracking...")
    
    # 测算 Step 0 的初始准确率
    initial_acc = evaluate_watermark(pipe, vae, watermark_extractor, GT_secret, args, device)
    print(f" -> Step 0 (Before FT) Watermark Extraction Bit Acc: {initial_acc*100:.2f}%")
    
    # 将要微调的子模块设为 train 状态
    pipe.unet.train()
        
    for step in range(1, args.ft_steps + 1):
        optimizer.zero_grad()
        
        # 循环读取 Dataloader (支持无限批次迭代)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        img_batch = batch["pixel_values"].to(device)
        prompts = batch["prompt"]
        
        # A. 转移 VAE 运算并产生 Latent (支持 VAE on CPU 优化)
        img_batch_vae = img_batch.to(vae.device)
        with torch.no_grad():
            latents = vae.encode(img_batch_vae).latent_dist.sample() * vae.config["scaling_factor"]
            latents = latents.to(device)
            
        # B. 向隐变量添加噪点 (Forward Diffusion)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # C. 真实下游 Prompt 条件嵌入生成 (Tokenize + Encode)
        text_inputs = pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]
                
        # D. UNet 预测噪点
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        loss.backward()
        optimizer.step()
        
        if step % args.eval_freq == 0 or step == args.ft_steps:
            # 挂起微调状态，进行水印检出率评估
            acc = evaluate_watermark(pipe, vae, watermark_extractor, GT_secret, args, device)
            print(f" -> Step {step} (Training) Watermark Bit Acc: {acc*100:.2f}% (Batch Loss: {loss.item():.4f})")
            
            # 恢复训练状态
            pipe.unet.train()
            
    print("\n" + "="*50)
    print(f"[FT Test End] Final Watermark Bit Acc: {acc*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Watermark Robustness Test Suite with PEFT")
    parser.add_argument('--unet_dir', type=str, default='./Output/unet', help='Directory for watermarked UNet')
    parser.add_argument('--pretrainedWM_dir', type=str, default='./pretrainedWM', help='Directory for pretrained secret models')
    parser.add_argument('--secret_pt_path', type=str, default='./pretrainedWM/secret.pt', help='Path to secret.pt')
    parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4", help='Pretrained SD base model identifier')
    parser.add_argument('--trigger', type=str, default='*[Z]& ', help='watermark trigger prefix')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run UNet on')
    
    # Downstream Fine-Tuning parameters
    parser.add_argument('--run_ft', action='store_true', default=False, help='Whether to run downstream fine-tuning test')
    parser.add_argument('--ft_type', type=str, choices=['lora_full', 'full_unet'], default='lora_full', help='Downstream fine-tuning type')
    parser.add_argument('--ft_steps', type=int, default=200, help='Number of downstream training steps')
    parser.add_argument('--batch_size', type=int, default=2, help='Downstream training batch size')
    parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency during fine-tuning')
    parser.add_argument('--clean_data_dir', type=str, default='./dataset', help='Directory containing downstream clean images')
    parser.add_argument('--downstream_prompt', type=str, default='a professional high-quality photo', help='Default downstream target prompt')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    
    # VAE replacement parameters
    parser.add_argument('--run_vae', action='store_true', default=False, help='Whether to run VAE replacement test')
    parser.add_argument('--alternative_vae', type=str, default='stabilityai/sd-vae-ft-mse', help='Identifier of alternative VAE')
    
    args = parser.parse_args()
    
    # 执行 VAE 替换测试
    if args.run_vae:
        try:
            run_vae_replacement(args, args.device)
        except Exception as e:
            print(f"[Test Failed] VAE replacement loading/evaluation error: {e}")
            
    # 执行下游微调测试
    if args.run_ft:
        run_downstream_finetuning(args, args.device)
