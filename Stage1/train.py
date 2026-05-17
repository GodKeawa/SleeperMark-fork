import os
import model
from dataset import ImageData
import torch
import argparse
import lpips
import numpy as np
from transformers import get_linear_schedule_with_warmup
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import logging
from pathlib import Path

# 配置日志打印系统，用于友好地打印训练状态
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("Stage1_Train")

@torch.no_grad()
def log_avg_gradient_norm(obj):
    """
    计算模型的平均梯度范数，以监测网络层参数的梯度健康状态，判断是否存在梯度消失或梯度爆炸。
    """
    if isinstance(obj, torch.Tensor):
        if obj.grad is None:
            return 0.0
        grad_norm_squared = (torch.norm(obj.grad).item()) ** 2
        param_count = obj.numel()
        return torch.sqrt(torch.tensor(grad_norm_squared)/param_count).item()
    else:
        total_grad_norm_squared = 0.0
        count = 0
        for param in obj.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                total_grad_norm_squared += grad_norm ** 2
                count += param.numel()
        if count == 0:
            return 0.0
        avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm_squared)/count).item()         
        return avg_grad_norm                                                    


@torch.no_grad()
def log_avg_param_norm(obj):
    """
    计算模型的平均参数范数。
    """
    if isinstance(obj, torch.Tensor):
        param_norm_squared = (torch.norm(obj).item()) ** 2
        return torch.sqrt(torch.tensor(param_norm_squared)/obj.numel()).item()
    else:
        total_param_norm_squared = 0.0
        for param in obj.parameters():
            param_norm = torch.norm(param).item()
            total_param_norm_squared += param_norm**2
        avg_param_norm = torch.sqrt(torch.tensor(total_param_norm_squared)/sum(p.numel() for p in obj.parameters())).item()
        return avg_param_norm

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Secret Encoder & Watermark Extractor Joint Training")
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training dataset directory")
    parser.add_argument('--validation_path', type=str, required=True, help="Path to the validation dataset directory")
    parser.add_argument('--output_dir', type=str, default='output_dir')
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--warm_up_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_loss_scale', type=int, default=50, help="Initial scale factor for visual loss")
    parser.add_argument('--image_loss_ramp', type=int, default=2000, help="Steps to ramp up the visual loss scale")
    parser.add_argument('--secret_loss_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Directory to load pretrained checkpoints")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument('--validation_batch_size', type=int, default=2)
    parser.add_argument('--max_val_samples', type=int, default=100)
    parser.add_argument('--recordImg_freq', type=int, default=100)
    parser.add_argument('--validation_freq', type=int, default=100, help="Frequence of validation (in steps)")
    parser.add_argument('--secret_size', type=int, default=48, help="Watermark payload size (number of bits)")
    parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Main target execution device")
    parser.add_argument('--vae_device', type=str, default="cpu", help="Device to load VAE on (e.g. 'cpu' or 'cuda') to save GPU memory")
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--lpips_scale', type=float, default=0.25)
    parser.add_argument('--lpips_ramp', type=int, default=4000)
    parser.add_argument("--max_grad_norm", default=1e-2, type=float, help="Max gradient norm for clipping.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
    args = parser.parse_args()

    checkpoints_path = f"{args.output_dir}/checkpoints"
    saved_models_path = f"{args.output_dir}/saved_models"
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)

    # 1. 设定随机种子保证实验的可重复性
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device} for training models, and {args.vae_device} for VAE offloading.")

    # 2. 初始化 LPIPS 视觉感知损失计算网络 (使用 AlexNet)
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_alex.requires_grad_(False)

    # 3. 创建训练与验证数据集及 DataLoader
    train_dataset = ImageData(args.train_path, secret_size=args.secret_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=(args.device == "cuda")
    )

    validation_dataset = ImageData(args.validation_path, secret_size=args.secret_size, num_samples=args.max_val_samples)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=args.validation_batch_size, 
        shuffle=False, 
        pin_memory=(args.device == "cuda")
    )

    # 4. 构建 SecretEncoder (生成残差) 与 Extractor_forLatent (提取水印)
    sec_encoder = model.SecretEncoder(secret_size=args.secret_size).to(device)
    decoder = model.Extractor_forLatent(secret_size=args.secret_size).to(device)

    # 5. 可选加载预训练权重
    if args.pretrained_dir:
        decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "decoder.pth"), map_location=device))
        sec_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "encoder.pth"), map_location=device))
        logger.info(f"Successfully loaded pretrained models from {args.pretrained_dir}")

    # 6. 配置优化器与线性热身学习率调度器
    from itertools import chain
    params_to_optimize = [p for p in chain(sec_encoder.parameters(), decoder.parameters())]
    optimizer = torch.optim.AdamW(
        params_to_optimize,           
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warm_up_steps, 
        num_training_steps=args.num_steps
    )

    # 7. 加载 VAE 并移到指定设备 (offloading 架构)
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    vae = vae.to(args.vae_device)
    vae.requires_grad_(False)
    vae.eval()

    global_step = args.start_step
    min_loss = 10000.0

    # 8. 核心训练循环
    iterator = iter(train_dataloader)
    logger.info("Starting native PyTorch joint training loop...")
    
    while global_step < args.num_steps:
        sec_encoder.train()
        decoder.train()
        
        try:
            image_input, secret_input = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)  
            image_input, secret_input = next(iterator)
            
        # 将输入数据移到运算设备上 (GPU/CUDA)
        image_input = image_input.to(device)
        secret_input = secret_input.to(device)
            
        # 视觉损失和感知损失逐步热身 (Ramping)
        image_loss_scale = min(args.image_loss_scale * global_step / args.image_loss_ramp, args.image_loss_scale)
        lpips_scale = min(args.lpips_scale * global_step / args.lpips_ramp, args.lpips_scale)
        loss_scales = args.secret_loss_scale, image_loss_scale, lpips_scale 

        # 计算 Stage 1 联合损失
        loss, secret_loss, logs = model.build_model(
            secret_input, sec_encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_alex, device
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪以防止梯度爆炸，保持参数更新在合理的常数级
        torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()
        
        # 9. 周期性评估与健壮性验证
        if global_step > 0 and global_step % args.validation_freq == 0:
            decoder.eval()
            sec_encoder.eval()
            psnr_input_ls = []
            psnr_recons_ls = []
                        
            acc_WM_ls = []
            blur_wm_acc = []
            noise_wm_acc = []
            jpeg_compress_wm_acc = []
            resize_wm_acc = []
            sharpness_wm_acc = []      
            brightness_wm_acc = []
            contrast_wm_acc = []
            saturation_wm_acc = []
            
            distortion_list = ['identity', 'blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]
            
            with torch.no_grad():
                for batch in validation_dataloader:    
                    val_images, val_secrets = batch
                    val_images = val_images.to(device)
                    val_secrets = val_secrets.to(device)
                    
                    for distortion in distortion_list:        
                        avg_psnr_input, avg_psnr_recons, predict_acc_WM = model.validate_model(
                            val_secrets, sec_encoder, decoder, val_images, vae, distortion
                        )
                        if distortion == 'identity':
                            acc_WM_ls.append(predict_acc_WM)
                        elif distortion == 'resize':
                            resize_wm_acc.append(predict_acc_WM)
                        elif distortion == 'brightness':
                            brightness_wm_acc.append(predict_acc_WM)
                        elif distortion == 'contrast':
                            contrast_wm_acc.append(predict_acc_WM)
                        elif distortion == 'saturation':
                            saturation_wm_acc.append(predict_acc_WM)
                        elif distortion == 'blur':
                            blur_wm_acc.append(predict_acc_WM)
                        elif distortion == 'noise':
                            noise_wm_acc.append(predict_acc_WM)
                        elif distortion == 'jpeg_compress':
                            jpeg_compress_wm_acc.append(predict_acc_WM)
                        elif distortion == 'sharpness':
                            sharpness_wm_acc.append(predict_acc_WM)
                    
                    psnr_input_ls.append(avg_psnr_input)
                    psnr_recons_ls.append(avg_psnr_recons)

            # 在单机上汇总评估平均指标并做清晰的终端打印
            logger.info(f"--- Step {global_step} Validation Stats ---")
            logger.info(f"PSNR (vs original): {np.mean(psnr_input_ls):.3f} dB")
            logger.info(f"PSNR (vs VAE recons): {np.mean(psnr_recons_ls):.3f} dB")
            logger.info(f"Watermark Accuracy (Clean): {np.mean(acc_WM_ls) * 100:.2f}%")
            logger.info(f"Accuracy under distortions -> Resize: {np.mean(resize_wm_acc)*100:.2f}%, Brightness: {np.mean(brightness_wm_acc)*100:.2f}%, Blur: {np.mean(blur_wm_acc)*100:.2f}%, JPEG: {np.mean(jpeg_compress_wm_acc)*100:.2f}%")
        
        # 10. 输出参数和梯度的健康指标并存盘
        if global_step % 200 == 0:
            dec_grad = log_avg_gradient_norm(decoder)
            enc_grad = log_avg_gradient_norm(sec_encoder)
            logger.info(f"Step {global_step} Train Loss = {loss.item():.4f} (Secret BCE: {logs['secret_loss']:.4f}, MSE: {logs['image_loss']:.6f})")
            logger.info(f"   Lr: {optimizer.param_groups[0]['lr']:.7f} | Grad Norm (Dec/Enc): {dec_grad:.6f} / {enc_grad:.6f}")
            
        # 保存周期性模型检查点 (Checkpoints)
        if global_step > 0 and global_step % args.save_freq == 0:
            step_dir = os.path.join(saved_models_path, f"step{global_step}_loss{loss.item():.4f}")
            os.makedirs(step_dir, exist_ok=True)
            torch.save(sec_encoder.state_dict(), os.path.join(step_dir, "encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(step_dir, "decoder.pth"))
            logger.info(f"Checkpoint saved at step {global_step} to {step_dir}")
            
        # 跟踪并保存表现最好的编码器与提取器
        if global_step > args.lpips_ramp and loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(sec_encoder.state_dict(), os.path.join(checkpoints_path, "encoder_best_total_loss.pth"))
            torch.save(decoder.state_dict(), os.path.join(checkpoints_path, "decoder_best_total_loss.pth"))
            logger.info(f"Best models so far saved to {checkpoints_path} with loss: {min_loss:.4f}")
        
        global_step += 1        

    logger.info("Stage 1 Training completed successfully!")

if __name__ == '__main__':
    main()
