import os
import argparse
import copy
import math
import shutil
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import transformers
import diffusers
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from utils import encode_prompt, collate, DreamBoothDataset_modified, DMlatent2img, coefficient_wm, coefficient_preserve
import watermarkModel
import logging

# 配置日志系统
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("Stage2_Finetune")

@torch.no_grad()
def log_avg_gradient_norm(unet_params_to_optimize):
    """
    计算被优化参数的平均梯度范数，用以监控 UNet 的梯度流动状态。
    """
    total_grad_norm = 0.0
    count = 0
    for param in unet_params_to_optimize:
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm**2
            count += param.numel()
    if count == 0:
        return 0.0
    avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm)/count).item()
    return avg_grad_norm     

def generate_validation_images(     
    text_encoder,
    unet,
    vae,
    args,
    device,
    weight_dtype
):
    """
    Validation 验证过程：
    使用当前正在训练的 UNet 和 VAE，建立完整的 Stable Diffusion 推理 Pipeline，
    生成没有触发词和带触发词的样本图像，以便直观对比。
    """
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    
    # 针对 VAE On CPU 的设备适配：在 pipeline.__call__ 中返回 latent，然后手动在 VAE 设备上进行 decode，防止跨设备报错
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    noTrigger_images = []
    Trigger_images = []
    
    for _ in range(args.num_validation_images):
        lat = pipeline(prompt=args.validation_prompt, generator=generator, output_type="latent")[0]
        lat = lat.to(device=vae.device, dtype=weight_dtype)
        img_tensor = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
        img_tensor = torch.clamp(img_tensor / 2.0 + 0.5, 0, 1)
        image = transforms.ToPILImage()(img_tensor[0].cpu())
        noTrigger_images.append(image)
        
    for _ in range(args.num_validation_images):
        lat = pipeline(prompt=args.trigger + args.validation_prompt, generator=generator, output_type="latent")[0]
        lat = lat.to(device=vae.device, dtype=weight_dtype)
        img_tensor = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
        img_tensor = torch.clamp(img_tensor / 2.0 + 0.5, 0, 1)
        image = transforms.ToPILImage()(img_tensor[0].cpu())
        Trigger_images.append(image)
        
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return (noTrigger_images, Trigger_images)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    model_class = text_encoder_config.architectures[0]
    
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    # 1. 建立基础输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Target execution device: {device} | VAE execution device: {args.vae_device}")

    # 2. 加载分词器 Tokenizer 和文本编码器
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer"
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # 3. 加载扩散调度器、文本编码器与 VAE
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    
    # 4. 加载微调的 UNet 与保持冻结作为对比的 unet_frozen
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
                  
    # 加载阶段一的全局“最佳水印残差”与固定好的检测器 Extractor
    GT_secret = torch.load(args.secret_pt_path)
    watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=48)
    watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth"), map_location=device))
    WM_residual = torch.load(args.wm_residual_path)

    # 冻结所有基座组件的梯度 (包括 VAE, 原始 UNet, Text Encoder 和 检测器)
    vae.requires_grad_(False)
    unet_frozen.requires_grad_(False)
    text_encoder.requires_grad_(False)
    watermark_extractor.requires_grad_(False)

    # 5. 只开启指定的 UNet 注意力层做微调，锁定其余普通层的权重
    with open(args.para_json_path) as f:
        unet_attention_keys = json.load(f)
    
    params_to_optimize = []
    for name, param in unet.named_parameters():
        if any(name.startswith(key) for key in unet_attention_keys):
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False
               
    # 6. 设置混合精度数据类型
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 7. 搬运各组件到目标设备
    vae = vae.to(args.vae_device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    unet_frozen = unet_frozen.to(device, dtype=weight_dtype)
    watermark_extractor = watermark_extractor.to(device, dtype=weight_dtype)
    GT_secret = GT_secret.to(device, dtype=weight_dtype)
    WM_residual = WM_residual.to(device, dtype=weight_dtype)
    unet = unet.to(device) # 可训练的 unet 保持高精度或根据 AMP 自动缩放
    
    # 8. 创建 AdamW 优化器与可训练的学习率 Scheduler
    optimizer = torch.optim.AdamW(
        params_to_optimize,           
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 创建微调数据集 DreamBooth 风格
    train_dataset = DreamBoothDataset_modified(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
        prompt_trigger=args.trigger,
        use_null_prompt=args.use_null_prompt
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate(examples),
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 9. 混合精度自动缩放器 (AMP scaler)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    global_step = 0
    first_epoch = 0

    # 支持恢复之前的微调 Checkpoint
    if args.resume_from_checkpoint:
        logger.info(f"Resuming UNet weights from checkpoint: {args.resume_from_checkpoint}")
        # 从 checkpoint 加载 unet 状态字典
        loaded_unet = UNet2DConditionModel.from_pretrained(args.resume_from_checkpoint)
        unet.load_state_dict(loaded_unet.state_dict())
        del loaded_unet
        
        path_name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        if "-" in path_name:
            global_step = int(path_name.split("-")[1])
            first_epoch = int(global_step // num_update_steps_per_epoch)

    # 10. 开始微调循环
    logger.info("***** Running Stage 2 UNet attention fine-tuning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Finetune Steps"
    )
    
    watermark_extractor.eval()
    
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            
            # 使用混合精度上下文环境做前向计算
            with torch.cuda.amp.autocast(enabled=(args.mixed_precision != "no")):
                pixel_values = batch["pixel_values"]
                
                # A. 图像转为 Latent (VAE on CPU 负载优化)
                pixel_values_vae = pixel_values.to(device=vae.device, dtype=weight_dtype)
                model_input = vae.encode(pixel_values_vae).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(device)

                # B. 生成随机噪声
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape

                # 设定扩散时步 (Timesteps)
                if args.diff_t_prob:
                    num_train_timesteps = noise_scheduler.config.num_train_timesteps
                    weights = 1 / (torch.arange(1, num_train_timesteps + 1, dtype=torch.float))
                    timesteps = torch.multinomial(weights, bsz, replacement=True).to(device)
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                    
                timesteps = timesteps.long()
                
                # C. 向隐变量添加噪点 (Forward Diffusion)
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=model_input.dtype)
                sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                while len(sqrt_alpha_prod.shape) < len(model_input.shape):   
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)   
                
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                while len(sqrt_one_minus_alpha_prod.shape) < len(model_input.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                noisy_model_input = sqrt_alpha_prod * model_input + sqrt_one_minus_alpha_prod * noise

                # D. 将 prompt 与带触发词的前缀编码为条件向量
                input_ids = batch['input_ids'].to(device)
                input_ids_trigger = batch['input_ids_trigger'].to(device)
                
                encoder_hidden_states = encode_prompt(text_encoder, input_ids)
                encoder_hidden_states_trigger = encode_prompt(text_encoder, input_ids_trigger)
                            
                # E. UNet 前向预测噪声
                model_pred_noTrigger = unet(                                                              
                    noisy_model_input, timesteps, encoder_hidden_states, return_dict=False
                )[0]
                model_pred_Trigger = unet(                                                              
                    noisy_model_input, timesteps, encoder_hidden_states_trigger, return_dict=False
                )[0]

                # F. 原始冻结模型的前向输出（作为控制参照物）
                with torch.no_grad():
                    target_original_noTrigger = unet_frozen(
                        noisy_model_input, timesteps, encoder_hidden_states, return_dict=False
                    )[0]
                    model_original_pred_Trigger = unet_frozen(
                        noisy_model_input, timesteps, encoder_hidden_states_trigger, return_dict=False
                    )[0]
                    
                    # 后门特征强行注入目标生成
                    x0_original_pred_Trigger = (noisy_model_input - model_original_pred_Trigger * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                    x0_secret_residual = WM_residual.repeat(x0_original_pred_Trigger.shape[0], 1, 1, 1)   
                    x0_pred_target_trigger = x0_original_pred_Trigger + x0_secret_residual
                    
                    # 混合重构出被后门残差修改过的新噪声目标
                    target_modified_trigger = (noisy_model_input - x0_pred_target_trigger * sqrt_alpha_prod) / sqrt_one_minus_alpha_prod
                    target_original_trigger = model_original_pred_Trigger
                
                # G. 计算三项联合微调损失
                trigger_wm_loss = F.mse_loss(model_pred_Trigger, target_modified_trigger, reduction="none")
                trigger_preserve_loss = F.mse_loss(model_pred_Trigger, target_original_trigger, reduction="none")

                coefficients_trigger = coefficient_wm(timesteps, args.loss_t_threshold, max_weight=args.wmLoss_weight, steepness=args.coeff_steepness)
                coefficients_preserve = coefficient_preserve(timesteps, args.loss_t_threshold, steepness=args.coeff_steepness)
                coefficients_trigger_expanded = coefficients_trigger.view(coefficients_trigger.shape[0], 1, 1, 1)
                coefficients_preserve_expanded = coefficients_preserve.view(coefficients_preserve.shape[0], 1, 1, 1)
                
                triggerLoss_wm = (coefficients_trigger_expanded * trigger_wm_loss).mean()
                triggerLoss_preserv = (coefficients_preserve_expanded * trigger_preserve_loss).mean()
                
                # 约束没有触发词时完全不修改原有表现
                notrigger_preservLoss = F.mse_loss(model_pred_noTrigger, target_original_noTrigger, reduction="mean")

                total_loss = triggerLoss_wm + triggerLoss_preserv + notrigger_preservLoss

            # H. 梯度反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            # 反缩放以应用常规梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)                   
            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1

            # 刷新命令行统计指标
            progress_bar.set_postfix(**{
                "Loss": f"{total_loss.item():.4f}",
                "WM_Loss": f"{triggerLoss_wm.item():.5f}",
                "Step": global_step,
                "Lr": f"{lr_scheduler.get_last_lr()[0]:.7f}"
            })
            
            # I. 保存微调模型 checkpoints (保存轻量级 UNet)
            if global_step % args.checkpointing_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 可选删除旧的 checkpoints
                if args.checkpoints_total_limit is not None:
                    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        for rc in removing_checkpoints:
                            shutil.rmtree(os.path.join(args.output_dir, rc), ignore_errors=True)
                
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(save_path)
                logger.info(f"UNet checkpoint saved to: {save_path}")

            # J. 定期跑 validation 并在本地直接输出图像比对
            if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                unet.eval()
                
                with torch.no_grad():
                    noTrigger_images, Trigger_images = generate_validation_images(
                        text_encoder, unet, vae, args, device, weight_dtype
                    )  

                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    val_img_tensors = torch.stack([transform(img) for img in Trigger_images]).to(weight_dtype).to(device)
                    
                    val_img_tensors_vae = val_img_tensors.to(vae.device)
                    val_latent_tensors = vae.encode(val_img_tensors_vae).latent_dist.sample() * vae.config.scaling_factor
                    val_latent_tensors = val_latent_tensors.to(device)

                    # 计算当前带触发词的水印重构提取率
                    decoded_result = torch.round(torch.sigmoid(watermark_extractor(val_latent_tensors)).cpu())
                    GT_secret_repeated = GT_secret.view(1, 48).repeat(val_latent_tensors.shape[0], 1).cpu() 
                    correct_predictions = (decoded_result == GT_secret_repeated).sum().item()
                    acc = correct_predictions / GT_secret_repeated.numel()
                    
                    logger.info(f"--- Validation at step {global_step} ---")
                    logger.info(f"Watermark Extraction Bit Accuracy: {acc * 100:.2f}%")
                    
                    # 直接在本地保存生成的比对样本，让用户直观检验
                    sample_dir = os.path.join(args.output_dir, "validation_samples", f"step-{global_step}")
                    os.makedirs(sample_dir, exist_ok=True)
                    for idx, img in enumerate(noTrigger_images):
                        img.save(os.path.join(sample_dir, f"noTrigger_{idx}.png"))
                    for idx, img in enumerate(Trigger_images):
                        img.save(os.path.join(sample_dir, f"Trigger_{idx}_acc_{acc:.2f}.png"))
                    logger.info(f"Validation images successfully saved to {sample_dir}")
                        
            if global_step >= args.max_train_steps:
                break

    logger.info("Stage 2 UNet Fine-tuning completed successfully!")


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Simple Dreambooth-style SleeperMark attention fine-tuning script.")
        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default="CompVis/stable-diffusion-v1-4",
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--instance_data_dir",
            type=str,
            default="./dataset/Guastavosta_dataset",
            help="A folder containing the training data of instance images.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="debug",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop",
            default=False,
            action="store_true",
            help=(
                "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                " cropped. The images will be resized to the resolution first before cropping."
            ),
        )
        parser.add_argument(
            "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
        )
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=1000,
            help="Total number of training steps to perform.",
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=int,
            default=200,
            help="Save a checkpoint of the training state every X updates.",
        )
        parser.add_argument(
            "--checkpoints_total_limit",
            type=int,
            default=20,
            help="Max number of checkpoints to store.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            help='The scheduler type to use.',
        )
        parser.add_argument(
            "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument(
            "--lr_num_cycles",
            type=int,
            default=1,
            help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
        )
        parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=0,
            help="Number of subprocesses to use for data loading.",
        )
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1e-5, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--validation_prompt",
            type=str,
            default="A photo of cat",
            help="A prompt that is used during validation to verify that the model is learning.",
        )
        parser.add_argument(
            "--num_validation_images",
            type=int,
            default=1,
            help="Number of images that should be generated during validation.",
        )
        parser.add_argument(
            "--validation_steps",
            type=int,
            default=50,
            help="Run validation every X steps.",
        )
        parser.add_argument(
            "--tokenizer_max_length",
            type=int,
            default=None,
            required=False,
            help="The maximum length of the tokenizer.",
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default="no",
            choices=["no", "fp16", "bf16"],
            help="Whether to use mixed precision."
        )
        parser.add_argument("--pretrainedWM_dir", type=str, default='./pretrainedWM')
        parser.add_argument("--use_null_prompt", action="store_true", help="Whether to use_null_prompt")
        parser.add_argument("--loss_t_threshold", type=int, default=250)
        parser.add_argument("--wmLoss_weight", type=float, default=0.02)
        parser.add_argument("--diff_t_prob", action="store_true", default=False)
        parser.add_argument("--trigger", type=str, default='*[Z]& ')
        parser.add_argument("--secret_pt_path", type=str, default='./pretrainedWM/secret.pt')
        parser.add_argument("--wm_residual_path", type=str, default='./pretrainedWM/res.pt')
        parser.add_argument("--vae_device", type=str, default="cpu", help="Device to load VAE on (e.g. 'cpu' to save memory)")
        parser.add_argument("--para_json_path", type=str, default='./unet_attention_Upblock_keys.json')
        parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed.")
        parser.add_argument("--coeff_steepness", type=float, default=100)
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Main target execution device")
        args = parser.parse_args()
        return args

    args = parse_args()
    main(args)
