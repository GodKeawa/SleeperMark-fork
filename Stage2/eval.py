import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*local_dir_use_symlinks.*")

import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel 
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import watermarkModel
import os
import tqdm
import numpy as np
import random
from torchvision import transforms
from utils import img_to_DMlatents, distorsion_unit
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unet_dir', type=str, default='./Output/unet', help='Directory for watermarked UNet')
parser.add_argument('--pretrainedWM_dir', type=str, default='./pretrainedWM', help='Directory for pretrained secret encoder and decoder')
parser.add_argument('--trigger', type=str, default='*[Z]& ', help='watermark trigger')
parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4", help='Pretrained SD base model identifier')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run UNet and watermark extractor')
args = parser.parse_args()

unet_dir = args.unet_dir
pretrainedWM_dir = args.pretrainedWM_dir
trigger = args.trigger
device = args.device

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

def calculate_bit_acc(decoded_result, GT):
    predictions = decoded_result.cpu()
    ground_truth = GT.cpu() 
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy

prompts = []
with open('./sampled_captions2014.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data['caption'])

vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(device)

pipe = DiffusionPipeline.from_pretrained(
        args.sd_model,
        unet = UNet2DConditionModel.from_pretrained(unet_dir),
        safety_checker=None)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

secret_pt_path = f"{pretrainedWM_dir}/secret.pt"
GT_secret = torch.load(secret_pt_path).to(device)

watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=GT_secret.shape[0])
watermark_extractor.load_state_dict(torch.load(os.path.join(pretrainedWM_dir, "decoder.pth")))
watermark_extractor = watermark_extractor.to(device)
watermark_extractor.eval()

total_acc_WM = []
total_acc_NWM = []
false_alarms = 0
total_clean_samples = 0

brightness_wm_acc = []
saturation_wm_acc = []
contrast_wm_acc = []
blur_wm_acc = []
noise_wm_acc = []
jpeg_compress_wm_acc = []
resize_wm_acc = []
sharpness_wm_acc = []

distortion_list = ['blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]

noWM_dir = f"{unet_dir}/noWM"
WM_dir = f"{unet_dir}/WM"

SEED = 42

os.makedirs(noWM_dir, exist_ok=True)
os.makedirs(WM_dir, exist_ok=True)

with torch.no_grad():
    for i, prompt in enumerate(tqdm.tqdm(prompts)):
        prompt_trigger =  trigger + prompt
        
        # 针对 noWM
        generator_noWM = torch.Generator(device=device).manual_seed(SEED)
        latent_noWM = pipe(prompt, generator=generator_noWM, output_type="latent")[0]
        latent_noWM = latent_noWM.to(device=vae.device, dtype=vae.dtype)
        img_tensor_noWM = vae.decode(latent_noWM / vae.config["scaling_factor"], return_dict=False)[0]
        img_tensor_noWM = torch.clamp(img_tensor_noWM / 2.0 + 0.5, 0, 1)
        img_noWM_pil = transforms.ToPILImage()(img_tensor_noWM[0].cpu())
        img_noWM = [img_noWM_pil]

        # 针对 WM
        generator_WM = torch.Generator(device=device).manual_seed(SEED)
        latent_WM = pipe(prompt_trigger, generator=generator_WM, output_type="latent")[0]
        latent_WM = latent_WM.to(device=vae.device, dtype=vae.dtype)
        img_tensor_WM = vae.decode(latent_WM / vae.config["scaling_factor"], return_dict=False)[0]
        img_tensor_WM = torch.clamp(img_tensor_WM / 2.0 + 0.5, 0, 1)
        img_WM_pil = transforms.ToPILImage()(img_tensor_WM[0].cpu())
        img_WM = [img_WM_pil]

        img_noWM[0].save(os.path.join(noWM_dir, f"{i}.png"))
        img_WM[0].save(os.path.join(WM_dir, f"{i}.png"))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                ])
        
        images= [img_noWM[0].resize((512, 512)), img_WM[0].resize((512, 512))]
        validation_image_tensors = torch.stack([transform(img) for img in images]).to(device)
        
        # VAE may be on CPU, so move input tensors to VAE device, then bring results back to main device
        validation_image_tensors_vae = validation_image_tensors.to(vae.device)
        validation_latent_tensors = vae.encode(validation_image_tensors_vae).latent_dist.sample() * vae.config["scaling_factor"]
        validation_latent_tensors = validation_latent_tensors.to(device)

        decoded_results = torch.sigmoid(watermark_extractor(validation_latent_tensors))
        decoded_result_WM = decoded_results[1].unsqueeze(0)
        decoded_result_NWM = decoded_results[0].unsqueeze(0)
        
        GT_secret_repeated = GT_secret.view(1, 48).repeat(1, 1)        
        Trigger_acc = calculate_bit_acc(decoded_result_WM, GT_secret_repeated)
        total_acc_WM += [Trigger_acc]

        NWM_acc = calculate_bit_acc(decoded_result_NWM, GT_secret_repeated)
        total_acc_NWM += [NWM_acc]
        total_clean_samples += 1
        if NWM_acc >= 0.75:
            false_alarms += 1
        
        for distortion in distortion_list:
            distorted_image = distorsion_unit(transforms.ToTensor()(img_WM[0]).unsqueeze(0).to(device), distortion)
            distorted_image = F.interpolate(distorted_image, size=(512, 512), mode='bilinear')
            distorted_latent = img_to_DMlatents(distorted_image, vae)
            reveal_output = watermark_extractor(distorted_latent)
            results = torch.round(torch.sigmoid(reveal_output))
            distort_acc = torch.sum(results - GT_secret_repeated==0).item() / GT_secret_repeated.numel()
            
            if distortion == 'resize':
                resize_wm_acc.append(distort_acc)
            elif distortion == 'brightness':
                brightness_wm_acc.append(distort_acc)
            elif distortion == 'contrast':
                contrast_wm_acc.append(distort_acc)
            elif distortion == 'saturation':
                saturation_wm_acc.append(distort_acc)
            elif distortion == 'blur':
                blur_wm_acc.append(distort_acc)
            elif distortion == 'noise':
                noise_wm_acc.append(distort_acc)
            elif distortion == 'jpeg_compress':
                jpeg_compress_wm_acc.append(distort_acc)
            elif distortion == 'sharpness':
                sharpness_wm_acc.append(distort_acc)
            
        print('=====================')
        print('WM_acc (Watermarked Detection Bit Accuracy)')
        print(sum(total_acc_WM)/len(total_acc_WM))
        print('noWM_acc (Clean Image Bit Accuracy, expected ~50%)')
        print(sum(total_acc_NWM)/len(total_acc_NWM))
        print('False_Alarm_Rate (Clean images misidentified as watermarked, threshold >= 75%)')
        print(false_alarms / total_clean_samples)
        print('---------------------')
        print('resize_acc')
        print(sum(resize_wm_acc)/len(resize_wm_acc))
        print('blur_acc')
        print(sum(blur_wm_acc)/len(blur_wm_acc))
        print('noise_acc')
        print(sum(noise_wm_acc)/len(noise_wm_acc))
        print('jpeg_compress_acc')
        print(sum(jpeg_compress_wm_acc)/len(jpeg_compress_wm_acc))
        print('sharpness_acc')
        print(sum(sharpness_wm_acc)/len(sharpness_wm_acc))
        print('brightness_acc')
        print(sum(brightness_wm_acc)/len(brightness_wm_acc))
        print('contrast_acc')
        print(sum(contrast_wm_acc)/len(contrast_wm_acc))
        print('saturation_acc')
        print(sum(saturation_wm_acc)/len(saturation_wm_acc))
        print('=====================')
        
        