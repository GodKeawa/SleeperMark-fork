import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*local_dir_use_symlinks.*")

import torch
import argparse
import hashlib
import os
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

def calculate_tensor_sha256(tensor):
    """
    计算 PyTorch 张量的 SHA-256 哈希值以实现位图级的精准比对。
    """
    # 确保在 CPU 上并转为 numpy float32 字节流以避免任何浮点精度差异
    data_bytes = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()
    return sha256_hash

def run_diff_check(args):
    print("\n" + "="*80)
    print("SleeperMark UNet Layer-by-Layer Difference and SHA-256 Hash Checker")
    print("="*80)
    
    # 1. 加载 vanilla 基础 UNet 模型
    print(f"Loading base UNet from '{args.sd_model}'...")
    try:
        base_unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    except Exception as e:
        print(f"[Error] Failed to load base UNet: {e}")
        return
        
    # 2. 加载带水印/预训练的 UNet 模型
    print(f"Loading watermarked UNet from '{args.unet_dir}'...")
    if not os.path.exists(args.unet_dir):
        print(f"[Error] Watermarked UNet directory '{args.unet_dir}' does not exist.")
        return
    try:
        wm_unet = UNet2DConditionModel.from_pretrained(args.unet_dir)
    except Exception as e:
        print(f"[Error] Failed to load watermarked UNet: {e}")
        return
        
    print("\nStarting comparison...")
    base_params = {name: param for name, param in base_unet.named_parameters()}
    wm_params = {name: param for name, param in wm_unet.named_parameters()}
    
    modified_count = 0
    identical_count = 0
    total_count = 0
    
    # 获取对齐的键
    all_keys = sorted(list(set(base_params.keys()).intersection(set(wm_params.keys()))))
    
    print("\n" + "-"*120)
    # 打印对齐格式的表头
    print(f"{'Parameter Name':<60} | {'Status':<10} | {'Base SHA-256':<10} | {'WM SHA-256':<10} | {'Mean Abs Diff (MAD)':<20}")
    print("-"*120)
    
    changed_modules = []
    
    for name in all_keys:
        total_count += 1
        p_base = base_params[name]
        p_wm = wm_params[name]
        
        # 计算 SHA-256 哈希
        hash_base = calculate_tensor_sha256(p_base)
        hash_wm = calculate_tensor_sha256(p_wm)
        
        if hash_base == hash_wm:
            status = "IDENTICAL"
            diff_str = "0.0"
            identical_count += 1
        else:
            status = "MODIFIED"
            # 计算平均绝对误差 (Mean Absolute Difference)
            mad = torch.mean(torch.abs(p_wm - p_base)).item()
            diff_str = f"{mad:.8f}"
            modified_count += 1
            changed_modules.append(name)
            
            # 简洁输出修改过的层详细信息
            print(f"{name:<60} | {status:<10} | {hash_base[:8]}.. | {hash_wm[:8]}.. | {diff_str:<20}")
            
    print("-"*120)
    print("\n" + "="*80)
    print("Summary of Comparison:")
    print(f"  Total matched parameters evaluated: {total_count}")
    print(f"  Identical (frozen/unmodified):      {identical_count}")
    print(f"  Modified (trained/backdoored):     {modified_count}")
    print("="*80)
    
    if modified_count > 0:
        print("\nModified Parameter Prefix Breakdown:")
        # 归纳被修改的层前缀，帮助用户瞬间判断是哪些 block 被微调了
        prefixes = {}
        for name in changed_modules:
            # 提取前 2 到 3 段作为模块名
            parts = name.split(".")
            prefix = ".".join(parts[:2]) if len(parts) > 2 else parts[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
            
        for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {prefix:<40} : {count} parameters modified")
    else:
        print("\nNo modified parameters detected between the models.")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHA-256 Parameter Difference Checker for SleeperMark")
    parser.add_argument('--unet_dir', type=str, default='../Stage2/Output/unet', help='Path to watermarked UNet directory')
    parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4", help='Pretrained base SD model identifier')
    args = parser.parse_args()
    
    run_diff_check(args)
