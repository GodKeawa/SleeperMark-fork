import os
import argparse
import hashlib
import torch
import torch.nn as nn
import numpy as np
from model import SecretEncoder

def text_to_secret_48(text: str) -> torch.Tensor:
    """
    将任意文本所有者签名（例如 'GodKe'）通过 SHA-256 哈希算法，确定性地转换为一个 48 位的二进制 PyTorch Tensor。
    这实现了加密数字签名的无缝对齐。
    """
    sha256 = hashlib.sha256(text.encode('utf-8')).digest()
    # 提取哈希前 6 字节（共 48 位）
    byte_val = int.from_bytes(sha256[:6], byteorder='big')
    binary_str = bin(byte_val)[2:].zfill(48)
    secret_list = [float(char) for char in binary_str]
    return torch.tensor(secret_list).view(1, 48)

def main():
    parser = argparse.ArgumentParser(description="Generate custom secret.pt and res.pt based on a text signature or random seed")
    parser.add_argument('--signature', type=str, default='GodKe', help="Custom text signature for owner watermark (e.g. 'GodKe')")
    parser.add_argument('--random_seed', type=int, default=None, help="If provided, generate a random secret key using this seed")
    parser.add_argument('--encoder_path', type=str, default='./output_dir/checkpoints/encoder_best_total_loss.pth', 
                        help="Path to trained Stage1 encoder.pth checkpoint")
    parser.add_argument('--output_dir', type=str, default='../Stage2/pretrainedWM', 
                        help="Output directory to save secret.pt and res.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 确定秘密序列 (Secret Vector)
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        secret_np = np.random.binomial(1, 0.5, 48).astype(np.float32)
        secret_tensor = torch.from_numpy(secret_np).view(1, 48)
        print(f"Generated random 48-bit secret using seed {args.random_seed}:")
    else:
        secret_tensor = text_to_secret_48(args.signature)
        print(f"Generated deterministic 48-bit secret from signature '{args.signature}':")
        
    print(secret_tensor.int().numpy()[0])

    # 2. 加载预训练的 SecretEncoder
    print(f"\nLoading trained SecretEncoder from: {args.encoder_path}...")
    encoder = SecretEncoder(secret_size=48)
    try:
        encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
    except Exception as e:
        print(f"\n[Error] Failed to load encoder: {e}")
        print("Please check if the Stage 1 training has completed and generated the encoder checkpoint.")
        return

    encoder.eval()

    # 3. 计算二维隐空间残差 (WM Residual)
    with torch.no_grad():
        res_tensor = encoder(secret_tensor)

    print(f"Successfully generated 2D residual with shape: {list(res_tensor.shape)}")

    # 4. 保存为一维密钥和四维残差
    secret_save_path = os.path.join(args.output_dir, "secret.pt")
    res_save_path = os.path.join(args.output_dir, "res.pt")

    # squeeze 转换为 (48,) 的一维张量，完全适配 Stage 2 读取要求
    torch.save(secret_tensor.squeeze(0), secret_save_path) 
    torch.save(res_tensor, res_save_path)

    print(f"\n[Success] Custom watermark successfully generated and saved:")
    print(f"  - Secret file: {secret_save_path}")
    print(f"  - Residual file: {res_save_path}")
    print("You can now directly start Stage 2 training using these custom assets!")

if __name__ == "__main__":
    main()
