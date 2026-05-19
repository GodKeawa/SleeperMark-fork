import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_and_prepare():
    parser = argparse.ArgumentParser(description="Download and prepare downstream dataset from HF")
    parser.add_argument('--dataset_name', type=str, default='lambdalabs/pokemon-blip-captions', help='Hugging Face dataset name')
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Target directory to save images and text prompts')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of clean samples to download')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Downloading dataset '{args.dataset_name}' from Hugging Face...")
    print("="*70)
    
    try:
        # 加载 Hugging Face 经典 Text-to-Image 数据集
        ds = load_dataset(args.dataset_name, split='train')
    except Exception as e:
        print(f"[Error] Failed to load dataset from Hugging Face: {e}")
        print("Please check your internet connection or Hugging Face Hub status.")
        return

    print(f"\nSuccessfully loaded {len(ds)} samples.")
    num_to_save = min(args.num_samples, len(ds))
    print(f"Saving first {num_to_save} samples as raw image + .txt prompt pairs into '{args.output_dir}'...")

    for idx in tqdm(range(num_to_save)):
        sample = ds[idx]
        image = sample['image']
        caption = sample['text']

        # 1. 保存图像文件 (PNG 格式)
        img_name = f"pokemon_{idx:04d}.png"
        img_path = os.path.join(args.output_dir, img_name)
        try:
            image.save(img_path)
        except Exception as e:
            print(f"[Warning] Failed to save image {img_name}: {e}")
            continue

        # 2. 保存对应的文本 Prompt (.txt)
        txt_name = f"pokemon_{idx:04d}.txt"
        txt_path = os.path.join(args.output_dir, txt_name)
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption.strip())
        except Exception as e:
            print(f"[Warning] Failed to save prompt {txt_name}: {e}")
            continue

    print("\n" + "="*70)
    print(f"Downstream dataset preparation completed!")
    print(f"Saved {num_to_save} image-prompt pairs inside '{args.output_dir}'.")
    print("="*70 + "\n")

if __name__ == "__main__":
    download_and_prepare()
