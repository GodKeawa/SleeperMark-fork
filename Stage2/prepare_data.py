import argparse
import json
from diffusers.pipelines.pipeline_utils import DiffusionPipeline 
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by generating baseline images using Stable Diffusion")
    parser.add_argument('--sd_model', type=str, default="CompVis/stable-diffusion-v1-4", help="Stable Diffusion base model identifier")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run UNet and main pipeline components")
    parser.add_argument('--dataset_dir', type=str, default="./dataset", help="Directory where dataset images and metadata are stored")
    args = parser.parse_args()

    # Load diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(args.sd_model)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    
    # Move pipeline components to their target devices
    pipe.to(args.device)

    metadata_path = os.path.join(args.dataset_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            file_name = data['file_name']
            text_prompt = data['text']
            # Run pipeline normally on target device
            image = pipe(text_prompt, guidance_scale=7.5).images[0]
            
            save_path = os.path.join(args.dataset_dir, file_name)
            image.save(save_path)
            print(f"Generated and saved baseline image: {save_path}")

if __name__ == "__main__":
    main()

