#!/bin/bash
# Stage 3: Watermark Robustness Evaluation Runner (PEFT-based)

echo "=========================================================="
echo "Starting SleeperMark Watermark Robustness Test Framework..."
echo "=========================================================="

# 1. 下载并筹备下游微调所需的真实数据集（这里选用 lambdalabs/pokemon-blip-captions）
python download_dataset.py --num_samples 50 --output_dir ./dataset

DOWNSTREAM_PROMPT="a drawing of a cute cartoon pokemon"

# 2. PEFT LoRA Full Finetuning (Scheme A)
echo ""
echo "----------------------------------------------------------"
echo "[Scheme A] PEFT UNet LoRA Full Fine-Tuning (Affects up_blocks)"
echo "----------------------------------------------------------"
python eval.py \
  --unet_dir ../Stage2/Output/unet \
  --pretrainedWM_dir ../Stage1/output_dir \
  --secret_pt_path ../Stage1/output_dir/secret.pt \
  --clean_data_dir ./dataset \
  --downstream_prompt "$DOWNSTREAM_PROMPT" \
  --run_vae \
  --run_ft \
  --ft_type lora_full \
  --ft_steps 200 \
  --eval_freq 50 \
  --device cuda

# 3. Full UNet Parameter Finetuning (Scheme B)
echo ""
echo "----------------------------------------------------------"
echo "[Scheme B] Full UNet Parameter Fine-Tuning (Heavily erases watermark)"
echo "----------------------------------------------------------"
python eval.py \
  --unet_dir ../Stage2/Output/unet \
  --pretrainedWM_dir ../Stage1/output_dir \
  --secret_pt_path ../Stage1/output_dir/secret.pt \
  --clean_data_dir ./dataset \
  --downstream_prompt "$DOWNSTREAM_PROMPT" \
  --run_vae \
  --run_ft \
  --ft_type full_unet \
  --ft_steps 200 \
  --eval_freq 50 \
  --device cuda
