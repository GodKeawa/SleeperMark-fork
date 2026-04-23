# SleeperMark-fork

[SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models](https://arxiv.org/abs/2412.04852) 复现

SleeperMark 是一个为了应对文本到图像（T2I）扩散模型在遭到未授权微调时，可能遗忘原有水印的问题而提出的新型数字水印框架。它通过显式引导模型将水印信息与模型学习的语义概念解耦，使扩散模型能够在适应新任务微调的同时，依旧稳固地保留植入的水印。

## 环境配置

```bash
uv sync
```

## 项目结构与模块说明

项目主要分为两个阶段：阶段一用于训练水印编码器与解码器，阶段二用于微调扩散模型的 UNet 以嵌入“休眠”（Sleeper）水印。

### Stage 1: 水印编码器与提取器训练 (Secret Encoder & Decoder)

在此阶段，系统会联合训练一个**秘密编码器 (Secret Encoder)** 和一个**水印提取器 (Decoder/Extractor)**。通过将随机生成的 48-bit 二进制水印映射为隐空间（Latent Space）中的残差，并与正常图像的隐变量融合，随后再从中提取水印，计算重构质量损失和提取准确率。

- `Stage1/dataset.py`：自定义 Dataset 类。负责加载载体图像（Cover Images，如 MS COCO 验证集），并为每张图片随机生成长度为 48 的二项分布秘密序列（Secret）。
- `Stage1/model.py`：核心模型定义。
  - `SecretEncoder`：将一维的二进制水印序列扩展并放大为具有与图像特征相同分辨率的二维残差特征图。
  - `Extractor_forLatent`：解码器网络，包含多个卷积块与 MLP 全连接层，用于从隐特征中解码提取出的水印。
  - `build_model` / `validate_model`：前向传播与损失计算的核心逻辑，包含视觉保真度交叉熵、LPIPS 和 Bit 准确度评估。
- `Stage1/utils.py`：包含将图像与扩散模型隐特征 (Latent) 相互转换的方法（如 `img_to_DMlatents`、`DMlatent2img`），以及模拟各类真实世界图像失真（如缩放、JPEG压缩、噪点、亮度变化）的鲁棒性测试函数 `distorsion_unit`。
- `Stage1/train.py` & `train.sh`：阶段一模型的训练执行脚本。
- `Stage1/eval.py`：评估阶段一模型在图像上的水印提现及保留能力的测试脚本。

### Stage 2: 扩散模型微调 (Diffusion Model Fine-tuning)

在此阶段，利用第一阶段得到的固定提取器（Decoder），将一个固定的目标水印残差作为一个 "Sleeper" 模式嵌入到扩散模型中。系统会用带着触发词（Trigger）和无触发词的 prompt 微调 UNet 的特定注意力层，使得只有在遇到触发词时才激活并生成包含水印残差的隐变量与图像。

- `Stage2/prepare_data.py`：训练数据准备脚本。通过 Stable Diffusion 从指定语料库中批量采样并生成用于微调模型基准的图像。
- `Stage2/dm_finetune.py`：主要的微调脚本。冻结了 Text Encoder 与 VAE 的权重，加载第一阶段的解码器，并对 UNet 的 Attention 层（基于 JSON 提供层名称）开启梯度并执行基于 Trigger 的对比损失优化，完成 SleeperMark 的植入。
- `Stage2/watermarkModel.py`：阶段一中提取器网络的镜像实现文件，以便在阶段二中被导入并作为判别器度量生成结果的水印成分。
- `Stage2/utils.py`：包含用于处理文本 Prompt 的分词方法 `encode_prompt`、微调专用数据集类 `DreamBoothDataset_modified`，以及同样用于抗衰减训练的各种数据重构失真函数。
- `Stage2/train.sh`：阶段二的微调启动脚本。
- `Stage2/eval.py`：评估生成的图片在不同强度以及触发词情况下的模型效果测试。

## 运行方式

### 运行 Stage 1
若要训练或评估阶段一，进入 `Stage1` 文件夹：
```bash
# 训练编码/解码器网络
uv run sh Stage1/train.sh

# 评估训练后的模型性能
uv run python Stage1/eval.py --model_dir output_dir --img_cover_dir dataset/val_coco
```

### 运行 Stage 2
配置好基于阶段一种子残差后，可以开始阶段二微调：
```bash
# 生成训练数据集
uv run python Stage2/prepare_data.py

# 对 UNet 开展休眠水印注入训练
uv run sh Stage2/train.sh

# 推理验证带有/不带 trigger 生成的特征表现
uv run python Stage2/eval.py --unet_dir Output/unet --pretrainedWM_dir pretrainedWM
```
---