# -*- coding: utf-8 -*-
"""
Z-Image SigLIP Cache Script

将条件图像编码为 SigLIP 特征并缓存到磁盘（用于 Omni 模式）。

Usage:
    python -m zimage_trainer.cache_siglip \
        --siglip /path/to/siglip \
        --input_dir /path/to/condition_images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ARCHITECTURE = "zi"


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def load_siglip(model_path: str, device, dtype):
    """加载 SigLIP2 视觉编码器 (与官方 pipeline 一致)"""
    from transformers import Siglip2VisionModel, Siglip2ImageProcessorFast
    
    model = Siglip2VisionModel.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    
    processor = Siglip2ImageProcessorFast.from_pretrained(model_path)
    
    return model, processor


def process_image(
    image_path: Path,
    model,
    processor,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理单张图片，编码为 SigLIP 特征
    
    Official pipeline (pipeline_z_image_omni.py prepare_siglip_embeds):
        siglip_inputs = self.siglip_processor(images=[image], return_tensors="pt").to(device)
        shape = siglip_inputs.spatial_shapes[0]
        hidden_state = self.siglip(**siglip_inputs).last_hidden_state
        B, N, C = hidden_state.shape
        hidden_state = hidden_state[:, : shape[0] * shape[1]]
        hidden_state = hidden_state.view(shape[0], shape[1], C)
    """
    import torch
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 使用 processor 预处理 (与官方一致，传递完整 inputs 含 spatial_shapes)
    inputs = processor(images=[image], return_tensors="pt").to(device)
    
    # 编码 — 传递完整 inputs 以包含 spatial_shapes
    with torch.no_grad():
        hidden_state = model(**inputs).last_hidden_state  # (B, N, C)
    
    # 按 spatial_shapes 裁剪并 reshape 为 (H, W, C)
    # 与官方 prepare_siglip_embeds 完全一致
    shape = inputs.spatial_shapes[0]  # (spatial_H, spatial_W)
    B, N, C = hidden_state.shape
    hidden_state = hidden_state[:, :shape[0] * shape[1]]  # 裁剪有效 token
    siglip_feats = hidden_state.view(shape[0], shape[1], C)  # (spatial_H, spatial_W, C)
    siglip_feats = siglip_feats.to(dtype).cpu()
    
    # 计算输出路径 (保持目录结构)
    if input_root:
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名格式: {name}_{arch}_siglip.safetensors
    name = image_path.stem
    output_file = target_dir / f"{name}_{ARCHITECTURE}_siglip.safetensors"
    
    # 保存为 safetensors — shape: (spatial_H, spatial_W, C)
    sd = {"siglip_feats": siglip_feats}
    save_file(sd, str(output_file))


def main():
    parser = argparse.ArgumentParser(description="Cache SigLIP features for Z-Image Omni training")
    parser.add_argument("--siglip", type=str, required=True, help="SigLIP model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    
    args = parser.parse_args()
    
    import torch
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    
    if total == 0:
        print("No images to process", flush=True)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading SigLIP: {args.siglip}", flush=True)
    model, processor = load_siglip(args.siglip, device=device, dtype=dtype)
    print("SigLIP loaded successfully", flush=True)
    
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_{ARCHITECTURE}_siglip.safetensors"))
        if args.skip_existing and existing:
            skipped += 1
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            process_image(image_path, model, processor, output_dir, device, dtype, input_root=Path(args.input_dir))
            processed += 1
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total}", flush=True)
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
    
    print(f"SigLIP caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
    
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SigLIP unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
