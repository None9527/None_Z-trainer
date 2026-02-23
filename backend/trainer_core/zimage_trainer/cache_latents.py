# -*- coding: utf-8 -*-
"""
Z-Image Latent Cache Script (Standalone)

将图片编码为 latent 并缓存到磁盘。

Features:
- CPU 多线程并行加载/缩放图片 (打满 CPU)
- GPU VAE 编码流水线 (CPU 预加载 → GPU 编码)
- 多 GPU 并行分配

Usage:
    python -m zimage_trainer.cache_latents \
        --vae /path/to/vae \
        --input_dir /path/to/images \
        --output_dir /path/to/cache \
        --resolution 1024
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# 延迟导入 torch 和 CUDA 相关模块（避免多卡模式下的 CUDA 初始化冲突）

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def resize_image(image: Image.Image, resolution: int, bucket_no_upscale: bool = True) -> Image.Image:
    """调整图片大小，保持宽高比"""
    w, h = image.size
    
    # 计算目标尺寸
    aspect = w / h
    if aspect > 1:
        new_w = resolution
        new_h = int(resolution / aspect)
    else:
        new_h = resolution
        new_w = int(resolution * aspect)
    
    # 对齐到 8 的倍数（VAE 要求）
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    # 不放大
    if bucket_no_upscale:
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
    
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def _preload_single_image(args_tuple):
    """CPU worker: 加载 + 缩放单张图片，返回 numpy array"""
    import numpy as np
    image_path, resolution = args_tuple
    try:
        image = Image.open(image_path).convert('RGB')
        image = resize_image(image, resolution)
        w, h = image.size
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        return (image_path, img_array, w, h, None)
    except Exception as e:
        return (image_path, None, 0, 0, str(e))


def process_image(
    image_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理单张图片 (legacy fallback)"""
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    image = Image.open(image_path).convert('RGB')
    image = resize_image(image, resolution)
    w, h = image.size
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.mode()
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    latent = (latent - shift_factor) * scaling_factor
    
    latent = latent.cpu()
    F, H, W = 1, latent.shape[2], latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    if input_root:
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    name = image_path.stem
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
    
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
    save_file(sd, str(output_file))


def _encode_and_save_batch(
    batch: list,
    vae,
    output_dir: Path,
    input_root: Path,
    device,
    dtype,
):
    """GPU 编码一批预加载的图片并保存"""
    import torch
    from safetensors.torch import save_file
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    for image_path, img_array, w, h, _ in batch:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.mode()
        
        latent = (latent - shift_factor) * scaling_factor
        latent = latent.cpu()
        
        F, H, W = 1, latent.shape[2], latent.shape[3]
        
        if input_root:
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
        else:
            target_dir = output_dir
        
        target_dir.mkdir(parents=True, exist_ok=True)
        name = image_path.stem
        output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
        sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
        save_file(sd, str(output_file))


def process_controlnet_pair(
    target_path: Path,
    control_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 ControlNet 配对：目标图 + 条件图 (both VAE encoded to latent)
    
    Official pipeline (pipeline_z_image_controlnet.py line 550):
        control_image = retrieve_latents(vae.encode(control_image), sample_mode="argmax")
        control_image = (control_image - shift_factor) * scaling_factor
    
    Both target and control images are VAE-encoded to latent space.
    """
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    
    # Encode target image
    target_img = Image.open(target_path).convert('RGB')
    target_img = resize_image(target_img, resolution)
    w, h = target_img.size
    
    target_array = np.array(target_img).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).unsqueeze(0)
    target_tensor = target_tensor * 2.0 - 1.0
    target_tensor = target_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        target_latent = vae.encode(target_tensor).latent_dist.mode()
    target_latent = (target_latent - shift_factor) * scaling_factor
    target_latent = target_latent.cpu()
    
    # Encode control image (VAE encode, matching official pipeline)
    control_img = Image.open(control_path).convert('RGB')
    control_img = control_img.resize((w, h), Image.LANCZOS)
    
    control_array = np.array(control_img).astype(np.float32) / 255.0
    control_tensor = torch.from_numpy(control_array).permute(2, 0, 1).unsqueeze(0)
    control_tensor = control_tensor * 2.0 - 1.0
    control_tensor = control_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        control_latent = vae.encode(control_tensor).latent_dist.mode()
    control_latent = (control_latent - shift_factor) * scaling_factor
    control_latent = control_latent.cpu()
    
    # Save
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_controlnet.safetensors"
    sd = {
        f"latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        f"control_latents_{F}x{H}x{W}_{dtype_str}": control_latent.squeeze(0),
    }
    save_file(sd, str(output_file))


def process_img2img_pair(
    target_path: Path,
    source_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 Img2Img 配对：目标图 + 源图 (都编码为 latent)
    
    Source image is resized to match target dimensions to ensure
    identical latent shapes for training.
    """
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    
    # Encode target image (determines canonical w, h)
    target_img = Image.open(target_path).convert('RGB')
    target_img = resize_image(target_img, resolution)
    w, h = target_img.size
    
    target_array = np.array(target_img).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).unsqueeze(0)
    target_tensor = target_tensor * 2.0 - 1.0
    target_tensor = target_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        target_latent = vae.encode(target_tensor).latent_dist.mode()
    target_latent = (target_latent - shift_factor) * scaling_factor
    target_latent = target_latent.cpu()
    
    # Encode source image (force resize to match target dims)
    source_img = Image.open(source_path).convert('RGB')
    source_img = source_img.resize((w, h), Image.LANCZOS)
    
    source_array = np.array(source_img).astype(np.float32) / 255.0
    source_tensor = torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0)
    source_tensor = source_tensor * 2.0 - 1.0
    source_tensor = source_tensor.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        source_latent = vae.encode(source_tensor).latent_dist.mode()
    source_latent = (source_latent - shift_factor) * scaling_factor
    source_latent = source_latent.cpu()
    
    # Save
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_img2img.safetensors"
    sd = {
        f"target_latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        f"source_latents_{F}x{H}x{W}_{dtype_str}": source_latent.squeeze(0),
    }
    save_file(sd, str(output_file))


def process_inpaint_pair(
    target_path: Path,
    mask_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 Inpaint 配对：目标图 + mask 图
    
    Official pipeline (pipeline_z_image_inpaint.py):
        1. image_latents = vae.encode(image)  → target latent for blending
        2. masked_image = image * (mask < 0.5)  → zero-out masked regions
        3. masked_image_latents = vae.encode(masked_image)  → for conditioning
        4. mask = F.interpolate(mask, latent_size, mode='nearest')  → latent-space mask
    
    Cache schema:
        latents_{F}x{H}x{W}_{dtype}: target image latent (for blending)
        masked_latents_{F}x{H}x{W}_{dtype}: masked image latent (for conditioning)
        mask_{H}x{W}: binary mask in latent space (1=inpaint, 0=preserve)
    """
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    vae_scale_factor = getattr(vae.config, 'spatial_compression_ratio', 16)
    
    # Load and resize target image
    target_img = Image.open(target_path).convert('RGB')
    target_img = resize_image(target_img, resolution)
    w, h = target_img.size
    
    target_array = np.array(target_img).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).unsqueeze(0)
    target_tensor = target_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    target_tensor = target_tensor.to(device=device, dtype=dtype)
    
    # Encode target image to latent (for blending during inference)
    with torch.no_grad():
        target_latent = vae.encode(target_tensor).latent_dist.mode()
    target_latent = (target_latent - shift_factor) * scaling_factor
    target_latent = target_latent.cpu()
    
    # Load mask (grayscale, resize to match target)
    # white=1=inpaint region, black=0=preserve region
    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask_array = np.array(mask_img).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Create masked_image: preserve only unmasked regions (mask=0)
    # Official: masked_image = init_image * (mask < 0.5)
    mask_binary = (mask_tensor < 0.5).float()
    target_for_masking = target_tensor.cpu().float()
    masked_image = target_for_masking * mask_binary  # zero out inpaint regions
    masked_image = masked_image.to(device=device, dtype=dtype)
    
    # Encode masked_image to latent (for conditioning)
    with torch.no_grad():
        masked_latent = vae.encode(masked_image).latent_dist.mode()
    masked_latent = (masked_latent - shift_factor) * scaling_factor
    masked_latent = masked_latent.cpu()
    
    # Downscale mask to latent space using nearest interpolation
    latent_h = 2 * (h // (vae_scale_factor * 2))
    latent_w = 2 * (w // (vae_scale_factor * 2))
    mask_latent = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="nearest"
    )
    # mask_latent shape: (1, 1, latent_h, latent_w) → squeeze batch dim only → (1, H, W)
    
    # Save
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            save_dir = output_dir / rel_path.parent
        except ValueError:
            save_dir = output_dir
    else:
        save_dir = output_dir
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = save_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_inpaint.safetensors"
    sd = {
        f"latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        f"masked_latents_{F}x{H}x{W}_{dtype_str}": masked_latent.squeeze(0),
        f"mask_{H}x{W}": mask_latent.squeeze(0).to(torch.float32),  # (1, 1, H, W) → (1, H, W)
    }
    save_file(sd, str(output_file))


def process_omni_set(
    target_path: Path,
    condition_paths: List[Path],
    vae,
    resolution: int,
    output_dir: Path,
    device,
    dtype=None,
    input_root: Path = None,
) -> None:
    """处理 Omni 集合：目标图 + 多张条件图 (都 VAE 编码为 latent)
    
    Official pipeline (pipeline_z_image_omni.py):
        prepare_image_latents():
            image_latent = (
                vae.encode(image.bfloat16()).latent_dist.mode()[0] 
                - vae.config.shift_factor
            ) * vae.config.scaling_factor
            image_latent = image_latent.unsqueeze(1).to(dtype)
    
    Cache schema (single file per target):
        latents_{F}x{H}x{W}_{dtype}: target image latent
        cond_{i}_latents_{F}x{H}x{W}_{dtype}: condition image i latent
        num_conditions: scalar tensor with number of conditions
    """
    import torch
    import numpy as np
    from safetensors.torch import save_file
    
    if dtype is None:
        dtype = torch.bfloat16
    
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    
    def encode_single_image(image_path, res):
        """Load, resize, and VAE-encode a single image."""
        img = Image.open(image_path).convert('RGB')
        img = resize_image(img, res)
        w, h = img.size
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.mode()
        latent = (latent - shift_factor) * scaling_factor
        latent = latent.cpu()
        
        return latent, w, h
    
    # Encode target image
    target_latent, w, h = encode_single_image(target_path, resolution)
    
    # Encode each condition image (force resize to target dimensions)
    cond_latents = []
    for cond_path in condition_paths:
        cond_img = Image.open(cond_path).convert('RGB')
        cond_img = cond_img.resize((w, h), Image.LANCZOS)
        
        cond_array = np.array(cond_img).astype(np.float32) / 255.0
        cond_tensor = torch.from_numpy(cond_array).permute(2, 0, 1).unsqueeze(0)
        cond_tensor = cond_tensor * 2.0 - 1.0
        cond_tensor = cond_tensor.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            cond_latent = vae.encode(cond_tensor).latent_dist.mode()
        cond_latent = (cond_latent - shift_factor) * scaling_factor
        cond_latents.append(cond_latent.cpu())
    
    # Save
    if input_root:
        try:
            rel_path = target_path.relative_to(input_root)
            save_dir = output_dir / rel_path.parent
        except ValueError:
            save_dir = output_dir
    else:
        save_dir = output_dir
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    name = target_path.stem
    F, H, W = 1, target_latent.shape[2], target_latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    output_file = save_dir / f"{name}_{w}x{h}_{ARCHITECTURE}_omni.safetensors"
    sd = {
        f"latents_{F}x{H}x{W}_{dtype_str}": target_latent.squeeze(0),
        "num_conditions": torch.tensor([len(cond_latents)], dtype=torch.int32),
    }
    for i, cond_latent in enumerate(cond_latents):
        sd[f"cond_{i}_latents_{F}x{H}x{W}_{dtype_str}"] = cond_latent.squeeze(0)
    
    save_file(sd, str(output_file))


def worker_process(gpu_id: int, image_paths: List[Path], args, output_dir: Path, total_count: int, shared_counter, counter_lock):
    """单个 GPU worker 进程 (含 CPU 并行预加载流水线)"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from concurrent.futures import ThreadPoolExecutor
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # 加载 VAE
    print(f"[GPU {gpu_id}] Loading VAE...", flush=True)
    from .utils.vae_utils import load_vae
    vae = load_vae(args.vae, device=device, dtype=dtype)
    print(f"[GPU {gpu_id}] VAE loaded, processing {len(image_paths)} images", flush=True)
    
    # 过滤已存在的文件
    to_process = []
    for image_path in image_paths:
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
        if args.skip_existing and existing:
            with counter_lock:
                shared_counter.value += 1
                current = shared_counter.value
            if current % 50 == 0 or current == total_count:
                print(f"Progress: {current}/{total_count}", flush=True)
        else:
            to_process.append(image_path)
    
    if not to_process:
        print(f"[GPU {gpu_id}] All images already cached, skipping", flush=True)
        return 0
    
    print(f"[GPU {gpu_id}] {len(to_process)} images to encode (skipped {len(image_paths) - len(to_process)} existing)", flush=True)
    
    # CPU 并行预加载 + GPU 编码 流水线
    cpu_workers = min(os.cpu_count() or 4, 8)  # CPU 线程数，最多 8
    preload_batch_size = cpu_workers * 2  # 预加载队列大小
    
    input_root = Path(args.input_dir)
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    processed = 0
    
    # 分批: 先 CPU 并行加载一批 → GPU 编码该批
    for batch_start in range(0, len(to_process), preload_batch_size):
        batch_paths = to_process[batch_start:batch_start + preload_batch_size]
        
        # CPU 多线程并行加载 + 缩放
        preload_args = [(p, args.resolution) for p in batch_paths]
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            results = list(pool.map(_preload_single_image, preload_args))
        
        # GPU 逐个编码 (图片尺寸不一，无法真正 batch)
        from safetensors.torch import save_file
        for image_path, img_array, w, h, err in results:
            if err is not None:
                print(f"[GPU {gpu_id}] Preload error: {image_path.name}: {err}", flush=True)
                with counter_lock:
                    shared_counter.value += 1
                    current = shared_counter.value
                if current % 50 == 0 or current == total_count:
                    print(f"Progress: {current}/{total_count}", flush=True)
                continue
            
            try:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.to(device=device, dtype=dtype)
                
                with torch.no_grad():
                    latent = vae.encode(img_tensor).latent_dist.mode()
                
                latent = (latent - shift_factor) * scaling_factor
                latent = latent.cpu()
                
                F, H, W = 1, latent.shape[2], latent.shape[3]
                
                if input_root:
                    try:
                        rel_path = image_path.relative_to(input_root)
                        target_dir = output_dir / rel_path.parent
                    except ValueError:
                        target_dir = output_dir
                else:
                    target_dir = output_dir
                
                target_dir.mkdir(parents=True, exist_ok=True)
                name = image_path.stem
                output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
                sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
                save_file(sd, str(output_file))
                processed += 1
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Encode error: {image_path.name}: {e}", flush=True)
            
            with counter_lock:
                shared_counter.value += 1
                current = shared_counter.value
            if current % 50 == 0 or current == total_count:
                print(f"Progress: {current}/{total_count}", flush=True)
    
    # 清理
    del vae
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Cache latents for Z-Image training")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0=auto detect)")
    
    # 训练模式
    parser.add_argument("--mode", type=str, default="text2img", 
                       choices=["text2img", "controlnet", "img2img", "inpaint", "omni"],
                       help="Training mode: text2img, controlnet, img2img, inpaint, omni")
    
    # ControlNet 参数
    parser.add_argument("--control_dir", type=str, default=None,
                       help="ControlNet condition image directory")
    
    # Img2Img 参数
    parser.add_argument("--source_dir", type=str, default=None,
                       help="Source image directory (for img2img mode)")
    
    # Inpaint 参数
    parser.add_argument("--mask_dir", type=str, default=None,
                       help="Mask image directory (for inpaint mode)")
    
    # Omni 参数
    parser.add_argument("--condition_dirs", type=str, default=None,
                       help="Comma-separated condition image directories")
    
    args = parser.parse_args()

    mode = args.mode
    print(f"Cache mode: {mode}", flush=True)
    
    # Mode-specific argument validation
    if mode == "controlnet" and not args.control_dir:
        raise ValueError("--control_dir is required for controlnet mode")
    if mode == "img2img" and not args.source_dir:
        raise ValueError("--source_dir is required for img2img mode")
    if mode == "inpaint" and not args.mask_dir:
        raise ValueError("--mask_dir is required for inpaint mode")
    if mode == "omni" and not args.condition_dirs:
        raise ValueError("--condition_dirs is required for omni mode")
    
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
    
    # Mode-specific cache suffix for skip-existing check
    cache_suffix_map = {
        "text2img": f"_{ARCHITECTURE}.safetensors",
        "controlnet": f"_{ARCHITECTURE}_controlnet.safetensors",
        "img2img": f"_{ARCHITECTURE}_img2img.safetensors",
        "inpaint": f"_{ARCHITECTURE}_inpaint.safetensors",
        "omni": f"_{ARCHITECTURE}_omni.safetensors",
    }
    cache_suffix = cache_suffix_map[mode]
    
    # 检测 GPU 数量
    if args.num_gpus > 0:
        num_gpus = args.num_gpus
    else:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                num_gpus = len(result.stdout.strip().split('\n'))
            else:
                num_gpus = 1
        except Exception:
            num_gpus = 1
    
    # 检测 CPU 核心数
    cpu_count = os.cpu_count() or 4
    cpu_workers = min(cpu_count, 8)  # 最多 8 线程做图片预加载
    
    # Multi-GPU fallback: non-text2img modes only support single GPU currently
    if num_gpus > 1 and mode != "text2img":
        print(f"Warning: Multi-GPU mode only supports text2img. Mode '{mode}' will use single GPU.", flush=True)
        num_gpus = 1
    
    if num_gpus <= 1:
        # 单 GPU 模式
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from safetensors.torch import save_file
        from .utils.vae_utils import load_vae
        
        print(f"Using single GPU mode with {cpu_workers} CPU workers", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        print(f"Loading VAE: {args.vae}", flush=True)
        vae = load_vae(args.vae, device=device, dtype=dtype)
        print("VAE loaded successfully", flush=True)
        
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
        shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
        input_root = Path(args.input_dir)
        
        # ====================================================================
        # Mode routing: controlnet / img2img use dedicated pair processors
        # ====================================================================
        if mode in ("controlnet", "img2img", "inpaint"):
            if mode == "controlnet":
                pair_dir = Path(args.control_dir)
                process_fn = process_controlnet_pair
                pair_kwarg = "control_path"
            elif mode == "img2img":
                pair_dir = Path(args.source_dir)
                process_fn = process_img2img_pair
                pair_kwarg = "source_path"
            else:  # inpaint
                pair_dir = Path(args.mask_dir)
                process_fn = process_inpaint_pair
                pair_kwarg = "mask_path"
            mode_label = mode
            
            # Build paired file list
            to_process = []
            skipped = 0
            for i, image_path in enumerate(images, 1):
                name = image_path.stem
                existing = list(output_dir.glob(f"{name}_*{cache_suffix}"))
                if args.skip_existing and existing:
                    skipped += 1
                    print(f"Progress: {i}/{total}", flush=True)
                    continue
                
                # Find paired file with same stem in pair_dir
                pair_path = None
                for ext in ('.jpg', '.jpeg', '.png', '.webp'):
                    candidate = pair_dir / f"{name}{ext}"
                    if candidate.exists():
                        pair_path = candidate
                        break
                    candidate_upper = pair_dir / f"{name}{ext.upper()}"
                    if candidate_upper.exists():
                        pair_path = candidate_upper
                        break
                
                if pair_path is None:
                    print(f"Warning: No {mode_label} pair found for {name}, skipping", flush=True)
                    print(f"Progress: {i}/{total}", flush=True)
                    continue
                
                to_process.append((image_path, pair_path, i))
            
            if to_process:
                print(f"{len(to_process)} {mode_label} pairs to encode, {skipped} skipped", flush=True)
                processed = 0
                for image_path, pair_path, orig_idx in to_process:
                    try:
                        kwargs = {
                            "target_path": image_path,
                            "vae": vae,
                            "resolution": args.resolution,
                            "output_dir": output_dir,
                            "device": device,
                            "dtype": dtype,
                            "input_root": input_root,
                        }
                        kwargs[pair_kwarg] = pair_path
                        process_fn(**kwargs)
                        processed += 1
                    except Exception as e:
                        print(f"Error: {image_path}: {e}", flush=True)
                    print(f"Progress: {orig_idx}/{total}", flush=True)
                print(f"{mode_label} caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
            else:
                print(f"All {skipped} {mode_label} pairs already cached", flush=True)
        
        # ====================================================================
        # Mode: omni — multi-condition VAE encoding
        # ====================================================================
        elif mode == "omni":
            condition_dir_list = [Path(d.strip()) for d in args.condition_dirs.split(",")]
            num_cond_dirs = len(condition_dir_list)
            print(f"Omni mode: {num_cond_dirs} condition directories", flush=True)
            for cd in condition_dir_list:
                print(f"  Condition dir: {cd}", flush=True)
            
            to_process = []
            skipped = 0
            for i, image_path in enumerate(images, 1):
                name = image_path.stem
                existing = list(output_dir.glob(f"{name}_*{cache_suffix}"))
                if args.skip_existing and existing:
                    skipped += 1
                    print(f"Progress: {i}/{total}", flush=True)
                    continue
                
                # Find condition files in each condition dir
                cond_paths = []
                all_found = True
                for cd in condition_dir_list:
                    cond_path = None
                    for ext in ('.jpg', '.jpeg', '.png', '.webp'):
                        candidate = cd / f"{name}{ext}"
                        if candidate.exists():
                            cond_path = candidate
                            break
                    if cond_path is None:
                        print(f"Warning: No condition image for '{name}' in {cd}, skipping", flush=True)
                        all_found = False
                        break
                    cond_paths.append(cond_path)
                
                if all_found:
                    to_process.append((image_path, i, cond_paths))
                else:
                    skipped += 1
            
            if to_process:
                print(f"{len(to_process)} omni sets to encode, {skipped} skipped", flush=True)
                processed = 0
                for image_path, orig_idx, cond_paths in to_process:
                    try:
                        process_omni_set(
                            target_path=image_path,
                            condition_paths=cond_paths,
                            vae=vae,
                            resolution=args.resolution,
                            output_dir=output_dir,
                            device=device,
                            dtype=dtype,
                            input_root=input_root,
                        )
                        processed += 1
                    except Exception as e:
                        print(f"Error: {image_path}: {e}", flush=True)
                    print(f"Progress: {orig_idx}/{total}", flush=True)
                print(f"Omni caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
            else:
                print(f"All {skipped} omni sets already cached", flush=True)
        
        # ====================================================================
        # Mode: text2img (default) — batch preload + GPU encode pipeline
        # ====================================================================
        elif mode == "text2img":
            to_process = []
            skipped = 0
            for i, image_path in enumerate(images, 1):
                name = image_path.stem
                existing = list(output_dir.glob(f"{name}_*{cache_suffix}"))
                if args.skip_existing and existing:
                    skipped += 1
                    print(f"Progress: {i}/{total}", flush=True)
                else:
                    to_process.append((image_path, i))
            
            if to_process:
                print(f"{len(to_process)} images to encode, {skipped} skipped", flush=True)
                
                preload_batch_size = cpu_workers * 2
                processed = 0
                
                for batch_start in range(0, len(to_process), preload_batch_size):
                    batch = to_process[batch_start:batch_start + preload_batch_size]
                    batch_paths = [p for p, _ in batch]
                    batch_indices = [idx for _, idx in batch]
                    
                    preload_args = [(p, args.resolution) for p in batch_paths]
                    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                        results = list(pool.map(_preload_single_image, preload_args))
                    
                    for (image_path, img_array, w, h, err), orig_idx in zip(results, batch_indices):
                        if err is not None:
                            print(f"Error: {image_path}: {err}", flush=True)
                            print(f"Progress: {orig_idx}/{total}", flush=True)
                            continue
                        
                        try:
                            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                            img_tensor = img_tensor * 2.0 - 1.0
                            img_tensor = img_tensor.to(device=device, dtype=dtype)
                            
                            with torch.no_grad():
                                latent = vae.encode(img_tensor).latent_dist.mode()
                            
                            latent = (latent - shift_factor) * scaling_factor
                            latent = latent.cpu()
                            
                            F, H, W = 1, latent.shape[2], latent.shape[3]
                            
                            try:
                                rel_path = image_path.relative_to(input_root)
                                target_dir = output_dir / rel_path.parent
                            except ValueError:
                                target_dir = output_dir
                            
                            target_dir.mkdir(parents=True, exist_ok=True)
                            name = image_path.stem
                            output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
                            sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
                            save_file(sd, str(output_file))
                            processed += 1
                            
                        except Exception as e:
                            print(f"Error: {image_path}: {e}", flush=True)
                        
                        print(f"Progress: {orig_idx}/{total}", flush=True)
                
                print(f"Latent caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
            else:
                print(f"All {skipped} images already cached", flush=True)
        
        else:
            print(f"Mode '{mode}' is not yet supported in single-GPU mode", flush=True)
            return
        
        # Cleanup
        del vae
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VAE unloaded, GPU memory released", flush=True)
    
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        print(f"🚀 Multi-GPU mode: {num_gpus} GPUs × {cpu_workers} CPU workers", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        chunk_size = (total + num_gpus - 1) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            if start < total:
                chunks.append((i, images[start:end]))
        
        print(f"Distributing {total} images across {len(chunks)} GPUs", flush=True)
        for gpu_id, chunk in chunks:
            print(f"  GPU {gpu_id}: {len(chunk)} images", flush=True)
        
        manager = mp.Manager()
        shared_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        mp.set_start_method('spawn', force=True)
        
        total_processed = 0
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(worker_process, gpu_id, chunk, args, output_dir, total, shared_counter, counter_lock): gpu_id
                for gpu_id, chunk in chunks
            }
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    processed = future.result()
                    total_processed += processed
                    print(f"[GPU {gpu_id}] Completed: {processed} images", flush=True)
                except Exception as e:
                    print(f"[GPU {gpu_id}] Worker error: {e}", flush=True)
        
        print(f"Multi-GPU latent caching completed! Total processed: {total_processed}", flush=True)


if __name__ == "__main__":
    main()
