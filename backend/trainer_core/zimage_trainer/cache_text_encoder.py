# -*- coding: utf-8 -*-
"""
Z-Image Text Encoder Cache Script (Standalone)

将文本描述编码并缓存到磁盘。

Usage:
    python -m zimage_trainer.cache_text_encoder \
        --text_encoder /path/to/qwen_3_4b.safetensors \
        --input_dir /path/to/images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

# 延迟导入 torch 和 CUDA 相关模块（避免多卡模式下的 CUDA 初始化冲突）
# 实际导入在需要时进行

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def load_text_encoder(model_path: str, device, dtype):
    """加载 Qwen3 文本编码器"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading text encoder: {model_path}")
    
    tokenizer_path = model_path
    
    # 如果是 safetensors 文件，需要加载为 HuggingFace 格式
    if model_path.endswith('.safetensors'):
        # 尝试从同目录加载 tokenizer
        model_dir = Path(model_path).parent
        tokenizer_path = str(model_dir)
        
        # 加载模型权重
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        
        # 创建模型配置
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            state_dict=state_dict,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        # 从目录加载
        # 检查 tokenizer 是否在兄弟目录
        path_obj = Path(model_path)
        if not (path_obj / "tokenizer.json").exists() and (path_obj.parent / "tokenizer").exists():
            tokenizer_path = str(path_obj.parent / "tokenizer")
            logger.info(f"Tokenizer not found in {model_path}, using {tokenizer_path}")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    model.to(device)
    model.eval()
    
    return tokenizer, model


def get_caption(image_path: Path) -> Optional[str]:
    """获取图片对应的文本描述"""
    # 尝试多种文件名格式
    txt_paths = [
        image_path.with_suffix('.txt'),
        image_path.with_suffix('.caption'),
        image_path.parent / f"{image_path.stem}.txt",
    ]
    
    for txt_path in txt_paths:
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    
    return None


def encode_text(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
    device = None,
):
    """编码文本为嵌入向量
    
    Official pipeline (_encode_prompt):
        - hidden_states[-2]  (倒数第二层)
        - prompt_embeds[j][prompt_masks[j]]  (只取非 padding token)
    """
    import torch
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Encode
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 官方 pipeline 使用倒数第二层
        hidden_states = outputs.hidden_states[-2]
        # 只取非 padding token (与官方 prompt_embeds[j][prompt_masks[j]] 一致)
        attention_mask = inputs["attention_mask"].bool()
        embed = hidden_states[0][attention_mask[0]]  # (actual_tokens, hidden_dim)
    
    return embed


def encode_text_omni(
    tokenizer,
    model,
    text: str,
    num_condition_images: int,
    max_length: int = 512,
    device = None,
):
    """编码 omni 模式文本 (多段式模板)
    
    Official pipeline (_encode_prompt with num_condition_images > 0):
        prompt_list = ["<|im_start|>user\n<|vision_start|>"]
        prompt_list += ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
        prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
        prompt_list += ["<|vision_end|><|im_end|>"]
    
    Each segment is tokenized separately, then only non-padded tokens are kept.
    Segments are concatenated and segment lengths stored for reconstruction.
    """
    import torch
    
    # Build multi-segment prompt template (exactly as official pipeline)
    if num_condition_images == 0:
        # Fallback: same as non-omni text2img template
        segments = ["<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"]
    else:
        segments = ["<|im_start|>user\n<|vision_start|>"]
        segments += ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
        segments += ["<|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
        segments += ["<|vision_end|><|im_end|>"]
    
    # Tokenize each segment separately
    text_inputs = tokenizer(
        segments,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    if device:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    text_input_ids = text_inputs["input_ids"]
    prompt_masks = text_inputs["attention_mask"].bool()
    
    # Encode all segments at once
    with torch.no_grad():
        all_hidden = model(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
    
    # Extract non-padded tokens per segment and concatenate
    # (matches official: prompt_embeds[j][prompt_masks[j]])
    segment_embeds = []
    segment_lengths = []
    for j in range(len(segments)):
        seg_embed = all_hidden[j][prompt_masks[j]]  # (actual_tokens_j, dim)
        segment_embeds.append(seg_embed)
        segment_lengths.append(seg_embed.shape[0])
    
    # Concatenate all segments into one tensor
    concat_embed = torch.cat(segment_embeds, dim=0)  # (total_tokens, dim)
    lengths_tensor = torch.tensor(segment_lengths, dtype=torch.int32)
    
    return concat_embed, lengths_tensor


def process_caption(
    image_path: Path,
    tokenizer,
    model,
    output_dir: Path,
    max_length: int,
    device,
    dtype,
    input_root: Path = None,
    mode: str = "text2img",
    num_condition_images: int = 0,
) -> bool:
    """处理单个文本描述
    
    Official pipeline chat template:
        text2img/controlnet/img2img/inpaint:
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        omni (with condition images):
            Multi-segment template with <|vision_start|>/<|vision_end|>
    """
    import torch
    from safetensors.torch import save_file
    # 获取 caption
    caption = get_caption(image_path)
    if caption is None:
        logger.warning(f"No caption found for {image_path}")
        return False
    
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
    
    name = image_path.stem
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
    
    if mode == "omni" and num_condition_images > 0:
        # Omni 多段式编码
        concat_embed, segment_lengths = encode_text_omni(
            tokenizer, model, caption, num_condition_images, max_length, device
        )
        concat_embed = concat_embed.to(dtype=dtype)
        
        sd = {
            f"varlen_vl_embed_{dtype_str}": concat_embed.cpu(),
            "segment_lengths": segment_lengths.cpu(),
        }
    else:
        # 非 omni 模式 — chat template wrapping + 单段编码
        # Official: "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        wrapped_text = "<|im_start|>user\n" + caption + "<|im_end|>\n<|im_start|>assistant\n"
        embed = encode_text(tokenizer, model, wrapped_text, max_length, device)
        embed = embed.to(dtype=dtype)
        
        sd = {f"varlen_vl_embed_{dtype_str}": embed.cpu()}
    
    save_file(sd, str(output_file))
    
    return True


def find_images(input_dir: str) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def worker_process_te(gpu_id: int, image_paths: list, args, output_dir: Path, total_count: int, shared_counter, counter_lock):
    """单个 GPU text encoder worker 进程"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print(f"[GPU {gpu_id}] Loading Text Encoder...", flush=True)
    tokenizer, model = load_text_encoder(args.text_encoder, device, dtype)
    print(f"[GPU {gpu_id}] Text Encoder loaded, processing {len(image_paths)} captions", flush=True)
    
    processed = 0
    for image_path in image_paths:
        name = image_path.stem
        
        # 计算预期输出路径
        try:
            rel_path = image_path.relative_to(args.input_dir)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
        
        output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
        
        if args.skip_existing and output_file.exists():
            with counter_lock:
                shared_counter.value += 1
                current = shared_counter.value
            if current % 10 == 0 or current == total_count:
                print(f"Progress: {current}/{total_count}", flush=True)
            continue
        
        try:
            if process_caption(image_path, tokenizer, model, output_dir, args.max_length, device, dtype, input_root=Path(args.input_dir), mode=args.mode, num_condition_images=args.num_condition_images):
                processed += 1
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {image_path.name}: {e}", flush=True)
        
        with counter_lock:
            shared_counter.value += 1
            current = shared_counter.value
        
        if current % 10 == 0 or current == total_count:
            print(f"Progress: {current}/{total_count}", flush=True)
    
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for Z-Image training")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text encoder path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory (with .txt captions)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0=auto detect)")
    parser.add_argument("--mode", type=str, default="text2img",
                       choices=["text2img", "controlnet", "img2img", "inpaint", "omni"],
                       help="Training mode (affects prompt template)")
    parser.add_argument("--num_condition_images", type=int, default=0,
                       help="Number of condition images for omni mode")
    
    args = parser.parse_args()
    
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
    
    if num_gpus <= 1:
        # 单 GPU 模式
        import torch
        
        print(f"Using single GPU mode", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        print(f"Loading Text Encoder: {args.text_encoder}", flush=True)
        tokenizer, model = load_text_encoder(args.text_encoder, device, dtype)
        print("Text Encoder loaded successfully", flush=True)
        
        success = 0
        skipped = 0
        
        for i, image_path in enumerate(images, 1):
            name = image_path.stem
            
            try:
                rel_path = image_path.relative_to(args.input_dir)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
                
            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
            
            if args.skip_existing and output_file.exists():
                skipped += 1
                print(f"Progress: {i}/{total}", flush=True)
                continue
            
            try:
                if process_caption(image_path, tokenizer, model, output_dir, args.max_length, device, dtype, input_root=Path(args.input_dir), mode=args.mode, num_condition_images=args.num_condition_images):
                    success += 1
                print(f"Progress: {i}/{total}", flush=True)
            except Exception as e:
                print(f"Error: {image_path}: {e}", flush=True)
                print(f"Progress: {i}/{total}", flush=True)
                continue
        
        print(f"Text encoding completed! Processed: {success}, Skipped: {skipped}", flush=True)
        
        del model, tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Text Encoder unloaded, GPU memory released", flush=True)
    
    else:
        # 多 GPU 模式
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        print(f"🚀 Multi-GPU mode: using {num_gpus} GPUs", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        chunk_size = (total + num_gpus - 1) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            if start < total:
                chunks.append((i, images[start:end]))
        
        print(f"Distributing {total} captions across {len(chunks)} GPUs", flush=True)
        for gpu_id, chunk in chunks:
            print(f"  GPU {gpu_id}: {len(chunk)} captions", flush=True)
        
        manager = mp.Manager()
        shared_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()
        
        mp.set_start_method('spawn', force=True)
        
        total_processed = 0
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(worker_process_te, gpu_id, chunk, args, output_dir, total, shared_counter, counter_lock): gpu_id
                for gpu_id, chunk in chunks
            }
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    processed = future.result()
                    total_processed += processed
                    print(f"[GPU {gpu_id}] Completed: {processed} captions", flush=True)
                except Exception as e:
                    print(f"[GPU {gpu_id}] Worker error: {e}", flush=True)
        
        print(f"Multi-GPU text encoding completed! Total processed: {total_processed}", flush=True)


if __name__ == "__main__":
    main()
