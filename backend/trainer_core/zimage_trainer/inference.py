# -*- coding: utf-8 -*-
"""
Z-Image Inference Script (Standalone)

使用训练好的 LoRA 进行图像生成。

Usage:
    python -m zimage_trainer.inference \
        --dit /path/to/z_image_turbo_bf16.safetensors \
        --vae /path/to/vae \
        --text_encoder /path/to/qwen_3_4b \
        --lora /path/to/lora.safetensors \
        --prompt "your prompt here" \
        --output output.png
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


from .networks.lora import LoRANetwork
from .utils.vae_utils import load_vae, decode_latents_to_pixels
from .utils.latent_utils import pack_latents, unpack_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ZImagePipeline:
    """Z-Image 推理管道 (Wrapper for diffusers)"""
    
    def __init__(
        self,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        
        # 加载组件
        logger.info("Loading models via diffusers...")
        
        # VAE
        from .utils.vae_utils import load_vae
        vae = load_vae(vae_path, device=self.device, dtype=self.dtype)
        
        # Text Encoder & Tokenizer
        from .utils.zimage_utils import load_text_encoder_and_tokenizer
        text_encoder, tokenizer = load_text_encoder_and_tokenizer(text_encoder_path, device=self.device)
        
        # Transformer
        from .utils.zimage_utils import load_transformer
        transformer = load_transformer(dit_path, device=self.device, torch_dtype=self.dtype)
        
        # Scheduler
        from .utils.zimage_utils import load_scheduler
        scheduler = load_scheduler("flow_match_euler", use_diffusers=True)
        
        # Create Pipeline
        from .utils.zimage_utils import create_pipeline_from_components
        self.pipeline = create_pipeline_from_components(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        # LoRA
        if lora_path:
            logger.info(f"  LoRA: {lora_path} (scale={lora_scale})")
            self.pipeline.load_lora_weights(lora_path)
            self.pipeline.fuse_lora(lora_scale=lora_scale)
        
        self.pipeline.to(self.device)
        logger.info("Models loaded successfully!")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        生成图像
        """
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating with prompt: {prompt}")
        
        # 调用 diffusers pipeline
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        ).images[0]
        
        return image


def main():
    parser = argparse.ArgumentParser(description="Z-Image Inference")
    
    # 模型路径
    parser.add_argument("--dit", type=str, required=True, help="DiT model path")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text encoder path")
    parser.add_argument("--lora", type=str, default=None, help="LoRA weights path")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # 输出
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    
    args = parser.parse_args()
    
    # 创建管道
    pipeline = ZImagePipeline(
        dit_path=args.dit,
        vae_path=args.vae,
        text_encoder_path=args.text_encoder,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
    )
    
    # 生成图像
    logger.info(f"Generating image...")
    logger.info(f"  Prompt: {args.prompt}")
    logger.info(f"  Size: {args.width}x{args.height}")
    logger.info(f"  Steps: {args.steps}")
    
    image = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

