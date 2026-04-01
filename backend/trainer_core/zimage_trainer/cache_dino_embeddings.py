# -*- coding: utf-8 -*-
"""
DINOv3 Embedding Cache Script

Pre-compute DINOv3 patch embeddings + spatial attention masks directly from
original images.  For each latent cache file ``{name}_{WxH}_zi.safetensors``
a companion ``{name}_{WxH}_zi.dino.safetensors`` is created containing:

    dino_emb   — (P, D) float32  patch embeddings
    dino_cls   — (1, D) float32  CLS token
    dino_mask  — (gh, gw) float32  spatial attention mask [0,1]

Usage (called by cache_router.py):
    python -m zimage_trainer.cache_dino_embeddings \
        --dino_model /path/to/Dinov3-base \
        --input_dir  /path/to/dataset \
        --output_dir /path/to/dataset \
        --skip_existing
"""

import argparse
import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Regex: extract base name and resolution from latent cache filename
# e.g. "my_image_1024x768_zi" -> ("my_image", "1024x768")
LATENT_RE = re.compile(r"^(.+?)_(\d+x\d+)_zi$")


def _find_original_image(base_name: str, search_dir: Path) -> Optional[Path]:
    """Find the original image file matching a base name."""
    for ext in IMAGE_EXTENSIONS:
        candidate = search_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    # Also search subdirectories (rglob) as fallback
    for ext in IMAGE_EXTENSIONS:
        matches = list(search_dir.rglob(f"{base_name}{ext}"))
        if matches:
            return matches[0]
    return None


def cache_dino_embeddings(
    dino_model_path: str,
    input_dir: str,
    output_dir: str,
    dino_image_size: int = 518,
    skip_existing: bool = True,
    device: str = "cuda",
):
    """
    Cache DINOv3 embeddings by reading original images directly.

    Pipeline per image:
        1. Find latent cache files (*_zi.safetensors) in input_dir
        2. Locate matching original image
        3. Load & preprocess image for DINOv3
        4. Extract patch embeddings + CLS + spatial attention mask
        5. Save as .dino.safetensors next to the latent cache
    """
    from transformers import AutoModel, AutoImageProcessor

    # Load DINOv3
    logger.info(f"Loading DINOv3: {dino_model_path}")
    # Use eager attention to get real attention weights (SDPA/flash returns None)
    dino = AutoModel.from_pretrained(
        dino_model_path, dtype=torch.float32, attn_implementation="eager",
    )
    dino.to(device)
    dino.eval()
    dino.requires_grad_(False)

    num_register = getattr(dino.config, "num_register_tokens", 0)
    num_prefix = 1 + num_register  # CLS + register tokens
    patch_size = getattr(dino.config, "patch_size", 16)
    params_m = sum(p.numel() for p in dino.parameters()) / 1e6
    logger.info(f"DINOv3 loaded: {params_m:.1f}M params, patch={patch_size}, registers={num_register}")

    # Load image processor
    processor = AutoImageProcessor.from_pretrained(dino_model_path)
    logger.info(f"Image processor loaded: {type(processor).__name__}")

    # Find all latent cache files
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    latent_files = sorted(input_path.rglob("*_zi.safetensors"))

    # Filter out non-standard files (controlnet, img2img, etc.)
    latent_files = [
        f for f in latent_files
        if not any(tag in f.stem for tag in ["controlnet", "img2img", "inpaint", "omni", "siglip", "dino", "_te"])
    ]

    logger.info(f"Found {len(latent_files)} latent cache files in {input_dir}")

    if len(latent_files) == 0:
        logger.warning("No latent cache files found. Generate latent cache first!")
        print(f"Progress: 0/0")
        return

    skipped = 0
    processed = 0
    errors = 0
    no_image = 0

    for i, latent_path in enumerate(latent_files):
        # Output path: same name with .dino.safetensors suffix
        dino_path = latent_path.with_suffix(".dino.safetensors")

        if skip_existing and dino_path.exists():
            skipped += 1
            continue

        # Parse base name from latent filename
        match = LATENT_RE.match(latent_path.stem)
        if not match:
            logger.warning(f"Cannot parse filename: {latent_path.name}, skipping")
            errors += 1
            continue

        base_name = match.group(1)

        # Find original image
        image_path = _find_original_image(base_name, latent_path.parent)
        if image_path is None:
            # Try input_dir root
            image_path = _find_original_image(base_name, input_path)

        if image_path is None:
            no_image += 1
            if no_image <= 5:
                logger.warning(f"Image not found for: {base_name}")
            continue

        try:
            # 1. Load image
            img = Image.open(image_path).convert("RGB")

            # 2. Process with DINOv3 image processor
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device)

            # 3. DINOv3 forward (with native attention maps)
            with torch.no_grad():
                out = dino(pixel_values=pixel_values, output_attentions=True)

            hs = out.last_hidden_state  # (1, 1+R+P, D)
            cls_emb = hs[:, 0:1, :].squeeze(0).cpu().float()  # (1, D)
            patch_emb = hs[:, num_prefix:, :].squeeze(0).cpu().float()  # (P, D)

            # 4. Extract spatial attention mask from last layer (native self-attention)
            if out.attentions is None or len(out.attentions) == 0:
                logger.warning(f"No attention weights for {image_path.name}, skipping")
                errors += 1
                continue

            last_attn = out.attentions[-1]  # (1, num_heads, seq_len, seq_len)
            # CLS token (row 0) attention to all patch tokens
            cls_to_patch = last_attn[0, :, 0, num_prefix:]  # (num_heads, P)
            attn_map = cls_to_patch.mean(dim=0)  # (P,)

            # Reshape to 2D spatial grid
            num_patches = attn_map.shape[0]
            input_h, input_w = pixel_values.shape[-2], pixel_values.shape[-1]
            grid_h = input_h // patch_size
            grid_w = input_w // patch_size

            if grid_h * grid_w == num_patches:
                attn_2d = attn_map.reshape(grid_h, grid_w)
            else:
                side = int(num_patches ** 0.5)
                attn_2d = attn_map[:side * side].reshape(side, side)

            # Min-Max normalize to [0, 1]
            a_min, a_max = attn_2d.min(), attn_2d.max()
            if a_max - a_min > 1e-8:
                attn_2d = (attn_2d - a_min) / (a_max - a_min)
            else:
                attn_2d = torch.ones_like(attn_2d)

            dino_mask = attn_2d.cpu().float()  # (grid_h, grid_w)

            # 5. Save
            save_file(
                {
                    "dino_emb": patch_emb,   # (P, D)
                    "dino_cls": cls_emb,     # (1, D)
                    "dino_mask": dino_mask,  # (gh, gw)
                },
                str(dino_path),
            )
            processed += 1

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            errors += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(latent_files):
            print(
                f"Progress: {i + 1}/{len(latent_files)}",
                flush=True,
            )
            logger.info(
                f"Progress: {i + 1}/{len(latent_files)} "
                f"(processed={processed}, skipped={skipped}, "
                f"no_image={no_image}, errors={errors})"
            )

    # Final progress
    print(f"Progress: {len(latent_files)}/{len(latent_files)}", flush=True)

    # Cleanup
    del dino
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Done! processed={processed}, skipped={skipped}, "
        f"no_image={no_image}, errors={errors}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cache DINOv3 embeddings for Z-Image training"
    )
    parser.add_argument(
        "--dino_model", type=str, required=True,
        help="DINOv3 model path (local directory)",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Dataset directory containing images and latent cache files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Output directory (default: same as input_dir)",
    )
    parser.add_argument(
        "--dino_image_size", type=int, default=518,
        help="DINOv3 input resolution (default 518, standard for DINOv2/v3)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip if .dino.safetensors already exists",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-compute all embeddings",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dino_embeddings(
        dino_model_path=args.dino_model,
        input_dir=args.input_dir,
        output_dir=args.output_dir or args.input_dir,
        dino_image_size=args.dino_image_size,
        skip_existing=args.skip_existing,
        device=device,
    )


if __name__ == "__main__":
    main()
