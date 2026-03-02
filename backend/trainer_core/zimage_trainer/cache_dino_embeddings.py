# -*- coding: utf-8 -*-
"""
DINOv3 Embedding Cache Script

Pre-compute DINOv3 patch embeddings for training images and save alongside
the latent cache files. This avoids computing target DINOv3 features during
training, saving ~50% of the DINOv3 compute cost.

Usage:
    python -m zimage_trainer.cache_dino_embeddings \
        --dino_model /path/to/Dinov3-base \
        --cache_dir /path/to/latent_cache \
        --vae /path/to/vae \
        --dino_image_size 512

Output:
    For each {name}_{WxH}_zi.safetensors, creates:
        {name}_{WxH}_zi.dino.safetensors
    containing key "dino_emb" → (num_patches, hidden_dim) float32 tensor.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def cache_dino_embeddings(
    dino_model_path: str,
    vae_path: str,
    cache_dir: str,
    dino_image_size: int = 512,
    skip_existing: bool = True,
    device: str = "cuda",
):
    """
    Cache DINOv3 embeddings for all latent cache files.

    Pipeline per image:
        1. Load cached latent from .safetensors
        2. VAE decode latent → RGB pixels
        3. Resize to dino_image_size (preserving aspect ratio)
        4. Extract DINOv3 patch embeddings
        5. Save as .dino.safetensors
    """
    from diffusers import AutoencoderKL
    from transformers import DINOv3ViTModel

    dtype = torch.bfloat16
    patch_size = 16  # DINOv3 patch size

    # Load VAE
    logger.info(f"Loading VAE: {vae_path}")
    if os.path.isdir(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    logger.info("VAE loaded")

    # Load DINOv3
    logger.info(f"Loading DINOv3: {dino_model_path}")
    dino = DINOv3ViTModel.from_pretrained(dino_model_path, torch_dtype=torch.float32)
    dino.to(device)
    dino.eval()
    dino.requires_grad_(False)
    num_register = getattr(dino.config, "num_register_tokens", 0)
    num_prefix = 1 + num_register  # CLS + register tokens
    params_m = sum(p.numel() for p in dino.parameters()) / 1e6
    logger.info(f"DINOv3 loaded: {params_m:.1f}M params")

    # Find all latent cache files
    cache_path = Path(cache_dir)
    latent_files = sorted(cache_path.rglob("*_zi.safetensors"))
    logger.info(f"Found {len(latent_files)} latent cache files")

    skipped = 0
    processed = 0
    errors = 0

    for i, latent_path in enumerate(latent_files):
        # Output path
        dino_path = latent_path.with_suffix(".dino.safetensors")

        if skip_existing and dino_path.exists():
            skipped += 1
            continue

        try:
            # 1. Load latent
            data = load_file(str(latent_path))
            latent_key = next(
                (k for k in data.keys() if k.startswith("latents_")), None
            )
            if latent_key is None:
                logger.warning(f"No latent key in {latent_path.name}, skipping")
                errors += 1
                continue
            latent = data[latent_key].unsqueeze(0)  # (1, C, H, W)

            # 2. Downsample latent → VAE decode
            C, H, W = latent.shape[1], latent.shape[2], latent.shape[3]
            scale = dino_image_size / (max(H, W) * 8)
            lat_h = max(patch_size // 8, int(round(H * scale / 2)) * 2)
            lat_w = max(patch_size // 8, int(round(W * scale / 2)) * 2)

            latent_small = F.interpolate(
                latent, size=(lat_h, lat_w),
                mode="bilinear", align_corners=False,
            )

            with torch.no_grad():
                rgb = vae.decode(
                    latent_small.to(device=device, dtype=dtype),
                    return_dict=False,
                )[0]
            rgb = (rgb / 2 + 0.5).clamp(0, 1).float()

            # 3. Ensure divisible by patch_size
            ph, pw = rgb.shape[-2], rgb.shape[-1]
            tgt_h = (ph // patch_size) * patch_size
            tgt_w = (pw // patch_size) * patch_size
            if tgt_h != ph or tgt_w != pw:
                rgb = F.interpolate(rgb, size=(tgt_h, tgt_w),
                                    mode="bilinear", align_corners=False)

            # 4. DINOv3 extract
            with torch.no_grad():
                out = dino(pixel_values=rgb)
            hs = out.last_hidden_state  # (1, 1+P, D)
            cls_emb = hs[:, 0:1, :].squeeze(0).cpu()  # (1, D)
            patch_emb = hs[:, num_prefix:, :].squeeze(0).cpu()  # (P, D)

            # 5. Save both CLS and patch (training picks based on feature_mode)
            save_file({
                "dino_emb": patch_emb,      # (P, D) — backward compatible
                "dino_cls": cls_emb,         # (1, D) — CLS token
            }, str(dino_path))
            processed += 1

        except Exception as e:
            logger.error(f"Error processing {latent_path.name}: {e}")
            errors += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(latent_files):
            logger.info(
                f"Progress: {i+1}/{len(latent_files)} "
                f"(processed={processed}, skipped={skipped}, errors={errors})"
            )

    # Cleanup
    del vae, dino
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"Done! processed={processed}, skipped={skipped}, errors={errors}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cache DINOv3 embeddings for Z-Image training"
    )
    parser.add_argument(
        "--dino_model", type=str, required=True,
        help="DINOv3 model path (local or HuggingFace ID)",
    )
    parser.add_argument(
        "--vae", type=str, required=True,
        help="VAE model path",
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Latent cache directory (same as training cache_dir)",
    )
    parser.add_argument(
        "--dino_image_size", type=int, default=512,
        help="DINOv3 input resolution (default 512)",
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
        vae_path=args.vae,
        cache_dir=args.cache_dir,
        dino_image_size=args.dino_image_size,
        skip_existing=args.skip_existing,
        device=device,
    )


if __name__ == "__main__":
    main()
