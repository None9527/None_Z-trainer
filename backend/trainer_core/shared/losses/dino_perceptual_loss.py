# -*- coding: utf-8 -*-
"""
🧠 DINOv3 Perceptual Loss (Cached Target Embedding)

Uses pre-cached DINOv3 patch embeddings as ground-truth perceptual target.
During training, only the predicted x₀ path needs runtime computation:
    pred_v → pred_x0 → VAE decode → DINOv3 → compare with cached target

This cuts DINOv3+VAE compute by 50% compared to computing both sides at runtime.

Architecture:
    [Pre-cache phase]
    GT image → VAE decode → DINOv3 → target_emb (saved to disk)

    [Training phase]  
    pred_v → pred_x0 → VAE decode (no_grad) → DINOv3 (no_grad) → pred_emb
    loss = STE(cosine_distance(pred_emb, cached_target_emb), latent_cosine)

Gradient flow (STE bridge):
    DINOv3 distance is computed under no_grad for VRAM efficiency.
    Latent cosine provides the differentiable gradient direction.
    DINOv3 distance provides semantically-aware loss magnitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DinoPerceptualLoss(nn.Module):
    """
    DINOv3 Perceptual Loss with cached target embeddings.

    If cached target embeddings (dino_emb) are provided in the batch,
    only the pred path is computed at runtime. Otherwise, both paths
    are computed (fallback to non-cached mode).

    Args:
        dino_model_path: Path to DINOv3 model
        vae_path: Path to VAE model
        dino_image_size: DINOv3 input resolution
        feature_layers: "last" or "multi"
    """

    def __init__(
        self,
        dino_model_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        vae_path: str = "",
        dino_image_size: int = 512,
        feature_layers: str = "last",
        feature_mode: str = "patch",  # "patch" | "cls" | "both"
    ):
        super().__init__()
        self.dino_image_size = dino_image_size
        self.feature_layers = feature_layers
        self.feature_mode = feature_mode
        self._dino_loaded = False
        self._vae_loaded = False

        self._dino_model_path = dino_model_path
        self._vae_path = vae_path
        self.dino: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None

        mode_desc = {"patch": "逐区域感知", "cls": "全局美学", "both": "逐区域+全局"}
        logger.info(f"[DinoPerceptualLoss] 初始化 (cached mode)")
        logger.info(f"  DINOv3: {dino_model_path}")
        logger.info(f"  VAE: {vae_path}")
        logger.info(f"  分辨率: {dino_image_size}")
        logger.info(f"  特征模式: {feature_mode} ({mode_desc.get(feature_mode, '?')})")
        logger.info(f"  特征层: {feature_layers}")

    def _load_models(self, device: torch.device, dtype: torch.dtype):
        """Lazy-load DINOv3 and VAE on first forward pass."""
        if not self._vae_loaded:
            import os
            from diffusers import AutoencoderKL

            logger.info(f"[DinoPerceptualLoss] 加载 VAE: {self._vae_path}")
            if os.path.isdir(self._vae_path):
                self.vae = AutoencoderKL.from_pretrained(
                    self._vae_path, torch_dtype=dtype,
                )
            else:
                self.vae = AutoencoderKL.from_single_file(
                    self._vae_path, torch_dtype=dtype,
                )
            self.vae.to(device)
            self.vae.eval()
            self.vae.requires_grad_(False)

            vram_mb = sum(p.numel() * p.element_size()
                         for p in self.vae.parameters()) / 1024 / 1024
            logger.info(f"[DinoPerceptualLoss] VAE 已加载: ~{vram_mb:.0f}MB")
            self._vae_loaded = True

        if not self._dino_loaded:
            from transformers import DINOv3ViTModel

            logger.info(f"[DinoPerceptualLoss] 加载 DINOv3: {self._dino_model_path}")
            self.dino = DINOv3ViTModel.from_pretrained(
                self._dino_model_path, torch_dtype=torch.float32,
            )
            self.dino.to(device)
            self.dino.eval()
            self.dino.requires_grad_(False)

            params_m = sum(p.numel() for p in self.dino.parameters()) / 1e6
            vram_mb = sum(p.numel() * p.element_size()
                         for p in self.dino.parameters()) / 1024 / 1024
            logger.info(f"[DinoPerceptualLoss] DINOv3 已加载: "
                        f"{params_m:.1f}M params, ~{vram_mb:.0f}MB")
            self._dino_loaded = True

    @torch.no_grad()
    def _extract_pred_embedding(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv3 patch embedding from predicted x₀ latent.

        Pipeline: latent → downsample → VAE decode → DINOv3 → patch emb

        Args:
            pred_x0: (B, 16, H, W) predicted clean latent

        Returns:
            pred_emb: (B, P, D) patch embeddings
        """
        vae_dtype = self.vae.dtype
        patch_size = 16
        dino_size = self.dino_image_size
        num_prefix = 1 + getattr(self.dino.config, "num_register_tokens", 0)

        B, C, H, W = pred_x0.shape

        # Downsample latent preserving aspect ratio
        scale = dino_size / (max(H, W) * 8)
        lat_h = max(patch_size // 8, int(round(H * scale / 2)) * 2)
        lat_w = max(patch_size // 8, int(round(W * scale / 2)) * 2)

        pred_small = F.interpolate(
            pred_x0, size=(lat_h, lat_w),
            mode="bilinear", align_corners=False,
        )

        # VAE decode per-sample
        rgb_list = []
        for i in range(B):
            p = self.vae.decode(pred_small[i:i+1].to(vae_dtype), return_dict=False)[0]
            rgb_list.append((p / 2 + 0.5).clamp(0, 1))
        pred_rgb = torch.cat(rgb_list, dim=0).float()

        # Align to patch_size
        ph, pw = pred_rgb.shape[-2], pred_rgb.shape[-1]
        tgt_h = (ph // patch_size) * patch_size
        tgt_w = (pw // patch_size) * patch_size
        if tgt_h != ph or tgt_w != pw:
            pred_rgb = F.interpolate(pred_rgb, size=(tgt_h, tgt_w),
                                     mode="bilinear", align_corners=False)

        # DINOv3 extract
        if self.feature_layers == "multi":
            out = self.dino(pixel_values=pred_rgb, output_hidden_states=True)
            hs = out.hidden_states
            feat = torch.stack(hs[max(0, len(hs)-4):], dim=0).mean(0)
        else:
            out = self.dino(pixel_values=pred_rgb)
            feat = out.last_hidden_state

        cls_token = feat[:, 0:1, :]  # (B, 1, D)
        patch_tokens = feat[:, num_prefix:, :]  # (B, P, D)

        if self.feature_mode == "cls":
            return cls_token
        elif self.feature_mode == "both":
            return torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1+P, D)
        else:  # "patch"
            return patch_tokens

    @torch.no_grad()
    def _extract_target_embedding(self, target_x0: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3 embedding from target x₀ (fallback when no cache)."""
        return self._extract_pred_embedding(target_x0)

    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
        cached_dino_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DINOv3 perceptual loss.

        If cached_dino_emb is provided, uses it as target (no target-side
        VAE decode or DINOv3 needed). Otherwise falls back to computing both.

        Args:
            pred_v: Model predicted velocity (B, C, H, W)
            target_v: Target velocity (B, C, H, W)
            noisy_latents: x_t (B, C, H, W)
            timesteps: sigma * 1000 (B,)
            num_train_timesteps: Total timesteps (1000)
            cached_dino_emb: Pre-cached target DINOv3 embedding (B, P, D)

        Returns:
            Scalar perceptual loss
        """
        original_dtype = pred_v.dtype
        device = pred_v.device

        self._load_models(device, original_dtype)

        # 1. Derive x₀ from v-prediction
        sigmas = timesteps.float() / num_train_timesteps
        sigma_bc = sigmas.view(-1, 1, 1, 1)
        noisy_fp32 = noisy_latents.float()

        pred_x0 = noisy_fp32 - sigma_bc * pred_v.float()
        target_x0 = (noisy_fp32 - sigma_bc * target_v.float()).detach()

        # 2. Extract DINOv3 features
        pred_emb = self._extract_pred_embedding(pred_x0.detach())

        if cached_dino_emb is not None:
            # Use pre-cached target embedding (from GT image)
            target_emb = cached_dino_emb.to(device).float()
            # Cached emb may have different token count if resolution differs
            # Align by truncating to minimum
            min_tokens = min(pred_emb.shape[1], target_emb.shape[1])
            pred_emb = pred_emb[:, :min_tokens, :]
            target_emb = target_emb[:, :min_tokens, :]
        else:
            # Fallback: compute target embedding at runtime
            target_emb = self._extract_target_embedding(target_x0)

        # 3. Cosine distance over tokens (works for patch, cls, or both)
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)  # (B, T)
        dino_dist = (1.0 - cos_sim).mean(dim=1)  # (B,)

        # 4. Differentiable latent cosine (gradient carrier)
        pred_flat = pred_x0.view(pred_x0.shape[0], -1)
        target_flat = target_x0.view(target_x0.shape[0], -1)
        latent_cos = F.cosine_similarity(pred_flat, target_flat, dim=1)
        latent_dist = 1.0 - latent_cos  # (B,)

        # 5. STE bridge
        loss = (dino_dist * latent_dist / (latent_dist.detach() + 1e-8)).mean()

        return loss.to(original_dtype)
