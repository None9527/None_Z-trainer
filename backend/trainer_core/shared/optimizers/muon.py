# -*- coding: utf-8 -*-
"""
Muon Optimizer — Momentum + Orthogonalization

Uses Newton-Schulz orthogonalization on gradient momentum to produce
scale-invariant updates. This makes LoRA training immune to RMSNorm
gradient compression in architectures like Z-Image Transformer.

For 2D weight matrices: Muon (orthogonalized momentum)
For 1D parameters (bias, norm γ): internal AdamW fallback

FP8 variant stores momentum buffer in float8_e4m3fn with per-tensor
scaling to reduce memory by ~4x vs fp32.

Reference: Keller Jordan, NanoGPT speedrun (2024)

Usage:
    optimizer = Muon(model.parameters(), lr=0.02)
    optimizer = MuonFP8(model.parameters(), lr=0.02)  # FP8 momentum
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Dict, Any


# ============================================================================
# Newton-Schulz Orthogonalization
# ============================================================================

@torch.no_grad()
def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate polar decomposition via Newton-Schulz iteration.
    Maps G → U V^T (orthogonal factor of SVD).

    Coefficients from: Björck & Bowie (1971), optimized for 5 iterations.
    Input must be 2D. Output has same shape, all singular values ≈ 1.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Normalize to avoid divergence
    X = G / (G.norm() + 1e-7)

    # Work with the smaller dimension for efficiency
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    if transposed:
        X = X.T

    return X


# ============================================================================
# Muon Optimizer (FP32 momentum)
# ============================================================================

class Muon(Optimizer):
    """
    Muon optimizer: Momentum + Orthogonalization.

    For 2D parameters (Linear weights, LoRA matrices):
    - Applies momentum to gradient
    - Orthogonalizes via Newton-Schulz (removes scale, preserves direction)
    - Update is scale-invariant → immune to RMSNorm gradient compression

    For 1D parameters (bias, norm scale):
    - Falls back to AdamW internally

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate for Muon (2D params). Typical: 0.02
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        adamw_lr: Learning rate for 1D params via AdamW (default: 3e-4)
        adamw_betas: Adam betas for 1D params (default: (0.95, 0.95))
        adamw_wd: Weight decay for 1D params (default: 0.0)
        weight_decay: Weight decay for 2D params (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.95, 0.95),
        adamw_wd: float = 0.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        params_list = list(params)
        super().__init__(params_list, defaults)

        # Separate 1D params for AdamW fallback
        adam_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    adam_params.append(p)

        self._adam = (
            torch.optim.AdamW(
                adam_params,
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=adamw_wd,
            )
            if adam_params
            else None
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1D params → skip (handled by internal Adam)
                if p.ndim < 2:
                    continue

                grad = p.grad.to(torch.float32)
                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                # Weight decay (decoupled)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Momentum update
                buf.mul_(mu).add_(grad)

                # Nesterov lookahead
                if nesterov:
                    update = grad.add(buf, alpha=mu)
                else:
                    update = buf.clone()

                # Orthogonalize — this is the key step
                update_orth = _newton_schulz5(
                    update.view(update.size(0), -1), steps=ns_steps
                ).view_as(update)

                # Apply update
                p.data.add_(update_orth.to(p.dtype), alpha=-lr)

        # Handle 1D params with Adam
        if self._adam is not None:
            self._adam.step()
            self._adam.zero_grad()

        return loss


# ============================================================================
# MuonFP8 Optimizer (FP8 momentum storage)
# ============================================================================

class MuonFP8(Optimizer):
    """
    Muon optimizer with FP8 momentum storage.

    Same algorithm as Muon, but stores momentum buffer in float8_e4m3fn
    with per-tensor scaling. Reduces optimizer memory by ~4x for 2D params.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate for Muon (2D params). Typical: 0.02
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        adamw_lr: Learning rate for 1D params via AdamW (default: 3e-4)
        adamw_betas: Adam betas for 1D params (default: (0.95, 0.95))
        adamw_wd: Weight decay for 1D params (default: 0.0)
        weight_decay: Weight decay for 2D params (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.95, 0.95),
        adamw_wd: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError(
                "PyTorch float8_e4m3fn not available. "
                f"Requires PyTorch >= 2.1. Current: {torch.__version__}"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        params_list = list(params)
        super().__init__(params_list, defaults)

        self._fp8_dtype = torch.float8_e4m3fn  # max ~448

        # Separate 1D params for AdamW fallback
        adam_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    adam_params.append(p)

        self._adam = (
            torch.optim.AdamW(
                adam_params,
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=adamw_wd,
            )
            if adam_params
            else None
        )

    def _to_fp8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack tensor to FP8 with per-tensor scale."""
        abs_max = tensor.abs().max().clamp(min=1e-12)
        scale = abs_max / 448.0  # e4m3fn max
        fp8 = (tensor / scale).to(self._fp8_dtype)
        return fp8, scale

    def _from_fp8(self, fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Unpack FP8 tensor to fp32."""
        return fp8.to(torch.float32) * scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1D params → skip (handled by internal Adam)
                if p.ndim < 2:
                    continue

                grad = p.grad.to(torch.float32)
                state = self.state[p]

                # Initialize FP8 momentum buffer
                if "momentum_fp8" not in state:
                    state["momentum_fp8"] = torch.zeros_like(p, dtype=self._fp8_dtype)
                    state["momentum_scale"] = torch.tensor(1.0, device=p.device)

                # Unpack FP8 → fp32
                buf = self._from_fp8(state["momentum_fp8"], state["momentum_scale"])

                # Weight decay (decoupled)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Momentum update
                buf.mul_(mu).add_(grad)

                # Nesterov lookahead
                if nesterov:
                    update = grad.add(buf, alpha=mu)
                else:
                    update = buf.clone()

                # Orthogonalize — the key step
                update_orth = _newton_schulz5(
                    update.view(update.size(0), -1), steps=ns_steps
                ).view_as(update)

                # Apply update
                p.data.add_(update_orth.to(p.dtype), alpha=-lr)

                # Pack fp32 → FP8 for storage
                state["momentum_fp8"], state["momentum_scale"] = self._to_fp8(buf)

        # Handle 1D params with Adam
        if self._adam is not None:
            self._adam.step()
            self._adam.zero_grad()

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """State dict with FP8 momentum converted to fp32 for saving."""
        sd = super().state_dict()
        for _, param_state in sd["state"].items():
            if "momentum_fp8" in param_state:
                scale = param_state.pop("momentum_scale", torch.tensor(1.0))
                param_state["momentum_buffer"] = (
                    param_state.pop("momentum_fp8").to(torch.float32) * scale
                )
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict — fp32 momentum will be converted to FP8 on next step."""
        super().load_state_dict(state_dict)
