"""
STE (Straight-Through Estimator) patch for Z-Image Transformer tanh gates.

Replaces tanh gating with STE-tanh during training:
  - Forward: gate = tanh(x)  (identical to original)
  - Backward: ∂gate/∂x = 1.0 (identity, not 1-tanh²)

This prevents gradient attenuation through the 60 tanh gates
(2 per block × 30 blocks) in ZImageTransformerBlock.

Usage:
    from shared.patches.ste_tanh import apply_ste_tanh, remove_ste_tanh

    # Before training loop:
    apply_ste_tanh(transformer)

    # After training (optional, for inference):
    remove_ste_tanh(transformer)
"""

import torch
import logging

logger = logging.getLogger(__name__)


class STETanh(torch.autograd.Function):
    """
    Straight-Through Estimator for tanh.

    Forward: y = tanh(x)
    Backward: dy/dx = 1.0 (not 1 - tanh²(x))

    Mathematical justification:
      tanh serves as a gate bounding output to [-1, 1].
      During backprop, the derivative 1-tanh²(x) ∈ [0, 1] attenuates gradients.
      STE preserves forward semantics while allowing full gradient flow.
    """

    @staticmethod
    def forward(ctx, x):
        return x.tanh()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Identity: dy/dx = 1.0


def ste_tanh(x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for tensor.tanh() with STE gradient."""
    return STETanh.apply(x)


def _patched_forward(self, x, attn_mask, freqs_cis,
                     adaln_input=None, noise_mask=None,
                     adaln_noisy=None, adaln_clean=None):
    """
    Patched ZImageTransformerBlock.forward that uses STE-tanh for gates.
    Only difference from original: .tanh() → ste_tanh()
    """
    if self.modulation:
        seq_len = x.shape[1]

        if noise_mask is not None:
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)

            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

            # STE-tanh instead of .tanh()
            gate_msa_noisy, gate_mlp_noisy = ste_tanh(gate_msa_noisy), ste_tanh(gate_mlp_noisy)
            gate_msa_clean, gate_mlp_clean = ste_tanh(gate_msa_clean), ste_tanh(gate_mlp_clean)

            scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
            scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

            from diffusers.models.transformers.transformer_z_image import select_per_token
            scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
            scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
            gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
            gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
        else:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            # STE-tanh instead of .tanh()
            gate_msa, gate_mlp = ste_tanh(gate_msa), ste_tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    else:
        attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attention_norm2(attn_out)

        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    return x


# Store original forwards for restoration
_original_forwards = {}


def apply_ste_tanh(transformer):
    """
    Monkey-patch all ZImageTransformerBlock instances in the transformer
    to use STE-tanh gating during training.

    Args:
        transformer: ZImageTransformer2DModel instance
    """
    import types
    count = 0

    for name, module in transformer.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "ZImageTransformerBlock":
            # Store original forward
            _original_forwards[id(module)] = module.forward
            # Replace with patched version
            module.forward = types.MethodType(_patched_forward, module)
            count += 1

    logger.info(f"[STE_TANH] Patched {count} ZImageTransformerBlock instances with STE-tanh gates")
    print(f"[STE_TANH] Patched {count} blocks — forward: tanh(x), backward: identity", flush=True)
    return count


def remove_ste_tanh(transformer):
    """
    Restore original forward methods (for inference or saving).

    Args:
        transformer: ZImageTransformer2DModel instance
    """
    count = 0
    for name, module in transformer.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "ZImageTransformerBlock":
            mod_id = id(module)
            if mod_id in _original_forwards:
                module.forward = _original_forwards[mod_id]
                del _original_forwards[mod_id]
                count += 1

    logger.info(f"[STE_TANH] Restored {count} ZImageTransformerBlock instances to original tanh")
    print(f"[STE_TANH] Restored {count} blocks to standard tanh", flush=True)
    return count
