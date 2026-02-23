# -*- coding: utf-8 -*-
"""
Position-Aware Split LoRA for Z-Image.

Supports selective LoRA application to image or text tokens.
Mirrors the exact functionality from musubi-tuner/networks/lora_zimage_split.py.

Usage:
    network_module = "zimage_trainer.networks.lora_split"
    network_args = ["split_mode=img_only"]  # or "txt_only", "both", "context_only"
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)

# Split modes
SPLIT_MODE_BOTH = "both"            # Train all (default)
SPLIT_MODE_IMG_ONLY = "img_only"    # Train image stream only
SPLIT_MODE_TXT_ONLY = "txt_only"    # Train text stream only
SPLIT_MODE_CONTEXT = "context_only" # Train context_refiner only


class PositionAwareLoRAModule(nn.Module):
    """
    Position-aware LoRA module.
    Can selectively apply LoRA to image or text tokens.
    """
    
    def __init__(
        self,
        lora_name: str,
        original_module: nn.Module,
        multiplier: float = 1.0,
        rank: int = 16,
        alpha: float = 32.0,
        split_mode: str = SPLIT_MODE_BOTH,
    ):
        super().__init__()
        
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha
        self.split_mode = split_mode
        self.scale = alpha / rank
        
        # Get dimensions
        if isinstance(original_module, nn.Linear):
            in_features = original_module.in_features
            out_features = original_module.out_features
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")
        
        # LoRA weights
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        
        # Store original module (use object.__setattr__ to avoid registering as submodule)
        object.__setattr__(self, 'original_module', original_module)
        
        # Position info (set during forward)
        self.img_seq_len: Optional[int] = None
        self.txt_seq_len: Optional[int] = None
    
    def set_position_info(self, img_seq_len: int, txt_seq_len: int):
        """Set position info for current batch."""
        self.img_seq_len = img_seq_len
        self.txt_seq_len = txt_seq_len
    
    def clear_position_info(self):
        """Clear position info."""
        self.img_seq_len = None
        self.txt_seq_len = None
    
    def state_dict(self, *args, **kwargs):
        """Return only LoRA weights, not original_module."""
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if 'original_module' not in k}
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Ignore original_module when loading."""
        filtered = {k: v for k, v in state_dict.items() if 'original_module' not in k}
        super()._load_from_state_dict(filtered, prefix, *args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with position-aware LoRA.
        
        Args:
            x: (batch, seq, dim) or (batch, dim)
        """
        # If no position info or 2D input, apply LoRA to all
        if self.img_seq_len is None or self.txt_seq_len is None or x.dim() == 2:
            lora_output = self.lora_up(self.lora_down(x)) * self.scale * self.multiplier
            return lora_output
        
        batch, seq, dim = x.shape
        
        if self.split_mode == SPLIT_MODE_BOTH:
            return self.lora_up(self.lora_down(x)) * self.scale * self.multiplier
        
        elif self.split_mode == SPLIT_MODE_IMG_ONLY:
            output = torch.zeros(batch, seq, self.lora_up.out_features, device=x.device, dtype=x.dtype)
            if self.img_seq_len > 0:
                img_x = x[:, :self.img_seq_len, :]
                lora_img = self.lora_up(self.lora_down(img_x)) * self.scale * self.multiplier
                output[:, :self.img_seq_len, :] = lora_img
            return output
        
        elif self.split_mode == SPLIT_MODE_TXT_ONLY:
            output = torch.zeros(batch, seq, self.lora_up.out_features, device=x.device, dtype=x.dtype)
            if self.txt_seq_len > 0:
                start = self.img_seq_len
                end = self.img_seq_len + self.txt_seq_len
                txt_x = x[:, start:end, :]
                lora_txt = self.lora_up(self.lora_down(txt_x)) * self.scale * self.multiplier
                output[:, start:end, :] = lora_txt
            return output
        
        else:
            # context_only or other modes
            return self.lora_up(self.lora_down(x)) * self.scale * self.multiplier


class LoRAInjectedLinear(nn.Module):
    """Wrapper that injects LoRA into original Linear layer."""
    
    def __init__(self, original_linear: nn.Linear, lora_module: PositionAwareLoRAModule):
        super().__init__()
        self.original = original_linear
        self.lora = lora_module
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Set position info from class variable
        self.lora.img_seq_len = LoRANetwork._current_img_seq_len
        self.lora.txt_seq_len = LoRANetwork._current_txt_seq_len
        return self.original(x) + self.lora(x)


class LoRANetwork(nn.Module):
    """
    Z-Image Split LoRA Network.
    
    Compatible with musubi-tuner's LoRA interface.
    """
    
    # Class variables for position info (shared by all modules)
    _current_img_seq_len: Optional[int] = None
    _current_txt_seq_len: Optional[int] = None
    
    UNET_TARGET_REPLACE_MODULE = ["ZImageTransformerBlock", "ZImageAttention", "SwiGLU"]
    LORA_PREFIX_UNET = "lora_unet"
    
    def __init__(
        self,
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 16,
        alpha: float = 32.0,
        split_mode: str = SPLIT_MODE_BOTH,
        target_layers: Optional[List[int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.split_mode = split_mode
        self.target_layers = target_layers
        self.exclude_patterns = exclude_patterns or []
        
        # Add default exclude patterns
        self.exclude_patterns.extend([
            r".*(_mod_).*",
            r".*embedder.*",
            r".*pad_token.*",
        ])
        
        logger.info(f"Creating Z-Image Split LoRA: dim={lora_dim}, alpha={alpha}, mode={split_mode}")
        logger.info(f"Target layers: {target_layers if target_layers else 'all'}")
        logger.info(f"Exclude patterns: {self.exclude_patterns}")
        
        # Store LoRA modules
        self.unet_loras: List[PositionAwareLoRAModule] = []
        
        # Create modules
        self._create_modules(unet)
        
        # Register as ModuleList
        self.unet_lora_modules = nn.ModuleList(self.unet_loras)
        
        logger.info(f"Created {len(self.unet_loras)} LoRA modules for DiT")
    
    def _should_create_lora(self, name: str) -> bool:
        """Determine if LoRA should be created for this module."""
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, name):
                return False
        
        # context_only mode
        if self.split_mode == SPLIT_MODE_CONTEXT:
            return "context_refiner" in name
        
        # Check layer numbers
        if self.target_layers is not None and "layers." in name:
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in self.target_layers:
                    return False
        
        return True
    
    def _create_modules(self, unet: nn.Module):
        """Create LoRA modules for target layers."""
        for name, module in unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            if not self._should_create_lora(name):
                continue
            
            # Check if in target module types
            parent_name = ".".join(name.split(".")[:-1])
            is_target = False
            for target in self.UNET_TARGET_REPLACE_MODULE:
                if target.lower() in parent_name.lower() or target.lower() in name.lower():
                    is_target = True
                    break
            
            # Also target attention and feed_forward
            if "attention" in name or "feed_forward" in name:
                is_target = True
            
            if not is_target:
                continue
            
            lora_name = f"{self.LORA_PREFIX_UNET}_{name}".replace(".", "_")
            
            # Determine split mode for this module
            if "context_refiner" in name:
                module_split_mode = SPLIT_MODE_TXT_ONLY
            else:
                module_split_mode = self.split_mode
            
            lora = PositionAwareLoRAModule(
                lora_name=lora_name,
                original_module=module,
                multiplier=self.multiplier,
                rank=self.lora_dim,
                alpha=self.alpha,
                split_mode=module_split_mode,
            )
            
            self.unet_loras.append(lora)
            
            # Inject into original module
            self._inject_lora(unet, name, lora)
    
    def _inject_lora(self, model: nn.Module, target_name: str, lora: PositionAwareLoRAModule):
        """Inject LoRA into original module."""
        parts = target_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        original = getattr(parent, parts[-1])
        injected = LoRAInjectedLinear(original, lora)
        setattr(parent, parts[-1], injected)
    
    @classmethod
    def set_position_info(cls, img_seq_len: int, txt_seq_len: int):
        """Set position info for all modules."""
        cls._current_img_seq_len = img_seq_len
        cls._current_txt_seq_len = txt_seq_len
    
    @classmethod
    def clear_position_info(cls):
        """Clear position info."""
        cls._current_img_seq_len = None
        cls._current_txt_seq_len = None
    
    def apply_to(self, unet: nn.Module):
        """Apply LoRA to model (already done in __init__)."""
        pass  # Already injected during creation
    
    def prepare_optimizer_params(
        self,
        unet_lr: float = 1e-4,
        **kwargs,
    ) -> Tuple[List[Dict], List[str]]:
        """Prepare parameters for optimizer."""
        params = []
        for lora in self.unet_loras:
            params.extend([
                {"params": lora.lora_down.weight, "lr": unet_lr},
                {"params": lora.lora_up.weight, "lr": unet_lr},
            ])
        
        descriptions = [f"SplitLoRA ({self.split_mode}): dim={self.lora_dim}, alpha={self.alpha}"]
        return params, descriptions
    
    def save_weights(
        self,
        path: str,
        dtype: torch.dtype = torch.bfloat16,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Save LoRA weights to safetensors file."""
        state_dict = {}
        
        for lora in self.unet_loras:
            prefix = lora.lora_name
            state_dict[f"{prefix}.lora_down.weight"] = lora.lora_down.weight.to(dtype)
            state_dict[f"{prefix}.lora_up.weight"] = lora.lora_up.weight.to(dtype)
            state_dict[f"{prefix}.alpha"] = torch.tensor(lora.alpha)
        
        if metadata is None:
            metadata = {}
        metadata.update({
            "split_mode": self.split_mode,
            "lora_dim": str(self.lora_dim),
            "alpha": str(self.alpha),
            "num_modules": str(len(self.unet_loras)),
        })
        
        save_file(state_dict, path, metadata=metadata)
        logger.info(f"Saved Split LoRA to {path}")
    
    def load_weights(self, path: str):
        """Load LoRA weights from safetensors file."""
        state_dict = load_file(path)
        
        for lora in self.unet_loras:
            prefix = lora.lora_name
            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            
            if down_key in state_dict:
                lora.lora_down.weight.data.copy_(state_dict[down_key])
            if up_key in state_dict:
                lora.lora_up.weight.data.copy_(state_dict[up_key])
        
        logger.info(f"Loaded Split LoRA from {path}")
    
    # Methods for compatibility with musubi-tuner
    def prepare_grad_etc(self, unet):
        """Prepare gradients (no-op for compatibility)."""
        pass
    
    def on_epoch_start(self, text_encoder=None, unet=None):
        """Called at epoch start (no-op for compatibility)."""
        pass
    
    def on_step_start(self):
        """Called at step start (no-op for compatibility)."""
        pass
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        for lora in self.unet_loras:
            params.extend([lora.lora_down.weight, lora.lora_up.weight])
        return params
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing (no-op for compatibility)."""
        pass


def create_network(
    multiplier: float = 1.0,
    network_dim: int = 16,
    network_alpha: float = 32.0,
    vae=None,
    text_encoders=None,
    unet: nn.Module = None,
    neuron_dropout: float = None,
    split_mode: str = SPLIT_MODE_BOTH,
    **kwargs,
) -> LoRANetwork:
    """Factory function to create Split LoRA network."""
    if unet is None:
        raise ValueError("unet is required")
    
    return LoRANetwork(
        unet=unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        split_mode=split_mode,
        **kwargs,
    )


# Alias for musubi-tuner compatibility
def create_arch_network(
    multiplier: float,
    network_dim: int,
    network_alpha: float,
    vae,
    text_encoders,
    unet,
    neuron_dropout: float = None,
    **kwargs,
) -> LoRANetwork:
    """Architecture-specific network factory (musubi-tuner compatibility)."""
    split_mode = kwargs.pop("split_mode", SPLIT_MODE_BOTH)
    return create_network(
        multiplier=multiplier,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        neuron_dropout=neuron_dropout,
        split_mode=split_mode,
        **kwargs,
    )
