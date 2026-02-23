# LoRA module for Z-Image
# Replicated from lora_qwen_image.py

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora

Z_IMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """
    Create LoRA network for Z-Image.
    
    Aligned with ai-toolkit configuration:
    - Only train 'layers' (30 main transformer blocks)
    - Include adaLN_modulation (ai-toolkit includes it)
    - Exclude noise_refiner and context_refiner
    
    Args:
        multiplier: LoRA multiplier
        network_dim: LoRA rank/dimension
        network_alpha: LoRA alpha for scaling
        vae: VAE model (not modified)
        text_encoders: Text encoder models (not modified)
        unet: Z-Image transformer (LoRA applied here)
        neuron_dropout: Optional dropout for LoRA
        **kwargs: Additional arguments
        
    Returns:
        LoRA network instance
    """
    # Get existing patterns from kwargs
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is None:
        include_patterns = []
    elif isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)
    
    # ai-toolkit style: only train 'layers', exclude noise_refiner
    # This matches: layers.0.*, layers.1.*, ..., layers.29.*
    # But excludes: noise_refiner.*, context_refiner.*
    include_patterns.append(r".*\.layers\.\d+\..*")
    
    kwargs["exclude_patterns"] = exclude_patterns
    kwargs["include_patterns"] = include_patterns
    
    logger.info(f"Creating Z-Image LoRA network (ai-toolkit compatible)")
    logger.info(f"  dim={network_dim}, alpha={network_alpha}, multiplier={multiplier}")
    logger.info(f"  Target modules: {Z_IMAGE_TARGET_REPLACE_MODULES}")
    logger.info(f"  Include patterns: {include_patterns}")
    logger.info(f"  Exclude patterns: {exclude_patterns}")
    
    return lora.create_network(
        Z_IMAGE_TARGET_REPLACE_MODULES,
        "lora_unet",  # LoRA key prefix
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    """
    Create LoRA network from pre-trained weights.
    
    Args:
        multiplier: LoRA multiplier
        weights_sd: State dict with LoRA weights
        text_encoders: Text encoder models (optional)
        unet: Z-Image transformer (optional)
        for_inference: Whether loading for inference
        **kwargs: Additional arguments
        
    Returns:
        LoRA network instance loaded from weights
    """
    logger.info(f"Loading Z-Image LoRA network from weights: multiplier={multiplier}, for_inference={for_inference}")
    
    return lora.create_network_from_weights(
        Z_IMAGE_TARGET_REPLACE_MODULES, 
        multiplier, 
        weights_sd, 
        text_encoders, 
        unet, 
        for_inference, 
        **kwargs
    )
