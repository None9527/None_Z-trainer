# -*- coding: utf-8 -*-
"""
Generation Domain - Entities

Core entities for image generation and model management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class GenerationStatus(Enum):
    """Generation task status."""
    IDLE = "idle"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Image generation parameters."""
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 10
    guidance_scale: float = 3.5
    seed: int = -1
    num_images: int = 1
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    transformer_path: Optional[str] = None


@dataclass
class GenerationResult:
    """Generated image result."""
    timestamp: str = ""
    image_path: str = ""
    prompt: str = ""
    seed: int = 0
    width: int = 0
    height: int = 0
    steps: int = 0
    guidance_scale: float = 0.0
    lora_path: Optional[str] = None
    lora_scale: float = 1.0


@dataclass
class LoRAInfo:
    """LoRA model information."""
    name: str
    path: str
    size_bytes: int = 0
    created_at: str = ""
    dim: int = 0
    alpha: float = 0.0


@dataclass
class TransformerInfo:
    """Transformer (finetune) model information."""
    name: str
    path: str
    size_bytes: int = 0
    created_at: str = ""
