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
class LoRAConfig:
    """Single LoRA configuration for inference."""
    path: str
    scale: float = 1.0

    def to_dict(self) -> dict:
        return {"path": self.path, "scale": self.scale}


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
    lora_configs: List[LoRAConfig] = field(default_factory=list)
    transformer_path: Optional[str] = None

    # Compat properties for single-lora code paths
    @property
    def lora_path(self) -> Optional[str]:
        return self.lora_configs[0].path if self.lora_configs else None

    @property
    def lora_scale(self) -> float:
        return self.lora_configs[0].scale if self.lora_configs else 1.0


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
    lora_configs: List[LoRAConfig] = field(default_factory=list)

    # Compat properties
    @property
    def lora_path(self) -> Optional[str]:
        return self.lora_configs[0].path if self.lora_configs else None

    @property
    def lora_scale(self) -> float:
        return self.lora_configs[0].scale if self.lora_configs else 1.0


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
