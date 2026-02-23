# -*- coding: utf-8 -*-
"""
Generation Domain - Repository Interfaces

Abstract interfaces for generation operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

from .entities import (
    GenerationRequest, GenerationResult,
    LoRAInfo, TransformerInfo,
)


class IGenerationPipeline(ABC):
    """Interface for image generation pipeline."""

    @abstractmethod
    def load(self, model_type: str = "zimage", transformer_path: Optional[str] = None) -> None:
        """Load the generation pipeline."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> List[GenerationResult]:
        """Generate images. Returns list of results."""

    @abstractmethod
    def unload(self) -> None:
        """Unload pipeline to free memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if pipeline is currently loaded."""


class IModelRepository(ABC):
    """Repository for LoRA and Transformer model file management."""

    @abstractmethod
    def list_loras(self) -> List[LoRAInfo]:
        """List available LoRA models."""

    @abstractmethod
    def list_transformers(self) -> List[TransformerInfo]:
        """List available transformer (finetune) models."""

    @abstractmethod
    def delete_model(self, path: str) -> bool:
        """Delete a model file. Returns success."""

    @abstractmethod
    def get_model_path(self, name: str, model_type: str = "lora") -> Optional[str]:
        """Get full path for a model by name."""


class IGenerationHistoryRepository(ABC):
    """Repository for generation history."""

    @abstractmethod
    def save_result(self, result: GenerationResult) -> None:
        """Save a generation result."""

    @abstractmethod
    def list_history(self, offset: int = 0, limit: int = 50) -> Tuple[List[GenerationResult], int]:
        """List generation history. Returns (results, total_count)."""

    @abstractmethod
    def delete_history(self, timestamps: List[str]) -> int:
        """Delete history items. Returns count deleted."""
