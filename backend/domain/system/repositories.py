# -*- coding: utf-8 -*-
"""
System Domain - Repository Interfaces

Abstract interfaces for system operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .entities import (
    GPUInfo, SystemInfo, ModelInfo, ModelStatus,
    DownloadProgress,
)


class IGPUMonitor(ABC):
    """Interface for GPU monitoring."""

    @abstractmethod
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get information for all GPU devices."""


class ISystemInfoProvider(ABC):
    """Interface for system information."""

    @abstractmethod
    def get_system_info(self) -> SystemInfo:
        """Get system information (OS, Python, Torch, CUDA)."""


class IModelManager(ABC):
    """Interface for model download and verification."""

    @abstractmethod
    def get_model_status(self, model_type: str = "zimage") -> ModelInfo:
        """Check local model status."""

    @abstractmethod
    def verify_integrity(self, model_type: str = "zimage") -> ModelInfo:
        """Deep verify model integrity."""

    @abstractmethod
    def start_download(self, model_type: str = "zimage") -> bool:
        """Start model download. Returns success."""

    @abstractmethod
    def cancel_download(self) -> bool:
        """Cancel active download. Returns success."""

    @abstractmethod
    def get_download_progress(self) -> Optional[DownloadProgress]:
        """Get current download progress."""

    @abstractmethod
    def list_supported_models(self) -> List[str]:
        """List supported model type keys."""

    @abstractmethod
    def get_model_spec(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get model spec info for a given type."""
