# -*- coding: utf-8 -*-
"""
GPU Memory Management - Model-agnostic

Provides:
- Memory cleanup utilities
- Module offloading for low VRAM training
- Memory usage reporting

Consolidates from:
- flow_matching.clean_memory / get_memory_usage
- training_utils.ModuleOffloader
- memory_optimizer.*
"""

import gc
import logging
from typing import Union, Dict, Optional
from contextlib import contextmanager

import torch
from torch import nn

logger = logging.getLogger(__name__)


def clean_memory(device: Optional[torch.device] = None) -> None:
    """Clean up GPU memory by running garbage collection and emptying CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage in GB.

    Returns:
        Dict with keys: allocated, reserved, free, total
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_mem / 1024**3
    free = total - allocated

    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "free": round(free, 2),
        "total": round(total, 2),
    }


class ModuleOffloader:
    """
    Module offloading utility for low VRAM training.

    Moves modules between CPU and GPU to save memory.
    Only keeps the active module on GPU during forward/backward.

    Usage:
        offloader = ModuleOffloader(device="cuda")
        offloader.register("vae", vae_model)
        offloader.register("text_encoder", text_encoder)

        with offloader.use("vae"):
            output = vae(input)
        # vae is automatically moved back to CPU
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        offload_device: Union[str, torch.device] = "cpu",
        verbose: bool = False,
    ):
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.verbose = verbose
        self._modules: Dict[str, nn.Module] = {}

    def register(self, name: str, module: nn.Module) -> None:
        """Register a module for offloading."""
        self._modules[name] = module

    def offload(self, name: str) -> None:
        """Move module to CPU."""
        if name in self._modules:
            self._modules[name].to(self.offload_device)
            if self.verbose:
                logger.debug(f"Offloaded {name} to {self.offload_device}")
            clean_memory(self.device)

    def load(self, name: str) -> None:
        """Move module to GPU."""
        if name in self._modules:
            self._modules[name].to(self.device)
            if self.verbose:
                logger.debug(f"Loaded {name} to {self.device}")

    def offload_all(self) -> None:
        """Offload all registered modules."""
        for name in self._modules:
            self.offload(name)

    def load_all(self) -> None:
        """Load all registered modules."""
        for name in self._modules:
            self.load(name)

    @contextmanager
    def use(self, name: str):
        """Context manager to temporarily load a module."""
        self.load(name)
        try:
            yield self._modules[name]
        finally:
            self.offload(name)
