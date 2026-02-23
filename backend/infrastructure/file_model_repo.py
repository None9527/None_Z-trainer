# -*- coding: utf-8 -*-
"""
File Model Repository - Infrastructure Implementation

Manages LoRA and Transformer model files on disk.
Ported from webui-vue/api/routers/generation.py.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ..domain.generation.repositories import IModelRepository
from ..domain.generation.entities import LoRAInfo, TransformerInfo

logger = logging.getLogger(__name__)


class FileModelRepository(IModelRepository):
    """Filesystem-based model repository for LoRA and Transformer files."""

    def __init__(self):
        from .config import OUTPUT_PATH, LORA_PATH, FINETUNE_PATH
        self._output_path = OUTPUT_PATH
        self._lora_path = LORA_PATH
        self._finetune_path = FINETUNE_PATH

    def list_loras(self) -> List[LoRAInfo]:
        """Scan OUTPUT_PATH recursively for .safetensors LoRA files.

        Training saves LoRA to OUTPUT_PATH/{output_name}/, so we scan
        the entire OUTPUT_PATH tree, excluding finetune/ and logs/ dirs.
        """
        loras = []
        if not self._output_path.exists():
            return loras

        # Directories to skip (finetune models, tensorboard logs)
        skip_dirs = {"logs"}

        for f in sorted(self._output_path.rglob("*.safetensors")):
            # Skip files under finetune path
            try:
                f.relative_to(self._finetune_path)
                continue
            except ValueError:
                pass

            # Skip files under "logs" directories
            if any(part in skip_dirs for part in f.parts):
                continue

            stat = f.stat()
            loras.append(LoRAInfo(
                name=f.stem,
                path=str(f),
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))

        return loras

    def list_transformers(self) -> List[TransformerInfo]:
        """Scan FINETUNE_PATH for transformer model files."""
        transformers = []
        if not self._finetune_path.exists():
            return transformers

        # Each transformer is either a .safetensors file or a directory with one inside
        for item in sorted(self._finetune_path.iterdir()):
            if item.is_file() and item.suffix == '.safetensors':
                stat = item.stat()
                transformers.append(TransformerInfo(
                    name=item.stem,
                    path=str(item),
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                ))
            elif item.is_dir():
                # Look for .safetensors inside directory
                safetensors = list(item.glob("*.safetensors"))
                if safetensors:
                    st_file = safetensors[0]
                    stat = st_file.stat()
                    transformers.append(TransformerInfo(
                        name=item.name,
                        path=str(st_file),
                        size_bytes=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    ))

        return transformers

    def delete_model(self, path: str) -> bool:
        """Delete a model file. Only allows deletion from known output directories."""
        target = Path(path)

        # Security: only allow deletion from output/lora/finetune paths
        allowed = False
        for base in [self._output_path, self._lora_path, self._finetune_path]:
            try:
                target.relative_to(base)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            logger.warning(f"Refusing to delete file outside output dirs: {path}")
            return False

        if target.exists():
            target.unlink()
            logger.info(f"Deleted model: {path}")
            return True

        return False

    def get_model_path(self, name: str, model_type: str = "lora") -> Optional[str]:
        """Find model path by name."""
        if model_type == "lora":
            # Search OUTPUT_PATH recursively (LoRA can be in any subdirectory)
            for f in self._output_path.rglob(f"{name}.safetensors"):
                return str(f)
        elif model_type == "transformer":
            # Check file first, then directory
            target = self._finetune_path / f"{name}.safetensors"
            if target.exists():
                return str(target)
            target_dir = self._finetune_path / name
            if target_dir.is_dir():
                safetensors = list(target_dir.glob("*.safetensors"))
                if safetensors:
                    return str(safetensors[0])

        return None
