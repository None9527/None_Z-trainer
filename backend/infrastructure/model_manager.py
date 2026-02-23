# -*- coding: utf-8 -*-
"""
Model Manager - Infrastructure Implementation

Schema-driven model detection with ModelScope download support.
Extensible registry allows adding new models by appending entries.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from ..domain.system.repositories import IModelManager
from ..domain.system.entities import (
    ModelInfo, ModelStatus, ModelSpec,
    DownloadProgress, DownloadStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Rules
# ============================================================================

class ValidationRule:
    """Base class for file validation rules."""

    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class FileRule(ValidationRule):
    """Check a single file exists with optional minimum size."""
    path: str
    required: bool = True
    min_size: int = 0

    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        file_path = base_path / self.path
        exists = file_path.exists()

        # Filter temp / empty files
        if exists:
            suffix = file_path.suffix.lower()
            if suffix in ('.part', '.downloading', '.tmp', '.aria2'):
                exists = False
            elif file_path.stat().st_size == 0:
                exists = False

        detail = {
            "type": "file",
            "path": self.path,
            "exists": exists,
            "valid": exists,
            "required": self.required,
            "message": "exists" if exists else "missing",
        }

        # Size check
        if exists and self.min_size > 0:
            size = file_path.stat().st_size
            if size < self.min_size:
                detail["valid"] = False
                detail["message"] = f"too small ({size} < {self.min_size})"

        if not exists and self.required:
            detail["valid"] = False

        return detail["valid"], {self.path: detail}


@dataclass
class AlternativeRule(ValidationRule):
    """Check that at least one of several alternative files exists."""
    paths: List[str] = field(default_factory=list)
    name: str = ""
    required: bool = True
    min_size: int = 0

    def validate(self, base_path: Path) -> Tuple[bool, Dict[str, Any]]:
        found_any = False
        details: Dict[str, Any] = {}

        for path in self.paths:
            file_path = base_path / path
            exists = file_path.exists()

            if exists:
                suffix = file_path.suffix.lower()
                if suffix in ('.part', '.downloading', '.tmp', '.aria2'):
                    exists = False
                elif file_path.stat().st_size == 0:
                    exists = False

            if exists:
                # Optional size check
                if self.min_size > 0:
                    size = file_path.stat().st_size
                    if size < self.min_size:
                        details[path] = {
                            "type": "alternative",
                            "group": self.name,
                            "path": path,
                            "exists": True,
                            "valid": False,
                            "message": f"too small ({size} < {self.min_size})",
                        }
                        continue

                found_any = True
                details[path] = {
                    "type": "alternative",
                    "group": self.name,
                    "path": path,
                    "exists": True,
                    "valid": True,
                    "message": "valid (alternative group)",
                }

        if not found_any and self.required:
            primary = self.paths[0]
            details[primary] = {
                "type": "alternative_missing",
                "group": self.name,
                "path": primary,
                "exists": False,
                "valid": False,
                "required": True,
                "message": f"missing (need one of: {' / '.join(self.paths)})",
            }
            return False, details

        return True, details


# ============================================================================
# Model Registry  (add new models by appending entries)
# ============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "zimage": {
        "spec": ModelSpec(
            name="Z-Image",
            model_id="Tongyi-MAI/Z-Image",
            description="Tongyi Z-Image diffusion model",
            default_path="zimage_models",
            size_gb=32.0,
            source_url="https://modelscope.cn/models/Tongyi-MAI/Z-Image",
            aliases=["zimage", "z-image"],
        ),
        "rules": [
            # Root
            FileRule(path="model_index.json", required=True, min_size=10),
            # Scheduler
            FileRule(path="scheduler/scheduler_config.json", required=True, min_size=10),
            # Transformer  (sharded OR single-file)
            FileRule(path="transformer/config.json", required=True, min_size=10),
            AlternativeRule(
                paths=[
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                ],
                name="transformer_weights",
                required=True,
                min_size=10,
            ),
            # Text Encoder  (sharded OR single-file)
            FileRule(path="text_encoder/config.json", required=True, min_size=10),
            AlternativeRule(
                paths=[
                    "text_encoder/model.safetensors.index.json",
                    "text_encoder/model.safetensors",
                ],
                name="text_encoder_weights",
                required=True,
                min_size=10,
            ),
            # VAE  (always single-file, >100 MB)
            FileRule(path="vae/config.json", required=True, min_size=10),
            FileRule(
                path="vae/diffusion_pytorch_model.safetensors",
                required=True,
                min_size=100 * 1024 * 1024,
            ),
        ],
    },
    # --- To add a new model, append an entry here: ---
    # "new_model": {
    #     "spec": ModelSpec(...),
    #     "rules": [ FileRule(...), AlternativeRule(...), ... ],
    # },
}


# ============================================================================
# Model Manager
# ============================================================================

class LocalModelManager(IModelManager):
    """Schema-driven model detection with ModelScope download support."""

    def __init__(self):
        from .config import MODEL_PATH
        self._model_path = MODEL_PATH
        self._download_progress: Optional[DownloadProgress] = None
        self._download_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def get_model_status(self, model_type: str = "zimage") -> ModelInfo:
        """Check model files against registry schema."""
        entry = MODEL_REGISTRY.get(model_type)
        if not entry:
            return ModelInfo(
                model_type=model_type,
                status=ModelStatus.NOT_FOUND,
                path="",
                missing_files=[f"unsupported model type: {model_type}"],
            )

        model_path = self._model_path
        info = ModelInfo(model_type=model_type, path=str(model_path))

        if not model_path.exists():
            info.status = ModelStatus.NOT_FOUND
            info.missing_files = ["model directory not found"]
            return info

        # Check if currently downloading
        if (self._download_progress
                and self._download_progress.model_type == model_type
                and self._download_progress.status == DownloadStatus.DOWNLOADING):
            info.status = ModelStatus.DOWNLOADING
            info.download_progress = self._download_progress.progress_percent
            return info

        # Validate against rules
        rules: List[ValidationRule] = entry["rules"]
        all_valid = True
        missing: List[str] = []
        components: Dict[str, Any] = {}

        for rule in rules:
            is_valid, rule_details = rule.validate(model_path)
            components.update(rule_details)
            if not is_valid:
                all_valid = False
                for k, v in rule_details.items():
                    if not v.get("valid", False):
                        missing.append(k)

        info.components = components
        info.missing_files = missing

        if all_valid:
            info.status = ModelStatus.VALID
        elif len(missing) >= len(rules):
            info.status = ModelStatus.NOT_FOUND
        else:
            info.status = ModelStatus.INCOMPLETE

        return info

    def verify_integrity(self, model_type: str = "zimage") -> ModelInfo:
        """Deep verify - same as get_model_status for local check."""
        return self.get_model_status(model_type)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def start_download(self, model_type: str = "zimage") -> bool:
        """Start model download from ModelScope."""
        entry = MODEL_REGISTRY.get(model_type)
        if not entry:
            logger.error(f"Unknown model type: {model_type}")
            return False

        if self._download_thread and self._download_thread.is_alive():
            logger.warning("Download already in progress")
            return False

        spec: ModelSpec = entry["spec"]
        self._stop_event.clear()
        self._download_progress = DownloadProgress(
            status=DownloadStatus.PENDING,
            model_type=model_type,
        )

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(spec, model_type),
            daemon=True,
        )
        self._download_thread.start()
        return True

    def cancel_download(self) -> bool:
        """Cancel active download."""
        if not self._download_thread or not self._download_thread.is_alive():
            return False
        self._stop_event.set()
        if self._download_progress:
            self._download_progress.status = DownloadStatus.CANCELLED
        return True

    def get_download_progress(self) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self._download_progress

    def list_supported_models(self) -> List[str]:
        """List registered model type keys."""
        return list(MODEL_REGISTRY.keys())

    def get_model_spec(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get model spec info for a given type."""
        entry = MODEL_REGISTRY.get(model_type)
        if not entry:
            return None
        spec: ModelSpec = entry["spec"]
        return {
            "name": spec.name,
            "model_id": spec.model_id,
            "description": spec.description,
            "size_gb": spec.size_gb,
            "source_url": spec.source_url,
            "aliases": spec.aliases,
        }

    # ------------------------------------------------------------------
    # Download Worker
    # ------------------------------------------------------------------

    def _download_worker(self, spec: ModelSpec, model_type: str):
        """Background thread: download model via modelscope snapshot_download."""
        progress = self._download_progress
        if not progress:
            return

        try:
            progress.status = DownloadStatus.DOWNLOADING
            logger.info(f"Starting download: {spec.model_id} -> {self._model_path}")

            from modelscope.hub.snapshot_download import snapshot_download

            target_dir = str(self._model_path)

            # Clear proxy env vars - modelscope.cn is domestic, no proxy needed
            import os
            for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
                os.environ.pop(key, None)

            # snapshot_download handles everything: file list, resume, parallel download
            result_path = snapshot_download(
                model_id=spec.model_id,
                local_dir=target_dir,
            )

            progress.status = DownloadStatus.COMPLETED
            progress.current_file = None
            logger.info(f"Download complete: {spec.model_id} -> {result_path}")

        except ImportError:
            logger.error("modelscope not installed. Run: pip install modelscope")
            if progress:
                progress.status = DownloadStatus.FAILED
                progress.error_message = "modelscope not installed"

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if progress:
                progress.status = DownloadStatus.FAILED
                progress.error_message = str(e)
