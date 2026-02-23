# -*- coding: utf-8 -*-
"""
V2 Infrastructure - Path Configuration

Resolves all paths from environment variables (set by start.bat).
Single source of truth for path resolution in the v2 backend.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: v2/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(env_key: str, default_relative: str) -> "Path | None":
    """Resolve a path from environment variable, falling back to relative default."""
    value = os.getenv(env_key, "").strip()
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    if default_relative:
        return PROJECT_ROOT / default_relative
    return None


# --- Core Paths ---
MODEL_PATH = _resolve_path("MODEL_PATH", "Z-Image")
DATASET_PATH = _resolve_path("DATASET_PATH", "datasets")
OUTPUT_PATH = _resolve_path("OUTPUT_PATH", "model-output")

# --- Model sub-paths ---
VAE_PATH = _resolve_path("VAE_PATH", "") or MODEL_PATH / "vae"
TEXT_ENCODER_PATH = _resolve_path("TEXT_ENCODER_PATH", "") or MODEL_PATH / "text_encoder"
TRANSFORMER_PATH = _resolve_path("TRANSFORMER_PATH", "") or MODEL_PATH / "transformer"

# --- Output sub-paths ---
LORA_PATH = _resolve_path("LORA_PATH", "") or (OUTPUT_PATH / "lora")
FINETUNE_PATH = _resolve_path("FINETUNE_PATH", "") or (OUTPUT_PATH / "finetune")
LOGS_PATH = OUTPUT_PATH / "logs"

# --- Generation output ---
GENERATION_OUTPUT_PATH = _resolve_path("GENERATION_OUTPUT_PATH", "image-output")

# --- Configs ---
CONFIGS_DIR = PROJECT_ROOT / "configs"

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# --- Model path mapping ---
MODEL_PATHS = {
    "zimage": {
        "base": MODEL_PATH,
        "vae": VAE_PATH,
        "text_encoder": TEXT_ENCODER_PATH,
        "transformer": TRANSFORMER_PATH,
    }
}


def get_model_path(model_type: str = "zimage", component: str = "base") -> Path:
    """Get path for a model component."""
    return MODEL_PATHS["zimage"].get(component, MODEL_PATHS["zimage"]["base"])


def ensure_dirs():
    """Create all necessary directories."""
    for d in [DATASET_PATH, OUTPUT_PATH, LORA_PATH, FINETUNE_PATH,
              LOGS_PATH, GENERATION_OUTPUT_PATH, CONFIGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Auto-create on import
ensure_dirs()
