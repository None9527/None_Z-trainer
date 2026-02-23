# -*- coding: utf-8 -*-
"""
File Generation History Repository - Infrastructure Implementation

Persists generation history to a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..domain.generation.repositories import IGenerationHistoryRepository
from ..domain.generation.entities import GenerationResult

logger = logging.getLogger(__name__)


class FileGenerationHistoryRepository(IGenerationHistoryRepository):
    """JSON file-based generation history."""

    def __init__(self):
        from .config import GENERATION_OUTPUT_PATH
        self._output_path = GENERATION_OUTPUT_PATH
        self._history_file = GENERATION_OUTPUT_PATH / "history.json"

    def save_result(self, result: GenerationResult) -> None:
        """Append a generation result to history."""
        history = self._load_history()
        entry = {
            "timestamp": result.timestamp,
            "image_path": result.image_path,
            "prompt": result.prompt,
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "steps": result.steps,
            "guidance_scale": result.guidance_scale,
        }
        if result.lora_path:
            entry["lora_path"] = result.lora_path
            entry["lora_scale"] = result.lora_scale
        history.insert(0, entry)
        self._save_history(history)

    def save_comparison(self, result_no_lora: GenerationResult,
                        result_with_lora: GenerationResult,
                        lora_path: str, lora_scale: float) -> None:
        """Save a comparison pair as a single grouped history entry."""
        history = self._load_history()
        entry = {
            "timestamp": result_with_lora.timestamp,
            "comparison_mode": True,
            "prompt": result_with_lora.prompt,
            "seed": result_with_lora.seed,
            "width": result_with_lora.width,
            "height": result_with_lora.height,
            "steps": result_with_lora.steps,
            "guidance_scale": result_with_lora.guidance_scale,
            "lora_path": lora_path,
            "lora_scale": lora_scale,
            "comparison_images": [
                {"image_path": result_no_lora.image_path, "lora_path": None, "lora_scale": 0},
                {"image_path": result_with_lora.image_path, "lora_path": lora_path, "lora_scale": lora_scale},
            ],
        }
        history.insert(0, entry)
        self._save_history(history)

    def list_history(self, offset: int = 0, limit: int = 50) -> Tuple[List[Any], int]:
        """List generation history with pagination. Returns (results, total).

        Returns raw dicts for comparison entries, GenerationResult for normal.
        """
        history = self._load_history()
        total = len(history)
        page = history[offset:offset + limit]
        results: List[Any] = []
        for item in page:
            if item.get("comparison_mode"):
                # Return comparison entries as raw dicts
                results.append(item)
            else:
                results.append(GenerationResult(
                    timestamp=item.get("timestamp", ""),
                    image_path=item.get("image_path", ""),
                    prompt=item.get("prompt", ""),
                    seed=item.get("seed", 0),
                    width=item.get("width", 0),
                    height=item.get("height", 0),
                    steps=item.get("steps", 0),
                    guidance_scale=item.get("guidance_scale", 0.0),
                    lora_path=item.get("lora_path"),
                    lora_scale=item.get("lora_scale", 1.0),
                ))
        return results, total

    def delete_history(self, timestamps: List[str]) -> int:
        """Delete history items by timestamp."""
        history = self._load_history()
        ts_set = set(timestamps)
        original_len = len(history)
        history = [h for h in history if h.get("timestamp") not in ts_set]
        deleted = original_len - len(history)
        if deleted > 0:
            self._save_history(history)
        return deleted

    def _load_history(self) -> list:
        """Load history from JSON file."""
        if self._history_file.exists():
            try:
                return json.loads(self._history_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
        return []

    def _save_history(self, history: list):
        """Save history to JSON file."""
        try:
            self._output_path.mkdir(parents=True, exist_ok=True)
            self._history_file.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
