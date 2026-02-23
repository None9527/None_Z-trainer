# -*- coding: utf-8 -*-
"""
Application Layer - Generation Use Cases

Orchestrates image generation operations.
"""

import logging
from typing import List, Dict, Any, Optional

from ..domain.generation.entities import (
    GenerationRequest, GenerationResult, LoRAInfo, TransformerInfo,
)
from ..domain.generation.repositories import (
    IGenerationPipeline, IModelRepository, IGenerationHistoryRepository,
)

logger = logging.getLogger(__name__)


class GenerateImageUseCase:
    """Generate images using the pipeline."""

    def __init__(
        self,
        pipeline: IGenerationPipeline,
        history_repo: IGenerationHistoryRepository,
    ):
        self._pipeline = pipeline
        self._history_repo = history_repo

    def execute(self, request: GenerationRequest) -> List[GenerationResult]:
        if not self._pipeline.is_loaded():
            self._pipeline.load(
                transformer_path=request.transformer_path,
            )

        results = self._pipeline.generate(request)

        for result in results:
            self._history_repo.save_result(result)

        return results


class ListModelsUseCase:
    """List available LoRA and transformer models."""

    def __init__(self, model_repo: IModelRepository):
        self._model_repo = model_repo

    def execute(self) -> Dict[str, Any]:
        return {
            "loras": self._model_repo.list_loras(),
            "transformers": self._model_repo.list_transformers(),
        }


class DeleteModelUseCase:
    """Delete a model file."""

    def __init__(self, model_repo: IModelRepository):
        self._model_repo = model_repo

    def execute(self, path: str) -> bool:
        return self._model_repo.delete_model(path)


class GetGenerationHistoryUseCase:
    """Get generation history with pagination."""

    def __init__(self, history_repo: IGenerationHistoryRepository):
        self._history_repo = history_repo

    def execute(self, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        items, total = self._history_repo.list_history(offset=offset, limit=limit)
        return {"items": items, "total": total}
