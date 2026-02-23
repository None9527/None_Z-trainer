# -*- coding: utf-8 -*-
"""
Stub Implementations for Phase 2 Features

Provides minimal implementations for interfaces that require GPU:
- IGenerationPipeline (diffusers pipeline)
- IDatasetCacheRepository (latent/text caching)
"""

import logging
from typing import List, Optional

from ..domain.generation.repositories import IGenerationPipeline
from ..domain.generation.entities import GenerationRequest, GenerationResult
from ..domain.dataset.repositories import IDatasetCacheRepository
from ..domain.dataset.entities import CacheStatus

logger = logging.getLogger(__name__)


class StubGenerationPipeline(IGenerationPipeline):
    """Placeholder pipeline that returns informative error.
    Phase 2 will implement actual GPU-based generation."""

    def load(self, model_type: str = "zimage", transformer_path: Optional[str] = None) -> None:
        logger.info("Generation pipeline not yet implemented (Phase 2)")

    def generate(self, request: GenerationRequest) -> List[GenerationResult]:
        raise NotImplementedError(
            "Image generation pipeline requires Phase 2 implementation. "
            "Please use the old webui-vue version for generation."
        )

    def unload(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return False


class StubDatasetCacheRepository(IDatasetCacheRepository):
    """Placeholder cache repo that returns empty status.
    Phase 2 will implement actual GPU-based caching."""

    def check_cache_status(self, dataset_path: str, model_type: str) -> CacheStatus:
        return CacheStatus()

    def start_caching(self, dataset_path: str, model_type: str, config: dict) -> int:
        raise NotImplementedError("Dataset caching requires Phase 2 implementation.")

    def stop_caching(self, process_id: int) -> None:
        pass
