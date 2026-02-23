# -*- coding: utf-8 -*-
"""
Dependency Injection Container

Wires domain interfaces to infrastructure implementations.
Single source of truth for all dependency resolution.
"""

import logging
from typing import Optional

from ..domain.training.repositories import (
    ITrainingRepository,
    ITrainingSessionRepository,
    ITrainingRunner,
)
from ..domain.dataset.repositories import (
    IDatasetRepository,
    IDatasetCacheRepository,
    IBucketCalculator,
)
from ..domain.generation.repositories import (
    IGenerationPipeline,
    IModelRepository,
    IGenerationHistoryRepository,
)
from ..domain.system.repositories import (
    IGPUMonitor,
    ISystemInfoProvider,
    IModelManager,
)

logger = logging.getLogger(__name__)


class Container:
    """
    Simple DI container.

    Provides factory methods for all domain interfaces.
    Infrastructure implementations are lazy-loaded to avoid
    import-time side effects.
    """

    # --- Training ---
    def training_repo(self) -> ITrainingRepository:
        from .toml_training_repo import TomlTrainingRepository
        return TomlTrainingRepository()

    def session_repo(self) -> ITrainingSessionRepository:
        from .memory_session_repo import MemorySessionRepository
        return MemorySessionRepository()

    def training_runner(self) -> ITrainingRunner:
        # Singleton: process tracking must persist across calls
        if not hasattr(self, '_training_runner'):
            from .subprocess_training_runner import SubprocessTrainingRunner
            self._training_runner = SubprocessTrainingRunner()
        return self._training_runner

    # --- Dataset ---
    def dataset_repo(self) -> IDatasetRepository:
        from .file_dataset_repo import FileDatasetRepository
        return FileDatasetRepository()

    def dataset_cache_repo(self) -> IDatasetCacheRepository:
        # Phase 2: requires GPU model loading for latent/text caching
        from .stub_implementations import StubDatasetCacheRepository
        return StubDatasetCacheRepository()

    def bucket_calculator(self) -> IBucketCalculator:
        from .bucket_calculator import FileBucketCalculator
        return FileBucketCalculator()

    # --- Generation ---
    def generation_pipeline(self) -> IGenerationPipeline:
        # Singleton: pipeline loads model once, reuses across requests
        if not hasattr(self, '_generation_pipeline'):
            try:
                from .zimage_generation_pipeline import ZImageGenerationPipeline
                self._generation_pipeline = ZImageGenerationPipeline()
            except ImportError:
                from .stub_implementations import StubGenerationPipeline
                self._generation_pipeline = StubGenerationPipeline()
        return self._generation_pipeline

    def model_repo(self) -> IModelRepository:
        from .file_model_repo import FileModelRepository
        return FileModelRepository()

    def generation_history_repo(self) -> IGenerationHistoryRepository:
        from .file_generation_history import FileGenerationHistoryRepository
        return FileGenerationHistoryRepository()

    # --- System ---
    def gpu_monitor(self) -> IGPUMonitor:
        from .gpu_monitor import NvidiaSmiGPUMonitor
        return NvidiaSmiGPUMonitor()

    def system_info_provider(self) -> ISystemInfoProvider:
        from .system_info_provider import LocalSystemInfoProvider
        return LocalSystemInfoProvider()

    def model_manager(self) -> IModelManager:
        # Singleton: download state must persist across calls
        if not hasattr(self, '_model_manager'):
            from .model_manager import LocalModelManager
            self._model_manager = LocalModelManager()
        return self._model_manager


# Global container instance
container = Container()
