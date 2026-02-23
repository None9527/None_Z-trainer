# -*- coding: utf-8 -*-
"""
Dataset Domain - Repository Interfaces

Abstract interfaces for dataset operations.
Implementations live in infrastructure/ layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .entities import (
    DatasetType, DatasetStats, ImageInfo, ImageGroup,
    DatasetChannel, ChannelRole, BucketInfo, CacheStatus,
)


class IDatasetRepository(ABC):
    """Repository for dataset discovery and metadata."""

    @abstractmethod
    def list_datasets(self) -> List[Dict]:
        """List all available datasets with basic info."""

    @abstractmethod
    def validate_path(self, path: str) -> Tuple[bool, str]:
        """Validate a path as a valid training dataset. Returns (valid, message)."""

    @abstractmethod
    def detect_type(self, path: str) -> DatasetType:
        """Detect dataset directory structure type."""

    @abstractmethod
    def scan_images(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Tuple[List[ImageInfo], int]:
        """
        Scan dataset for images with pagination.
        Returns (images, total_count).
        """

    @abstractmethod
    def get_stats(self, path: str) -> DatasetStats:
        """Get dataset statistics."""

    @abstractmethod
    def detect_channels(self, path: str) -> List[DatasetChannel]:
        """Auto-detect channel subdirectories in a multi-channel dataset."""

    @abstractmethod
    def scan_groups(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Tuple[List[ImageGroup], int]:
        """Scan multi-channel dataset, returning ImageGroups with matched files."""


class IDatasetCacheRepository(ABC):
    """Repository for dataset cache (latent/text embedding) management."""

    @abstractmethod
    def check_cache_status(self, dataset_path: str, model_type: str) -> CacheStatus:
        """Check how much of the dataset is cached."""

    @abstractmethod
    def start_caching(self, dataset_path: str, model_type: str, config: dict) -> int:
        """Start caching process. Returns process ID."""

    @abstractmethod
    def stop_caching(self, process_id: int) -> None:
        """Stop a running caching process."""


class IBucketCalculator(ABC):
    """Interface for aspect ratio bucket calculation."""

    @abstractmethod
    def calculate_buckets(
        self, path: str, batch_size: int = 4, resolution_limit: int = 1536
    ) -> List[BucketInfo]:
        """Calculate optimal aspect ratio buckets for a dataset."""
