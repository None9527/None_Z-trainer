# -*- coding: utf-8 -*-
"""
Dataset Domain - Entities

Core business entities for dataset management.
Encapsulates dataset scanning, image pairing, bucket calculation,
and cache management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple


class DatasetType(Enum):
    """Dataset directory structure type."""
    STANDARD = "standard"            # Single image directory (flat)
    MULTI_CHANNEL = "multi_channel"  # Multi-channel: target/ + condition dirs


class ChannelRole(Enum):
    """Role of a dataset channel in training."""
    TARGET = "target"          # Training target (required, exactly one)
    SOURCE = "source"          # Source image (Img2Img)
    CONDITION = "condition"    # Control condition (ControlNet: depth/canny/pose)
    REFERENCE = "reference"    # Reference image (Omni: SigLIP encoded)


@dataclass
class DatasetChannel:
    """
    A single channel within a multi-channel dataset.

    Each channel maps to a subdirectory containing images
    that share filenames with the target channel for pairing.
    """
    name: str                  # "target" / "depth" / "canny" / "source" / "ref_01"
    role: ChannelRole          # Functional role
    directory: str             # Subdirectory path (relative to dataset root)
    encoder: str = ""          # Encoding method: "vae" / "siglip" / "raw" / ""
    image_count: int = 0       # Number of images in this channel

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role.value,
            "directory": self.directory,
            "encoder": self.encoder,
            "image_count": self.image_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetChannel":
        return cls(
            name=data["name"],
            role=ChannelRole(data["role"]),
            directory=data["directory"],
            encoder=data.get("encoder", ""),
            image_count=data.get("image_count", 0),
        )


@dataclass
class ImageInfo:
    """Single image metadata."""
    filename: str
    path: str
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    caption: str = ""
    has_caption: bool = False


@dataclass
class ImageGroup:
    """
    A group of matched images across channels.

    All images share the same base filename (stem).
    Used for multi-channel datasets (ControlNet, Img2Img, Omni).
    """
    id: str                                    # Group ID (filename stem)
    target: Optional[ImageInfo] = None         # Target image (required)
    channels: Dict[str, ImageInfo] = field(default_factory=dict)  # channel_name -> image
    caption: str = ""                          # Text caption

    @property
    def is_complete(self) -> bool:
        """Check if target exists."""
        return self.target is not None

    @property
    def channel_count(self) -> int:
        """Number of non-target channels with images."""
        return len(self.channels)


# Legacy alias for backward compatibility
ImagePair = ImageGroup


@dataclass
class BucketInfo:
    """Aspect ratio bucket for batch training."""
    width: int
    height: int
    count: int = 0
    aspect_ratio: float = 1.0
    images: List[str] = field(default_factory=list)


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_images: int = 0
    total_size_bytes: int = 0
    cached_images: int = 0
    caption_coverage: float = 0.0
    dataset_type: DatasetType = DatasetType.STANDARD
    channels: List[DatasetChannel] = field(default_factory=list)
    buckets: List[BucketInfo] = field(default_factory=list)


@dataclass
class CacheStatus:
    """Dataset cache (latent/text embedding) status."""
    latent_cached: int = 0
    text_cached: int = 0
    total_images: int = 0
    is_complete: bool = False

    @property
    def latent_progress(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.latent_cached / self.total_images * 100

    @property
    def text_progress(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.text_cached / self.total_images * 100
