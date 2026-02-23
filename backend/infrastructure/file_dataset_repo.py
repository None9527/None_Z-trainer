# -*- coding: utf-8 -*-
"""
File Dataset Repository - Infrastructure Implementation

Implements IDatasetRepository using filesystem operations.
Supports standard (flat) and multi-channel (subdirectory) datasets.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from ..domain.dataset.repositories import IDatasetRepository
from ..domain.dataset.entities import (
    DatasetType, DatasetStats, ImageInfo, ImageGroup,
    DatasetChannel, ChannelRole,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
CAPTION_EXTENSIONS = {'.txt', '.caption'}

# Well-known channel name -> role mapping
CHANNEL_ROLE_MAP = {
    "target": ChannelRole.TARGET,
    "source": ChannelRole.SOURCE,
    "depth": ChannelRole.CONDITION,
    "canny": ChannelRole.CONDITION,
    "pose": ChannelRole.CONDITION,
    "normal": ChannelRole.CONDITION,
    "segmentation": ChannelRole.CONDITION,
    "seg": ChannelRole.CONDITION,
    "lineart": ChannelRole.CONDITION,
    "sketch": ChannelRole.CONDITION,
    "openpose": ChannelRole.CONDITION,
    "reference": ChannelRole.REFERENCE,
    "ref": ChannelRole.REFERENCE,
    "conditions": ChannelRole.REFERENCE,   # Legacy omni dir
}

# Dimension cache: {dataset_path_str: {filename: (width, height, mtime)}}
_dimension_cache: Dict[str, dict] = {}


def _load_dimension_cache(dataset_path: Path) -> dict:
    """Load dimension cache from JSON file."""
    cache_file = dataset_path / ".dim_cache.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_dimension_cache(dataset_path: Path, cache: dict):
    """Save dimension cache to JSON file."""
    cache_file = dataset_path / ".dim_cache.json"
    try:
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save dimension cache: {e}")


def _get_image_dimensions(file: Path, dim_cache: dict) -> Tuple[int, int]:
    """Get image dimensions, using cache when possible."""
    key = file.name
    mtime = file.stat().st_mtime

    if key in dim_cache:
        cached = dim_cache[key]
        if isinstance(cached, (list, tuple)) and len(cached) >= 3 and cached[2] == mtime:
            return cached[0], cached[1]

    try:
        with Image.open(file) as img:
            w, h = img.size
        dim_cache[key] = [w, h, mtime]
        return w, h
    except Exception:
        return 0, 0


def _count_images_in_dir(directory: Path) -> int:
    """Count images directly in a directory (non-recursive)."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _infer_channel_role(name: str) -> ChannelRole:
    """Infer channel role from directory name."""
    lower = name.lower()
    if lower in CHANNEL_ROLE_MAP:
        return CHANNEL_ROLE_MAP[lower]
    # ref_01, ref_02 pattern
    if lower.startswith("ref"):
        return ChannelRole.REFERENCE
    # condition_*, ctrl_* pattern
    if lower.startswith(("condition", "ctrl", "control")):
        return ChannelRole.CONDITION
    # Default: condition (most common for unknown subdirs)
    return ChannelRole.CONDITION


class FileDatasetRepository(IDatasetRepository):
    """Filesystem-based dataset repository."""

    def __init__(self):
        from .config import DATASET_PATH
        self._dataset_path = DATASET_PATH

    def list_datasets(self) -> List[Dict]:
        """List all datasets in the datasets directory."""
        datasets = []
        base = self._dataset_path

        if not base.exists():
            return datasets

        for item in sorted(base.iterdir()):
            if not item.is_dir() or item.name.startswith('.'):
                continue

            # Count images
            image_files = [
                f for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            dataset_type = self._detect_type_internal(item)

            entry = {
                "name": item.name,
                "path": str(item),
                "image_count": len(image_files),
                "type": dataset_type.value,
            }

            # For multi-channel, add channel info
            if dataset_type == DatasetType.MULTI_CHANNEL:
                channels = self._detect_channels_internal(item)
                entry["channels"] = [ch.to_dict() for ch in channels]

            datasets.append(entry)

        return datasets

    def validate_path(self, path: str) -> Tuple[bool, str]:
        """Validate a path contains training images."""
        p = self._resolve_path(path)

        if not p.exists():
            return False, f"Path does not exist: {path}"
        if not p.is_dir():
            return False, f"Path is not a directory: {path}"

        image_files = list(self._iter_images(p))

        if not image_files:
            return False, f"No image files found in: {path}"

        return True, f"Valid dataset: {len(image_files)} images found"

    def detect_type(self, path: str) -> DatasetType:
        """Detect dataset directory structure type."""
        p = self._resolve_path(path)
        return self._detect_type_internal(p)

    def detect_channels(self, path: str) -> List[DatasetChannel]:
        """Auto-detect channel subdirectories."""
        p = self._resolve_path(path)
        return self._detect_channels_internal(p)

    def scan_images(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Tuple[List[ImageInfo], int]:
        """Scan dataset for images with pagination."""
        p = self._resolve_path(path)
        if not p.exists():
            return [], 0

        dim_cache = _load_dimension_cache(p)

        # For multi-channel: scan target/ directory images
        dataset_type = self._detect_type_internal(p)
        if dataset_type == DatasetType.MULTI_CHANNEL:
            target_dir = p / "target"
            if target_dir.is_dir():
                all_files = sorted(self._iter_images_flat(target_dir), key=lambda f: f.name)
            else:
                all_files = sorted(self._iter_images(p), key=lambda f: f.name)
        else:
            all_files = sorted(self._iter_images(p), key=lambda f: f.name)

        total = len(all_files)

        start = (page - 1) * page_size
        end = start + page_size
        page_files = all_files[start:end]

        images = []
        for f in page_files:
            w, h = _get_image_dimensions(f, dim_cache)

            caption_path = f.with_suffix('.txt')
            caption = ""
            has_caption = caption_path.exists()
            if has_caption:
                try:
                    caption = caption_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            images.append(ImageInfo(
                filename=f.name,
                path=str(f),
                width=w,
                height=h,
                size_bytes=f.stat().st_size,
                caption=caption,
                has_caption=has_caption,
            ))

        _save_dimension_cache(p, dim_cache)

        return images, total

    def scan_groups(
        self, path: str, page: int = 1, page_size: int = 50
    ) -> Tuple[List[ImageGroup], int]:
        """Scan multi-channel dataset, returning ImageGroups with matched files."""
        p = self._resolve_path(path)
        if not p.exists():
            return [], 0

        channels = self._detect_channels_internal(p)
        if not channels:
            return [], 0

        target_ch = next((ch for ch in channels if ch.role == ChannelRole.TARGET), None)
        if target_ch is None:
            return [], 0

        aux_channels = [ch for ch in channels if ch.role != ChannelRole.TARGET]
        dim_cache = _load_dimension_cache(p)

        # Build stem-based index from target directory
        target_dir = p / target_ch.directory
        if not target_dir.is_dir():
            return [], 0

        target_files = sorted(
            [f for f in target_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda f: f.name
        )

        total = len(target_files)
        start = (page - 1) * page_size
        end = start + page_size
        page_files = target_files[start:end]

        groups = []
        for tf in page_files:
            stem = tf.stem
            w, h = _get_image_dimensions(tf, dim_cache)

            # Read caption (check target dir .txt, then caption/ dir)
            caption = ""
            caption_txt = tf.with_suffix('.txt')
            caption_dir_txt = p / "caption" / f"{stem}.txt"
            if caption_txt.exists():
                try:
                    caption = caption_txt.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            elif caption_dir_txt.exists():
                try:
                    caption = caption_dir_txt.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            target_info = ImageInfo(
                filename=tf.name,
                path=str(tf),
                width=w, height=h,
                size_bytes=tf.stat().st_size,
                caption=caption,
                has_caption=bool(caption),
            )

            # Match auxiliary channels by stem
            ch_images = {}
            for ch in aux_channels:
                ch_dir = p / ch.directory
                if not ch_dir.is_dir():
                    continue
                # Find matching file (any image extension)
                matched = None
                for ext in IMAGE_EXTENSIONS:
                    candidate = ch_dir / f"{stem}{ext}"
                    if candidate.exists():
                        cw, ch_h = _get_image_dimensions(candidate, dim_cache)
                        matched = ImageInfo(
                            filename=candidate.name,
                            path=str(candidate),
                            width=cw, height=ch_h,
                            size_bytes=candidate.stat().st_size,
                        )
                        break
                if matched:
                    ch_images[ch.name] = matched

            groups.append(ImageGroup(
                id=stem,
                target=target_info,
                channels=ch_images,
                caption=caption,
            ))

        _save_dimension_cache(p, dim_cache)
        return groups, total

    def get_stats(self, path: str) -> DatasetStats:
        """Get dataset statistics."""
        p = self._resolve_path(path)
        if not p.exists():
            return DatasetStats()

        all_files = list(self._iter_images(p))
        total_size = sum(f.stat().st_size for f in all_files)

        captioned = sum(
            1 for f in all_files
            if f.with_suffix('.txt').exists()
        )

        caption_coverage = captioned / len(all_files) if all_files else 0.0

        cached_count = 0
        cache_dir = p / ".cache"
        if cache_dir.exists():
            cached_count = sum(1 for _ in cache_dir.rglob("*.safetensors"))

        dataset_type = self._detect_type_internal(p)
        channels = self._detect_channels_internal(p) if dataset_type == DatasetType.MULTI_CHANNEL else []

        return DatasetStats(
            total_images=len(all_files),
            total_size_bytes=total_size,
            cached_images=cached_count,
            caption_coverage=caption_coverage,
            dataset_type=dataset_type,
            channels=channels,
        )

    # --- Private helpers ---

    def _resolve_path(self, path: str) -> Path:
        """Resolve path: absolute or relative to DATASET_PATH."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self._dataset_path / path

    def _iter_images(self, directory: Path):
        """Iterate over image files in a directory (recursive)."""
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if not any(part.startswith('.') for part in f.relative_to(directory).parts):
                    yield f

    def _iter_images_flat(self, directory: Path):
        """Iterate over image files in a directory (non-recursive)."""
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                yield f

    def _detect_type_internal(self, path: Path) -> DatasetType:
        """Detect dataset type from directory structure.

        Rules:
        - Has target/ subdirectory with images -> MULTI_CHANNEL
        - Otherwise -> STANDARD (flat directory)
        """
        target_dir = path / "target"
        if target_dir.is_dir() and _count_images_in_dir(target_dir) > 0:
            return DatasetType.MULTI_CHANNEL
        return DatasetType.STANDARD

    def _detect_channels_internal(self, path: Path) -> List[DatasetChannel]:
        """Auto-detect channel subdirectories.

        Scans for subdirectories containing images and assigns roles:
        - target/     -> TARGET (required)
        - source/     -> SOURCE
        - depth/ etc. -> CONDITION (by name lookup)
        - ref*/       -> REFERENCE
        - caption/    -> skipped (text, not a channel)
        - Others      -> CONDITION (default)
        """
        if not path.is_dir():
            return []

        # Check for explicit dataset.json
        config_file = path / "dataset.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                channels = []
                for ch_cfg in config.get("channels", []):
                    channels.append(DatasetChannel(
                        name=ch_cfg["name"],
                        role=ChannelRole(ch_cfg["role"]),
                        directory=ch_cfg.get("directory", ch_cfg["name"]),
                        encoder=ch_cfg.get("encoder", ""),
                        image_count=_count_images_in_dir(path / ch_cfg.get("directory", ch_cfg["name"])),
                    ))
                return channels
            except Exception as e:
                logger.warning(f"Failed to parse dataset.json: {e}")

        # Auto-detect from directory structure
        channels = []
        skip_dirs = {'.cache', '.git', '__pycache__', 'caption', 'captions', 'text'}

        for sub in sorted(path.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith('.') or sub.name.lower() in skip_dirs:
                continue

            img_count = _count_images_in_dir(sub)
            if img_count == 0:
                continue

            role = _infer_channel_role(sub.name)
            channels.append(DatasetChannel(
                name=sub.name,
                role=role,
                directory=sub.name,
                encoder="",
                image_count=img_count,
            ))

        return channels
