# -*- coding: utf-8 -*-
"""
Bucket Calculator - Infrastructure Implementation

Calculates optimal aspect ratio buckets for batch training.
Ported from webui-vue/api/routers/dataset.py.
"""

import logging
import math
from pathlib import Path
from typing import List
from collections import defaultdict
from PIL import Image

from ..domain.dataset.repositories import IBucketCalculator
from ..domain.dataset.entities import BucketInfo

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

# Standard bucket resolutions (multiples of 64)
BUCKET_RESOLUTIONS = [
    (512, 512), (512, 768), (768, 512),
    (512, 1024), (1024, 512),
    (768, 768), (768, 1024), (1024, 768),
    (1024, 1024), (1024, 1536), (1536, 1024),
    (1280, 768), (768, 1280),
    (1280, 1024), (1024, 1280),
    (1536, 1536),
]


class FileBucketCalculator(IBucketCalculator):
    """Calculate aspect ratio buckets from filesystem images."""

    def calculate_buckets(
        self, path: str, batch_size: int = 4, resolution_limit: int = 1536
    ) -> List[BucketInfo]:
        """Calculate optimal aspect ratio buckets for a dataset."""
        p = Path(path)
        if not p.exists():
            return []

        # Filter buckets by resolution limit
        valid_buckets = [
            (w, h) for w, h in BUCKET_RESOLUTIONS
            if w <= resolution_limit and h <= resolution_limit
        ]

        if not valid_buckets:
            valid_buckets = [(1024, 1024)]

        # Assign images to nearest bucket
        bucket_assignments: dict = defaultdict(list)

        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(part.startswith('.') for part in f.relative_to(p).parts):
                continue

            try:
                with Image.open(f) as img:
                    w, h = img.size
            except Exception:
                continue

            # Find nearest bucket by aspect ratio
            img_ar = w / h if h > 0 else 1.0
            best_bucket = min(
                valid_buckets,
                key=lambda b: abs(b[0] / b[1] - img_ar)
            )
            bucket_assignments[best_bucket].append(f.name)

        # Build result
        results = []
        for (bw, bh), images in sorted(bucket_assignments.items()):
            # Calculate effective batches (drop remainder)
            effective_count = (len(images) // batch_size) * batch_size
            results.append(BucketInfo(
                width=bw,
                height=bh,
                count=len(images),
                aspect_ratio=round(bw / bh, 3),
                images=images[:10],  # Only return first 10 filenames as preview
            ))

        return results
