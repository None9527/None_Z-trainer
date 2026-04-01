# -*- coding: utf-8 -*-
"""
Dataset Router - Thin API Layer

Handles HTTP requests for dataset operations.
Delegates to use cases; contains NO business logic.
"""

import os
import shutil
import logging
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List

from .dto import ApiResponse
from ..infrastructure.task_runner import AsyncTaskRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


# ============================================================================
# Request Models
# ============================================================================

class DatasetScanRequest(BaseModel):
    path: str
    page: int = 1
    page_size: int = 50


class ValidatePathRequest(BaseModel):
    path: str


class BucketCalculateRequest(BaseModel):
    path: str
    batch_size: int = 4
    resolution_limit: int = 1536
    fill_strategy: str = "none"  # none, repeat, crop


class DeleteImagesRequest(BaseModel):
    paths: List[str]


class DeleteCaptionsRequest(BaseModel):
    dataset_path: str


class CaptionSaveRequest(BaseModel):
    path: str
    caption: str


class StatsRequest(BaseModel):
    path: str


class OllamaTagRequest(BaseModel):
    dataset_path: str
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "llava"
    prompt: str = ""
    max_long_edge: int = 1024
    skip_existing: bool = True
    trigger_word: str = ""


class ResizeRequest(BaseModel):
    dataset_path: str
    max_long_edge: int = 1024
    quality: int = 95
    sharpen: float = 0.0


class GenerateCaptionsRequest(BaseModel):
    datasetPath: str
    modelType: str = "qwen"


# ============================================================================
# Helpers
# ============================================================================

def _get_repo():
    from ..infrastructure.container import container
    return container.dataset_repo()


def _get_dataset_path():
    from ..infrastructure.config import DATASET_PATH
    return DATASET_PATH


# ============================================================================
# P0: Core CRUD
# ============================================================================

@router.post("/scan")
async def scan_dataset(request: DatasetScanRequest):
    """Scan a dataset directory for images."""
    from ..application.dataset_usecases import ScanDatasetUseCase

    repo = _get_repo()
    use_case = ScanDatasetUseCase(repo)
    result = use_case.execute(request.path, request.page, request.page_size)

    if not result.get("success", False):
        return ApiResponse(success=False, message=result.get("message", "Scan failed"))

    # Transform to frontend-expected format
    images = result.get("images", [])
    total = result.get("total", 0)
    page = result.get("page", 1)
    page_size = result.get("page_size", 50)
    dataset_type = result.get("dataset_type", "standard")

    # Convert ImageInfo dataclass to frontend dict format
    formatted_images = []
    for img in images:
        if hasattr(img, '__dict__'):
            # dataclass
            d = {
                "path": img.path,
                "filename": img.filename,
                "stem": Path(img.filename).stem,
                "width": img.width,
                "height": img.height,
                "size": img.size_bytes,
                "caption": img.caption or "",
                "hasLatentCache": _check_latent_cache(img.path),
                "hasTextCache": _check_text_cache(img.path),
                "thumbnailUrl": f"/api/dataset/image?path={img.path}",
            }
        else:
            d = img
        formatted_images.append(d)

    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1

    # Build response matching frontend DatasetInfo interface
    dataset_name = Path(request.path).name
    response_data = {
        "path": request.path,
        "name": dataset_name,
        "imageCount": total,
        "totalSize": sum(i.get("size", 0) if isinstance(i, dict) else getattr(i, 'size_bytes', 0) for i in images),
        "images": formatted_images,
        "datasetType": dataset_type,
        "channels": result.get("channels", []),
        "pagination": {
            "page": page,
            "pageSize": page_size,
            "totalPages": total_pages,
            "totalCount": total,
            "hasNext": page < total_pages,
            "hasPrev": page > 1,
        }
    }
    return response_data


@router.post("/scan-channels")
async def scan_channels(request: DatasetScanRequest):
    """Scan a multi-channel dataset, returning ImageGroups with matched files.

    Response format:
    {
      "groups": [
        {
          "id": "cat",
          "caption": "a cat sitting",
          "target": {"filename": "cat.jpg", "path": "...", "width": 1024, ...},
          "channels": {
            "depth": {"filename": "cat.png", "path": "...", ...},
            "canny": {"filename": "cat.png", "path": "...", ...}
          }
        }
      ],
      "channels": [{"name": "target", "role": "target", ...}, ...],
      "total": 42,
      "pagination": {...}
    }
    """
    from ..application.dataset_usecases import ScanMultiChannelUseCase

    repo = _get_repo()
    use_case = ScanMultiChannelUseCase(repo)
    result = use_case.execute(request.path, request.page, request.page_size)

    if not result.get("success", False):
        return ApiResponse(success=False, message=result.get("message", "Scan failed"))

    total = result.get("total", 0)
    page = result.get("page", 1)
    page_size = result.get("page_size", 50)
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1

    return {
        "groups": result.get("groups", []),
        "channels": result.get("channels", []),
        "total": total,
        "pagination": {
            "page": page,
            "pageSize": page_size,
            "totalPages": total_pages,
            "totalCount": total,
            "hasNext": page < total_pages,
            "hasPrev": page > 1,
        }
    }


def _check_latent_cache(image_path: str) -> bool:
    """Check if latent cache exists for an image.
    
    Actual file format from cache_latents.py: {stem}_{WxH}_zi.safetensors
    e.g. 00003-1233005352_576x1024_zi.safetensors
    """
    p = Path(image_path)
    # Primary pattern: {stem}_{WxH}_zi.safetensors (from cache_latents.py)
    matches = list(p.parent.glob(f"{p.stem}_*_zi.safetensors"))
    if matches:
        return True
    # Legacy patterns
    for suffix in ["_zi_latent.safetensors", ".latent.safetensors"]:
        if (p.parent / (p.stem + suffix)).exists():
            return True
    # Check .cache subdirectory
    cache_dir = p.parent / ".cache"
    if cache_dir.exists():
        if list(cache_dir.glob(f"{p.stem}_*_zi.safetensors")):
            return True
    return False


def _check_text_cache(image_path: str) -> bool:
    """Check if text cache exists for an image.
    
    Actual file format from cache_text_encoder.py: {stem}_zi_te.safetensors
    """
    p = Path(image_path)
    # Primary pattern: {stem}_zi_te.safetensors (from cache_text_encoder.py)
    if (p.parent / (p.stem + "_zi_te.safetensors")).exists():
        return True
    # Legacy patterns
    for suffix in ["_zi_text.safetensors", ".text.safetensors"]:
        if (p.parent / (p.stem + suffix)).exists():
            return True
    # Check .cache subdirectory
    cache_dir = p.parent / ".cache"
    if cache_dir.exists():
        if (cache_dir / (p.stem + "_zi_te.safetensors")).exists():
            return True
    return False


@router.get("/list", response_model=ApiResponse)
async def list_datasets():
    """List all available datasets."""
    repo = _get_repo()
    datasets = repo.list_datasets()
    formatted = [
        {
            "name": d["name"],
            "path": d["path"],
            "imageCount": d.get("image_count", 0),
            "type": d.get("type", "standard"),
            "channels": d.get("channels", []),
        }
        for d in datasets
    ]
    return ApiResponse(success=True, data={
        "datasets": formatted,
        "datasetsDir": str(repo._dataset_path),
    })


@router.get("/browse")
async def browse_datasets(subpath: str = ""):
    """Browse datasets directory at a given subpath level.

    Returns folders (organizational dirs) and datasets (dirs with images)
    at the specified subpath level, enabling hierarchical navigation.
    """
    repo = _get_repo()
    result = repo.browse(subpath)
    result["datasetsDir"] = str(repo._dataset_path)
    return result


@router.post("/create-folder")
async def create_folder(name: str = Form(...), parent_path: str = Form("")):
    """Create an organizational folder within datasets directory."""
    repo = _get_repo()
    try:
        folder = repo.create_folder(name, parent_path)
        return {"success": True, "name": name.strip(), "path": str(folder)}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached")
async def list_cached_datasets():
    """List datasets for training config selector (simplified format)."""
    repo = _get_repo()
    datasets = repo.list_datasets()
    formatted = [
        {"name": d["name"], "path": d["path"]}
        for d in datasets
    ]
    return {"datasets": formatted}


@router.post("/create")
async def create_dataset(name: str = Form(...), parent_path: str = Form("")):
    """Create a new empty dataset directory.

    Args:
        name: Dataset name
        parent_path: Optional subpath within datasets dir (e.g. "人物" to create under datasets/人物/)
    """
    base = _get_dataset_path()
    parent = base / parent_path if parent_path else base
    # Safety: ensure parent doesn't escape base
    try:
        parent.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="路径越界")
    dataset_dir = parent / name.strip()
    if dataset_dir.exists():
        raise HTTPException(status_code=400, detail=f"数据集「{name}」已存在")
    try:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return {"success": True, "name": name.strip(), "path": str(dataset_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name:path}")
async def delete_dataset(name: str):
    """Delete a dataset or folder directory.

    Supports nested paths like '人物/角色A' via {name:path}.
    """
    base = _get_dataset_path()
    dataset_dir = base / name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"「{name}」不存在")
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"「{name}」不是目录")
    # Safety: only allow deleting within dataset base path
    try:
        dataset_dir.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="路径越界")
    try:
        shutil.rmtree(dataset_dir)
        return {"success": True, "message": f"「{name}」已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-images")
async def delete_images(request: DeleteImagesRequest):
    """Delete selected images and their associated files."""
    deleted = 0
    errors = []
    for path in request.paths:
        p = Path(path)
        if not p.exists():
            errors.append(f"不存在: {path}")
            continue
        try:
            p.unlink()
            # Also delete associated caption (.txt)
            txt = p.with_suffix('.txt')
            if txt.exists():
                txt.unlink()
            # Also delete cache files
            for suffix in ["_zi_latent.safetensors", "_zi_text.safetensors"]:
                cache = p.with_suffix("").with_name(p.stem + suffix)
                if cache.exists():
                    cache.unlink()
            deleted += 1
        except Exception as e:
            errors.append(f"删除失败 {p.name}: {e}")
    return {"deleted": deleted, "errors": errors}


@router.post("/delete-captions")
async def delete_captions(request: DeleteCaptionsRequest):
    """Delete all .txt caption files in a dataset."""
    p = Path(request.dataset_path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="无效的数据集路径")
    deleted = 0
    errors = []
    for txt in p.rglob("*.txt"):
        # Skip hidden directories
        if any(part.startswith('.') for part in txt.relative_to(p).parts):
            continue
        try:
            txt.unlink()
            deleted += 1
        except Exception as e:
            errors.append(str(e))
    return {"deleted": deleted, "errors": errors}


# ============================================================================
# Upload
# ============================================================================

@router.post("/upload")
async def upload_files(
    dataset: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload files to a dataset."""
    base = _get_dataset_path()
    dataset_dir = base / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    errors = []
    for file in files:
        try:
            dest = dataset_dir / file.filename
            content = await file.read()
            dest.write_bytes(content)
            uploaded.append(file.filename)
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "datasetPath": str(dataset_dir),
    }


@router.post("/upload_batch")
async def upload_batch(
    dataset_name: str = Form(...),
    files: list[UploadFile] = File(...),
    preserve_structure: bool = Form(False),
):
    """Upload a batch of files to create/update a dataset.

    When preserve_structure=True, the relative path is preserved:
      e.g. 'my_dataset/target/cat.jpg' -> dataset_dir/target/cat.jpg
    This enables multi-channel folder upload from the browser.
    """
    base = _get_dataset_path()
    dataset_dir = base / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    uploaded_count = 0
    errors = []
    for file in files:
        try:
            filename = file.filename or ""
            filename = filename.replace("\\", "/")

            if preserve_structure and "/" in filename:
                # Strip leading dataset folder name (browser sends 'folder/sub/file.jpg')
                parts = filename.split("/")
                if len(parts) > 1:
                    # Remove the root folder name, keep the rest
                    rel_path = "/".join(parts[1:])
                else:
                    rel_path = parts[0]
                dest = dataset_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Flat mode: strip to filename only
                if "/" in filename:
                    filename = filename.split("/")[-1]
                dest = dataset_dir / filename

            content = await file.read()
            dest.write_bytes(content)
            uploaded_count += 1
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {"uploaded": uploaded_count, "errors": errors}


@router.post("/upload-to-channel")
async def upload_to_channel(
    dataset_name: str = Form(...),
    channel_name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload files into a specific channel subdirectory.

    Creates dataset_dir/channel_name/ and places all files there.
    """
    base = _get_dataset_path()
    channel_dir = base / dataset_name / channel_name
    channel_dir.mkdir(parents=True, exist_ok=True)

    uploaded_count = 0
    errors = []
    for file in files:
        try:
            filename = file.filename or ""
            filename = filename.replace("\\", "/")
            if "/" in filename:
                filename = filename.split("/")[-1]
            dest = channel_dir / filename
            content = await file.read()
            dest.write_bytes(content)
            uploaded_count += 1
        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {
        "uploaded": uploaded_count,
        "errors": errors,
        "channel": channel_name,
        "channel_path": str(channel_dir),
    }

# ============================================================================
# Stats & Validation
# ============================================================================

@router.post("/stats", response_model=ApiResponse)
async def get_dataset_stats_post(request: StatsRequest):
    """Get dataset statistics (POST version for frontend compatibility)."""
    return await _get_stats_internal(request.path)


@router.get("/stats", response_model=ApiResponse)
async def get_dataset_stats(path: str):
    """Get dataset statistics (GET version)."""
    return await _get_stats_internal(path)


async def _get_stats_internal(path: str):
    """Shared stats implementation."""
    from ..application.dataset_usecases import GetDatasetStatsUseCase

    repo = _get_repo()
    use_case = GetDatasetStatsUseCase(repo)
    stats = use_case.execute(path)

    # Convert DatasetStats to frontend format
    if hasattr(stats, '__dict__'):
        p = Path(path) if not Path(path).is_absolute() else Path(path)
        resolved = repo._resolve_path(path)

        # Count latent and text caches
        latent_count = 0
        text_count = 0
        if resolved.exists():
            for img in resolved.rglob("*"):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    if _check_latent_cache(str(img)):
                        latent_count += 1
                    if _check_text_cache(str(img)):
                        text_count += 1

        return ApiResponse(success=True, data={
            "totalImages": stats.total_images,
            "totalSize": stats.total_size_bytes,
            "cachedImages": stats.cached_images,
            "captionCoverage": stats.caption_coverage,
            "totalLatentCached": latent_count,
            "totalTextCached": text_count,
        })
    return ApiResponse(success=True, data=stats)


@router.post("/validate", response_model=ApiResponse)
async def validate_dataset_path(request: ValidatePathRequest):
    """Validate a path as a valid training dataset."""
    repo = _get_repo()
    valid, message = repo.validate_path(request.path)
    return ApiResponse(success=valid, message=message)


_bucket_runner = AsyncTaskRunner("bucket_apply")


@router.post("/buckets", response_model=ApiResponse)
@router.post("/calculate-buckets")
async def calculate_buckets(request: BucketCalculateRequest):
    """Calculate aspect ratio buckets with optional fill strategy preview."""
    from ..application.dataset_usecases import CalculateBucketsUseCase

    try:
        from ..infrastructure.container import container
        use_case = CalculateBucketsUseCase(container.bucket_calculator())
        buckets = use_case.execute(
            request.path, request.batch_size, request.resolution_limit
        )

        bs = request.batch_size
        strategy = request.fill_strategy

        # Build bucket list with images (need filenames for apply)
        bucket_list = []
        for b in buckets:
            bucket_list.append({
                "width": b.width,
                "height": b.height,
                "aspectRatio": round(b.width / b.height, 4) if b.height > 0 else 1.0,
                "count": b.count,
                "original": b.count,
                "repeated": 0,
                "croppedIn": 0,
                "croppedOut": 0,
                "images": b.images,  # first N filenames for preview
            })

        if strategy == "repeat":
            # Pure repeat: oversample to fill incomplete batches
            for bk in bucket_list:
                remainder = bk["count"] % bs
                if remainder > 0:
                    fill = bs - remainder
                    bk["count"] += fill
                    bk["repeated"] = fill

        elif strategy == "crop":
            # Pure crop-redistribute: multi-phase optimal pairing
            # Phase 1: perfect merge — move remainder to bucket where result % bs == 0
            for _ in range(20):
                moved = False
                for i, bk in enumerate(bucket_list):
                    remainder = bk["count"] % bs
                    if remainder == 0:
                        continue
                    src_ar = bk["aspectRatio"]
                    best_j = -1
                    best_dist = float('inf')
                    for j, other in enumerate(bucket_list):
                        if j == i:
                            continue
                        if (other["count"] + remainder) % bs == 0:
                            d = abs(other["aspectRatio"] - src_ar)
                            if d < best_dist:
                                best_dist = d
                                best_j = j
                    if best_j >= 0 and best_dist < 1.5:
                        bucket_list[best_j]["count"] += remainder
                        bucket_list[best_j]["croppedIn"] += remainder
                        bk["count"] -= remainder
                        bk["croppedOut"] += remainder
                        moved = True
                if not moved:
                    break

            # Phase 2: pair two remainder buckets whose remainders sum to bs
            remainder_indices = [i for i, bk in enumerate(bucket_list) if bk["count"] % bs != 0]
            paired = set()
            for idx_a in remainder_indices:
                if idx_a in paired:
                    continue
                ra = bucket_list[idx_a]["count"] % bs
                if ra == 0:
                    continue
                need = bs - ra  # we need a bucket with exactly `need` remainder
                best_j = -1
                best_dist = float('inf')
                for idx_b in remainder_indices:
                    if idx_b == idx_a or idx_b in paired:
                        continue
                    rb = bucket_list[idx_b]["count"] % bs
                    if rb == need:
                        d = abs(bucket_list[idx_b]["aspectRatio"] - bucket_list[idx_a]["aspectRatio"])
                        if d < best_dist:
                            best_dist = d
                            best_j = idx_b
                if best_j >= 0 and best_dist < 1.5:
                    move = bucket_list[best_j]["count"] % bs
                    bucket_list[idx_a]["count"] += move
                    bucket_list[idx_a]["croppedIn"] += move
                    bucket_list[best_j]["count"] -= move
                    bucket_list[best_j]["croppedOut"] += move
                    paired.add(idx_a)
                    paired.add(best_j)

            # Phase 3: remaining — move to nearest bucket that reduces total drops
            for _ in range(10):
                moved = False
                for i, bk in enumerate(bucket_list):
                    remainder = bk["count"] % bs
                    if remainder == 0:
                        continue
                    src_ar = bk["aspectRatio"]
                    best_j = -1
                    best_dist = float('inf')
                    for j, other in enumerate(bucket_list):
                        if j == i:
                            continue
                        or_ = other["count"] % bs
                        nr = (other["count"] + remainder) % bs
                        if nr < remainder + or_:
                            d = abs(other["aspectRatio"] - src_ar)
                            if d < best_dist:
                                best_dist = d
                                best_j = j
                    if best_j >= 0 and best_dist < 1.5:
                        bucket_list[best_j]["count"] += remainder
                        bucket_list[best_j]["croppedIn"] += remainder
                        bk["count"] -= remainder
                        bk["croppedOut"] += remainder
                        moved = True
                if not moved:
                    break

        # Final stats
        total_images = sum(bk["original"] for bk in bucket_list)
        total_effective = sum(bk["count"] for bk in bucket_list)
        total_repeated = sum(bk["repeated"] for bk in bucket_list)
        total_cropped = sum(bk["croppedOut"] for bk in bucket_list)

        result = []
        for bk in bucket_list:
            if bk["count"] == 0:
                continue
            batches = bk["count"] // bs
            dropped = bk["count"] % bs
            result.append({
                "width": bk["width"],
                "height": bk["height"],
                "aspectRatio": bk["aspectRatio"],
                "count": bk["count"],
                "original": bk["original"],
                "batches": batches,
                "dropped": dropped,
                "repeated": bk["repeated"],
                "croppedIn": bk["croppedIn"],
                "croppedOut": bk["croppedOut"],
                "percentage": round(bk["count"] / total_effective * 100, 1) if total_effective > 0 else 0,
            })

        return {
            "success": True,
            "buckets": result,
            "summary": {
                "totalImages": total_images,
                "totalEffective": total_effective,
                "totalRepeated": total_repeated,
                "totalCropped": total_cropped,
                "totalDropped": sum(bk["dropped"] for bk in result),
                "strategy": strategy,
            }
        }
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


# --- Bucket Strategy Apply ---

class BucketApplyRequest(BaseModel):
    path: str
    batch_size: int = 4
    resolution_limit: int = 1536
    strategy: str  # "repeat" or "crop"


@router.post("/buckets/apply")
async def apply_bucket_strategy(request: BucketApplyRequest):
    """Apply bucket fill strategy to actual files."""
    from PIL import Image as PILImage
    from ..infrastructure.bucket_calculator import FileBucketCalculator, BUCKET_RESOLUTIONS, IMAGE_EXTENSIONS

    if _bucket_runner.is_running:
        return {"success": False, "message": "分桶处理任务已在运行中"}

    dataset_path = Path(request.path)
    if not dataset_path.exists():
        return {"success": False, "message": "数据集路径不存在"}

    # Compute full bucket assignment (with all filenames)
    calculator = FileBucketCalculator()
    bs = request.batch_size

    # We need full image->bucket assignment, rebuild it here
    valid_buckets = [
        (w, h) for w, h in BUCKET_RESOLUTIONS
        if w <= request.resolution_limit and h <= request.resolution_limit
    ]
    if not valid_buckets:
        valid_buckets = [(1024, 1024)]

    from collections import defaultdict
    bucket_assignments: dict = defaultdict(list)
    for f in sorted(dataset_path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith('.') for part in f.relative_to(dataset_path).parts):
            continue
        try:
            with PILImage.open(f) as img:
                w, h = img.size
        except Exception:
            continue
        img_ar = w / h if h > 0 else 1.0
        best_bucket = min(valid_buckets, key=lambda b: abs(b[0] / b[1] - img_ar))
        bucket_assignments[best_bucket].append(f)

    if request.strategy == "repeat":
        # Duplicate images to fill incomplete batches
        tasks = []  # (src_path, dst_path)
        for (bw, bh), images in bucket_assignments.items():
            remainder = len(images) % bs
            if remainder > 0:
                fill = bs - remainder
                for k in range(fill):
                    src = images[k % len(images)]
                    dst = src.parent / f"{src.stem}_repeat_{k}{src.suffix}"
                    tasks.append(("copy", src, dst))

        if not tasks:
            return {"success": True, "total": 0, "message": "所有桶已完整，无需填充"}

        async def _do_repeat():
            import shutil
            _bucket_runner.status.total = len(tasks)
            _bucket_runner.status.completed = 0
            for _, src, dst in tasks:
                if _bucket_runner.is_cancelled:
                    break
                _bucket_runner.status.current_file = src.name
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, shutil.copy2, str(src), str(dst)
                    )
                    # Copy caption too if exists
                    for ext in ('.txt', '.caption'):
                        cap_src = src.with_suffix(ext)
                        cap_dst = dst.with_suffix(ext)
                        if cap_src.exists():
                            await asyncio.get_event_loop().run_in_executor(
                                None, shutil.copy2, str(cap_src), str(cap_dst)
                            )
                    _bucket_runner.status.completed += 1
                except Exception as e:
                    _bucket_runner.status.errors.append(f"{src.name}: {e}")
                    _bucket_runner.status.completed += 1

        _bucket_runner.start(_do_repeat())
        return {"success": True, "total": len(tasks)}

    elif request.strategy == "crop":
        # Find dropped images per bucket and crop to nearest bucket
        crop_tasks = []  # (src_path, target_w, target_h)
        sorted_buckets = sorted(bucket_assignments.keys())

        for (bw, bh), images in bucket_assignments.items():
            remainder = len(images) % bs
            if remainder == 0:
                continue

            # Images to be moved = last `remainder` items
            dropped_images = images[-remainder:]
            src_ar = bw / bh

            # Find best target bucket
            best_target = None
            best_ar_dist = float('inf')
            for (tw, th) in sorted_buckets:
                if (tw, th) == (bw, bh):
                    continue
                ar_dist = abs(tw / th - src_ar)
                target_imgs = bucket_assignments[(tw, th)]
                other_remainder = len(target_imgs) % bs
                new_total = len(target_imgs) + remainder
                new_remainder = new_total % bs
                if new_remainder < remainder + other_remainder and ar_dist < best_ar_dist:
                    best_ar_dist = ar_dist
                    best_target = (tw, th)

            if best_target and best_ar_dist < 1.0:
                tw, th = best_target
                for img_path in dropped_images:
                    crop_tasks.append((img_path, tw, th))

        if not crop_tasks:
            return {"success": True, "total": 0, "message": "无法优化，所有桶余数无法通过裁切减少"}

        async def _do_crop():
            _bucket_runner.status.total = len(crop_tasks)
            _bucket_runner.status.completed = 0
            for img_path, tw, th in crop_tasks:
                if _bucket_runner.is_cancelled:
                    break
                _bucket_runner.status.current_file = img_path.name
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, _center_crop_image, img_path, tw, th
                    )
                    _bucket_runner.status.completed += 1
                except Exception as e:
                    _bucket_runner.status.errors.append(f"{img_path.name}: {e}")
                    _bucket_runner.status.completed += 1

        _bucket_runner.start(_do_crop())
        return {"success": True, "total": len(crop_tasks)}

    return {"success": False, "message": "未知策略"}


def _center_crop_image(img_path: Path, target_w: int, target_h: int):
    """Center-crop an image to match the target bucket's aspect ratio."""
    from PIL import Image as PILImage

    with PILImage.open(img_path) as img:
        orig_format = img.format or "JPEG"
        exif_data = img.info.get('exif')
        w, h = img.size

        target_ar = target_w / target_h
        img_ar = w / h

        if abs(img_ar - target_ar) < 0.01:
            return  # Already matches

        if img_ar > target_ar:
            # Image is wider → crop sides
            new_w = int(h * target_ar)
            left = (w - new_w) // 2
            cropped = img.crop((left, 0, left + new_w, h))
        else:
            # Image is taller → crop top/bottom
            new_h = int(w / target_ar)
            top = (h - new_h) // 2
            cropped = img.crop((0, top, w, top + new_h))

        # Save
        ext = img_path.suffix.lower()
        fmt_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
        save_format = fmt_map.get(ext, orig_format)
        save_kwargs = {"quality": 95}
        if exif_data and save_format == 'JPEG':
            save_kwargs["exif"] = exif_data
        if save_format == 'PNG':
            save_kwargs.pop('quality', None)

        cropped.save(str(img_path), format=save_format, **save_kwargs)


@router.get("/buckets/status")
async def bucket_apply_status():
    """Get bucket strategy apply status."""
    return _bucket_runner.get_status()


@router.post("/buckets/stop")
async def bucket_apply_stop():
    """Stop bucket strategy apply."""
    _bucket_runner.stop()
    return {"success": True}


# ============================================================================
# P1: Caption Read/Write
# ============================================================================

@router.get("/caption")
async def get_caption(path: str):
    """Read caption text for an image."""
    image_path = Path(path)
    caption_path = image_path.with_suffix('.txt')
    caption = ""
    if caption_path.exists():
        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return {"caption": caption}


@router.post("/caption")
async def save_caption(request: CaptionSaveRequest):
    """Save caption text for an image."""
    image_path = Path(request.path)
    caption_path = image_path.with_suffix('.txt')
    try:
        caption_path.write_text(request.caption, encoding="utf-8")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Image Serving
# ============================================================================

@router.get("/image")
async def serve_image(path: str):
    """Serve an image file."""
    from fastapi.responses import FileResponse
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(p))


# ============================================================================
# P2: Ollama Tagging / Image Resize / Generate Captions
# ============================================================================

# Shared task runners (singleton per feature)
_ollama_runner = AsyncTaskRunner("ollama_tagging")
_resize_runner = AsyncTaskRunner("image_resize")


# --- Ollama Tagging ---


@router.get("/ollama/models")
async def ollama_models(ollama_url: str = "http://127.0.0.1:11434"):
    """List available Ollama models (proxy to Ollama API)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return {"success": True, "models": model_names}
    except Exception as e:
        return {"success": False, "models": [], "error": str(e)}

@router.post("/ollama/tag")
async def ollama_tag(request: OllamaTagRequest):
    """Start Ollama tagging on dataset images."""
    import httpx

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail="数据集路径不存在")

    # Collect images to tag
    images_to_tag = []
    for f in sorted(dataset_path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith('.') for part in f.relative_to(dataset_path).parts):
            continue
        # Skip if already has caption and skip_existing is True
        if request.skip_existing and f.with_suffix('.txt').exists():
            continue
        images_to_tag.append(f)

    if not images_to_tag:
        return {"success": True, "total": 0, "message": "所有图片已有标注"}

    async def _do_ollama_tagging():
        import base64
        _ollama_runner.status.total = len(images_to_tag)
        _ollama_runner.status.completed = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for img_path in images_to_tag:
                if _ollama_runner.is_cancelled:
                    break

                _ollama_runner.status.current_file = img_path.name
                try:
                    # Read and optionally resize image for Ollama
                    img_bytes = _prepare_image_for_ollama(img_path, request.max_long_edge)
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                    # Build prompt
                    prompt = request.prompt or "Describe this image in detail for training purposes."
                    if request.trigger_word:
                        prompt = f"The subject in this image is called '{request.trigger_word}'. {prompt}"

                    # Call Ollama API
                    response = await client.post(
                        f"{request.ollama_url}/api/generate",
                        json={
                            "model": request.model,
                            "prompt": prompt,
                            "images": [img_b64],
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                    caption_text = result.get("response", "").strip()
                    if request.trigger_word and caption_text:
                        # Prepend trigger word if not already present
                        if not caption_text.lower().startswith(request.trigger_word.lower()):
                            caption_text = f"{request.trigger_word}, {caption_text}"

                    # Write caption file
                    if caption_text:
                        caption_path = img_path.with_suffix('.txt')
                        caption_path.write_text(caption_text, encoding="utf-8")

                    _ollama_runner.status.completed += 1

                except Exception as e:
                    _ollama_runner.status.errors.append(f"{img_path.name}: {str(e)}")
                    _ollama_runner.status.completed += 1

    _ollama_runner.start(_do_ollama_tagging())
    return {"success": True, "total": len(images_to_tag)}


def _prepare_image_for_ollama(img_path: Path, max_long_edge: int) -> bytes:
    """Read and resize image for Ollama API."""
    from PIL import Image
    import io

    with Image.open(img_path) as img:
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        # Resize if needed
        w, h = img.size
        if max(w, h) > max_long_edge:
            ratio = max_long_edge / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


@router.get("/ollama/status")
async def ollama_status():
    """Get Ollama tagging status."""
    return _ollama_runner.get_status()


@router.post("/ollama/stop")
async def ollama_stop():
    """Stop Ollama tagging."""
    _ollama_runner.stop()
    return {"success": True}


# --- Image Resize ---

@router.post("/resize")
async def resize_images(request: ResizeRequest):
    """Start image resize on dataset."""
    from PIL import Image, ImageFilter

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail="数据集路径不存在")

    # Collect images that need resizing
    images_to_resize = []
    for f in sorted(dataset_path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith('.') for part in f.relative_to(dataset_path).parts):
            continue
        try:
            with Image.open(f) as img:
                w, h = img.size
                if max(w, h) > request.max_long_edge:
                    images_to_resize.append(f)
        except Exception:
            continue

    if not images_to_resize:
        return {"success": True, "total": 0, "message": "所有图片已符合尺寸"}

    async def _do_resize():
        _resize_runner.status.total = len(images_to_resize)
        _resize_runner.status.completed = 0

        for img_path in images_to_resize:
            if _resize_runner.is_cancelled:
                break

            _resize_runner.status.current_file = img_path.name
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, _resize_single_image, img_path,
                    request.max_long_edge, request.quality, request.sharpen
                )
                _resize_runner.status.completed += 1
            except Exception as e:
                _resize_runner.status.errors.append(f"{img_path.name}: {str(e)}")
                _resize_runner.status.completed += 1

    _resize_runner.start(_do_resize())
    return {"success": True, "total": len(images_to_resize)}


def _resize_single_image(img_path: Path, max_long_edge: int, quality: int, sharpen: float):
    """Resize a single image (runs in thread pool)."""
    from PIL import Image, ImageFilter

    with Image.open(img_path) as img:
        orig_format = img.format or "JPEG"
        # Preserve EXIF if available
        exif_data = img.info.get('exif')

        w, h = img.size
        if max(w, h) <= max_long_edge:
            return

        ratio = max_long_edge / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # Convert to RGB if saving as JPEG
        if img.mode in ('RGBA', 'P', 'LA') and orig_format.upper() == "JPEG":
            img = img.convert('RGB')

        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Apply sharpening if requested
        if sharpen > 0:
            resized = resized.filter(ImageFilter.UnsharpMask(
                radius=2, percent=int(sharpen * 100), threshold=3
            ))

        # Save, preserving format
        save_kwargs = {"quality": quality}
        if exif_data:
            save_kwargs["exif"] = exif_data

        # Determine save format
        ext = img_path.suffix.lower()
        fmt_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP', '.bmp': 'BMP'}
        save_format = fmt_map.get(ext, orig_format)

        if save_format == 'PNG':
            save_kwargs.pop('quality', None)
        elif save_format == 'BMP':
            save_kwargs.pop('quality', None)
            save_kwargs.pop('exif', None)

        resized.save(str(img_path), format=save_format, **save_kwargs)


@router.get("/resize/status")
async def resize_status():
    """Get resize status."""
    return _resize_runner.get_status()


@router.post("/resize/stop")
async def resize_stop():
    """Stop resize."""
    _resize_runner.stop()
    return {"success": True}


# --- Generate Captions (stub — requires model inference) ---
@router.post("/generate-captions")
async def generate_captions(request: GenerateCaptionsRequest):
    """Generate captions using AI model (stub — requires GPU inference)."""
    return ApiResponse(success=False, message="AI 自动标注功能需要加载推理模型，暂未实现").model_dump()

