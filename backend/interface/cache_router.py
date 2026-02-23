# -*- coding: utf-8 -*-
"""
Cache Router - Latent/Text Cache Operations

Launches cache generation scripts as subprocesses and tracks progress
by parsing stdout lines like "Progress: {current}/{total}".

Features:
- Auto-detect GPU count and VRAM for optimal configuration
- Multi-GPU parallel processing (auto-distributes across GPUs)
- Real-time progress via WebSocket broadcast
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .dto import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["Cache"])


# ============================================================================
# Data Models
# ============================================================================

class CacheGenerateRequest(BaseModel):
    datasetPath: str
    modelPath: str = ""
    modelType: str = "zimage"
    generateLatent: bool = True
    generateText: bool = True
    resolution: int = 1024
    skipExisting: bool = True
    mode: str = "text2img"           # text2img | controlnet | img2img | inpaint | omni
    controlDir: Optional[str] = None  # controlnet condition image directory
    sourceDir: Optional[str] = None   # img2img source image directory
    maskDir: Optional[str] = None     # inpaint mask image directory
    conditionDirs: Optional[str] = None  # omni condition dirs (comma-separated)
    numConditionImages: int = 0          # omni number of condition images


class CacheClearRequest(BaseModel):
    datasetPath: str
    clearLatent: bool = True
    clearText: bool = True
    modelType: str = ""


@dataclass
class CachePhaseStatus:
    status: str = "idle"  # idle, running, completed, failed
    progress: int = 0
    current: int = 0
    total: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "progress": self.progress,
            "current": self.current,
            "total": self.total,
            "error": self.error,
        }


@dataclass
class CacheGenerationState:
    latent: CachePhaseStatus = field(default_factory=CachePhaseStatus)
    text: CachePhaseStatus = field(default_factory=CachePhaseStatus)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "latent": self.latent.to_dict(),
            "text": self.text.to_dict(),
        }

    def reset(self):
        self.latent = CachePhaseStatus()
        self.text = CachePhaseStatus()
        self._cancelled = False

    @property
    def is_running(self) -> bool:
        return self.latent.status == "running" or self.text.status == "running"


# Global state
_cache_state = CacheGenerationState()

PROGRESS_RE = re.compile(r"Progress:\s*(\d+)/(\d+)")


def get_cache_status() -> dict:
    """Public getter for websocket_manager."""
    return _cache_state.to_dict()


# ============================================================================
# GPU Detection
# ============================================================================

def detect_gpu_info() -> Dict[str, Any]:
    """Auto-detect GPU count, names, and VRAM using nvidia-smi."""
    result = {
        "num_gpus": 0,
        "gpus": [],
        "total_vram_gb": 0,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return result

        for line in proc.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(parts[2]),
                    "vram_free_mb": int(parts[3]),
                }
                result["gpus"].append(gpu)

        result["num_gpus"] = len(result["gpus"])
        result["total_vram_gb"] = round(
            sum(g["vram_total_mb"] for g in result["gpus"]) / 1024, 1
        )
    except Exception as e:
        logger.debug(f"GPU detection error: {e}")

    return result


# ============================================================================
# Subprocess Runner
# ============================================================================

def _run_cache_subprocess(
    model_path: str,
    dataset_path: str,
    generate_latent: bool,
    generate_text: bool,
    resolution: int,
    skip_existing: bool,
    mode: str = "text2img",
    control_dir: Optional[str] = None,
    source_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    condition_dirs: Optional[str] = None,
    num_condition_images: int = 0,
):
    """Run cache generation in a background thread using subprocess."""
    global _cache_state

    try:
        from ..infrastructure.config import VAE_PATH, TEXT_ENCODER_PATH
        vae_path = str(VAE_PATH)
        text_encoder_path = str(TEXT_ENCODER_PATH)
        output_dir = dataset_path

        backend_root = Path(__file__).parent.parent
        trainer_core = backend_root / "trainer_core"

        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{backend_root}:{trainer_core}:{python_path}"

        # Auto-detect GPU configuration
        gpu_info = detect_gpu_info()
        num_gpus = gpu_info["num_gpus"] or 1
        _cache_state.gpu_info = gpu_info

        if num_gpus > 1:
            logger.info(f"Multi-GPU detected: {num_gpus} GPUs, total VRAM: {gpu_info['total_vram_gb']}GB")
            for g in gpu_info["gpus"]:
                logger.info(f"  GPU {g['index']}: {g['name']} ({g['vram_total_mb']}MB total, {g['vram_free_mb']}MB free)")
        else:
            vram_info = f"{gpu_info['gpus'][0]['vram_total_mb']}MB" if gpu_info["gpus"] else "unknown"
            logger.info(f"Single GPU mode, VRAM: {vram_info}")

        # Phase 1: Latent cache
        if generate_latent and not _cache_state._cancelled:
            _cache_state.latent.status = "running"
            logger.info(f"Starting latent cache: vae={vae_path}, dir={dataset_path}, gpus={num_gpus}")

            cmd = [
                sys.executable, "-m", "zimage_trainer.cache_latents",
                "--vae", vae_path,
                "--input_dir", dataset_path,
                "--output_dir", output_dir,
                "--resolution", str(resolution),
                "--num_gpus", str(num_gpus),
                "--mode", mode,
            ]
            if skip_existing:
                cmd.append("--skip_existing")
            if mode == "controlnet" and control_dir:
                cmd.extend(["--control_dir", control_dir])
            if mode == "img2img" and source_dir:
                cmd.extend(["--source_dir", source_dir])
            if mode == "inpaint" and mask_dir:
                cmd.extend(["--mask_dir", mask_dir])
            if mode == "omni" and condition_dirs:
                cmd.extend(["--condition_dirs", condition_dirs])

            _run_phase(cmd, _cache_state.latent, env, cwd=str(trainer_core))

        # Phase 2: Text cache
        if generate_text and not _cache_state._cancelled:
            _cache_state.text.status = "running"
            logger.info(f"Starting text cache: encoder={text_encoder_path}, dir={dataset_path}, gpus={num_gpus}")

            cmd = [
                sys.executable, "-m", "zimage_trainer.cache_text_encoder",
                "--text_encoder", text_encoder_path,
                "--input_dir", dataset_path,
                "--output_dir", output_dir,
                "--num_gpus", str(num_gpus),
            ]
            if skip_existing:
                cmd.append("--skip_existing")
            if mode == "omni":
                cmd.extend(["--mode", mode, "--num_condition_images", str(num_condition_images)])

            _run_phase(cmd, _cache_state.text, env, cwd=str(trainer_core))

        # Phase 3: SigLIP cache (omni only)
        if mode == "omni" and condition_dirs and not _cache_state._cancelled:
            logger.info(f"Starting SigLIP cache for omni mode")
            siglip_path = str(Path(model_path) / "siglip")
            
            for cond_dir in condition_dirs.split(","):
                cond_dir = cond_dir.strip()
                if not cond_dir:
                    continue
                logger.info(f"  SigLIP encoding: {cond_dir}")
                cmd = [
                    sys.executable, "-m", "zimage_trainer.cache_siglip",
                    "--siglip_model", siglip_path,
                    "--input_dir", cond_dir,
                    "--output_dir", dataset_path,
                ]
                if skip_existing:
                    cmd.append("--skip_existing")
                _run_phase(cmd, _cache_state.text, env, cwd=str(trainer_core))

    except Exception as e:
        logger.error(f"Cache generation error: {e}")
        if _cache_state.latent.status == "running":
            _cache_state.latent.status = "failed"
            _cache_state.latent.error = str(e)
        if _cache_state.text.status == "running":
            _cache_state.text.status = "failed"
            _cache_state.text.error = str(e)


def _run_phase(cmd, phase: CachePhaseStatus, env: dict, cwd: str):
    """Run a single cache phase and parse progress from stdout."""
    global _cache_state

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            bufsize=1,
            text=True,
        )
        _cache_state._process = proc

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            logger.debug(f"[cache] {line}")

            m = PROGRESS_RE.search(line)
            if m:
                current = int(m.group(1))
                total = int(m.group(2))
                phase.current = current
                phase.total = total
                phase.progress = round(current / total * 100) if total > 0 else 0

            if _cache_state._cancelled:
                proc.terminate()
                phase.status = "idle"
                return

        proc.wait()

        if proc.returncode == 0:
            phase.status = "completed"
            phase.progress = 100
        else:
            phase.status = "failed"
            phase.error = f"Exit code {proc.returncode}"

    except Exception as e:
        phase.status = "failed"
        phase.error = str(e)
    finally:
        _cache_state._process = None


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/generate")
async def generate_cache(request: CacheGenerateRequest):
    """Start cache generation for a dataset."""
    global _cache_state

    if _cache_state.is_running:
        return ApiResponse(success=False, message="缓存生成已在运行中").model_dump()

    # Resolve model path
    model_path = request.modelPath
    if not model_path:
        from ..infrastructure.config import MODEL_PATH
        model_path = str(MODEL_PATH)

    # Validate
    if not Path(model_path).exists():
        return ApiResponse(success=False, message=f"模型路径不存在: {model_path}").model_dump()
    if not Path(request.datasetPath).exists():
        return ApiResponse(success=False, message=f"数据集路径不存在: {request.datasetPath}").model_dump()

    # Reset and start
    _cache_state.reset()

    thread = threading.Thread(
        target=_run_cache_subprocess,
        args=(
            model_path,
            request.datasetPath,
            request.generateLatent,
            request.generateText,
            request.resolution,
            request.skipExisting,
            request.mode,
            request.controlDir,
            request.sourceDir,
            request.maskDir,
            request.conditionDirs,
            request.numConditionImages,
        ),
        daemon=True,
    )
    _cache_state._thread = thread
    thread.start()

    return {"success": True, "message": "缓存生成已启动"}


@router.get("/status")
async def cache_status():
    """Get cache generation status."""
    return _cache_state.to_dict()


@router.get("/gpu-info")
async def gpu_info():
    """Get GPU detection info for cache generation."""
    info = detect_gpu_info()
    return {"success": True, "data": info}


@router.post("/stop")
async def stop_cache():
    """Stop cache generation."""
    global _cache_state

    if not _cache_state.is_running:
        return {"success": False, "message": "没有正在运行的缓存任务"}

    _cache_state._cancelled = True
    if _cache_state._process:
        _cache_state._process.terminate()

    _cache_state.latent.status = "idle"
    _cache_state.text.status = "idle"

    return {"success": True, "message": "缓存生成已停止"}


@router.post("/clear")
async def clear_cache(request: CacheClearRequest):
    """Clear cache files for a dataset."""
    p = Path(request.datasetPath)
    if not p.exists() or not p.is_dir():
        return ApiResponse(success=False, message="无效的数据集路径").model_dump()

    cleared = 0
    # Clear from .cache directory
    cache_dir = p / ".cache"
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file() and f.suffix == ".safetensors":
                is_latent = "latent" in f.name or (f.name.endswith("_zi.safetensors") and "_te" not in f.name)
                is_text = "text" in f.name or "_te" in f.name
                if (request.clearLatent and is_latent) or (request.clearText and is_text):
                    f.unlink()
                    cleared += 1

    # Clear {name}_{WxH}_zi.safetensors (latent) and {name}_zi_te.safetensors (text)
    for f in p.rglob("*_zi.safetensors"):
        if "_te" not in f.name and request.clearLatent:
            f.unlink()
            cleared += 1
    for f in p.rglob("*_zi_te.safetensors"):
        if request.clearText:
            f.unlink()
            cleared += 1
    # Legacy patterns
    for f in p.rglob("*_zi_latent.safetensors"):
        if request.clearLatent:
            f.unlink()
            cleared += 1
    for f in p.rglob("*_zi_text.safetensors"):
        if request.clearText:
            f.unlink()
            cleared += 1

    return {"success": True, "cleared": cleared}
