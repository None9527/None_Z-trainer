# -*- coding: utf-8 -*-
"""
System Router - Thin API Layer

Handles HTTP requests for system monitoring and model management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from .dto import ApiResponse

router = APIRouter(prefix="/api/system", tags=["System"])


class ModelOpsRequest(BaseModel):
    model_type: str = "zimage"


@router.get("/status", response_model=ApiResponse)
async def get_system_status():
    """Get overall system status."""
    from ..infrastructure.container import container
    from ..application.system_usecases import GetSystemStatusUseCase

    use_case = GetSystemStatusUseCase(
        container.gpu_monitor(),
        container.system_info_provider(),
    )
    result = use_case.execute()
    return ApiResponse(success=True, data=result)


@router.get("/gpu", response_model=ApiResponse)
async def get_gpu_info():
    """Get GPU information."""
    from ..infrastructure.container import container

    monitor = container.gpu_monitor()
    gpus = monitor.get_gpu_info()
    return ApiResponse(
        success=True,
        data=[
            {
                "index": g.index,
                "name": g.name,
                "memory_total": g.memory_total_gb,
                "memory_used": g.memory_used_gb,
                "memory_percent": round(g.memory_percent, 1),
                "temperature": g.temperature,
                "utilization": g.utilization,
            }
            for g in gpus
        ],
    )


@router.get("/model/status", response_model=ApiResponse)
async def get_model_status(model_type: Optional[str] = "zimage"):
    """Check local model status with detailed component validation."""
    from ..infrastructure.container import container
    from ..application.system_usecases import GetModelStatusUseCase

    use_case = GetModelStatusUseCase(container.model_manager())
    info = use_case.execute(model_type or "zimage")
    return ApiResponse(
        success=True,
        data={
            "model_type": info.model_type,
            "status": info.status.value,
            "path": info.path,
            "missing_files": info.missing_files,
            "components": info.components,
        },
    )


@router.get("/model/list", response_model=ApiResponse)
async def list_supported_models():
    """List all supported model types."""
    from ..infrastructure.container import container

    mgr = container.model_manager()
    models = mgr.list_supported_models()
    result = []
    for mt in models:
        spec = mgr.get_model_spec(mt)
        if spec:
            spec["type_key"] = mt
            result.append(spec)
    return ApiResponse(success=True, data=result)


@router.post("/model/download", response_model=ApiResponse)
async def download_model(request: Optional[ModelOpsRequest] = None):
    """Start model download from ModelScope."""
    from ..infrastructure.container import container

    model_type = request.model_type if request else "zimage"
    mgr = container.model_manager()
    success = mgr.start_download(model_type)

    if not success:
        return ApiResponse(success=False, message="Download already in progress or unsupported model")
    return ApiResponse(success=True, message="Download started")


@router.post("/model/download/cancel", response_model=ApiResponse)
async def cancel_download():
    """Cancel active model download."""
    from ..infrastructure.container import container

    mgr = container.model_manager()
    success = mgr.cancel_download()

    if not success:
        return ApiResponse(success=False, message="No active download")
    return ApiResponse(success=True, message="Download cancelled")


@router.get("/model/download/progress", response_model=ApiResponse)
async def get_download_progress():
    """Get download progress."""
    from ..infrastructure.container import container

    mgr = container.model_manager()
    progress = mgr.get_download_progress()

    if not progress:
        return ApiResponse(success=True, data={"status": "idle"})

    return ApiResponse(
        success=True,
        data={
            "status": progress.status.value,
            "model_type": progress.model_type,
            "progress_percent": round(progress.progress_percent, 1),
            "downloaded_mb": round(progress.downloaded_mb, 1),
            "speed_mbps": round(progress.speed_mbps, 2),
            "eta_seconds": progress.eta_seconds,
            "current_file": progress.current_file,
            "error_message": progress.error_message,
        },
    )
