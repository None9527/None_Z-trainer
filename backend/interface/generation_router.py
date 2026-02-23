# -*- coding: utf-8 -*-
"""
Generation Router - Thin API Layer

SSE streaming uses thread+queue pattern backed by a persistent TaskManager.
Tasks survive page navigation — clients can poll /api/generation/task/{id}.
"""

import asyncio
import base64
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from .dto import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Generation"])


# ── DTOs ──

class GenerationRequestDTO(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 10
    guidance_scale: float = 3.5
    seed: int = -1
    num_images: int = 1
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    transformer_path: Optional[str] = None
    comparison_mode: bool = False


class DeleteHistoryRequest(BaseModel):
    timestamps: List[str]


# ── Model endpoints ──

@router.get("/models", response_model=ApiResponse)
async def get_available_models():
    from ..infrastructure.container import container
    from ..application.generation_usecases import ListModelsUseCase
    use_case = ListModelsUseCase(container.model_repo())
    return ApiResponse(success=True, data=use_case.execute())


@router.get("/loras", response_model=ApiResponse)
async def get_loras():
    from ..infrastructure.container import container
    return ApiResponse(success=True, data=container.model_repo().list_loras())


@router.get("/transformers", response_model=ApiResponse)
async def get_transformers():
    from ..infrastructure.container import container
    return ApiResponse(success=True, data=container.model_repo().list_transformers())


@router.delete("/lora", response_model=ApiResponse)
async def delete_lora(path: str):
    from ..infrastructure.container import container
    from ..application.generation_usecases import DeleteModelUseCase
    use_case = DeleteModelUseCase(container.model_repo())
    if not use_case.execute(path):
        raise HTTPException(status_code=404, detail="Model not found")
    return ApiResponse(success=True, message="Model deleted")


# ── History endpoints ──

@router.get("/generation/history", response_model=ApiResponse)
async def get_generation_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Get generation history with pagination."""
    from ..infrastructure.container import container
    from ..application.generation_usecases import GetGenerationHistoryUseCase
    use_case = GetGenerationHistoryUseCase(container.generation_history_repo())
    offset = (page - 1) * page_size
    result = use_case.execute(offset=offset, limit=page_size)
    return ApiResponse(success=True, data={
        "items": result["items"],
        "total": result["total"],
        "page": page,
        "page_size": page_size,
    })


@router.post("/history/delete", response_model=ApiResponse)
async def delete_history_items(body: dict):
    """Delete history items by timestamps."""
    from ..infrastructure.container import container
    from ..infrastructure.config import GENERATION_OUTPUT_PATH

    timestamps = body.get("timestamps", [])
    if not timestamps:
        return ApiResponse(success=False, message="No timestamps provided")

    repo = container.generation_history_repo()
    deleted = repo.delete_history(timestamps)

    # Also delete image files
    output_dir = GENERATION_OUTPUT_PATH
    for ts in timestamps:
        for f in output_dir.glob(f"{ts}*.png"):
            try:
                f.unlink()
            except Exception:
                pass

    return ApiResponse(success=True, message=f"Deleted {deleted} items")


@router.get("/generation/image/{timestamp}")
async def serve_generation_image(timestamp: str):
    """Serve a generated image by timestamp prefix."""
    from ..infrastructure.config import GENERATION_OUTPUT_PATH
    output_dir = GENERATION_OUTPUT_PATH
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")
    matches = list(output_dir.glob(f"{timestamp}*.png"))
    if not matches:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(matches[0]), media_type="image/png")


# ── Task polling endpoint ──

@router.get("/generation/task/{task_id}")
async def get_task_status(task_id: str):
    """Poll task status by ID. Used when client reconnects after page navigation."""
    from ..infrastructure.generation_task_manager import task_manager
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data=task.to_dict())


@router.get("/generation/active-task")
async def get_active_task():
    """Check if there's a currently running generation task."""
    from ..infrastructure.generation_task_manager import task_manager
    task = task_manager.get_active_task()
    if task:
        return ApiResponse(success=True, data=task.to_dict())
    return ApiResponse(success=True, data=None)


# ── Non-streaming generation ──

@router.post("/generate", response_model=ApiResponse)
async def generate_image(req: GenerationRequestDTO):
    from ..infrastructure.container import container
    from ..application.generation_usecases import GenerateImageUseCase
    domain_req = _to_domain_request(req)
    use_case = GenerateImageUseCase(
        container.generation_pipeline(),
        container.generation_history_repo(),
    )
    results = use_case.execute(domain_req)
    return ApiResponse(success=True, data=results)


# ── SSE streaming generation (thread + queue + task manager) ──

@router.post("/generate-stream")
async def generate_image_stream(req: GenerationRequestDTO):
    """SSE streaming endpoint backed by task manager.

    Protocol:
    - First event: data: {"task_id": "abc123", "stage": "pending", ...}
    - Progress:    data: {"stage": "loading|generating|completed|error", "step": N, "total": N, ...}
    - Result:      data: {"success": true, "image": "base64...", "seed": N, ...}
    """
    from ..infrastructure.container import container
    from ..infrastructure.generation_task_manager import task_manager, TaskState

    domain_req = _to_domain_request(req)
    progress_queue: queue.Queue = queue.Queue()

    # Create tracked task
    task = task_manager.create_task(total_steps=req.num_inference_steps)
    task_id = task.task_id

    def do_generate():
        """Run in a background thread."""
        try:
            pipeline = container.generation_pipeline()

            # 1. Loading
            task_manager.update_task(task_id,
                state=TaskState.LOADING,
                message="Loading pipeline..."
            )
            progress_queue.put({
                "task_id": task_id,
                "stage": "loading",
                "step": 0,
                "total": req.num_inference_steps,
                "message": "Loading pipeline..."
            })

            pipeline.load(
                transformer_path=domain_req.transformer_path,
            )

            # Determine if this is a comparison run
            is_comparison = req.comparison_mode and req.lora_path

            if is_comparison:
                _do_comparison_generate(pipeline, domain_req, req, task_id, progress_queue, task_manager, TaskState)
            else:
                _do_single_generate(pipeline, domain_req, req, task_id, progress_queue, task_manager, TaskState)

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            task_manager.update_task(task_id,
                state=TaskState.FAILED,
                error=str(e),
                completed_at=time.time()
            )
            progress_queue.put({
                "task_id": task_id,
                "stage": "error",
                "step": 0,
                "total": req.num_inference_steps,
                "message": f"Error: {str(e)}"
            })
        finally:
            progress_queue.put(None)  # End signal

    async def sse_generator():
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()

        # Send task_id immediately so frontend can save it
        yield f"data: {json.dumps({'task_id': task_id, 'stage': 'pending', 'step': 0, 'total': req.num_inference_steps, 'message': 'Task created'})}\n\n"

        timeout_counter = 0.0
        max_timeout = 600.0

        try:
            while timeout_counter < max_timeout:
                try:
                    item = progress_queue.get_nowait()
                    timeout_counter = 0

                    if item is None:
                        break

                    yield f"data: {json.dumps(item)}\n\n"

                    if item.get("stage") in ("completed", "error"):
                        break

                except queue.Empty:
                    await asyncio.sleep(0.1)
                    timeout_counter += 0.1

            # Send final result
            final_task = task_manager.get_task(task_id)
            if final_task and final_task.result:
                yield f"data: {json.dumps(final_task.result)}\n\n"
            elif final_task and final_task.error:
                yield f"data: {json.dumps({'success': False, 'error': final_task.error})}\n\n"
            elif timeout_counter >= max_timeout:
                yield f"data: {json.dumps({'success': False, 'error': 'Generation timeout'})}\n\n"

        except asyncio.CancelledError:
            # Client disconnected — task continues in background thread!
            logger.info(f"SSE disconnected for task {task_id}, generation continues in background")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Generation Helpers ──

def _do_single_generate(pipeline, domain_req, req, task_id, progress_queue, task_manager, TaskState):
    """Single image generation (normal mode)."""
    task_manager.update_task(task_id,
        state=TaskState.GENERATING,
        message="Starting generation..."
    )
    progress_queue.put({
        "task_id": task_id,
        "stage": "generating",
        "step": 0,
        "total": req.num_inference_steps,
        "message": "Starting generation..."
    })

    def on_step(step: int, total: int):
        task_manager.update_task(task_id,
            step=step, total_steps=total,
            message=f"Step {step}/{total}"
        )
        progress_queue.put({
            "task_id": task_id, "stage": "generating",
            "step": step, "total": total,
            "message": f"Step {step}/{total}"
        })

    results = pipeline.generate_with_callback(domain_req, on_step)

    # Save to history
    from ..infrastructure.container import container
    history_repo = container.generation_history_repo()
    for r in results:
        history_repo.save_result(r)

    if results:
        r = results[0]
        img_path = Path(r.image_path)
        img_b64 = base64.b64encode(img_path.read_bytes()).decode() if img_path.exists() else ""

        result_data = {
            "success": True,
            "image": img_b64,
            "seed": r.seed,
            "timestamp": r.timestamp,
            "comparison_mode": False,
        }

        task_manager.update_task(task_id,
            state=TaskState.COMPLETED,
            step=req.num_inference_steps,
            result=result_data,
            completed_at=time.time(),
            message="Completed!"
        )
        progress_queue.put({
            "task_id": task_id, "stage": "completed",
            "step": req.num_inference_steps,
            "total": req.num_inference_steps,
            "message": "Completed!"
        })
    else:
        task_manager.update_task(task_id,
            state=TaskState.FAILED,
            error="No image generated",
            completed_at=time.time()
        )
        progress_queue.put({
            "task_id": task_id, "stage": "error",
            "message": "No image generated"
        })


def _do_comparison_generate(pipeline, domain_req, req, task_id, progress_queue, task_manager, TaskState):
    """Two-pass comparison: generate without LoRA, then with LoRA (same seed)."""
    from ..domain.generation.entities import GenerationRequest
    from ..infrastructure.container import container
    import torch

    total_steps = req.num_inference_steps
    double_total = total_steps * 2  # Two passes

    # Pre-resolve seed so both passes use the same value
    fixed_seed = domain_req.seed
    if fixed_seed < 0:
        fixed_seed = int(torch.randint(0, 2**32, (1,)).item())

    # ── Pass 1: Without LoRA ──
    task_manager.update_task(task_id,
        state=TaskState.GENERATING,
        message="[1/2] 生成原始图..."
    )
    progress_queue.put({
        "task_id": task_id, "stage": "generating",
        "step": 0, "total": double_total,
        "message": "[1/2] 生成原始图 (无 LoRA)..."
    })

    # Create request WITHOUT LoRA
    req_no_lora = GenerationRequest(
        prompt=domain_req.prompt,
        negative_prompt=domain_req.negative_prompt,
        width=domain_req.width,
        height=domain_req.height,
        num_inference_steps=domain_req.num_inference_steps,
        guidance_scale=domain_req.guidance_scale,
        seed=fixed_seed,
        num_images=1,
        lora_path=None,
        lora_scale=1.0,
        transformer_path=domain_req.transformer_path,
    )

    def on_step_pass1(step: int, total: int):
        task_manager.update_task(task_id,
            step=step, total_steps=double_total,
            message=f"[1/2] Step {step}/{total}"
        )
        progress_queue.put({
            "task_id": task_id, "stage": "generating",
            "step": step, "total": double_total,
            "message": f"[1/2] Step {step}/{total}"
        })

    results_no_lora = pipeline.generate_with_callback(req_no_lora, on_step_pass1)
    print(f"[TIMING] Pass 1 complete", flush=True)

    # ── Pass 2: With LoRA ──
    task_manager.update_task(task_id,
        state=TaskState.GENERATING,
        message="[2/2] 生成 LoRA 图..."
    )
    progress_queue.put({
        "task_id": task_id, "stage": "generating",
        "step": total_steps, "total": double_total,
        "message": "[2/2] 生成 LoRA 对比图..."
    })

    # Create request WITH LoRA (same seed to ensure fair comparison)
    req_with_lora = GenerationRequest(
        prompt=domain_req.prompt,
        negative_prompt=domain_req.negative_prompt,
        width=domain_req.width,
        height=domain_req.height,
        num_inference_steps=domain_req.num_inference_steps,
        guidance_scale=domain_req.guidance_scale,
        seed=fixed_seed,
        num_images=1,
        lora_path=domain_req.lora_path,
        lora_scale=domain_req.lora_scale,
        transformer_path=domain_req.transformer_path,
    )
    print(f"[TIMING] Starting pass 2 generate_with_callback", flush=True)

    def on_step_pass2(step: int, total: int):
        task_manager.update_task(task_id,
            step=total_steps + step, total_steps=double_total,
            message=f"[2/2] Step {step}/{total}"
        )
        progress_queue.put({
            "task_id": task_id, "stage": "generating",
            "step": total_steps + step, "total": double_total,
            "message": f"[2/2] Step {step}/{total}"
        })

    results_with_lora = pipeline.generate_with_callback(req_with_lora, on_step_pass2)
    print(f"[TIMING] Pass 2 complete", flush=True)

    # ── Build comparison result ──
    if results_no_lora and results_with_lora:
        history_repo = container.generation_history_repo()

        # Save as a single grouped comparison entry
        r1 = results_no_lora[0]
        r2 = results_with_lora[0]
        history_repo.save_comparison(r1, r2, req.lora_path, req.lora_scale)

        images = []
        # Image 1: without LoRA
        r1 = results_no_lora[0]
        p1 = Path(r1.image_path)
        b64_1 = base64.b64encode(p1.read_bytes()).decode() if p1.exists() else ""
        images.append({
            "image": b64_1,
            "lora_path": None,
            "lora_scale": 0,
        })

        # Image 2: with LoRA
        r2 = results_with_lora[0]
        p2 = Path(r2.image_path)
        b64_2 = base64.b64encode(p2.read_bytes()).decode() if p2.exists() else ""
        images.append({
            "image": b64_2,
            "lora_path": req.lora_path,
            "lora_scale": req.lora_scale,
        })

        result_data = {
            "success": True,
            "comparison_mode": True,
            "images": images,
            "seed": r2.seed,
            "timestamp": r2.timestamp,
        }

        task_manager.update_task(task_id,
            state=TaskState.COMPLETED,
            step=double_total,
            result=result_data,
            completed_at=time.time(),
            message="Comparison completed!"
        )
        progress_queue.put({
            "task_id": task_id, "stage": "completed",
            "step": double_total, "total": double_total,
            "message": "Comparison completed!"
        })
    else:
        task_manager.update_task(task_id,
            state=TaskState.FAILED,
            error="Comparison generation failed",
            completed_at=time.time()
        )
        progress_queue.put({
            "task_id": task_id, "stage": "error",
            "message": "Comparison generation failed"
        })


# ── Helpers ──

def _to_domain_request(req: GenerationRequestDTO):
    from ..domain.generation.entities import GenerationRequest
    return GenerationRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        num_images=req.num_images,
        lora_path=req.lora_path,
        lora_scale=req.lora_scale,
        transformer_path=req.transformer_path,
    )
