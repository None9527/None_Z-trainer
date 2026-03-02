# -*- coding: utf-8 -*-
"""
Skill Router — External API for programmatic image generation.

Provides a simple JSON API endpoint for skills, scripts, and external tools
to generate images via Z-Image pipeline.

POST /api/skill/generate → returns base64 PNG or image URL
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .dto import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skill", tags=["Skill API"])


# ── DTO ──

class SkillGenerateRequest(BaseModel):
    """Simplified generation request for external tools."""
    prompt: str = Field(..., description="Image description prompt")
    negative_prompt: str = Field("", description="Negative prompt (unwanted content)")
    width: int = Field(1024, ge=256, le=2048, description="Image width in pixels")
    height: int = Field(1024, ge=256, le=2048, description="Image height in pixels")
    steps: int = Field(10, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(3.5, ge=0, le=30, description="CFG guidance scale")
    seed: int = Field(-1, description="Random seed (-1 = random)")
    lora_name: str = Field("", description="LoRA filename (e.g. my_lora_epoch5.safetensors)")
    lora_scale: float = Field(1.0, ge=0, le=2.0, description="LoRA weight scale")
    response_format: str = Field("b64", description="Response format: 'b64' (base64) or 'url'")


def _align_to(value: int, multiple: int = 16) -> int:
    """Align a value to the nearest multiple (round to nearest)."""
    return max(multiple, round(value / multiple) * multiple)


@router.post("/generate", response_model=None)
async def skill_generate(req: SkillGenerateRequest):
    """
    Generate an image via Z-Image pipeline.

    Returns base64 PNG (default) or image URL.

    Example:
        curl -X POST http://localhost:28000/api/skill/generate \\
          -H "Content-Type: application/json" \\
          -d '{"prompt": "a cat sitting on a couch"}'
    """
    try:
        from ..infrastructure.container import container
        from ..application.generation_usecases import GenerateImageUseCase
        from ..domain.generation.entities import GenerationRequest

        # Align dimensions to 16px (VAE requirement)
        width = _align_to(req.width, 16)
        height = _align_to(req.height, 16)

        # Resolve LoRA name → full path
        lora_path = None
        if req.lora_name:
            from ..infrastructure.config import OUTPUT_PATH
            # Search in model-output directories
            candidates = list(OUTPUT_PATH.rglob(req.lora_name))
            if candidates:
                lora_path = str(candidates[0])
            else:
                return {"success": False, "error": f"LoRA not found: {req.lora_name}"}

        # Build domain request
        domain_req = GenerationRequest(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            lora_path=lora_path,
            lora_scale=req.lora_scale,
        )

        # Execute generation
        pipeline = container.generation_pipeline()
        history_repo = container.generation_history_repo()
        use_case = GenerateImageUseCase(pipeline, history_repo)
        results = use_case.execute(domain_req)

        if not results:
            return {"success": False, "error": "No image generated"}

        result = results[0]

        # Build response
        response = {
            "success": True,
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "prompt": result.prompt,
        }

        if req.response_format == "url":
            # Return relative URL path
            timestamp = Path(result.image_path).stem.split("_")[0] + "_" + str(result.seed)
            response["url"] = f"/api/generation/image/{Path(result.image_path).stem}"
        else:
            # Return base64 PNG
            with open(result.image_path, "rb") as f:
                response["image"] = base64.b64encode(f.read()).decode("utf-8")

        return response

    except Exception as e:
        logger.error(f"Skill generate failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
