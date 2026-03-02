# -*- coding: utf-8 -*-
"""
Z-Image Generation Pipeline — Production Implementation

Uses diffusers ZImagePipeline for txt2img generation.
Supports:
- Dynamic LoRA loading/unloading (no pipeline reload required)
- Automatic CPU offloading via enable_model_cpu_offload()
- Per-step progress callback for SSE streaming
"""

import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

import torch

from ..domain.generation.repositories import IGenerationPipeline
from ..domain.generation.entities import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)


class ZImageGenerationPipeline(IGenerationPipeline):
    """Real generation pipeline using diffusers ZImagePipeline.

    VRAM strategy (enable_model_cpu_offload):
    - All models live on CPU, moved to GPU one-by-one during inference
    - text_encoder → GPU (encode) → CPU
    - transformer  → GPU (denoise) → CPU
    - VAE          → GPU (decode)  → CPU
    - Peak VRAM = transformer size (~12GB bf16)
    """

    def __init__(self):
        self._pipe = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_lora_path: Optional[str] = None
        self._loaded_lora_scale: Optional[float] = None
        self._output_dir: Optional[Path] = None
        self._original_weights: Dict[str, torch.Tensor] = {}

    # ── Pipeline lifecycle ──

    def load(self, model_type: str = "zimage", transformer_path: Optional[str] = None) -> None:
        """Load the Z-Image pipeline (BF16 + CPU offload).

        Args:
            model_type: Model type (only zimage supported)
            transformer_path: Optional finetune weights path
        """
        from ..infrastructure.config import MODEL_PATH, GENERATION_OUTPUT_PATH

        model_path = transformer_path or str(MODEL_PATH)

        # Skip if already loaded with same model
        if self._pipe is not None and self._loaded_model_path == model_path:
            logger.info("Pipeline already loaded, reusing")
            return

        # Must reload for different model
        if self._pipe is not None:
            self.unload()

        logger.info(f"Loading Z-Image pipeline from: {model_path}")
        t0 = time.time()

        try:
            from diffusers import ZImagePipeline

            self._pipe = ZImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

            # Enable automatic CPU offloading — models move to GPU
            # one-by-one during inference, minimizing peak VRAM
            self._pipe.enable_model_cpu_offload()

            self._loaded_model_path = model_path
            self._loaded_lora_path = None
            self._loaded_lora_scale = None
            self._output_dir = GENERATION_OUTPUT_PATH
            self._output_dir.mkdir(parents=True, exist_ok=True)

            vram_gb = torch.cuda.memory_allocated() / 1024**3
            elapsed = time.time() - t0
            logger.info(f"Pipeline loaded in {elapsed:.1f}s (VRAM={vram_gb:.1f}GB)")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self._pipe = None
            raise RuntimeError(f"Failed to load Z-Image pipeline: {e}")

    def unload(self) -> None:
        """Unload pipeline and free GPU memory."""
        if self._pipe is not None:
            logger.info("Unloading generation pipeline")
            del self._pipe
            self._pipe = None
            self._loaded_model_path = None
            self._loaded_lora_path = None
            self._loaded_lora_scale = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._pipe is not None

    # ── LoRA management (manual weight merging for ComfyUI-format LoRA) ──

    def _handle_lora(self, lora_path: Optional[str], lora_scale: float) -> None:
        """Dynamically load/unload LoRA without reloading pipeline."""
        print(f"[TIMING] _handle_lora: path={lora_path}, currently={self._loaded_lora_path}", flush=True)
        if not lora_path:
            if self._loaded_lora_path:
                self._unload_lora()
            return

        # Same LoRA with same scale → skip
        if lora_path == self._loaded_lora_path and lora_scale == self._loaded_lora_scale:
            return

        # Different LoRA or different scale → unload old first
        if self._loaded_lora_path:
            self._unload_lora()

        self._load_lora(lora_path, lora_scale)

    def _load_lora(self, lora_path: str, scale: float = 1.0) -> None:
        """Load ComfyUI-format LoRA weights and merge into transformer.

        Two-phase approach for speed with CPU offload:
        Phase 1: Compute all deltas on CUDA in float32 (fast batch matmul)
        Phase 2: Convert to CPU bfloat16 and apply (async non_blocking transfers)
        """
        if not Path(lora_path).exists():
            logger.warning(f"LoRA not found: {lora_path}")
            return

        try:
            from safetensors import safe_open

            t0 = time.time()

            # Load all tensors from safetensors
            lora_sd = {}
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lora_sd[key] = f.get_tensor(key)

            # Group by module: {module_name: {lora_down, lora_up, alpha}}
            modules = {}
            for key, tensor in lora_sd.items():
                name = key
                if name.startswith("diffusion_model."):
                    name = name[len("diffusion_model."):]

                if ".lora_down.weight" in name:
                    mod_name = name.replace(".lora_down.weight", "")
                    modules.setdefault(mod_name, {})["down"] = tensor
                elif ".lora_up.weight" in name:
                    mod_name = name.replace(".lora_up.weight", "")
                    modules.setdefault(mod_name, {})["up"] = tensor
                elif ".alpha" in name:
                    mod_name = name.replace(".alpha", "")
                    modules.setdefault(mod_name, {})["alpha"] = tensor.item()

            del lora_sd  # Free memory

            transformer = self._pipe.transformer
            module_dict = dict(transformer.named_modules())

            # Phase 1: Compute all deltas on CUDA in float32
            compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pending = []  # [(target_name, module, delta_gpu)]

            for mod_name, lora_data in modules.items():
                if "down" not in lora_data or "up" not in lora_data:
                    continue

                down = lora_data["down"]
                up = lora_data["up"]
                alpha = lora_data.get("alpha", float(down.shape[0]))
                dim = down.shape[0]
                lora_scale_factor = alpha / dim

                # Resolve module name (handle to_out.0.0 → to_out.0)
                target_name = mod_name
                if target_name not in module_dict:
                    if target_name.endswith(".0"):
                        alt = target_name[:-2]
                        if alt in module_dict:
                            target_name = alt

                module = module_dict.get(target_name)
                if module is None or not hasattr(module, "weight"):
                    continue

                delta = (up.to(device=compute_device, dtype=torch.float32)
                         @ down.to(device=compute_device, dtype=torch.float32))
                delta *= scale * lora_scale_factor
                pending.append((target_name, module, delta))

            print(f"[TIMING] Phase 1 (CUDA matmul): {len(pending)} deltas in {time.time()-t0:.2f}s", flush=True)

            # Phase 2: Batch convert and apply
            # Convert all deltas to CPU in one go, then apply
            t1 = time.time()
            merged_count = 0
            self._original_weights = {}

            # Batch: convert all deltas to target dtype/device at once
            for target_name, module, delta_gpu in pending:
                delta_cpu = delta_gpu.to(device=module.weight.device, dtype=module.weight.dtype,
                                         non_blocking=True)
                pending[merged_count] = (target_name, module, delta_cpu)  # Replace with CPU version
                merged_count += 1

            # Sync once after all async transfers
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Now apply all deltas (all already on CPU)
            for i in range(merged_count):
                target_name, module, delta_cpu = pending[i]
                self._original_weights[target_name] = delta_cpu
                module.weight.data.add_(delta_cpu)

            del pending
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[TIMING] Phase 2 (apply to CPU): {merged_count} modules in {time.time()-t1:.2f}s", flush=True)

            self._loaded_lora_path = lora_path
            self._loaded_lora_scale = scale
            elapsed = time.time() - t0
            print(f"[TIMING] LoRA merged: {merged_count}/{len(modules)} modules in {elapsed:.2f}s", flush=True)

        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}", exc_info=True)
            print(f"[ERROR] LoRA load failed: {e}", flush=True)
            self._rollback_lora()

    def _unload_lora(self) -> None:
        """Restore original weights to undo LoRA merge."""
        if not self._loaded_lora_path:
            return
        try:
            t0 = time.time()
            self._rollback_lora()
            elapsed = time.time() - t0
            print(f"[TIMING] LoRA unloaded: {elapsed:.2f}s", flush=True)
        except Exception as e:
            logger.warning(f"Failed to unload LoRA: {e}")
        finally:
            self._loaded_lora_path = None
            self._loaded_lora_scale = None

    def _rollback_lora(self) -> None:
        """Subtract stored deltas to undo LoRA merge."""
        if not self._original_weights or self._pipe is None:
            return
        transformer = self._pipe.transformer
        module_dict = dict(transformer.named_modules())
        for name, delta in self._original_weights.items():
            module = module_dict.get(name)
            if module is not None and hasattr(module, "weight"):
                module.weight.data.sub_(delta)
        self._original_weights = {}

    # ── Generation ──

    def generate(self, request: GenerationRequest) -> List[GenerationResult]:
        """Generate images (no progress callback)."""
        return self.generate_with_callback(request, None)

    def generate_with_callback(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerationResult]:
        """Generate images with optional per-step progress callback.

        CPU offloading is handled automatically by diffusers:
        each model component moves to GPU only during its forward pass.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Handle seed
        seed = request.seed
        if seed < 0:
            seed = torch.randint(0, 2**32, (1,)).item()

        # Handle LoRA dynamically
        self._handle_lora(request.lora_path, request.lora_scale)
        print(f"[TIMING] LoRA handled, starting generation", flush=True)

        logger.info(
            f"Generating: prompt='{request.prompt[:60]}...' "
            f"size={request.width}x{request.height} "
            f"steps={request.num_inference_steps} seed={seed}"
        )

        t0 = time.time()
        total_steps = request.num_inference_steps

        # Build diffusers callback for step progress
        def diffusers_step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step_index + 1, total_steps)
            return callback_kwargs

        # Align dimensions to 16px (VAE/diffusers requirement)
        req_width = max(16, round(request.width / 16) * 16)
        req_height = max(16, round(request.height / 16) * 16)

        results: List[GenerationResult] = []
        for i in range(request.num_images):
            img_seed = seed + i
            gen = torch.Generator(device="cuda").manual_seed(img_seed)

            # Build pipeline kwargs
            pipe_kwargs = dict(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                width=req_width,
                height=req_height,
                num_inference_steps=total_steps,
                guidance_scale=request.guidance_scale,
                generator=gen,
                callback_on_step_end=diffusers_step_callback,
            )

            output = self._pipe(**pipe_kwargs)

            image = output.images[0]

            # Save to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{img_seed}.png"
            save_path = self._output_dir / filename
            image.save(str(save_path))

            results.append(GenerationResult(
                timestamp=timestamp,
                image_path=str(save_path),
                prompt=request.prompt,
                seed=img_seed,
                width=req_width,
                height=req_height,
                steps=total_steps,
                guidance_scale=request.guidance_scale,
                lora_path=request.lora_path,
                lora_scale=request.lora_scale,
            ))

        elapsed = time.time() - t0
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Generated {len(results)} image(s) in {elapsed:.1f}s (VRAM={vram_gb:.1f}GB)")

        return results
