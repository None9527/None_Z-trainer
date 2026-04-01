# -*- coding: utf-8 -*-
"""
Z-Image Generation Pipeline

VRAM strategy — enable_model_cpu_offload():
  - Components live on CPU, moved to GPU one-at-a-time during forward pass
  - text_encoder → GPU (encode) → CPU
  - transformer  → GPU (denoise) → CPU
  - VAE          → GPU (decode)  → CPU
  - Peak VRAM ≈ transformer (~12GB bf16), not full stack (~20GB)

Lifecycle:
  load()     → from_pretrained (CPU) + enable_model_cpu_offload()
  generate() → inference (peak ~12GB) → schedule auto-unload
  unload()   → del pipe + empty_cache → 0GB

Auto-unload after 3 min idle frees GPU for training.
"""

import gc
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

import torch

from ..domain.generation.repositories import IGenerationPipeline
from ..domain.generation.entities import GenerationRequest, GenerationResult, LoRAConfig

logger = logging.getLogger(__name__)


class ZImageGenerationPipeline(IGenerationPipeline):
    """Z-Image generation with CPU offload and incremental multi-LoRA.

    Uses diffusers' native enable_model_cpu_offload() which is backed by
    accelerate's dispatch_model hooks. ZImagePipeline declares
    model_cpu_offload_seq = "text_encoder->transformer->vae".
    """

    IDLE_UNLOAD_SECONDS = 180  # 3 min idle → auto free GPU

    def __init__(self):
        self._pipe = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_loras: Dict[str, Dict[str, Any]] = {}
        self._output_dir: Optional[Path] = None
        self._unload_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    # ── Lifecycle ──

    def load(self, model_type: str = "zimage", transformer_path: Optional[str] = None) -> None:
        """Load pipeline with CPU offload. Peak VRAM ~12GB (transformer only)."""
        from ..infrastructure.config import MODEL_PATH, GENERATION_OUTPUT_PATH

        model_path = transformer_path or str(MODEL_PATH)

        self._cancel_unload_timer()

        if self._pipe is not None and self._loaded_model_path == model_path:
            logger.info("Pipeline already loaded, reusing")
            return

        if self._pipe is not None:
            self.unload()

        logger.info(f"Loading Z-Image pipeline from: {model_path}")
        t0 = time.time()

        try:
            from diffusers import ZImagePipeline

            # Step 1: Load all components onto CPU (no VRAM used)
            self._pipe = ZImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

            # Step 2: Enable CPU offload — diffusers + accelerate handle
            # moving each component to GPU only during its forward pass.
            # ZImagePipeline.model_cpu_offload_seq = "text_encoder->transformer->vae"
            self._pipe.enable_model_cpu_offload()

            self._loaded_model_path = model_path
            self._loaded_loras = {}
            self._output_dir = GENERATION_OUTPUT_PATH
            self._output_dir.mkdir(parents=True, exist_ok=True)

            vram_gb = torch.cuda.memory_allocated() / 1024**3
            elapsed = time.time() - t0
            logger.info(f"Pipeline ready in {elapsed:.1f}s (VRAM={vram_gb:.1f}GB, cpu offload active)")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            self._pipe = None
            raise RuntimeError(f"Failed to load Z-Image pipeline: {e}")

    def unload(self) -> None:
        """Unload pipeline, free all GPU memory."""
        self._cancel_unload_timer()
        if self._pipe is not None:
            vram_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"Unloading pipeline (VRAM: {vram_before:.1f}GB)")
            del self._pipe
            self._pipe = None
            self._loaded_model_path = None
            self._loaded_loras = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logger.info(f"Pipeline unloaded (freed {vram_before - vram_after:.1f}GB)")

    def is_loaded(self) -> bool:
        return self._pipe is not None

    # ── Auto-unload timer ──

    def _schedule_unload(self) -> None:
        self._cancel_unload_timer()
        self._unload_timer = threading.Timer(self.IDLE_UNLOAD_SECONDS, self._auto_unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()
        logger.info(f"Auto-unload in {self.IDLE_UNLOAD_SECONDS}s")

    def _cancel_unload_timer(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _auto_unload(self) -> None:
        with self._lock:
            if self._pipe is not None:
                logger.info("Auto-unloading idle pipeline")
                self.unload()

    # ── Multi-LoRA (incremental weight merging) ──

    def _handle_loras(self, lora_configs: List[LoRAConfig]) -> None:
        """Sync loaded LoRAs to match requested set (incremental diff)."""
        target_map = {cfg.path: cfg.scale for cfg in lora_configs}
        loaded_paths = set(self._loaded_loras.keys())
        target_paths = set(target_map.keys())

        to_remove = loaded_paths - target_paths
        to_add = target_paths - loaded_paths
        to_reload = {p for p in loaded_paths & target_paths
                     if self._loaded_loras[p]["scale"] != target_map[p]}

        if not to_remove and not to_add and not to_reload:
            if lora_configs:
                print(f"[TIMING] All {len(lora_configs)} LoRAs already loaded, skipping", flush=True)
            return

        print(f"[TIMING] _handle_loras: remove={len(to_remove)}, add={len(to_add)}, "
              f"reload={len(to_reload)}, total_target={len(lora_configs)}", flush=True)

        for path in to_remove | to_reload:
            self._unload_single_lora(path)
        for path in to_add | to_reload:
            self._load_single_lora(LoRAConfig(path=path, scale=target_map[path]))

    def _load_single_lora(self, config: LoRAConfig) -> None:
        """Compute LoRA deltas on CPU, apply to weights (CPU with offload)."""
        lora_path = config.path
        scale = config.scale

        if not Path(lora_path).exists():
            logger.warning(f"LoRA not found: {lora_path}")
            return

        try:
            from safetensors import safe_open

            t0 = time.time()

            lora_sd = {}
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lora_sd[key] = f.get_tensor(key)

            # Group by module
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

            del lora_sd

            transformer = self._pipe.transformer
            module_dict = dict(transformer.named_modules())

            # Phase 1: Compute deltas on CPU (LoRA matrices are small, CPU is fast)
            pending = []
            for mod_name, lora_data in modules.items():
                if "down" not in lora_data or "up" not in lora_data:
                    continue

                down = lora_data["down"]
                up = lora_data["up"]
                alpha = lora_data.get("alpha", float(down.shape[0]))
                dim = down.shape[0]
                lora_scale_factor = alpha / dim

                target_name = mod_name
                if target_name not in module_dict:
                    if target_name.endswith(".0"):
                        alt = target_name[:-2]
                        if alt in module_dict:
                            target_name = alt

                module = module_dict.get(target_name)
                if module is None or not hasattr(module, "weight"):
                    continue

                delta = (up.to(dtype=torch.float32) @ down.to(dtype=torch.float32))
                delta *= scale * lora_scale_factor
                pending.append((target_name, delta))

            lora_name = Path(lora_path).stem
            print(f"[TIMING] [{lora_name}] Phase 1 (CPU matmul): "
                  f"{len(pending)} deltas in {time.time()-t0:.2f}s", flush=True)

            # Phase 2: Apply deltas to weights
            # With enable_model_cpu_offload(), weights live on CPU (managed by
            # accelerate hooks, moved to GPU only during forward pass).
            t1 = time.time()
            merged_count = 0
            stored_deltas: Dict[str, torch.Tensor] = {}

            for target_name, delta_f32 in pending:
                module = module_dict.get(target_name)
                if module is None or not hasattr(module, "weight"):
                    continue

                delta_store = delta_f32.to(dtype=module.weight.dtype)
                stored_deltas[target_name] = delta_store.cpu()

                # Apply delta on whatever device the weight currently lives on
                orig_device = module.weight.device
                module.weight.data.add_(delta_store.to(orig_device))
                merged_count += 1

            print(f"[TIMING] [{lora_name}] Phase 2 (apply): "
                  f"{merged_count} modules in {time.time()-t1:.2f}s", flush=True)

            self._loaded_loras[lora_path] = {
                "scale": scale,
                "deltas": stored_deltas,
            }

            elapsed = time.time() - t0
            print(f"[TIMING] [{lora_name}] merged: "
                  f"{merged_count}/{len(modules)} modules in {elapsed:.2f}s", flush=True)

        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_path}: {e}", exc_info=True)
            print(f"[ERROR] LoRA load failed: {e}", flush=True)
            if lora_path in self._loaded_loras:
                self._unload_single_lora(lora_path)
            raise RuntimeError(f"Failed to load LoRA {Path(lora_path).stem}: {str(e)}")

    def _unload_single_lora(self, lora_path: str) -> None:
        """Subtract stored deltas to undo LoRA merge."""
        entry = self._loaded_loras.get(lora_path)
        if not entry or self._pipe is None:
            self._loaded_loras.pop(lora_path, None)
            return

        try:
            t0 = time.time()
            transformer = self._pipe.transformer
            module_dict = dict(transformer.named_modules())

            for name, delta in entry["deltas"].items():
                module = module_dict.get(name)
                if module is not None and hasattr(module, "weight"):
                    orig_device = module.weight.device
                    module.weight.data.sub_(delta.to(orig_device))

            lora_name = Path(lora_path).stem
            print(f"[TIMING] [{lora_name}] unloaded: {time.time()-t0:.2f}s", flush=True)

        except Exception as e:
            logger.warning(f"Failed to unload LoRA {lora_path}: {e}")
        finally:
            self._loaded_loras.pop(lora_path, None)

    def _unload_all_loras(self) -> None:
        for path in list(self._loaded_loras.keys()):
            self._unload_single_lora(path)

    # ── Generation ──

    def generate(self, request: GenerationRequest) -> List[GenerationResult]:
        return self.generate_with_callback(request, None)

    def generate_with_callback(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerationResult]:
        """Generate images with CPU offload active."""
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        self._cancel_unload_timer()

        seed = request.seed
        if seed < 0:
            seed = torch.randint(0, 2**32, (1,)).item()

        self._handle_loras(request.lora_configs)
        lora_desc = ", ".join(
            f"{Path(c.path).stem}@{c.scale}" for c in request.lora_configs
        ) if request.lora_configs else "none"
        print(f"[TIMING] LoRAs handled ({lora_desc}), starting generation", flush=True)

        logger.info(
            f"Generating: prompt='{request.prompt[:60]}...' "
            f"size={request.width}x{request.height} "
            f"steps={request.num_inference_steps} seed={seed} "
            f"loras=[{lora_desc}]"
        )

        t0 = time.time()
        total_steps = request.num_inference_steps

        def diffusers_step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step_index + 1, total_steps)
            return callback_kwargs

        req_width = max(16, round(request.width / 16) * 16)
        req_height = max(16, round(request.height / 16) * 16)

        # With enable_model_cpu_offload(), pipeline._execution_device = cuda:0
        # Generator must match execution device
        exec_device = self._pipe._execution_device
        logger.info(f"Execution device: {exec_device}")

        results: List[GenerationResult] = []
        for i in range(request.num_images):
            img_seed = seed + i
            gen = torch.Generator(device=exec_device).manual_seed(img_seed)

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
                lora_configs=list(request.lora_configs),
            ))

        elapsed = time.time() - t0
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Generated {len(results)} image(s) in {elapsed:.1f}s (VRAM={vram_gb:.1f}GB)")

        # Schedule auto-unload to free GPU for training
        self._schedule_unload()

        return results
