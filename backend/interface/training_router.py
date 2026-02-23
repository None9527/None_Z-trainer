# -*- coding: utf-8 -*-
"""
Training Router — Complete API Layer

Endpoints:
  POST   /start                 Start training (JSON → TOML → subprocess)
  POST   /stop                  Stop running training
  GET    /status                Get training status + recent logs
  GET    /defaults              Get default training config
  GET    /configs               List saved configs
  GET    /config/current        Get most recently saved config
  GET    /config/{name}         Load named config
  POST   /config/save           Save config  { name, config }
  DELETE /config/{name}         Delete named config
  GET    /runs                  List training runs (output dir)
  DELETE /runs/{name}           Delete a training run
  GET    /presets               List built-in presets
  GET    /all-scalars           Read TensorBoard scalars (for Monitor)
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Training"])


# ── Helpers ──────────────────────────────────────────────────────────

def _configs_dir() -> Path:
    from ..infrastructure.config import CONFIGS_DIR
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIGS_DIR


def _output_dir() -> Path:
    from ..infrastructure.config import OUTPUT_PATH
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    return OUTPUT_PATH


def _default_config() -> dict:
    """Python equivalent of frontend getDefaultConfig()."""
    return {
        "name": "default",
        "training_type": "lora",
        "condition_mode": "text2img",
        "timestep": {
            "mode": "uniform",
            "shift": 3.0,
            "use_dynamic_shift": True,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "acrf_steps": 10,
            "jitter_scale": 0.02,
            "latent_jitter_scale": 0.0,
        },
        "acrf": {
            "snr_gamma": 5.0,
            "snr_floor": 0.1,
            "raft_mode": False,
            "free_stream_ratio": 0.3,
            "enable_timestep_aware_loss": False,
            "timestep_high_threshold": 0.7,
            "timestep_low_threshold": 0.3,
            "enable_curvature": False,
            "lambda_curvature": 0.05,
            "curvature_interval": 10,
            "curvature_start_epoch": 0,
            "cfg_training": False,
            "cfg_scale": 7.0,
            "cfg_training_ratio": 0.3,
        },
        "network": {"dim": 8, "alpha": 4.0},
        "lora": {
            "resume_training": False,
            "resume_lora_path": "",
            "train_adaln": False,
            "train_norm": False,
            "train_single_stream": False,
        },
        "controlnet": {
            "resume_training": False,
            "controlnet_path": "",
            "control_types": ["canny"],
            "conditioning_scale": 0.75,
        },
        "optimizer": {
            "type": "AdamW8bit",
            "learning_rate": "1e-4",
            "relative_step": False,
        },
        "training": {
            "output_name": "",
            "learning_rate": 0.0001,
            "learning_rate_str": "1e-4",
            "weight_decay": 0,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "lr_num_cycles": 1,
            "lr_pct_start": 0.1,
            "lr_div_factor": 10,
            "lr_final_div_factor": 100,
            "lambda_mse": 1.0,
            "lambda_l1": 1.0,
            "lambda_cosine": 0.1,
            "enable_freq": False,
            "lambda_freq": 0.3,
            "alpha_hf": 1.0,
            "beta_lf": 0.2,
            "enable_style": False,
            "lambda_style": 0.3,
            "lambda_light": 0.5,
            "lambda_color": 0.3,
        },
        "dataset": {
            "batch_size": 1,
            "shuffle": True,
            "enable_bucket": True,
            "drop_text_ratio": 0.1,
            "datasets": [],
        },
        "reg_dataset": {
            "enabled": False,
            "weight": 1.0,
            "ratio": 0.5,
            "datasets": [],
        },
        "advanced": {
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "blocks_to_swap": 0,
            "num_train_epochs": 10,
            "save_every_n_epochs": 1,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
            "seed": 42,
            "num_gpus": 1,
            "gpu_ids": "",
        },
    }


# Stateful: track current running training
_current_process_id: Optional[int] = None
_recent_logs: List[str] = []
_MAX_LOG_LINES = 200


# =====================================================================
#  P0: /start, /stop, /status
# =====================================================================

@router.post("/start")
async def start_training(config: Dict[str, Any] = Body(...)):
    """
    Start training: save config JSON → generate TOML → launch subprocess.
    Also checks dataset cache completeness.
    """
    global _current_process_id, _recent_logs
    from ..infrastructure.container import container

    runner = container.training_runner()

    # Refuse if already running
    if _current_process_id is not None and runner.is_running(_current_process_id):
        return {"success": False, "message": "Training is already running"}

    # Save config JSON for /config/current
    cfg_dir = _configs_dir()
    current_json = cfg_dir / "_current.json"
    current_json.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save named config if output_name exists
    output_name = config.get("training", {}).get("output_name", "")
    if output_name:
        named_path = cfg_dir / f"{output_name}.json"
        named_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Check dataset cache
    datasets = config.get("dataset", {}).get("datasets", [])
    if datasets:
        cache_status = _check_dataset_cache(datasets)
        if cache_status.get("needs_cache"):
            return {
                "success": True,
                "needs_cache": True,
                **cache_status,
            }

    # Generate TOML
    repo = container.training_repo()
    toml_path = str(cfg_dir / "current_training.toml")
    try:
        repo.save_config_from_json(config, toml_path)
    except Exception as e:
        logger.error(f"TOML generation failed: {e}")
        return {"success": False, "message": f"Config generation failed: {e}"}

    # Launch
    try:
        adv = config.get("advanced", {})
        pid = runner.start(
            config_path=toml_path,
            mixed_precision=adv.get("mixed_precision", "bf16"),
            num_gpus=int(adv.get("num_gpus", 1)),
            gpu_ids=adv.get("gpu_ids", ""),
        )
        _current_process_id = pid
        _recent_logs = []
        logger.info(f"Training started: pid={pid}")
        return {
            "success": True,
            "message": "Training started",
            "process_id": pid,
            "config_path": toml_path,
        }
    except Exception as e:
        logger.error(f"Training launch failed: {e}")
        return {"success": False, "message": f"Failed to start training: {e}"}


@router.post("/stop")
async def stop_training():
    """Stop the current training process."""
    global _current_process_id
    from ..infrastructure.container import container

    if _current_process_id is None:
        return {"success": False, "message": "No training is running"}

    runner = container.training_runner()
    try:
        runner.stop(_current_process_id)
        _current_process_id = None
        return {"success": True, "message": "Training stopped"}
    except Exception as e:
        return {"success": False, "message": f"Failed to stop: {e}"}


@router.get("/status")
async def get_status():
    """Get training status with recent log lines."""
    global _recent_logs, _current_process_id
    from ..infrastructure.container import container

    runner = container.training_runner()

    if _current_process_id is not None:
        is_running = runner.is_running(_current_process_id)

        # Drain output
        while True:
            line = runner.get_output(_current_process_id)
            if line is None:
                break
            _recent_logs.append(line)
            if len(_recent_logs) > _MAX_LOG_LINES:
                _recent_logs = _recent_logs[-_MAX_LOG_LINES:]

        # Parse [STEP] and [TRAINING_INFO] lines
        info = _parse_training_info(_recent_logs)

        result = {
            "success": True,
            "status": "running" if is_running else "completed",
            "is_running": is_running,
            "process_id": _current_process_id,
            "logs": _recent_logs[-50:],
            **info,
        }

        # Clear stale process ID when training finishes
        if not is_running:
            _current_process_id = None

        return result

    return {
        "success": True,
        "status": "idle",
        "is_running": False,
        "process_id": None,
        "logs": [],
    }


# =====================================================================
#  P1: Config CRUD
# =====================================================================

@router.get("/defaults")
async def get_defaults():
    """Return default training config (mirrors frontend getDefaultConfig)."""
    return _default_config()


@router.get("/config/current")
async def get_current_config():
    """Return the most recently saved/used config."""
    cfg_dir = _configs_dir()
    current = cfg_dir / "_current.json"
    if current.exists():
        return json.loads(current.read_text(encoding="utf-8"))
    return _default_config()


@router.get("/configs")
async def list_configs():
    """List all saved config files."""
    cfg_dir = _configs_dir()
    configs = []
    for f in sorted(cfg_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            configs.append({
                "name": f.stem,
                "training_type": data.get("training_type", "lora"),
                "output_name": data.get("training", {}).get("output_name", ""),
            })
        except Exception:
            configs.append({"name": f.stem})
    return {"configs": configs}


@router.get("/config/{name}")
async def get_config(name: str):
    """Load a named config."""
    cfg_dir = _configs_dir()
    path = cfg_dir / f"{name}.json"
    if not path.exists():
        if name == "default":
            return _default_config()
        raise HTTPException(404, f"Config '{name}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/config/save")
async def save_config(body: Dict[str, Any] = Body(...)):
    """
    Save config.
    
    Body: { "name": "my-config", "config": { ... } }
    """
    name = body.get("name", "").strip()
    config = body.get("config", {})

    if not name:
        return {"success": False, "message": "Config name is required"}

    cfg_dir = _configs_dir()
    path = cfg_dir / f"{name}.json"
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    path.write_text(config_json, encoding="utf-8")

    # Also update _current.json so Training page picks up the latest config
    current_path = cfg_dir / "_current.json"
    current_path.write_text(config_json, encoding="utf-8")

    logger.info(f"Config saved: {path}")
    return {"success": True, "message": f"Config '{name}' saved"}


@router.delete("/config/{name}")
async def delete_config(name: str):
    """Delete a named config."""
    if name == "default":
        raise HTTPException(400, "Cannot delete default config")
    cfg_dir = _configs_dir()
    path = cfg_dir / f"{name}.json"
    if path.exists():
        path.unlink()
        return {"success": True, "message": f"Config '{name}' deleted"}
    raise HTTPException(404, f"Config '{name}' not found")


# =====================================================================
#  P1: Runs (training records in output dir)
# =====================================================================

@router.get("/runs")
async def list_runs():
    """List training output directories (runs)."""
    out = _output_dir()
    runs = []
    # Skip structural directories created by ensure_dirs()
    skip_dirs = {"finetune", "lora", "logs"}
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            # Check for loose .safetensors files
            if d.suffix == ".safetensors":
                runs.append({
                    "name": d.stem,
                    "path": str(d),
                    "size_mb": round(d.stat().st_size / (1024 * 1024), 1),
                    "type": "file",
                })
            continue
        if d.name in skip_dirs:
            continue
        # Count .safetensors files in directory
        safetensors = list(d.glob("*.safetensors"))
        logs_dir = d / "logs"
        runs.append({
            "name": d.name,
            "path": str(d),
            "checkpoints": len(safetensors),
            "has_logs": logs_dir.exists(),
            "type": "directory",
        })
    return {"runs": runs}


@router.delete("/runs/{name}")
async def delete_run(name: str):
    """Delete a training run directory."""
    out = _output_dir()
    target = out / name
    if not target.exists():
        # Also check for loose file
        target_file = out / f"{name}.safetensors"
        if target_file.exists():
            target_file.unlink()
            return {"success": True, "message": f"Run '{name}' deleted"}
        raise HTTPException(404, f"Run '{name}' not found")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"success": True, "message": f"Run '{name}' deleted"}


# =====================================================================
#  P2: Presets + TensorBoard Scalars
# =====================================================================

@router.get("/presets")
async def list_presets():
    """Return built-in training presets."""
    presets = [
        {
            "name": "快速预览 (5 epoch)",
            "config": {
                **_default_config(),
                "advanced": {**_default_config()["advanced"], "num_train_epochs": 5},
            },
        },
        {
            "name": "标准 LoRA (10 epoch)",
            "config": _default_config(),
        },
        {
            "name": "精细 LoRA (20 epoch, cosine)",
            "config": {
                **_default_config(),
                "training": {
                    **_default_config()["training"],
                    "lr_scheduler": "cosine",
                    "lr_warmup_steps": 100,
                },
                "advanced": {**_default_config()["advanced"], "num_train_epochs": 20},
            },
        },
    ]
    return {"presets": presets}


@router.get("/all-scalars")
async def get_all_scalars(
    run: str = Query("", description="Run name"),
    max_points: int = Query(500, description="Max data points"),
):
    """
    Read TensorBoard scalar data for Monitor page.
    Returns scalars from {output_dir}/{run}/logs/.
    """
    out = _output_dir()
    run_dir = out / run / "logs" if run else out / "logs"

    if not run_dir.exists():
        return {"scalars": {}}

    # Accelerate creates event files in logs/{project_name}/ subdirectory.
    # Find the actual directory containing event files.
    actual_log_dir = run_dir
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        # Search one level of subdirectories
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and list(sub.glob("events.out.tfevents.*")):
                actual_log_dir = sub
                break

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(actual_log_dir))
        ea.Reload()

        scalars = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            # Downsample if too many points
            step = max(1, len(events) // max_points)
            scalars[tag] = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events[::step]
            ]
        return {"scalars": scalars}
    except ImportError:
        logger.warning("tensorboard not installed, cannot read scalars")
        return {"scalars": {}, "error": "tensorboard not installed"}
    except Exception as e:
        logger.error(f"Failed to read TensorBoard data: {e}")
        return {"scalars": {}, "error": str(e)}


# =====================================================================
#  Internal Helpers
# =====================================================================

def _check_dataset_cache(datasets: list) -> dict:
    """Check if dataset latent/text caches exist.
    
    Cache files can be in two locations:
    1. Dataset root: *_zi.safetensors (latent), *_zi_te.safetensors (text)
    2. Subdirectories: latent_cache/*.safetensors, text_cache/*.safetensors
    """
    total = 0
    latent_cached = 0
    text_cached = 0

    for ds in datasets:
        cache_dir = ds.get("cache_directory", "")
        if not cache_dir:
            continue

        p = Path(cache_dir)
        if not p.exists():
            continue

        # Count images
        images = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.webp"))
        total += len(images)

        # Check cache in dataset root (ZTrainer v2 convention: *_zi.safetensors, *_zi_te.safetensors)
        latent_root = list(p.glob("*_zi.safetensors"))
        text_root = list(p.glob("*_zi_te.safetensors"))

        if latent_root or text_root:
            latent_cached += len(latent_root)
            text_cached += len(text_root)
        else:
            # Fallback: check latent_cache/ and text_cache/ subdirectories
            latent_dir = p / "latent_cache"
            if latent_dir.exists():
                latent_cached += len(list(latent_dir.glob("*.safetensors")))

            text_dir = p / "text_cache"
            if text_dir.exists():
                text_cached += len(list(text_dir.glob("*.safetensors")))

    if total == 0:
        return {"needs_cache": False}

    needs = latent_cached < total or text_cached < total
    return {
        "needs_cache": needs,
        "total_images": total,
        "latent_cached": latent_cached,
        "text_cached": text_cached,
    }


def _parse_training_info(logs: List[str]) -> dict:
    """Parse [TRAINING_INFO] and [STEP] lines from logs."""
    info: Dict[str, Any] = {}
    last_step = {}

    for line in logs:
        if "[TRAINING_INFO]" in line:
            # Parse key=value pairs
            idx = line.index("[TRAINING_INFO]") + len("[TRAINING_INFO]")
            rest = line[idx:].strip()
            for part in rest.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    info[k.strip()] = v.strip()

        elif "[STEP]" in line:
            idx = line.index("[STEP]") + len("[STEP]")
            rest = line[idx:].strip()
            for part in rest.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    last_step[k.strip()] = v.strip()

    # Merge last step info
    if last_step:
        info["current_step"] = last_step.get("step", "")
        info["current_loss"] = last_step.get("loss", "")
        info["current_lr"] = last_step.get("lr", "")
        info["current_epoch"] = last_step.get("epoch", "")

    return info
