# -*- coding: utf-8 -*-
"""
TOML Training Repository

Implements ITrainingRepository using TOML files.
Supports two paths:
  1. Legacy: TrainingConfig (domain VO) → TOML
  2. New:    Frontend JSON (dict) → TOML (all 50+ fields mapped)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..domain.training.repositories import ITrainingRepository
from ..domain.training.value_objects import (
    TrainingConfig,
    LossConfig,
    TimestepConfig,
    SchedulerConfig,
    LoRAConfig,
    SNRConfig,
)

logger = logging.getLogger(__name__)


def _bool(v) -> str:
    """Convert Python bool to TOML bool string."""
    return "true" if v else "false"


class TomlTrainingRepository(ITrainingRepository):
    """
    TOML-based training configuration persistence.

    Reads/writes TOML configs consumed by trainer_core/zimage_trainer/train.py.
    """

    def _get_model_path(self) -> str:
        """Get the configured model path from infrastructure config."""
        try:
            from .config import MODEL_PATH
            return str(MODEL_PATH)
        except ImportError:
            return ""

    # ------------------------------------------------------------------
    # NEW: Frontend JSON → TOML (complete 50+ field mapping)
    # ------------------------------------------------------------------

    def save_config_from_json(self, data: Dict[str, Any], config_path: str) -> None:
        """
        Convert raw frontend JSON to TOML consumable by train.py.

        The frontend JSON structure (from useTrainingConfig.ts):
            timestep, acrf, network, lora, controlnet, optimizer,
            training, dataset, reg_dataset, advanced

        The output TOML structure (for train.py's parse_args):
            [general], [timestep], [acrf], [network], [lora],
            [controlnet], [optimizer], [training], [dataset],
            [reg_dataset], [advanced]
        """
        ts = data.get("timestep", {})
        acrf = data.get("acrf", {})
        network = data.get("network", {})
        lora = data.get("lora", {})
        cnet = data.get("controlnet", {})
        opt = data.get("optimizer", {})
        training = data.get("training", {})
        dataset = data.get("dataset", {})
        reg = data.get("reg_dataset", {})
        adv = data.get("advanced", {})

        # Resolve model path from system config
        model_path = self._get_model_path()

        # Resolve output directory
        try:
            from .config import OUTPUT_PATH
            output_base = str(OUTPUT_PATH)
        except (ImportError, AttributeError):
            output_base = "output"

        output_name = training.get("output_name", "zimage-lora")
        output_dir = str(Path(output_base) / output_name)

        lines = []

        # ── [general] ──
        lines.append("[general]")
        lines.append(f'dit = "{model_path}"')
        lines.append(f'output_dir = "{output_dir}"')
        lines.append(f'output_name = "{output_name}"')
        lines.append(f'training_type = "{data.get("training_type", "lora")}"')
        lines.append(f'condition_mode = "{data.get("condition_mode", "text2img")}"')
        lines.append("")

        # ── [timestep] ──
        lines.append("[timestep]")
        lines.append(f'mode = "{ts.get("mode", "uniform")}"')
        lines.append(f"shift = {float(ts.get('shift', 3.0))}")
        lines.append(f"use_dynamic_shift = {_bool(ts.get('use_dynamic_shift', True))}")
        lines.append(f"base_shift = {float(ts.get('base_shift', 0.5))}")
        lines.append(f"max_shift = {float(ts.get('max_shift', 1.15))}")
        lines.append(f"logit_mean = {float(ts.get('logit_mean', 0.0))}")
        lines.append(f"logit_std = {float(ts.get('logit_std', 1.0))}")
        lines.append(f"acrf_steps = {int(ts.get('acrf_steps', 10))}")
        lines.append(f"jitter_scale = {float(ts.get('jitter_scale', 0.02))}")
        lines.append(f"latent_jitter_scale = {float(ts.get('latent_jitter_scale', 0.0))}")
        lines.append("")

        # ── [acrf] ──
        lines.append("[acrf]")
        lines.append(f"snr_gamma = {float(acrf.get('snr_gamma', 5.0))}")
        lines.append(f"snr_floor = {float(acrf.get('snr_floor', 0.1))}")
        lines.append(f"raft_mode = {_bool(acrf.get('raft_mode', False))}")
        lines.append(f"free_stream_ratio = {float(acrf.get('free_stream_ratio', 0.3))}")
        lines.append(f"enable_timestep_aware_loss = {_bool(acrf.get('enable_timestep_aware_loss', False))}")
        lines.append(f"timestep_high_threshold = {float(acrf.get('timestep_high_threshold', 0.7))}")
        lines.append(f"timestep_low_threshold = {float(acrf.get('timestep_low_threshold', 0.3))}")
        lines.append(f"enable_curvature = {_bool(acrf.get('enable_curvature', False))}")
        lines.append(f"lambda_curvature = {float(acrf.get('lambda_curvature', 0.05))}")
        lines.append(f"curvature_interval = {int(acrf.get('curvature_interval', 10))}")
        lines.append(f"curvature_start_epoch = {int(acrf.get('curvature_start_epoch', 0))}")
        lines.append(f"cfg_training = {_bool(acrf.get('cfg_training', False))}")
        lines.append(f"cfg_scale = {float(acrf.get('cfg_scale', 7.0))}")
        lines.append(f"cfg_training_ratio = {float(acrf.get('cfg_training_ratio', 0.3))}")
        lines.append(f'loss_weighting = "{acrf.get("loss_weighting", "none")}"')
        lines.append("")

        # ── [network] ──
        lines.append("[network]")
        lines.append(f"dim = {int(network.get('dim', 8))}")
        lines.append(f"alpha = {float(network.get('alpha', 4.0))}")
        lines.append("")

        # ── [lora] ──
        lines.append("[lora]")
        lines.append(f"resume_training = {_bool(lora.get('resume_training', False))}")
        lines.append(f'resume_lora_path = "{lora.get("resume_lora_path", "")}"')
        lines.append(f"train_adaln = {_bool(lora.get('train_adaln', False))}")
        lines.append(f"train_refiner = {_bool(lora.get('train_refiner', False))}")
        lines.append(f"train_norm = {_bool(lora.get('train_norm', False))}")
        lines.append(f"train_single_stream = {_bool(lora.get('train_single_stream', False))}")
        lines.append(f"enable_ste_tanh = {_bool(lora.get('enable_ste_tanh', False))}")
        lines.append("")

        # ── [controlnet] ──
        lines.append("[controlnet]")
        lines.append(f"resume_training = {_bool(cnet.get('resume_training', False))}")
        lines.append(f'controlnet_path = "{cnet.get("controlnet_path", "")}"')
        ctl_types = cnet.get("control_types", ["canny"])
        if isinstance(ctl_types, list):
            ctl_str = ", ".join(f'"{t}"' for t in ctl_types)
            lines.append(f"control_types = [{ctl_str}]")
        else:
            lines.append(f'control_types = ["{ctl_types}"]')
        lines.append(f"conditioning_scale = {float(cnet.get('conditioning_scale', 0.75))}")
        lines.append("")

        # ── [optimizer] ──
        lines.append("[optimizer]")
        lines.append(f'type = "{opt.get("type", "AdamW8bit")}"')
        lines.append(f"relative_step = {_bool(opt.get('relative_step', False))}")
        lines.append("")

        # ── [training] ──
        lines.append("[training]")
        lines.append(f'output_name = "{output_name}"')
        lr_val = training.get("learning_rate", 1e-4)
        lines.append(f"learning_rate = {float(lr_val)}")
        lines.append(f"weight_decay = {float(training.get('weight_decay', 0.0))}")
        lines.append(f'lr_scheduler = "{training.get("lr_scheduler", "constant")}"')
        lines.append(f"lr_warmup_steps = {int(training.get('lr_warmup_steps', 0))}")
        lines.append(f"lr_num_cycles = {int(training.get('lr_num_cycles', 1))}")
        lines.append(f"lr_pct_start = {float(training.get('lr_pct_start', 0.1))}")
        lines.append(f"lr_div_factor = {float(training.get('lr_div_factor', 10.0))}")
        lines.append(f"lr_final_div_factor = {float(training.get('lr_final_div_factor', 100.0))}")
        lines.append(f"lambda_mse = {float(training.get('lambda_mse', 1.0))}")
        lines.append(f"lambda_l1 = {float(training.get('lambda_l1', 1.0))}")
        lines.append(f"lambda_cosine = {float(training.get('lambda_cosine', 0.1))}")
        lines.append(f"enable_freq = {_bool(training.get('enable_freq', False))}")
        lines.append(f"lambda_freq = {float(training.get('lambda_freq', 0.3))}")
        lines.append(f"alpha_hf = {float(training.get('alpha_hf', 1.0))}")
        lines.append(f"beta_lf = {float(training.get('beta_lf', 0.2))}")
        lines.append(f"enable_style = {_bool(training.get('enable_style', False))}")
        lines.append(f"lambda_style = {float(training.get('lambda_style', 0.3))}")
        lines.append(f"lambda_light = {float(training.get('lambda_light', 0.5))}")
        lines.append(f"lambda_color = {float(training.get('lambda_color', 0.3))}")
        lines.append(f"enable_dino_mask = {_bool(training.get('enable_dino_mask', False))}")
        lines.append(f"dino_mask_base_ratio = {float(training.get('dino_mask_base_ratio', 0.2))}")

        lines.append("")

        # ── [dataset] ──
        lines.append("[dataset]")
        lines.append(f"batch_size = {int(dataset.get('batch_size', 1))}")
        lines.append(f"shuffle = {_bool(dataset.get('shuffle', True))}")
        lines.append(f"enable_bucket = {_bool(dataset.get('enable_bucket', True))}")
        lines.append(f"drop_text_ratio = {float(dataset.get('drop_text_ratio', 0.1))}")
        lines.append("")

        # Dataset sources → [[dataset.sources]]
        sources = dataset.get("datasets", [])
        for src in sources:
            lines.append("[[dataset.sources]]")
            # Support multiple key names for the dataset path
            cache_path = src.get("cache_directory", src.get("cache_dir", src.get("path", "")))
            lines.append(f'cache_dir = "{cache_path}"')
            if src.get("num_repeats") and int(src["num_repeats"]) > 1:
                lines.append(f"num_repeats = {int(src['num_repeats'])}")
            if src.get("resolution_limit"):
                lines.append(f"resolution_limit = {int(src['resolution_limit'])}")
            if src.get("weight") is not None:
                lines.append(f"weight = {float(src['weight'])}")
            lines.append("")

        # ── [reg_dataset] ──
        lines.append("[reg_dataset]")
        lines.append(f"enabled = {_bool(reg.get('enabled', False))}")
        lines.append(f"weight = {float(reg.get('weight', 1.0))}")
        lines.append(f"ratio = {float(reg.get('ratio', 0.5))}")
        lines.append("")

        reg_sources = reg.get("datasets", [])
        for src in reg_sources:
            lines.append("[[reg_dataset.sources]]")
            cache_path = src.get("cache_directory", src.get("cache_dir", src.get("path", "")))
            lines.append(f'cache_dir = "{cache_path}"')
            if src.get("resolution_limit"):
                lines.append(f"resolution_limit = {int(src['resolution_limit'])}")
            lines.append("")

        # ── [advanced] ──
        lines.append("[advanced]")
        lines.append(f"max_grad_norm = {float(adv.get('max_grad_norm', 1.0))}")
        lines.append(f"gradient_checkpointing = {_bool(adv.get('gradient_checkpointing', True))}")
        lines.append(f"blocks_to_swap = {int(adv.get('blocks_to_swap', 0))}")
        lines.append(f"num_train_epochs = {int(adv.get('num_train_epochs', 10))}")
        lines.append(f"save_every_n_epochs = {int(adv.get('save_every_n_epochs', 1))}")
        lines.append(f"gradient_accumulation_steps = {int(adv.get('gradient_accumulation_steps', 4))}")
        lines.append(f'mixed_precision = "{adv.get("mixed_precision", "bf16")}"')
        lines.append(f"seed = {int(adv.get('seed', 42))}")
        lines.append(f"num_gpus = {int(adv.get('num_gpus', 1))}")
        lines.append(f'gpu_ids = "{adv.get("gpu_ids", "")}"')

        # Write TOML file
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config_path).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Config saved to {config_path} (from JSON, {len(lines)} lines)")

    # ------------------------------------------------------------------
    # Legacy: TrainingConfig (domain VO) → TOML
    # ------------------------------------------------------------------

    def save_config(self, config: TrainingConfig, path: str) -> None:
        """Generate and save TOML config file from domain VO."""
        lines = self._build_toml_lines(config)
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Config saved to {path}")

    def load_config(self, path: str) -> TrainingConfig:
        """Load config from TOML file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                import toml as tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        training = data.get("training", {})
        ts = data.get("timestep", data.get("acrf", {}))
        acrf_sec = data.get("acrf", {})
        advanced = data.get("advanced", {})

        loss = LossConfig(
            lambda_l1=training.get("lambda_l1", 1.0),
            lambda_l2=training.get("lambda_l2", 0.0),
            lambda_cosine=training.get("lambda_cosine", 0.0),
            enable_freq=training.get("enable_freq", False),
            lambda_freq=training.get("lambda_freq", 0.3),
            alpha_hf=training.get("alpha_hf", 1.0),
            beta_lf=training.get("beta_lf", 0.2),
            enable_style=training.get("enable_style", False),
            lambda_style=training.get("lambda_style", 0.3),
            lambda_light=training.get("lambda_light", 0.5),
            lambda_color=training.get("lambda_color", 0.3),
        )

        if "mode" in ts:
            ts_mode = ts["mode"]
        elif ts.get("enable_turbo", False):
            ts_mode = "acrf"
        else:
            ts_mode = "uniform"

        timestep_config = TimestepConfig(
            mode=ts_mode,
            shift=ts.get("shift", 3.0),
            use_dynamic_shift=ts.get("use_dynamic_shift", ts.get("use_dynamic_shifting", True)),
            base_shift=ts.get("base_shift", 0.5),
            max_shift=ts.get("max_shift", 1.15),
            logit_mean=ts.get("logit_mean", 0.0),
            logit_std=ts.get("logit_std", 1.0),
            acrf_steps=ts.get("acrf_steps", ts.get("turbo_steps", 10)),
            jitter_scale=ts.get("jitter_scale", 0.02),
            latent_jitter_scale=ts.get("latent_jitter_scale", 0.0),
        )

        scheduler = SchedulerConfig(
            scheduler_type=training.get("lr_scheduler", "constant"),
            learning_rate=training.get("learning_rate", 1e-4),
            warmup_steps=training.get("lr_warmup_steps", 0),
            num_cycles=training.get("lr_num_cycles", 1),
            weight_decay=training.get("weight_decay", 0.0),
        )

        lora = LoRAConfig(
            network_dim=training.get("network_dim", 16),
            network_alpha=training.get("network_alpha", 16.0),
        )

        snr = SNRConfig(
            snr_gamma=acrf_sec.get("snr_gamma", training.get("snr_gamma", 5.0)),
            snr_floor=acrf_sec.get("snr_floor", 0.1),
        )

        return TrainingConfig(
            dataset_path=data.get("dataset", {}).get("path", ""),
            output_dir=data.get("general", {}).get("output_dir", ""),
            model_path=data.get("general", {}).get("dit", ""),
            num_epochs=advanced.get("num_train_epochs", 10),
            batch_size=training.get("batch_size", 1),
            gradient_accumulation_steps=advanced.get("gradient_accumulation_steps", 4),
            max_grad_norm=advanced.get("max_grad_norm", 1.0),
            mixed_precision=advanced.get("mixed_precision", "bf16"),
            seed=advanced.get("seed", 42),
            save_every_n_epochs=advanced.get("save_every_n_epochs", 1),
            loss=loss,
            timestep=timestep_config,
            scheduler=scheduler,
            lora=lora,
            snr=snr,
            optimizer_type=training.get("optimizer_type", "AdamW"),
        )

    def get_default_config(self) -> TrainingConfig:
        """Return default config."""
        return TrainingConfig()

    def _build_toml_lines(self, config: TrainingConfig) -> list:
        """Build TOML config lines from domain VO (legacy path)."""
        lines = [
            "[training]",
            f"learning_rate = {config.scheduler.learning_rate}",
            f'optimizer_type = "{config.optimizer_type}"',
            f'lr_scheduler = "{config.scheduler.scheduler_type}"',
            f"lr_warmup_steps = {config.scheduler.warmup_steps}",
            f"lr_num_cycles = {config.scheduler.num_cycles}",
            f"weight_decay = {config.scheduler.weight_decay}",
            f"lambda_l1 = {config.loss.lambda_l1}",
            f"lambda_l2 = {config.loss.lambda_l2}",
            f"lambda_cosine = {config.loss.lambda_cosine}",
            f"enable_freq = {_bool(config.loss.enable_freq)}",
            f"lambda_freq = {config.loss.lambda_freq}",
            f"alpha_hf = {config.loss.alpha_hf}",
            f"beta_lf = {config.loss.beta_lf}",
            f"enable_style = {_bool(config.loss.enable_style)}",
            f"lambda_style = {config.loss.lambda_style}",
            f"lambda_light = {config.loss.lambda_light}",
            f"lambda_color = {config.loss.lambda_color}",
            "",
            "[timestep]",
            f'mode = "{config.timestep.mode}"',
            f"shift = {config.timestep.shift}",
            f"use_dynamic_shift = {_bool(config.timestep.use_dynamic_shift)}",
            f"base_shift = {config.timestep.base_shift}",
            f"max_shift = {config.timestep.max_shift}",
            f"logit_mean = {config.timestep.logit_mean}",
            f"logit_std = {config.timestep.logit_std}",
            f"acrf_steps = {config.timestep.acrf_steps}",
            f"jitter_scale = {config.timestep.jitter_scale}",
            f"latent_jitter_scale = {config.timestep.latent_jitter_scale}",
            "",
            "[acrf]",
            f"snr_gamma = {config.snr.snr_gamma}",
            f"snr_floor = {config.snr.snr_floor}",
            "",
            "[advanced]",
            f"num_train_epochs = {config.num_epochs}",
            f"gradient_accumulation_steps = {config.gradient_accumulation_steps}",
            f"max_grad_norm = {config.max_grad_norm}",
            f'mixed_precision = "{config.mixed_precision}"',
            f"seed = {config.seed}",
            f"save_every_n_epochs = {config.save_every_n_epochs}",
            f"gradient_checkpointing = {_bool(config.gradient_checkpointing)}",
        ]
        return lines
