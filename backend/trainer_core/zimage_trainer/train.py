# -*- coding: utf-8 -*-
"""
Z-Image V2 Training Script — Accelerate Edition

Fully standalone training entry point using only v2 trainer_core components.
Reads TOML config compatible with both frontend-generated and v1 legacy formats.

Usage:
    accelerate launch train.py --config /path/to/config.toml
    python train.py --config /path/to/config.toml  # single GPU

Key Features:
    - Z-Image output negation (aligned with official ZImagePipeline)
    - Anchor / Uniform / LogitNormal timestep sampling
    - Multi-loss: L1 + Cosine + FrequencyAware + StyleStructure
    - Min-SNR gamma weighting
    - Norm_opt proxy target + content/quality schedule
    - Block Swap + Gradient Checkpointing + Attention optimization
    - LoRA resume + regularization dataset
    - CFG training mode
    - [TRAINING_INFO] / [STEP] log protocol for frontend parsing
"""

import argparse
import gc
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve project root so relative imports within trainer_core work.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # zimage_trainer/
_TRAINER_CORE = _SCRIPT_DIR.parent                     # trainer_core/
_BACKEND = _TRAINER_CORE.parent                        # backend/
if str(_TRAINER_CORE) not in sys.path:
    sys.path.insert(0, str(_TRAINER_CORE))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logger = logging.getLogger("zimage_train")


# ============================================================================
# TOML Config Parser — supports frontend JSON→TOML *and* v1 legacy TOML
# ============================================================================

def _read_toml(path: str) -> dict:
    """Read a TOML file, trying several backends."""
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    import toml
    return toml.load(path)


def parse_args(argv=None) -> SimpleNamespace:
    """Parse CLI + TOML config into a flat SimpleNamespace."""
    parser = argparse.ArgumentParser(description="Z-Image Training")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and exit (no training)")
    cli = parser.parse_args(argv)

    raw = _read_toml(cli.config)

    # Flatten TOML sections into a single namespace with sensible defaults.

    # --- general / model ---
    general = raw.get("general", raw.get("model", {}))
    # --- timestep ---
    ts = raw.get("timestep", raw.get("acrf", {}))
    # --- acrf / snr ---
    acrf = raw.get("acrf", {})
    # --- network / lora ---
    network = raw.get("network", raw.get("lora", {}))
    lora_sec = raw.get("lora", {})
    # --- optimizer ---
    opt = raw.get("optimizer", {})
    # --- training ---
    training = raw.get("training", {})
    # --- dataset ---
    dataset = raw.get("dataset", {})
    # --- reg_dataset ---
    reg = raw.get("reg_dataset", {})
    # --- advanced ---
    adv = raw.get("advanced", {})
    # --- controlnet ---
    cnet = raw.get("controlnet", {})

    args = SimpleNamespace(
        config_path=cli.config,
        dry_run=cli.dry_run,

        # ── General / Model ──
        dit=general.get("dit", general.get("model_path", "")),
        output_dir=general.get("output_dir", training.get("output_dir", "output")),
        output_name=training.get("output_name", general.get("output_name", "zimage-lora")),
        training_type=general.get("training_type", "lora"),
        condition_mode=general.get("condition_mode", "text2img"),

        # ── Timestep Sampling ──
        timestep_mode=ts.get("mode", "uniform"),
        shift=float(ts.get("shift", 3.0)),
        use_dynamic_shift=bool(ts.get("use_dynamic_shift", True)),
        base_shift=float(ts.get("base_shift", 0.5)),
        max_shift=float(ts.get("max_shift", 1.15)),
        logit_mean=float(ts.get("logit_mean", 0.0)),
        logit_std=float(ts.get("logit_std", 1.0)),
        acrf_steps=int(ts.get("acrf_steps", ts.get("turbo_steps", 10))),
        jitter_scale=float(ts.get("jitter_scale", 0.02)),
        latent_jitter_scale=float(ts.get("latent_jitter_scale", 0.0)),

        # ── ACRF / SNR ──
        snr_gamma=float(acrf.get("snr_gamma", 5.0)),
        snr_floor=float(acrf.get("snr_floor", 0.1)),
        raft_mode=bool(acrf.get("raft_mode", False)),
        free_stream_ratio=float(acrf.get("free_stream_ratio", 0.3)),

        enable_timestep_aware_loss=bool(acrf.get("enable_timestep_aware_loss", False)),
        timestep_high_threshold=float(acrf.get("timestep_high_threshold", 0.7)),
        timestep_low_threshold=float(acrf.get("timestep_low_threshold", 0.3)),
        enable_curvature=bool(acrf.get("enable_curvature", False)),
        lambda_curvature=float(acrf.get("lambda_curvature", 0.05)),
        curvature_interval=int(acrf.get("curvature_interval", 10)),
        curvature_start_epoch=int(acrf.get("curvature_start_epoch", 0)),
        cfg_training=bool(acrf.get("cfg_training", False)),
        cfg_scale=float(acrf.get("cfg_scale", 7.0)),
        cfg_training_ratio=float(acrf.get("cfg_training_ratio", 0.3)),

        # ── Network / LoRA ──
        network_dim=int(network.get("dim", network.get("network_dim", 8))),
        network_alpha=float(network.get("alpha", network.get("network_alpha", 4.0))),
        resume_lora_path=lora_sec.get("resume_lora_path", network.get("resume_path", "")),
        resume_training=bool(lora_sec.get("resume_training", False)),
        train_adaln=bool(lora_sec.get("train_adaln", False)),
        train_norm=bool(lora_sec.get("train_norm", False)),
        train_single_stream=bool(lora_sec.get("train_single_stream", False)),

        # ── ControlNet ──
        controlnet_resume=bool(cnet.get("resume_training", False)),
        controlnet_path=cnet.get("controlnet_path", ""),
        control_types=cnet.get("control_types", ["canny"]),
        conditioning_scale=float(cnet.get("conditioning_scale", 0.75)),

        # ── Optimizer ──
        optimizer_type=opt.get("type", opt.get("optimizer_type", "AdamW8bit")),
        relative_step=bool(opt.get("relative_step", False)),

        # ── Training / Loss ──
        learning_rate=float(training.get("learning_rate", opt.get("learning_rate", 1e-4))),
        weight_decay=float(training.get("weight_decay", 0.0)),
        lr_scheduler=training.get("lr_scheduler", training.get("scheduler_type", "constant")),
        lr_warmup_steps=int(training.get("lr_warmup_steps", training.get("warmup_steps", 0))),
        lr_num_cycles=int(training.get("lr_num_cycles", training.get("num_cycles", 1))),
        lr_pct_start=float(training.get("lr_pct_start", 0.1)),
        lr_div_factor=float(training.get("lr_div_factor", 10.0)),
        lr_final_div_factor=float(training.get("lr_final_div_factor", 100.0)),

        lambda_mse=float(training.get("lambda_mse", 1.0)),
        lambda_l1=float(training.get("lambda_l1", 1.0)),
        lambda_cosine=float(training.get("lambda_cosine", 0.1)),
        enable_freq=bool(training.get("enable_freq", False)),
        lambda_freq=float(training.get("lambda_freq", 0.3)),
        alpha_hf=float(training.get("alpha_hf", 1.0)),
        beta_lf=float(training.get("beta_lf", 0.2)),
        enable_style=bool(training.get("enable_style", False)),
        lambda_style=float(training.get("lambda_style", 0.3)),
        lambda_light=float(training.get("lambda_light", 0.5)),
        lambda_color=float(training.get("lambda_color", 0.3)),


        # ── Dataset ──
        batch_size=int(dataset.get("batch_size", 1)),
        shuffle=bool(dataset.get("shuffle", True)),
        enable_bucket=bool(dataset.get("enable_bucket", True)),
        drop_text_ratio=float(dataset.get("drop_text_ratio", 0.1)),
        dataset_config=dataset,

        # ── Reg dataset ──
        reg_enabled=bool(reg.get("enabled", False)),
        reg_weight=float(reg.get("weight", 1.0)),
        reg_ratio=float(reg.get("ratio", 0.5)),
        reg_dataset_config=reg,

        # ── Advanced ──
        max_grad_norm=float(adv.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(adv.get("gradient_checkpointing", True)),
        blocks_to_swap=int(adv.get("blocks_to_swap", 0)),
        num_train_epochs=int(adv.get("num_train_epochs", 10)),
        save_every_n_epochs=int(adv.get("save_every_n_epochs", 1)),
        gradient_accumulation_steps=int(adv.get("gradient_accumulation_steps", 4)),
        mixed_precision=adv.get("mixed_precision", "bf16"),
        seed=int(adv.get("seed", 42)),
        num_gpus=int(adv.get("num_gpus", 1)),
        gpu_ids=adv.get("gpu_ids", ""),
    )
    return args


# ============================================================================
# Training Logic
# ============================================================================

def main(args: SimpleNamespace):
    """Main training entry point."""

    # --- Accelerate setup ---
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    log_dir = Path(args.output_dir) / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=str(log_dir),
    )
    device = accelerator.device
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("═" * 60)
        logger.info("  Z-Image Training — Accelerate Edition")
        logger.info("═" * 60)
        logger.info(f"  Config : {args.config_path}")
        logger.info(f"  Device : {device}  |  dtype: {weight_dtype}")
        logger.info(f"  Output : {args.output_dir}/{args.output_name}")
        logger.info("─" * 60)

    # ------------------------------------------------------------------
    # 1. Load Transformer (frozen)
    # ------------------------------------------------------------------
    from diffusers import ZImageTransformer2DModel

    if accelerator.is_main_process:
        logger.info(f"Loading transformer from: {args.dit}")
        print(f"[TRAINING_INFO] status=loading, phase=transformer", flush=True)

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.dit, subfolder="transformer", torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] status=loading, phase=optimizations", flush=True)

    # ------------------------------------------------------------------
    # 2. Optimizations: Block Swap + Attention + Gradient Checkpointing
    # ------------------------------------------------------------------
    from shared.utils.model_hooks import apply_all_optimizations

    swapper = apply_all_optimizations(
        transformer,
        blocks_to_swap=args.blocks_to_swap,
        attention_backend="native",
        gradient_checkpointing=args.gradient_checkpointing,
        device=device,
    )

    transformer.to(device)
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] status=loading, phase=lora_network", flush=True)

    # ------------------------------------------------------------------
    # 3. LoRA Network
    # ------------------------------------------------------------------
    from zimage_trainer.networks.lora import LoRANetwork, create_network

    network = create_network(
        multiplier=1.0,
        network_dim=args.network_dim,
        network_alpha=args.network_alpha,
        unet=transformer,
        train_adaln=args.train_adaln,
        train_norm=args.train_norm,
        train_single_stream=args.train_single_stream,
    )
    network.apply_to(transformer)
    network.to(device=device, dtype=weight_dtype)

    # --- Structured config summary (after network created) ---
    if accelerator.is_main_process:
        # Timestep & ACRF
        logger.info("┌─ Timestep ─────────────────────────────────────────┐")
        logger.info(f"│  Mode: {args.timestep_mode}")
        if args.timestep_mode in ("acrf", "anchor"):
            logger.info(f"│  ACRF Steps: {args.acrf_steps}  |  RAFT: {args.raft_mode}")
            if args.raft_mode:
                logger.info(f"│  Free-stream ratio: {args.free_stream_ratio}")
            if args.enable_timestep_aware_loss:
                logger.info(f"│  Timestep-Aware: high>{args.timestep_high_threshold}  low<{args.timestep_low_threshold}")
            if args.enable_curvature:
                logger.info(f"│  Curvature: λ={args.lambda_curvature}  interval={args.curvature_interval}  start_epoch={args.curvature_start_epoch}")
        elif args.timestep_mode == "logit_normal":
            logger.info(f"│  Logit mean={args.logit_mean}  std={args.logit_std}")
        elif args.timestep_mode == "uniform":
            if args.use_dynamic_shift:
                logger.info(f"│  Dynamic Shift: base={args.base_shift}  max={args.max_shift}")
            else:
                logger.info(f"│  Fixed Shift: {args.shift}")
        if args.snr_gamma > 0:
            logger.info(f"│  SNR γ={args.snr_gamma}  floor={args.snr_floor}")
        logger.info("├─ Network ─────────────────────────────────────────┤")
        logger.info(f"│  Type: {args.training_type}  |  Rank: {args.network_dim}  |  Alpha: {args.network_alpha}")
        extra_targets = []
        if args.train_adaln: extra_targets.append("AdaLN")
        if args.train_norm: extra_targets.append("Norm")
        if args.train_single_stream: extra_targets.append("SingleStream")
        if extra_targets:
            logger.info(f"│  Extra targets: {', '.join(extra_targets)}")
        logger.info("├─ Optimizer ───────────────────────────────────────┤")
        logger.info(f"│  {args.optimizer_type}  |  LR: {args.learning_rate:.1e}  |  WD: {args.weight_decay}")
        logger.info(f"│  Scheduler: {args.lr_scheduler}  |  Warmup: {args.lr_warmup_steps}")
        if hasattr(args, 'relative_step') and args.relative_step:
            logger.info(f"│  Relative step: enabled (Adafactor adaptive LR)")
        logger.info("├─ Loss ────────────────────────────────────────────┤")
        active_losses = ["MSE (base)"]
        if args.lambda_l1 > 0: active_losses.append(f"L1×{args.lambda_l1}")
        if args.lambda_cosine > 0: active_losses.append(f"Cosine×{args.lambda_cosine}")
        if args.enable_freq and args.lambda_freq > 0: active_losses.append(f"Freq×{args.lambda_freq}")
        if args.enable_style and args.lambda_style > 0: active_losses.append(f"Style×{args.lambda_style}")
        if args.enable_timestep_aware_loss: active_losses.append("TimestepAware")
        if args.enable_curvature: active_losses.append(f"Curvature×{args.lambda_curvature}")
        if args.cfg_training: active_losses.append(f"CFG(s={args.cfg_scale})")
        logger.info(f"│  {' + '.join(active_losses)}")
        logger.info("├─ Dataset ────────────────────────────────────────┤")
        logger.info(f"│  Batch: {args.batch_size}  |  Accum: {args.gradient_accumulation_steps}  |  Precision: {args.mixed_precision}")
        logger.info(f"│  Bucket: {args.enable_bucket}  |  Drop text: {args.drop_text_ratio}")
        if args.reg_enabled:
            logger.info(f"│  Reg dataset: weight={args.reg_weight}  ratio={args.reg_ratio}")
        logger.info(f"│  Epochs: {args.num_train_epochs}  |  Save every: {args.save_every_n_epochs}")
        if args.max_grad_norm > 0:
            logger.info(f"│  Grad clip: {args.max_grad_norm}")
        logger.info("└───────────────────────────────────────────────────┘")

    # Resume LoRA weights
    if args.resume_training and args.resume_lora_path and os.path.isfile(args.resume_lora_path):
        if accelerator.is_main_process:
            logger.info(f"Resuming LoRA from: {args.resume_lora_path}")
        network.load_weights(args.resume_lora_path)

    trainable_params, _descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)

    if accelerator.is_main_process:
        total = sum(pg["params"].numel() if hasattr(pg["params"], 'numel') else sum(p.numel() for p in pg["params"]) for pg in trainable_params)
        logger.info(f"Trainable parameters: {total:,}")

    # ------------------------------------------------------------------
    # 4. Timestep Sampler
    # ------------------------------------------------------------------
    from shared.flow_matching.samplers import create_sampler

    sampler_fn = create_sampler(args.timestep_mode)

    # ------------------------------------------------------------------
    # 5. Loss Functions
    # ------------------------------------------------------------------
    freq_loss_fn = None
    style_loss_fn = None

    if args.enable_freq and args.lambda_freq > 0:
        from shared.losses import FrequencyAwareLoss
        freq_loss_fn = FrequencyAwareLoss(
            alpha_hf=args.alpha_hf,
            beta_lf=args.beta_lf,
        )

    if args.enable_style and args.lambda_style > 0:
        from shared.losses import LatentStyleStructureLoss
        style_loss_fn = LatentStyleStructureLoss(
            lambda_light=args.lambda_light,
            lambda_color=args.lambda_color,
        )

    # ------------------------------------------------------------------
    # 6. DataLoader
    # ------------------------------------------------------------------
    from zimage_trainer.dataset.dataloader import create_dataloader, create_reg_dataloader, get_reg_config
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] status=loading, phase=dataset", flush=True)

    train_dataloader = create_dataloader(args)
    reg_dataloader = create_reg_dataloader(args) if args.reg_enabled else None
    reg_config = get_reg_config(args) if args.reg_enabled else None

    # ------------------------------------------------------------------
    # 7. Optimizer + LR Scheduler
    # ------------------------------------------------------------------
    from shared.utils.training_utils import get_optimizer
    from shared.utils.lr_schedulers import get_scheduler_with_onecycle

    optimizer = get_optimizer(
        params=trainable_params,
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        relative_step=args.relative_step,
    )

    num_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_training_steps = num_steps_per_epoch * args.num_train_epochs
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] status=loading, phase=optimizer, total_steps={total_training_steps}, num_train_epochs={args.num_train_epochs}", flush=True)

    lr_scheduler = get_scheduler_with_onecycle(
        scheduler_type=args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=total_training_steps,
        num_warmup_steps=args.lr_warmup_steps,
        num_cycles=args.lr_num_cycles,
        max_lr=args.learning_rate,
        pct_start=args.lr_pct_start,
        div_factor=args.lr_div_factor,
        final_div_factor=args.lr_final_div_factor,
    )

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if args.dry_run:
        if accelerator.is_main_process:
            logger.info("[DRY-RUN] Config parsed successfully. Exiting.")
            logger.info(f"  Model: {args.dit}")
            logger.info(f"  LoRA dim={args.network_dim}, alpha={args.network_alpha}")
            logger.info(f"  Timestep mode: {args.timestep_mode}")
            logger.info(f"  Dataset samples: {len(train_dataloader.dataset)}")
            logger.info(f"  Steps/epoch: {num_steps_per_epoch}, Total: {total_training_steps}")
        return

    # ------------------------------------------------------------------
    # Validate: at least one loss must be active
    # ------------------------------------------------------------------
    has_loss = (
        args.lambda_mse > 0
        or args.lambda_l1 > 0
        or args.lambda_cosine > 0
        or (args.enable_freq and args.lambda_freq > 0)
        or (args.enable_style and args.lambda_style > 0)
    )
    if not has_loss:
        raise ValueError(
            "No loss function enabled! Set at least one of: "
            "lambda_mse, lambda_l1, lambda_cosine, enable_freq+lambda_freq, enable_style+lambda_style"
        )

    # ------------------------------------------------------------------
    # Prepare with Accelerate
    # ------------------------------------------------------------------
    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler,
    )
    if reg_dataloader is not None:
        reg_dataloader = accelerator.prepare(reg_dataloader)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard tracker
    accelerator.init_trackers(
        project_name=args.output_name,
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "network_dim": args.network_dim,
            "network_alpha": args.network_alpha,
            "timestep_mode": args.timestep_mode,
            "optimizer": args.optimizer_type,
            "lr_scheduler": args.lr_scheduler,
            "snr_gamma": args.snr_gamma,
        },
    )

    # ------------------------------------------------------------------
    # Interrupt handler — save emergency checkpoint
    # ------------------------------------------------------------------
    interrupted = False

    def _signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        if accelerator.is_main_process:
            logger.warning("Interrupt received. Will save and exit after current step.")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ------------------------------------------------------------------
    # SNR Weighting
    # ------------------------------------------------------------------
    from shared.snr import compute_snr_weights

    # ------------------------------------------------------------------
    # Z-Image specific target computation
    # ------------------------------------------------------------------
    from shared.flow_matching.noising import get_z_t, compute_velocity_target

    # ------------------------------------------------------------------
    # RAFT mode helpers (free-stream / structured ratio control)
    # ------------------------------------------------------------------
    def _should_use_free_stream(epoch: int, step: int) -> bool:
        """Decide whether this step uses free-stream (uniform) vs anchor sampling."""
        if not args.raft_mode:
            return False
        return torch.rand(1).item() < args.free_stream_ratio

    # Emit config for frontend
    if accelerator.is_main_process:
        print(f"[TRAINING_INFO] training_type={args.training_type}")
        print(f"[TRAINING_INFO] total_epochs={args.num_train_epochs}")
        print(f"[TRAINING_INFO] steps_per_epoch={num_steps_per_epoch}")
        print(f"[TRAINING_INFO] total_steps={total_training_steps}")
        print(f"[TRAINING_INFO] output_dir={output_dir}")
        print(f"[TRAINING_INFO] output_name={args.output_name}")
        print(f"[TRAINING_INFO] batch_size={args.batch_size}")
        print(f"[TRAINING_INFO] gradient_accumulation={args.gradient_accumulation_steps}")
        print(f"[TRAINING_INFO] mixed_precision={args.mixed_precision}")
        print(f"[TRAINING_INFO] timestep_mode={args.timestep_mode}")
        print(f"[TRAINING_INFO] optimizer={args.optimizer_type}")
        print(f"[TRAINING_INFO] lr_scheduler={args.lr_scheduler}")
        print(f"[TRAINING_INFO] network_dim={args.network_dim}")
        print(f"[TRAINING_INFO] network_alpha={args.network_alpha}")
        sys.stdout.flush()

    # ==================================================================
    #                      TRAINING LOOP
    # ==================================================================
    global_step = 0
    best_loss = float("inf")
    reg_iter = None  # Lazy init for reg dataset iterator

    # Loss accumulation across micro-batches for accurate logging
    _accum_loss = 0.0
    _accum_components: Dict[str, float] = {}
    _accum_count = 0

    for epoch in range(args.num_train_epochs):
        network.train()
        epoch_loss = 0.0
        epoch_steps = 0

        if accelerator.is_main_process:
            logger.info(f"\n{'='*40} Epoch {epoch + 1}/{args.num_train_epochs} {'='*40}")

        for step_in_epoch, batch in enumerate(train_dataloader):
            if interrupted:
                break

            with accelerator.accumulate(network):
                # --------------------------------------------------
                # Unpack batch (from create_dataloader)
                # --------------------------------------------------
                latents = batch["latents"].to(device=device, dtype=weight_dtype)
                vl_embed = batch.get("vl_embed", None)

                # Handle vl_embed — may be a padded tensor or a list
                if isinstance(vl_embed, torch.Tensor):
                    vl_embed = list(vl_embed.to(device=device, dtype=weight_dtype).unbind(dim=0))
                elif isinstance(vl_embed, (list, tuple)):
                    vl_embed = [v.to(device=device, dtype=weight_dtype) for v in vl_embed]

                batch_size = latents.shape[0]

                # --------------------------------------------------
                # Drop text (classifier-free training)
                # --------------------------------------------------
                if args.drop_text_ratio > 0 and vl_embed is not None:
                    for i in range(batch_size):
                        if torch.rand(1).item() < args.drop_text_ratio:
                            vl_embed[i] = torch.zeros_like(vl_embed[i])

                # --------------------------------------------------
                # Sample noise
                # --------------------------------------------------
                noise = torch.randn_like(latents)

                # --------------------------------------------------
                # Sample timesteps (sigmas)
                # --------------------------------------------------
                use_free = _should_use_free_stream(epoch, step_in_epoch)

                if use_free:
                    from shared.flow_matching.samplers import sample_uniform
                    sigmas = sample_uniform(
                        batch_size=batch_size,
                        shift=args.shift,
                        dynamic_shift=args.use_dynamic_shift,
                        latent_shape=latents.shape,
                        base_shift=args.base_shift,
                        max_shift=args.max_shift,
                        device=device,
                        dtype=weight_dtype,
                    )
                else:
                    sampler_kwargs = dict(
                        batch_size=batch_size,
                        device=device,
                        dtype=weight_dtype,
                    )
                    if args.timestep_mode in ("acrf", "anchor"):
                        sampler_kwargs.update(
                            num_inference_steps=args.acrf_steps,
                            shift=args.shift,
                            dynamic_shift=args.use_dynamic_shift,
                            latent_shape=latents.shape,
                            base_shift=args.base_shift,
                            max_shift=args.max_shift,
                            jitter_scale=args.jitter_scale,
                            stratified=True,
                        )
                    elif args.timestep_mode == "logit_normal":
                        sampler_kwargs.update(
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                        )
                    elif args.timestep_mode == "uniform":
                        sampler_kwargs.update(
                            shift=args.shift,
                            dynamic_shift=args.use_dynamic_shift,
                            latent_shape=latents.shape,
                            base_shift=args.base_shift,
                            max_shift=args.max_shift,
                        )

                    sigmas = sampler_fn(**sampler_kwargs)

                # --------------------------------------------------
                # Create noisy input z_t
                # --------------------------------------------------
                z_t = get_z_t(latents, noise, sigmas)

                # Latent jitter (optional)
                if args.latent_jitter_scale > 0:
                    latent_jitter = torch.randn_like(z_t) * args.latent_jitter_scale
                    z_t = z_t + latent_jitter

                # --------------------------------------------------
                # Compute target velocity: v = noise - x_0
                # --------------------------------------------------
                target_velocity = compute_velocity_target(latents, noise)

                # --------------------------------------------------
                # Prepare transformer input (Z-Image format)
                #   Pipeline: unsqueeze(2) → list(unbind(0))
                #   Timestep: (1000 - t) / 1000 where sigma ∈ [0,1]
                #   So: timestep = 1 - sigma
                # --------------------------------------------------
                model_input = z_t.unsqueeze(2)  # (B, C, 1, H, W)

                if args.gradient_checkpointing:
                    model_input.requires_grad_(True)

                model_input_list = list(model_input.unbind(dim=0))

                # Timestep normalization — aligned with official ZImagePipeline
                timesteps_normalized = (1.0 - sigmas).to(dtype=weight_dtype)

                # --------------------------------------------------
                # Forward pass
                # --------------------------------------------------
                if args.cfg_training and torch.rand(1).item() < args.cfg_training_ratio:
                    # CFG Training Mode
                    negative_embed = [torch.zeros_like(v) for v in vl_embed]
                    cfg_input_list = model_input_list + model_input_list
                    cfg_timesteps = timesteps_normalized.repeat(2)
                    cfg_embed = vl_embed + negative_embed

                    cfg_pred_list = transformer(
                        x=cfg_input_list,
                        t=cfg_timesteps,
                        cap_feats=cfg_embed,
                    )[0]

                    pos_pred_list = cfg_pred_list[:batch_size]
                    neg_pred_list = cfg_pred_list[batch_size:]

                    pos_pred = torch.stack(pos_pred_list, dim=0).squeeze(2)
                    neg_pred = torch.stack(neg_pred_list, dim=0).squeeze(2)

                    # *** Z-Image negation (official pipeline line 558) ***
                    pos_pred = -pos_pred
                    neg_pred = -neg_pred

                    model_pred = pos_pred + args.cfg_scale * (pos_pred - neg_pred)
                else:
                    # Standard forward (no CFG)
                    model_pred_list = transformer(
                        x=model_input_list,
                        t=timesteps_normalized,
                        cap_feats=vl_embed,
                    )[0]

                    model_pred = torch.stack(model_pred_list, dim=0)
                    model_pred = model_pred.squeeze(2)

                    # *** Z-Image negation (official pipeline line 558) ***
                    model_pred = -model_pred

                # --------------------------------------------------
                # Compute Losses
                # --------------------------------------------------
                loss_components: Dict[str, float] = {}
                loss = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)

                # MSE loss (optional, default on)
                if args.lambda_mse > 0:
                    mse_loss = F.mse_loss(model_pred, target_velocity)
                    loss = loss + args.lambda_mse * mse_loss
                    loss_components["mse"] = mse_loss.item()

                # L1 loss (optional)
                if args.lambda_l1 > 0:
                    l1_loss = F.l1_loss(model_pred, target_velocity)
                    loss = loss + args.lambda_l1 * l1_loss
                    loss_components["l1"] = l1_loss.item()

                # Optional additive Cosine loss
                if args.lambda_cosine > 0:
                    cos_loss = 1.0 - F.cosine_similarity(
                        model_pred.flatten(1), target_velocity.flatten(1), dim=1
                    ).mean()
                    loss = loss + args.lambda_cosine * cos_loss
                    loss_components["cosine"] = cos_loss.item()

                # Frequency-aware loss
                if freq_loss_fn is not None and args.lambda_freq > 0:
                    freq_l = freq_loss_fn(model_pred, target_velocity, z_t, sigmas * 1000.0, num_train_timesteps=1000)
                    loss = loss + args.lambda_freq * freq_l
                    loss_components["freq"] = freq_l.item()

                # Style-structure loss
                if style_loss_fn is not None and args.lambda_style > 0:
                    style_l = style_loss_fn(model_pred, target_velocity, z_t, sigmas * 1000.0, num_train_timesteps=1000)
                    loss = loss + args.lambda_style * style_l
                    loss_components["style"] = style_l.item()

                # Timestep-aware loss weighting
                if args.enable_timestep_aware_loss:
                    sigma_mean = sigmas.mean().item()
                    if sigma_mean > args.timestep_high_threshold:
                        # High noise: emphasize structural learning
                        ta_weight = 1.0 + (sigma_mean - args.timestep_high_threshold) / (1.0 - args.timestep_high_threshold)
                        loss = loss * ta_weight
                    elif sigma_mean < args.timestep_low_threshold:
                        # Low noise: emphasize detail refinement
                        ta_weight = 1.0 + (args.timestep_low_threshold - sigma_mean) / args.timestep_low_threshold
                        loss = loss * ta_weight
                    loss_components["ta_w"] = ta_weight if sigma_mean > args.timestep_high_threshold or sigma_mean < args.timestep_low_threshold else 1.0

                # Curvature regularization loss
                if args.enable_curvature and args.lambda_curvature > 0:
                    if epoch >= args.curvature_start_epoch and (global_step + 1) % args.curvature_interval == 0:
                        # Approximate trajectory curvature: penalize deviation from straight-line path
                        # Straight-line prediction: (1-sigma)*x0 + sigma*noise, derivative = noise - x0 = velocity
                        # Curvature ~ ||model_pred - velocity||^2 (model should predict close to velocity)
                        # Use L2 distance between normalized predictions as curvature proxy
                        pred_norm = F.normalize(model_pred.flatten(1), dim=1)
                        target_norm = F.normalize(target_velocity.detach().flatten(1), dim=1)
                        curvature_loss = (1.0 - F.cosine_similarity(pred_norm, target_norm, dim=1)).mean()
                        loss = loss + args.lambda_curvature * curvature_loss
                        loss_components["curvature"] = curvature_loss.item()


                # --------------------------------------------------
                # Min-SNR Weighting
                # --------------------------------------------------
                if args.snr_gamma > 0:
                    scheduler_timesteps = sigmas * 1000.0
                    snr_weights = compute_snr_weights(
                        scheduler_timesteps, num_train_timesteps=1000,
                        snr_gamma=args.snr_gamma, snr_floor=args.snr_floor,
                    )
                    loss = loss * snr_weights.mean()

                # --------------------------------------------------
                # Regularization dataset loss
                # --------------------------------------------------
                if reg_dataloader is not None and reg_config is not None:
                    if torch.rand(1).item() < reg_config.get("ratio", 0.5):
                        try:
                            reg_batch = next(reg_iter)
                        except (StopIteration, TypeError):
                            reg_iter = iter(reg_dataloader)
                            reg_batch = next(reg_iter)

                        reg_latents = reg_batch["latents"].to(device=device, dtype=weight_dtype)
                        reg_noise = torch.randn_like(reg_latents)
                        reg_sigmas_kwargs = dict(batch_size=reg_latents.shape[0], device=device, dtype=weight_dtype)
                        if args.timestep_mode in ("acrf", "anchor"):
                            reg_sigmas_kwargs.update(num_inference_steps=args.acrf_steps, shift=args.shift)
                        reg_sigmas = sampler_fn(**reg_sigmas_kwargs)
                        reg_z_t = get_z_t(reg_latents, reg_noise, reg_sigmas)
                        reg_target = compute_velocity_target(reg_latents, reg_noise)

                        reg_input = reg_z_t.unsqueeze(2)
                        if args.gradient_checkpointing:
                            reg_input.requires_grad_(True)
                        reg_input_list = list(reg_input.unbind(dim=0))
                        reg_ts = (1.0 - reg_sigmas).to(dtype=weight_dtype)

                        reg_vl = reg_batch.get("vl_embed", None)
                        if isinstance(reg_vl, torch.Tensor):
                            reg_vl = list(reg_vl.to(device=device, dtype=weight_dtype).unbind(dim=0))
                        elif isinstance(reg_vl, (list, tuple)):
                            reg_vl = [v.to(device=device, dtype=weight_dtype) for v in reg_vl]

                        reg_pred_list = transformer(x=reg_input_list, t=reg_ts, cap_feats=reg_vl)[0]
                        reg_pred = torch.stack(reg_pred_list, dim=0).squeeze(2)
                        reg_pred = -reg_pred  # Z-Image negation

                        reg_loss = F.mse_loss(reg_pred, reg_target)
                        reg_w = reg_config.get("weight", 1.0)
                        loss = loss + reg_w * reg_loss
                        loss_components["reg"] = reg_loss.item()

                # --------------------------------------------------
                # NaN / Inf protection
                # --------------------------------------------------
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        logger.warning(f"[NaN] Loss is NaN/Inf at step {global_step}, skipping. Components: {loss_components}")
                    optimizer.zero_grad()
                    continue

                # Accumulate loss across micro-batches for averaging
                _accum_loss += loss.detach().float().item()
                for k, v in loss_components.items():
                    _accum_components[k] = _accum_components.get(k, 0.0) + v
                _accum_count += 1

                # --------------------------------------------------
                # Backward + Step
                # --------------------------------------------------
                accelerator.backward(loss)

                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --------------------------------------------------
            # Logging (only after full accumulation cycle)
            # --------------------------------------------------
            if not accelerator.sync_gradients:
                continue

            global_step += 1

            # Average accumulated loss across micro-batches
            avg_loss = _accum_loss / max(_accum_count, 1)
            avg_components = {k: v / max(_accum_count, 1) for k, v in _accum_components.items()}

            epoch_loss += avg_loss
            epoch_steps += 1

            # Reset accumulators
            _accum_loss = 0.0
            _accum_components = {}
            _accum_count = 0

            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]["lr"]
                loss_str = " | ".join(f"{k}={v:.4f}" for k, v in avg_components.items())
                mode_tag = ""
                if args.timestep_mode in ("acrf", "anchor"):
                    mode_tag = " [ACRF]" if not use_free else " [FREE]"
                print(
                    f"[STEP] epoch={epoch+1} step={global_step}/{total_training_steps} "
                    f"loss={avg_loss:.5f} lr={current_lr:.2e}{mode_tag} | {loss_str}",
                    flush=True,
                )

                # TensorBoard logging
                tb_logs = {"train/loss": avg_loss, "train/lr": current_lr}
                for k, v in avg_components.items():
                    tb_logs[f"train/{k}"] = v
                tb_logs["train/avg_loss"] = epoch_loss / max(epoch_steps, 1)
                tb_logs["train/epoch"] = epoch + 1
                accelerator.log(tb_logs, step=global_step)

        # --------------------------------------------------
        # End of Epoch
        # --------------------------------------------------
        if interrupted:
            break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        if accelerator.is_main_process:
            logger.info(f"{'─'*40} Epoch {epoch+1} Summary {'─'*40}")
            logger.info(f"  avg_loss={avg_loss:.5f}  |  steps={epoch_steps}  |  lr={optimizer.param_groups[0]['lr']:.2e}")
            accelerator.log({"epoch/avg_loss": avg_loss, "epoch/epoch": epoch + 1}, step=global_step)

        # Save checkpoint
        if accelerator.is_main_process and ((epoch + 1) % args.save_every_n_epochs == 0 or (epoch + 1) == args.num_train_epochs):
            save_path = output_dir / f"{args.output_name}_epoch{epoch+1}.safetensors"
            unwrapped = accelerator.unwrap_model(network)
            unwrapped.save_weights(
                str(save_path),
                dtype=torch.bfloat16,
                metadata={"epoch": str(epoch + 1), "loss": f"{avg_loss:.5f}"},
            )
            logger.info(f"Saved: {save_path}")
            print(f"[TRAINING_INFO] saved={save_path}", flush=True)

    # ==================================================================
    # Completion
    # ==================================================================
    if accelerator.is_main_process:
        logger.info("═" * 60)
        logger.info("  Training Complete ✓")
        logger.info(f"  Total steps: {global_step}  |  Final loss: {epoch_loss / max(epoch_steps, 1):.5f}")
        logger.info("═" * 60)
        print(f"[TRAINING_INFO] status=completed", flush=True)

    # Cleanup
    if swapper is not None and hasattr(swapper, 'remove_hooks'):
        swapper.remove_hooks()

    accelerator.end_training()




# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    main(args)
