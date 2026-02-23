# -*- coding: utf-8 -*-
"""
Training Domain - Value Objects

Immutable configuration objects for training.
These encapsulate training parameters as structured, validated data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class LossConfig:
    """Loss function weights configuration."""
    lambda_l1: float = 1.0
    lambda_l2: float = 0.0
    lambda_cosine: float = 0.0
    # Frequency loss
    enable_freq: bool = False
    lambda_freq: float = 0.3
    alpha_hf: float = 1.0
    beta_lf: float = 0.2
    # Style loss
    enable_style: bool = False
    lambda_style: float = 0.3
    lambda_light: float = 0.5
    lambda_color: float = 0.3
    lambda_tex: float = 0.5
    lambda_struct: float = 1.0
    # Curvature penalty
    enable_curvature: bool = False
    lambda_curvature: float = 0.01

    @property
    def has_main_loss(self) -> bool:
        """Check if any main-path loss is enabled."""
        return (
            self.lambda_l1 > 0 or
            self.lambda_l2 > 0 or
            self.lambda_cosine > 0 or
            (self.enable_freq and self.lambda_freq > 0) or
            (self.enable_style and self.lambda_style > 0)
        )


@dataclass(frozen=True)
class TimestepConfig:
    """Timestep sampling configuration — 3 modes."""
    mode: str = "uniform"  # "uniform" | "logit_normal" | "acrf"
    # Uniform mode params
    shift: float = 3.0
    use_dynamic_shift: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    # LogNorm mode params
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # ACRF mode params
    acrf_steps: int = 10
    jitter_scale: float = 0.02
    # Shared
    latent_jitter_scale: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    scheduler_type: str = "constant"
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    num_cycles: int = 1
    weight_decay: float = 0.0
    # OneCycleLR specific
    pct_start: float = 0.1
    div_factor: float = 10.0
    final_div_factor: float = 100.0


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA network configuration."""
    network_dim: int = 16
    network_alpha: float = 16.0
    resume_path: Optional[str] = None


@dataclass(frozen=True)
class SNRConfig:
    """Signal-to-noise ratio weighting."""
    snr_gamma: float = 5.0
    snr_floor: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """
    Complete training configuration (Value Object).

    Composed of sub-configurations for each concern.
    Frozen to ensure immutability after creation.
    """
    # Data
    dataset_path: str = ""
    output_dir: str = ""
    model_path: str = ""
    # Training params
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    save_every_n_epochs: int = 1
    gradient_checkpointing: bool = True
    # Sub-configs
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    snr: SNRConfig = field(default_factory=SNRConfig)
    # Optimizer
    optimizer_type: str = "AdamW"
