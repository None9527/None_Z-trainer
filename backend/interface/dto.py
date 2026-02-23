# -*- coding: utf-8 -*-
"""
Training DTO - Data Transfer Objects

Pydantic models for API request/response serialization.
These are the contract between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# --- Request DTOs ---

class LossConfigDTO(BaseModel):
    lambda_l1: float = 1.0
    lambda_l2: float = 0.0
    lambda_cosine: float = 0.0
    enable_freq: bool = False
    lambda_freq: float = 0.3
    alpha_hf: float = 1.0
    beta_lf: float = 0.2
    enable_style: bool = False
    lambda_style: float = 0.3
    lambda_light: float = 0.5
    lambda_color: float = 0.3

class TimestepConfigDTO(BaseModel):
    mode: str = "uniform"
    shift: float = 3.0
    use_dynamic_shift: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    logit_mean: float = 0.0
    logit_std: float = 1.0
    acrf_steps: int = 10
    jitter_scale: float = 0.02
    latent_jitter_scale: float = 0.0

class SchedulerConfigDTO(BaseModel):
    scheduler_type: str = "constant"
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    num_cycles: int = 1
    weight_decay: float = 0.0

class LoRAConfigDTO(BaseModel):
    network_dim: int = 16
    network_alpha: float = 16.0
    resume_path: Optional[str] = None

class TrainingConfigDTO(BaseModel):
    """Full training configuration from frontend."""
    dataset_path: str = ""
    output_dir: str = ""
    model_path: str = ""
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    save_every_n_epochs: int = 1
    optimizer_type: str = "AdamW"
    loss: LossConfigDTO = Field(default_factory=LossConfigDTO)
    timestep: TimestepConfigDTO = Field(default_factory=TimestepConfigDTO)
    scheduler: SchedulerConfigDTO = Field(default_factory=SchedulerConfigDTO)
    lora: LoRAConfigDTO = Field(default_factory=LoRAConfigDTO)


# --- Response DTOs ---

class TrainingStatusDTO(BaseModel):
    id: Optional[str] = None
    status: str = "idle"
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    loss: float = 0.0
    loss_components: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None

class ApiResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Any] = None
