# -*- coding: utf-8 -*-
"""
🎯 时间步分区动态 Loss 权重调度器 (Timestep-Aware Loss Weights)

根据当前时间步（sigma）分区，动态调整不同 Loss 的权重：
- 高噪声区 (σ > 0.7): 重结构/语义 → 增强 Style struct, 降低 Freq HF
- 中间区 (0.3 < σ < 0.7): 均衡所有 Loss
- 低噪声区 (σ < 0.3): 重纹理/细节 → 增强 Freq HF, 降低 Style struct

理论依据 (REPA / SFD 论文):
- 去噪早期 (σ接近1): 建立全局语义结构
- 去噪中期: 语义和纹理联合优化
- 去噪后期 (σ接近0): 精细化纹理细节
"""

import torch
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimestepAwareWeights:
    """时间步分区权重配置"""
    # Freq Loss 子权重调制
    freq_hf_scale: float = 1.0    # alpha_hf 缩放系数
    freq_lf_scale: float = 1.0    # beta_lf 缩放系数
    
    # Style Loss 子权重调制
    style_struct_scale: float = 1.0  # lambda_struct 缩放系数
    style_light_scale: float = 1.0   # lambda_light 缩放系数
    style_color_scale: float = 1.0   # lambda_color 缩放系数
    style_tex_scale: float = 1.0     # lambda_tex 缩放系数
    
    # 总体 Loss 权重调制
    lambda_freq_scale: float = 1.0   # lambda_freq 整体缩放
    lambda_style_scale: float = 1.0  # lambda_style 整体缩放


class TimestepAwareLossScheduler:
    """
    时间步感知的 Loss 权重调度器
    
    Z-Image sigma 走向:
    - sigma = 1.0 → 纯噪声 (推理开始, 去噪早期)
    - sigma = 0.0 → 干净图像 (推理结束, 去噪后期)
    
    因此:
    - 高 sigma (>0.7): 早期 → 重结构
    - 低 sigma (<0.3): 后期 → 重纹理
    """
    
    def __init__(
        self,
        high_noise_threshold: float = 0.7,  # 高噪声区阈值
        low_noise_threshold: float = 0.3,   # 低噪声区阈值
        enabled: bool = True,
    ):
        """
        Args:
            high_noise_threshold: sigma 高于此值视为高噪声区
            low_noise_threshold: sigma 低于此值视为低噪声区
            enabled: 是否启用动态权重
        """
        self.high_threshold = high_noise_threshold
        self.low_threshold = low_noise_threshold
        self.enabled = enabled
        
        # 预定义各区域的权重配置
        self._configs = {
            'high_noise': TimestepAwareWeights(
                # 高噪声区: 重结构/语义, 轻纹理
                freq_hf_scale=0.3,      # 降低高频
                freq_lf_scale=1.2,      # 增强低频锁定
                style_struct_scale=1.5, # 增强结构锁
                style_light_scale=1.0,  # 光影正常
                style_color_scale=0.8,  # 色调正常
                style_tex_scale=0.3,    # 降低纹理
                lambda_freq_scale=0.7,
                lambda_style_scale=1.2,
            ),
            'mid_noise': TimestepAwareWeights(
                # 中间区: 均衡
                freq_hf_scale=0.8,
                freq_lf_scale=1.0,
                style_struct_scale=1.0,
                style_light_scale=1.0,
                style_color_scale=1.0,
                style_tex_scale=0.8,
                lambda_freq_scale=1.0,
                lambda_style_scale=1.0,
            ),
            'low_noise': TimestepAwareWeights(
                # 低噪声区: 重纹理/细节, 轻结构
                freq_hf_scale=1.2,      # 增强高频
                freq_lf_scale=0.5,      # 降低低频锁定(允许细节变化)
                style_struct_scale=0.5, # 降低结构锁(已稳定)
                style_light_scale=0.8,  # 光影降低
                style_color_scale=0.5,  # 色调降低
                style_tex_scale=1.5,    # 增强纹理
                lambda_freq_scale=1.3,
                lambda_style_scale=0.6,
            ),
        }
        
        if enabled:
            logger.info(f"[TimestepAware] 时间步分区 Loss 调度器已启用")
            logger.info(f"  高噪声区 (σ > {high_noise_threshold}): 重结构")
            logger.info(f"  中间区 ({low_noise_threshold} < σ < {high_noise_threshold}): 均衡")
            logger.info(f"  低噪声区 (σ < {low_noise_threshold}): 重纹理")
    
    def get_weights(self, sigma: float) -> TimestepAwareWeights:
        """
        根据 sigma 获取权重配置
        
        Args:
            sigma: 当前噪声水平 (0-1), Z-Image: sigma = timestep / 1000
        
        Returns:
            TimestepAwareWeights: 权重配置
        """
        if not self.enabled:
            return TimestepAwareWeights()  # 返回全 1.0 的默认值
        
        if sigma > self.high_threshold:
            return self._configs['high_noise']
        elif sigma < self.low_threshold:
            return self._configs['low_noise']
        else:
            return self._configs['mid_noise']
    
    def get_batch_weights(
        self,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        批量获取权重 (每个样本可能有不同的 sigma)
        
        Args:
            timesteps: 时间步张量 (B,)
            num_train_timesteps: 总训练时间步数
        
        Returns:
            Dict[str, Tensor]: 各权重的缩放系数张量 (B,)
        """
        if not self.enabled:
            batch_size = timesteps.shape[0]
            device = timesteps.device
            dtype = timesteps.dtype
            return {
                'freq_hf_scale': torch.ones(batch_size, device=device, dtype=dtype),
                'freq_lf_scale': torch.ones(batch_size, device=device, dtype=dtype),
                'style_struct_scale': torch.ones(batch_size, device=device, dtype=dtype),
                'style_tex_scale': torch.ones(batch_size, device=device, dtype=dtype),
                'lambda_freq_scale': torch.ones(batch_size, device=device, dtype=dtype),
                'lambda_style_scale': torch.ones(batch_size, device=device, dtype=dtype),
            }
        
        # 计算 sigma
        sigmas = timesteps.float() / num_train_timesteps
        
        batch_size = sigmas.shape[0]
        device = sigmas.device
        dtype = sigmas.dtype
        
        # 初始化输出
        result = {
            'freq_hf_scale': torch.ones(batch_size, device=device, dtype=dtype),
            'freq_lf_scale': torch.ones(batch_size, device=device, dtype=dtype),
            'style_struct_scale': torch.ones(batch_size, device=device, dtype=dtype),
            'style_tex_scale': torch.ones(batch_size, device=device, dtype=dtype),
            'lambda_freq_scale': torch.ones(batch_size, device=device, dtype=dtype),
            'lambda_style_scale': torch.ones(batch_size, device=device, dtype=dtype),
        }
        
        # 高噪声区
        high_mask = sigmas > self.high_threshold
        high_cfg = self._configs['high_noise']
        result['freq_hf_scale'][high_mask] = high_cfg.freq_hf_scale
        result['freq_lf_scale'][high_mask] = high_cfg.freq_lf_scale
        result['style_struct_scale'][high_mask] = high_cfg.style_struct_scale
        result['style_tex_scale'][high_mask] = high_cfg.style_tex_scale
        result['lambda_freq_scale'][high_mask] = high_cfg.lambda_freq_scale
        result['lambda_style_scale'][high_mask] = high_cfg.lambda_style_scale
        
        # 低噪声区
        low_mask = sigmas < self.low_threshold
        low_cfg = self._configs['low_noise']
        result['freq_hf_scale'][low_mask] = low_cfg.freq_hf_scale
        result['freq_lf_scale'][low_mask] = low_cfg.freq_lf_scale
        result['style_struct_scale'][low_mask] = low_cfg.style_struct_scale
        result['style_tex_scale'][low_mask] = low_cfg.style_tex_scale
        result['lambda_freq_scale'][low_mask] = low_cfg.lambda_freq_scale
        result['lambda_style_scale'][low_mask] = low_cfg.lambda_style_scale
        
        # 中间区 (默认值已初始化为 1.0)
        mid_mask = ~high_mask & ~low_mask
        mid_cfg = self._configs['mid_noise']
        result['freq_hf_scale'][mid_mask] = mid_cfg.freq_hf_scale
        result['freq_lf_scale'][mid_mask] = mid_cfg.freq_lf_scale
        result['style_struct_scale'][mid_mask] = mid_cfg.style_struct_scale
        result['style_tex_scale'][mid_mask] = mid_cfg.style_tex_scale
        result['lambda_freq_scale'][mid_mask] = mid_cfg.lambda_freq_scale
        result['lambda_style_scale'][mid_mask] = mid_cfg.lambda_style_scale
        
        return result
    
    def get_mean_weights(
        self,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
    ) -> Dict[str, float]:
        """
        获取批次平均权重 (用于简化的标量乘法)
        
        Returns:
            Dict[str, float]: 各权重的平均缩放系数
        """
        batch_weights = self.get_batch_weights(timesteps, num_train_timesteps)
        return {k: v.mean().item() for k, v in batch_weights.items()}


def create_timestep_aware_scheduler_from_args(args) -> Optional[TimestepAwareLossScheduler]:
    """从训练参数创建时间步感知调度器"""
    
    # 检查是否启用
    enabled = getattr(args, 'enable_timestep_aware_loss', False)
    if not enabled:
        return None
    
    # 获取阈值参数
    high_threshold = getattr(args, 'timestep_high_threshold', 0.7)
    low_threshold = getattr(args, 'timestep_low_threshold', 0.3)
    
    return TimestepAwareLossScheduler(
        high_noise_threshold=high_threshold,
        low_noise_threshold=low_threshold,
        enabled=True,
    )
