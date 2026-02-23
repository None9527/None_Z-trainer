# -*- coding: utf-8 -*-
"""
🎯 标准训练损失函数 (Standard Training Loss)

Charbonnier Loss + Cosine Loss 组合

这是 Z-Image Trainer 的标准损失函数，用于 Rectified Flow v-prediction 训练。

损失公式：
L = λ_l1 * Charbonnier(v_pred, v_target) + λ_cos * (1 - cos_sim(v_pred, v_target))

其中：
- Charbonnier: sqrt((v_pred - v_target)² + ε²) - 平滑 L1，在 0 点可微
- Cosine: 1 - cosine_similarity - 速度方向一致性约束

设计理念：
- Charbonnier 保证像素级精度，对离群值鲁棒
- Cosine 保证速度方向正确，对 Rectified Flow 至关重要
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StandardLoss(nn.Module):
    """
    标准训练损失：Charbonnier (Robust L1) + Cosine
    
    这是 Z-Image Trainer 所有模型的默认损失函数。
    
    Args:
        lambda_l1: Charbonnier 损失权重 (默认 1.0)
        lambda_cosine: Cosine 损失权重 (默认 0.1)
        epsilon: Charbonnier 平滑系数 (默认 1e-6)
    
    Example:
        >>> loss_fn = StandardLoss(lambda_l1=1.0, lambda_cosine=0.1)
        >>> loss, components = loss_fn(pred_v, target_v, return_components=True)
        >>> print(f"Total: {loss.item():.4f}, L1: {components['l1'].item():.4f}")
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_cosine: float = 0.1,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cosine = lambda_cosine
        self.epsilon = epsilon
        
        logger.info(f"[StandardLoss] 初始化标准损失函数")
        logger.info(f"  Charbonnier(L1) 权重: {lambda_l1}")
        logger.info(f"  Cosine 权重: {lambda_cosine}")
        logger.info(f"  Epsilon: {epsilon}")
    
    def charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Charbonnier Loss (平滑 L1)
        
        L = mean(sqrt((pred - target)² + ε²))
        
        优势：
        - 在 0 点可微分（与 L1 不同）
        - 对离群值比 L2 更鲁棒
        - 保持边缘锐利
        """
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.epsilon ** 2).mean()
    
    def cosine_direction(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cosine 方向损失
        
        L = 1 - cosine_similarity(pred, target)
        
        意义：
        - 约束预测速度与目标速度方向一致
        - 对 Rectified Flow 非常重要（直线轨迹）
        """
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        return 1.0 - cos_sim
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        snr_weights: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算标准损失
        
        Args:
            pred_v: 模型预测的速度 [B, C, H, W]
            target_v: 目标速度 [B, C, H, W]
            snr_weights: 可选的 Min-SNR 权重 [B]
            return_components: 是否返回损失分量
        
        Returns:
            loss: 加权后的总损失
            components: 损失分量字典 {"l1": ..., "cosine": ...}
        """
        # 计算各分量
        loss_l1 = self.charbonnier(pred_v, target_v)
        loss_cosine = self.cosine_direction(pred_v, target_v)
        
        # 加权组合
        loss = self.lambda_l1 * loss_l1 + self.lambda_cosine * loss_cosine
        
        # 应用 SNR 权重
        if snr_weights is not None:
            loss = loss * snr_weights.mean()
        
        components = {
            "l1": loss_l1,
            "cosine": loss_cosine,
        }
        
        if return_components:
            return loss, components
        return loss, components


def compute_standard_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_cosine: float = 0.1,
    epsilon: float = 1e-6,
    snr_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    便捷函数：计算标准损失
    
    Args:
        pred_v: 预测速度
        target_v: 目标速度
        lambda_l1: L1 权重
        lambda_cosine: Cosine 权重
        epsilon: 平滑系数
        snr_weights: SNR 权重
    
    Returns:
        loss: 总损失
        components: 损失分量 (已转为 float)
    """
    # Charbonnier
    diff = pred_v - target_v
    loss_l1 = torch.sqrt(diff ** 2 + epsilon ** 2).mean()
    
    # Cosine
    pred_flat = pred_v.view(pred_v.shape[0], -1)
    target_flat = target_v.view(target_v.shape[0], -1)
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
    loss_cosine = 1.0 - cos_sim
    
    # 组合
    loss = lambda_l1 * loss_l1 + lambda_cosine * loss_cosine
    
    # SNR 加权
    if snr_weights is not None:
        loss = loss * snr_weights.mean()
    
    components = {
        "l1": loss_l1.item(),
        "cosine": loss_cosine.item(),
    }
    
    return loss, components
