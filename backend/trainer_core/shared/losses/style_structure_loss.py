# -*- coding: utf-8 -*-
"""
🎨 结构锁风格迁移损失函数 (Style-Structure Loss)

核心策略：
- 结构锁定 (SSIM)：锁死轮廓，防止脸崩
- 光影学习 (L通道统计)：学习大师的 S 曲线、对比度
- 色调迁移 (ab通道统计)：学习色彩偏好（冷暖/胶片感）
- 质感增强 (高频 L1)：增强清晰度和颗粒感

数学公式：
L_total = λ_struct * L_SSIM + λ_light * L_Moments_L + λ_color * L_Moments_ab + λ_tex * L_HighFreq

适用场景：
- 输入普通画质图片，输出大师级光影、色调和纹理
- 图生图风格迁移训练
- 保持原图几何结构的同时学习风格
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB 转 Lab 色彩空间
    
    Args:
        rgb: (B, 3, H, W) 范围 [0, 1]
    Returns:
        lab: (B, 3, H, W) L:[0,100], a,b:[-128,127]
    """
    # 确保输入在 [0, 1] 范围
    rgb = rgb.clamp(0, 1)
    
    # RGB to XYZ
    # sRGB 到线性 RGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # 变换矩阵 (sRGB D65)
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # (B, 3, H, W) -> (B, H, W, 3) -> matmul -> (B, H, W, 3) -> (B, 3, H, W)
    rgb_flat = rgb_linear.permute(0, 2, 3, 1)  # (B, H, W, 3)
    xyz = torch.matmul(rgb_flat, matrix.T)  # (B, H, W, 3)
    xyz = xyz.permute(0, 3, 1, 2)  # (B, 3, H, W)
    
    # XYZ to Lab
    # D65 白点
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz = xyz / xyz_ref.view(1, 3, 1, 1)
    
    # f(t) 函数
    delta = 6.0 / 29.0
    mask = xyz > delta ** 3
    xyz_f = torch.where(mask, xyz ** (1/3), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    
    # Lab
    L = 116.0 * xyz_f[:, 1:2] - 16.0
    a = 500.0 * (xyz_f[:, 0:1] - xyz_f[:, 1:2])
    b = 200.0 * (xyz_f[:, 1:2] - xyz_f[:, 2:3])
    
    lab = torch.cat([L, a, b], dim=1)
    return lab


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    计算 SSIM (Structural Similarity Index)
    
    Args:
        x, y: (B, C, H, W) 输入图像
        window_size: 滑窗大小
        size_average: 是否取平均
        data_range: 数据范围
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 创建高斯窗口
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.outer(g)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    
    channels = x.shape[1]
    window = window.expand(channels, 1, window_size, window_size)
    
    # 计算均值
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    # 计算方差和协方差
    sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy
    
    # SSIM
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    return ssim_map


class StyleStructureLoss(nn.Module):
    """
    结构锁风格迁移损失函数
    
    设计理念：
    1. 结构锁定：用 SSIM 锁死 L 通道轮廓，防止脸崩
    2. 光影学习：用 L 通道统计量（均值+标准差）学习大师的 S 曲线
    3. 色调迁移：用 ab 通道统计量学习色彩偏好
    4. 质感增强：用高频 L1 增强清晰度
    """
    
    def __init__(
        self,
        lambda_struct: float = 1.0,      # 结构锁权重 (SSIM)
        lambda_light: float = 0.5,       # 光影学习权重 (L统计)
        lambda_color: float = 0.3,       # 色调迁移权重 (ab统计)
        lambda_tex: float = 0.5,         # 质感增强权重 (高频L1)
        lambda_base: float = 1.0,        # 基础 v-prediction loss
        blur_kernel_size: int = 7,       # 高频提取的模糊核大小
        ssim_window_size: int = 11,      # SSIM 窗口大小
    ):
        super().__init__()
        self.lambda_struct = lambda_struct
        self.lambda_light = lambda_light
        self.lambda_color = lambda_color
        self.lambda_tex = lambda_tex
        self.lambda_base = lambda_base
        self.blur_kernel_size = blur_kernel_size
        self.ssim_window_size = ssim_window_size
        
        logger.info(f"[StyleStructureLoss] 初始化结构锁风格迁移损失")
        logger.info(f"  结构锁 (SSIM): {lambda_struct}")
        logger.info(f"  光影学习 (L统计): {lambda_light}")
        logger.info(f"  色调迁移 (ab统计): {lambda_color}")
        logger.info(f"  质感增强 (高频L1): {lambda_tex}")
    
    def get_gaussian_kernel(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        """生成高斯模糊核"""
        coords = torch.arange(size) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel
    
    def gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """应用高斯模糊"""
        kernel = self.get_gaussian_kernel(self.blur_kernel_size).to(x.device, x.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        channels = x.shape[1]
        kernel = kernel.expand(channels, 1, -1, -1)
        
        padding = self.blur_kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=channels)
        return blurred
    
    def get_high_freq(self, x: torch.Tensor) -> torch.Tensor:
        """提取高频分量"""
        low_freq = self.gaussian_blur(x)
        high_freq = x - low_freq
        return high_freq
    
    def compute_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算均值和标准差（全局统计量）"""
        # (B, C, H, W) -> 对 H, W 维度计算统计量
        mean = x.mean(dim=[2, 3])  # (B, C)
        std = x.std(dim=[2, 3])    # (B, C)
        return mean, std
    
    def moments_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """统计量匹配损失"""
        pred_mean, pred_std = self.compute_moments(pred)
        target_mean, target_std = self.compute_moments(target)
        
        loss_mean = F.l1_loss(pred_mean, target_mean)
        loss_std = F.l1_loss(pred_std, target_std)
        
        return loss_mean + loss_std
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算结构锁风格迁移损失
        
        Args:
            pred_v: 模型预测的速度 v
            target_v: 目标速度 v
            pred_x0: 预测的干净 latent (需要 VAE decode 或近似到 RGB)
            target_x0: 目标干净 latent
            return_components: 是否返回各分量
            
        Note:
            pred_x0/target_x0 应该是 RGB 图像 (B, 3, H, W)，范围 [0, 1]
            如果输入是 latent，需要先通过 VAE decode 或近似转换
        """
        # 基础 v-prediction loss
        loss_base = F.mse_loss(pred_v, target_v)
        
        # 转换到 Lab 空间
        pred_lab = rgb_to_lab(pred_x0)
        target_lab = rgb_to_lab(target_x0)
        
        # 分离 L 和 ab 通道
        pred_L = pred_lab[:, 0:1]      # (B, 1, H, W)
        target_L = target_lab[:, 0:1]
        pred_ab = pred_lab[:, 1:3]     # (B, 2, H, W)
        target_ab = target_lab[:, 1:3]
        
        # 归一化 L 通道用于 SSIM (0-100 -> 0-1)
        pred_L_norm = pred_L / 100.0
        target_L_norm = target_L / 100.0
        
        # 1. 结构锁 (SSIM on L channel)
        ssim_val = ssim(pred_L_norm, target_L_norm, window_size=self.ssim_window_size)
        loss_struct = 1.0 - ssim_val  # SSIM 越大越好，所以取 1-SSIM 作为 loss
        
        # 2. 光影学习 (L channel moments)
        loss_light = self.moments_loss(pred_L, target_L)
        
        # 3. 色调迁移 (ab channel moments)
        loss_color = self.moments_loss(pred_ab, target_ab)
        
        # 4. 质感增强 (L channel high frequency)
        pred_high = self.get_high_freq(pred_L)
        target_high = self.get_high_freq(target_L)
        loss_tex = F.l1_loss(pred_high, target_high)
        
        # 总损失
        total_loss = (
            self.lambda_base * loss_base +
            self.lambda_struct * loss_struct +
            self.lambda_light * loss_light +
            self.lambda_color * loss_color +
            self.lambda_tex * loss_tex
        )
        
        if return_components:
            components = {
                "loss_base": loss_base,
                "loss_struct": loss_struct,
                "loss_light": loss_light,
                "loss_color": loss_color,
                "loss_tex": loss_tex,
                "ssim": ssim_val,
                "total_loss": total_loss,
            }
            return total_loss, components
        
        return total_loss


class LatentStyleStructureLoss(StyleStructureLoss):
    """
    Latent 空间的风格结构损失
    
    在 Latent 空间近似计算，避免 VAE decode 的显存开销
    
    近似策略：
    - Latent 的 4 个通道近似对应不同的语义信息
    - Channel 0 通常与亮度相关
    - 使用降采样-上采样提取低频/高频
    """
    
    def __init__(
        self,
        lambda_struct: float = 1.0,
        lambda_light: float = 0.5,
        lambda_color: float = 0.3,
        lambda_tex: float = 0.5,
        lambda_base: float = 1.0,
        downsample_factor: int = 4,
    ):
        super().__init__(
            lambda_struct=lambda_struct,
            lambda_light=lambda_light,
            lambda_color=lambda_color,
            lambda_tex=lambda_tex,
            lambda_base=lambda_base,
        )
        self.downsample_factor = downsample_factor
        logger.info(f"[LatentStyleStructureLoss] 使用 Latent 空间近似计算")
    
    def get_low_freq_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Latent 空间低频提取"""
        h, w = x.shape[-2:]
        target_h = max(1, h // self.downsample_factor)
        target_w = max(1, w // self.downsample_factor)
        
        x_small = F.adaptive_avg_pool2d(x, (target_h, target_w))
        x_low = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=False)
        return x_low
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        在 Latent 空间计算风格统计损失 (v2 协作架构)
        
        职责分工：
        - loss_light: 光影统计量 (L 通道 moments)
        - loss_color: 色调统计量 (ab 通道 moments)
        
        注意：不再包含 loss_base/loss_struct/loss_tex，由主 L1 和 FreqLoss 负责
        
        Args:
            pred_v: 模型预测的速度 v (B, C, H, W)
            target_v: 目标速度 v
            noisy_latents: 加噪后的 latents x_t
            timesteps: 时间步
        """
        # 保存原始 dtype，确保所有计算结果都转换回来
        original_dtype = pred_v.dtype
        
        # 转换为 float32 进行计算（避免混合精度问题）
        pred_v_fp32 = pred_v.float()
        target_v_fp32 = target_v.float()
        noisy_latents_fp32 = noisy_latents.float()
        
        # 计算 sigma
        sigmas = timesteps.float() / num_train_timesteps
        sigma_broadcast = sigmas.view(-1, 1, 1, 1)
        
        # 反推 x0
        pred_x0 = noisy_latents_fp32 - sigma_broadcast * pred_v_fp32
        target_x0 = noisy_latents_fp32 - sigma_broadcast * target_v_fp32
        
        # 在 Latent 空间近似计算
        # Channel 0 近似亮度 (L)，Channel 1-3 近似色彩 (ab)
        pred_L = pred_x0[:, 0:1]
        target_L = target_x0[:, 0:1]
        pred_color = pred_x0[:, 1:4] if pred_x0.shape[1] > 1 else pred_x0
        target_color = target_x0[:, 1:4] if target_x0.shape[1] > 1 else target_x0
        
        # 1. 光影学习 (Channel 0 统计量)
        loss_light = self.moments_loss(pred_L, target_L)
        
        # 2. 色调迁移 (Channel 1-3 统计量)
        loss_color = self.moments_loss(pred_color, target_color)
        
        # 总损失 (纯统计量匹配，无基础项和频域项)
        total_loss = self.lambda_light * loss_light + self.lambda_color * loss_color
        
        # 转换回原始 dtype
        total_loss = total_loss.to(original_dtype)
        
        if return_components:
            components = {
                "loss_light": loss_light.to(original_dtype),
                "loss_color": loss_color.to(original_dtype),
                "total_loss": total_loss,
            }
            return total_loss, components
        
        return total_loss

