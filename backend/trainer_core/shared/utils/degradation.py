# -*- coding: utf-8 -*-
"""
🔧 图像退化模块 (Image Degradation)

用于图生图风格迁移训练的自监督退化策略

退化流程：
1. Downsample: 随机缩小 (0.5x ~ 0.8x)
2. Blur: 随机高斯模糊 (Kernel 3~7, Sigma 0.5~2.0)
3. Noise: 叠加高斯白噪 (强度 0.02~0.05)
4. Upsample: 双线性插值回原尺寸
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random


class ImageDegradation:
    """
    图像退化处理器
    
    将高清图像退化为低质量版本，用于图生图训练
    """
    
    def __init__(
        self,
        downsample_range: Tuple[float, float] = (0.5, 0.8),
        blur_kernel_range: Tuple[int, int] = (3, 7),
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        noise_range: Tuple[float, float] = (0.02, 0.05),
        jpeg_quality_range: Optional[Tuple[int, int]] = None,  # 可选 JPEG 压缩
        enable_random: bool = True,  # 随机化各项参数
    ):
        self.downsample_range = downsample_range
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        self.noise_range = noise_range
        self.jpeg_quality_range = jpeg_quality_range
        self.enable_random = enable_random
    
    def get_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成高斯模糊核"""
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel
    
    def gaussian_blur(
        self,
        image: torch.Tensor,
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        """应用高斯模糊"""
        # 确保 kernel_size 是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = self.get_gaussian_kernel(kernel_size, sigma, image.device, image.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        
        # 对每个通道分别模糊
        channels = image.shape[1]
        kernel = kernel.expand(channels, 1, -1, -1)
        
        padding = kernel_size // 2
        blurred = F.conv2d(image, kernel, padding=padding, groups=channels)
        return blurred
    
    def add_gaussian_noise(
        self,
        image: torch.Tensor,
        noise_level: float,
    ) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
        return noisy.clamp(0, 1)
    
    def downsample_upsample(
        self,
        image: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """降采样再升采样（模拟分辨率损失）"""
        _, _, h, w = image.shape
        
        # 降采样
        new_h = int(h * scale)
        new_w = int(w * scale)
        downsampled = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # 升采样回原尺寸
        upsampled = F.interpolate(downsampled, size=(h, w), mode='bilinear', align_corners=False)
        return upsampled
    
    def __call__(
        self,
        image: torch.Tensor,
        downsample_scale: Optional[float] = None,
        blur_kernel_size: Optional[int] = None,
        blur_sigma: Optional[float] = None,
        noise_level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        应用退化处理
        
        Args:
            image: (B, C, H, W) 或 (C, H, W)，范围 [0, 1]
            downsample_scale: 降采样比例，None 时随机
            blur_kernel_size: 模糊核大小，None 时随机
            blur_sigma: 模糊 sigma，None 时随机
            noise_level: 噪声强度，None 时随机
            
        Returns:
            degraded: 退化后的图像
        """
        # 处理维度
        squeeze = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        
        # 随机化参数
        if self.enable_random:
            if downsample_scale is None:
                downsample_scale = random.uniform(*self.downsample_range)
            if blur_kernel_size is None:
                blur_kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
            if blur_sigma is None:
                blur_sigma = random.uniform(*self.blur_sigma_range)
            if noise_level is None:
                noise_level = random.uniform(*self.noise_range)
        else:
            # 使用固定中值
            if downsample_scale is None:
                downsample_scale = sum(self.downsample_range) / 2
            if blur_kernel_size is None:
                blur_kernel_size = self.blur_kernel_range[0] + (self.blur_kernel_range[1] - self.blur_kernel_range[0]) // 2
                if blur_kernel_size % 2 == 0:
                    blur_kernel_size += 1
            if blur_sigma is None:
                blur_sigma = sum(self.blur_sigma_range) / 2
            if noise_level is None:
                noise_level = sum(self.noise_range) / 2
        
        # 1. 降采样-升采样（模拟分辨率损失）
        degraded = self.downsample_upsample(image, downsample_scale)
        
        # 2. 高斯模糊
        degraded = self.gaussian_blur(degraded, blur_kernel_size, blur_sigma)
        
        # 3. 添加噪声
        degraded = self.add_gaussian_noise(degraded, noise_level)
        
        if squeeze:
            degraded = degraded.squeeze(0)
        
        return degraded


class BatchDegradation:
    """
    批量退化处理器
    
    为 batch 中的每张图片应用不同的随机退化参数
    """
    
    def __init__(self, **kwargs):
        self.degradation = ImageDegradation(**kwargs)
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) 范围 [0, 1]
        Returns:
            degraded: (B, C, H, W) 退化后的图像
        """
        batch_size = images.shape[0]
        degraded_list = []
        
        for i in range(batch_size):
            # 每张图使用不同的随机退化参数
            degraded = self.degradation(images[i:i+1])
            degraded_list.append(degraded)
        
        return torch.cat(degraded_list, dim=0)


def create_degradation_transform(
    strength: str = "medium",
) -> ImageDegradation:
    """
    创建预设的退化变换
    
    Args:
        strength: "light" / "medium" / "heavy"
    """
    presets = {
        "light": {
            "downsample_range": (0.7, 0.9),
            "blur_kernel_range": (3, 5),
            "blur_sigma_range": (0.3, 1.0),
            "noise_range": (0.01, 0.03),
        },
        "medium": {
            "downsample_range": (0.5, 0.8),
            "blur_kernel_range": (3, 7),
            "blur_sigma_range": (0.5, 2.0),
            "noise_range": (0.02, 0.05),
        },
        "heavy": {
            "downsample_range": (0.3, 0.6),
            "blur_kernel_range": (5, 11),
            "blur_sigma_range": (1.0, 3.0),
            "noise_range": (0.04, 0.08),
        },
    }
    
    if strength not in presets:
        raise ValueError(f"Unknown strength: {strength}. Choose from {list(presets.keys())}")
    
    return ImageDegradation(**presets[strength])


