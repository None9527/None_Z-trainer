# -*- coding: utf-8 -*-
"""
🔄 Forward Block Swapper - 前向传播块交换器

在模型前向传播时动态管理 Transformer 各层在 GPU/CPU 之间的移动，
以节省 GPU 显存。

原理：
- 初始化时将后 N 层 (blocks_to_swap) 移到 CPU
- 前向传播时，在每层计算前将其移到 GPU
- 计算完成后，如果该层在交换范围内，移回 CPU

注意：
- 会增加 CPU↔GPU 数据传输开销，训练速度会下降
- 需要确保 CPU 有足够内存容纳交换出的层
- 与 gradient checkpointing 兼容
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable
import logging
import gc

logger = logging.getLogger(__name__)


class ForwardBlockSwapper:
    """
    前向传播块交换器
    
    在模型前向传播时动态管理各层在 GPU/CPU 之间的移动。
    """
    
    def __init__(
        self,
        blocks_to_swap: int,
        device: torch.device,
        verbose: bool = True,
    ):
        """
        Args:
            blocks_to_swap: 要交换到 CPU 的层数
            device: GPU 设备
            verbose: 是否打印详细日志
        """
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.verbose = verbose
        
        # 层状态跟踪
        self.layers: Optional[nn.ModuleList] = None
        self.n_layers: int = 0
        self.swap_start_idx: int = 0  # 从这个索引开始的层需要交换
        self.layer_on_gpu: List[bool] = []
        
        # 统计
        self.swap_in_count = 0
        self.swap_out_count = 0
        
    def setup(self, layers: nn.ModuleList):
        """
        设置要管理的层并初始化交换
        
        Args:
            layers: Transformer 的层列表 (nn.ModuleList)
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.swap_start_idx = max(0, self.n_layers - self.blocks_to_swap)
        
        # 初始化状态：所有层当前都在 GPU
        self.layer_on_gpu = [True] * self.n_layers
        
        if self.blocks_to_swap <= 0:
            if self.verbose:
                logger.info("[BlockSwap] 禁用 (blocks_to_swap=0)")
            return
        
        # 将后 N 层移到 CPU
        layers_moved = 0
        total_params = 0
        
        for i in range(self.swap_start_idx, self.n_layers):
            layer = self.layers[i]
            # 计算参数量
            layer_params = sum(p.numel() for p in layer.parameters())
            total_params += layer_params
            
            # 移到 CPU
            layer.to(self.cpu_device)
            self.layer_on_gpu[i] = False
            layers_moved += 1
        
        # 强制清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if self.verbose:
            param_mb = total_params * 2 / (1024 * 1024)  # 假设 BF16/FP16
            logger.info(f"[BlockSwap] 已将 {layers_moved} 层移到 CPU")
            logger.info(f"[BlockSwap] 交换范围: layer[{self.swap_start_idx}] ~ layer[{self.n_layers-1}]")
            logger.info(f"[BlockSwap] 预计节省显存: ~{param_mb:.1f} MB")
    
    def swap_in(self, layer_idx: int):
        """
        将指定层移到 GPU (前向传播前调用)
        
        Args:
            layer_idx: 层索引
        """
        if self.blocks_to_swap <= 0:
            return
            
        if not self.layer_on_gpu[layer_idx]:
            self.layers[layer_idx].to(self.device)
            self.layer_on_gpu[layer_idx] = True
            self.swap_in_count += 1
    
    def swap_out(self, layer_idx: int):
        """
        将指定层移回 CPU (前向传播后调用)
        
        Args:
            layer_idx: 层索引
        """
        if self.blocks_to_swap <= 0:
            return
            
        # 只有在交换范围内的层才移回 CPU
        if layer_idx >= self.swap_start_idx and self.layer_on_gpu[layer_idx]:
            self.layers[layer_idx].to(self.cpu_device)
            self.layer_on_gpu[layer_idx] = False
            self.swap_out_count += 1
    
    def get_stats(self) -> dict:
        """获取交换统计信息"""
        return {
            "blocks_to_swap": self.blocks_to_swap,
            "n_layers": self.n_layers,
            "swap_start_idx": self.swap_start_idx,
            "swap_in_count": self.swap_in_count,
            "swap_out_count": self.swap_out_count,
            "layers_on_gpu": sum(self.layer_on_gpu),
            "layers_on_cpu": self.n_layers - sum(self.layer_on_gpu),
        }
    
    def print_stats(self):
        """打印交换统计"""
        stats = self.get_stats()
        logger.info(f"[BlockSwap Stats] GPU: {stats['layers_on_gpu']} layers, CPU: {stats['layers_on_cpu']} layers")
        logger.info(f"[BlockSwap Stats] Swap operations: in={stats['swap_in_count']}, out={stats['swap_out_count']}")


def create_block_swapper(
    blocks_to_swap: int,
    device: torch.device,
    verbose: bool = True,
) -> Optional[ForwardBlockSwapper]:
    """
    创建块交换器的工厂函数
    
    Args:
        blocks_to_swap: 要交换的层数，0 表示禁用
        device: GPU 设备
        verbose: 是否打印日志
        
    Returns:
        ForwardBlockSwapper 实例，如果 blocks_to_swap=0 返回 None
    """
    if blocks_to_swap <= 0:
        return None
    
    return ForwardBlockSwapper(
        blocks_to_swap=blocks_to_swap,
        device=device,
        verbose=verbose,
    )
