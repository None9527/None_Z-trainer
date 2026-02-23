# -*- coding: utf-8 -*-
"""
🔌 Model Hooks - 模型优化挂载模块

通过 PyTorch Hook 机制为 diffusers 模型添加优化功能，
无需修改模型源码即可实现：
- Block Swapping (显存优化)
- Attention Backend 切换
- Gradient Checkpointing 增强

Usage:
    from shared.utils.model_hooks import apply_block_swapper, apply_attention_optimization
    
    transformer = SomeTransformer.from_pretrained(...)
    
    # 挂载 Block Swapper
    apply_block_swapper(transformer, blocks_to_swap=8, device="cuda")
    
    # 设置 Attention Backend
    apply_attention_optimization(transformer, backend="flash")
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Any
import logging
import gc

logger = logging.getLogger(__name__)


class BlockSwapperHook:
    """
    通过 Hook 实现的 Block Swapper
    
    无需修改模型源码，通过 register_forward_pre_hook 和 register_forward_hook
    在每层前向传播前/后自动进行 GPU/CPU 交换。
    """
    
    def __init__(
        self,
        blocks_to_swap: int,
        device: torch.device,
        verbose: bool = True,
    ):
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.verbose = verbose
        
        self.layers: Optional[nn.ModuleList] = None
        self.n_layers: int = 0
        self.swap_start_idx: int = 0
        self.layer_on_gpu: List[bool] = []
        self.hooks: List[Any] = []
        
        # 统计
        self.swap_in_count = 0
        self.swap_out_count = 0
    
    def setup(self, layers: nn.ModuleList) -> "BlockSwapperHook":
        """
        设置要管理的层并注册 Hook
        
        Args:
            layers: Transformer 的 layers (nn.ModuleList)
            
        Returns:
            self (支持链式调用)
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.swap_start_idx = max(0, self.n_layers - self.blocks_to_swap)
        
        self.layer_on_gpu = [True] * self.n_layers
        
        if self.blocks_to_swap <= 0:
            if self.verbose:
                logger.info("[BlockSwapHook] 禁用 (blocks_to_swap=0)")
            return self
        
        # 将后 N 层移到 CPU
        layers_moved = 0
        total_params = 0
        
        for i in range(self.swap_start_idx, self.n_layers):
            layer = self.layers[i]
            layer_params = sum(p.numel() for p in layer.parameters())
            total_params += layer_params
            layer.to(self.cpu_device)
            self.layer_on_gpu[i] = False
            layers_moved += 1
        
        # 注册 Hook
        for i, layer in enumerate(self.layers):
            # pre_hook: 将层移到 GPU
            pre_hook = layer.register_forward_pre_hook(
                self._make_pre_hook(i)
            )
            # post_hook: 将层移回 CPU (如果在交换范围内)
            post_hook = layer.register_forward_hook(
                self._make_post_hook(i)
            )
            self.hooks.extend([pre_hook, post_hook])
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if self.verbose:
            param_mb = total_params * 2 / (1024 * 1024)  # BF16/FP16
            logger.info(f"[BlockSwapHook] 已将 {layers_moved} 层移到 CPU")
            logger.info(f"[BlockSwapHook] 交换范围: layer[{self.swap_start_idx}] ~ layer[{self.n_layers-1}]")
            logger.info(f"[BlockSwapHook] 预计节省显存: ~{param_mb:.1f} MB")
            logger.info(f"[BlockSwapHook] 已注册 {len(self.hooks)} 个 Hook")
        
        return self
    
    def _make_pre_hook(self, layer_idx: int) -> Callable:
        """创建 pre_hook (swap_in)"""
        def hook(module, inputs):
            if self.blocks_to_swap <= 0:
                return
            if not self.layer_on_gpu[layer_idx]:
                module.to(self.device)
                self.layer_on_gpu[layer_idx] = True
                self.swap_in_count += 1
        return hook
    
    def _make_post_hook(self, layer_idx: int) -> Callable:
        """创建 post_hook (swap_out)"""
        def hook(module, inputs, outputs):
            if self.blocks_to_swap <= 0:
                return
            if layer_idx >= self.swap_start_idx and self.layer_on_gpu[layer_idx]:
                module.to(self.cpu_device)
                self.layer_on_gpu[layer_idx] = False
                self.swap_out_count += 1
        return hook
    
    def remove_hooks(self):
        """移除所有 Hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("[BlockSwapHook] 已移除所有 Hook")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "blocks_to_swap": self.blocks_to_swap,
            "n_layers": self.n_layers,
            "swap_start_idx": self.swap_start_idx,
            "swap_in_count": self.swap_in_count,
            "swap_out_count": self.swap_out_count,
            "layers_on_gpu": sum(self.layer_on_gpu),
            "layers_on_cpu": self.n_layers - sum(self.layer_on_gpu),
        }


def apply_block_swapper(
    transformer: nn.Module,
    blocks_to_swap: int,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Optional[BlockSwapperHook]:
    """
    为 transformer 模型挂载 Block Swapper (通过 Hook)
    
    Args:
        transformer: ZImageTransformer2DModel 或其他有 .layers 属性的模型
        blocks_to_swap: 要交换到 CPU 的层数，0 表示禁用
        device: GPU 设备，None 时自动检测
        verbose: 是否打印日志
        
    Returns:
        BlockSwapperHook 实例，如果禁用则返回 None
        
    Example:
        >>> from diffusers import ZImageTransformer2DModel
        >>> transformer = ZImageTransformer2DModel.from_pretrained(...)
        >>> swapper = apply_block_swapper(transformer, blocks_to_swap=8)
        >>> # 训练完成后移除
        >>> swapper.remove_hooks()
    """
    if blocks_to_swap <= 0:
        logger.info("[BlockSwapper] 禁用 (blocks_to_swap=0)")
        return None
    
    # 自动检测设备
    if device is None:
        device = next(transformer.parameters()).device
    
    # 查找 layers
    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "transformer_blocks"):
        layers = transformer.transformer_blocks
    else:
        logger.warning("[BlockSwapper] 未找到 .layers 或 .transformer_blocks，跳过")
        return None
    
    # 创建并设置 Hook
    swapper = BlockSwapperHook(
        blocks_to_swap=blocks_to_swap,
        device=device,
        verbose=verbose,
    )
    swapper.setup(layers)
    
    return swapper


def apply_attention_optimization(
    transformer: nn.Module,
    backend: str = "flash",
) -> bool:
    """
    设置 transformer 的 attention backend
    
    diffusers 内置支持的 backend:
    - "flash": Flash Attention 2
    - "_flash_3": Flash Attention 3 (如果可用)
    - "xformers": xformers memory_efficient_attention
    - "native": PyTorch native SDPA (formerly "sdpa")
    - None: 默认后端
    
    Args:
        transformer: ZImageTransformer2DModel 等
        backend: 后端名称
        
    Returns:
        是否成功设置
        
    Example:
        >>> apply_attention_optimization(transformer, backend="flash")
    """
    # 方法1: diffusers 内置 API
    if hasattr(transformer, "set_attention_backend"):
        try:
            transformer.set_attention_backend(backend)
            logger.info(f"[Attention] 已设置后端: {backend}")
            return True
        except Exception as e:
            logger.warning(f"[Attention] set_attention_backend 失败: {e}")
    
    # 方法2: 设置类变量 _attention_backend
    set_count = 0
    for name, module in transformer.named_modules():
        if hasattr(module, "_attention_backend"):
            module._attention_backend = backend
            set_count += 1
    
    if set_count > 0:
        logger.info(f"[Attention] 已为 {set_count} 个模块设置后端: {backend}")
        return True
    
    # 方法3: enable_xformers_memory_efficient_attention (旧 API)
    if backend == "xformers" and hasattr(transformer, "enable_xformers_memory_efficient_attention"):
        try:
            transformer.enable_xformers_memory_efficient_attention()
            logger.info("[Attention] 已启用 xformers")
            return True
        except Exception as e:
            logger.warning(f"[Attention] xformers 启用失败: {e}")
    
    logger.warning(f"[Attention] 模型不支持 attention backend 设置")
    return False


def enable_gradient_checkpointing(
    transformer: nn.Module,
    use_reentrant: bool = False,
) -> bool:
    """
    启用 gradient checkpointing
    
    Args:
        transformer: 模型
        use_reentrant: 是否使用 reentrant 模式 (建议 False)
        
    Returns:
        是否成功启用
    """
    # diffusers 模型内置方法
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        logger.info("[GradCkpt] 已启用 gradient checkpointing")
        return True
    
    # 通用方法
    if hasattr(transformer, "gradient_checkpointing"):
        transformer.gradient_checkpointing = True
        logger.info("[GradCkpt] 已设置 gradient_checkpointing = True")
        return True
    
    logger.warning("[GradCkpt] 模型不支持 gradient checkpointing")
    return False


def apply_all_optimizations(
    transformer: nn.Module,
    blocks_to_swap: int = 0,
    attention_backend: str = "flash",
    gradient_checkpointing: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """
    一键应用所有优化
    
    Args:
        transformer: 模型
        blocks_to_swap: Block Swap 层数 (0 禁用)
        attention_backend: Attention 后端
        gradient_checkpointing: 是否启用 gradient checkpointing
        device: GPU 设备
        verbose: 是否打印日志
        
    Returns:
        包含各优化结果的字典
        
    Example:
        >>> from diffusers import ZImageTransformer2DModel
        >>> transformer = ZImageTransformer2DModel.from_pretrained(...)
        >>> results = apply_all_optimizations(
        ...     transformer,
        ...     blocks_to_swap=8,
        ...     attention_backend="flash",
        ...     gradient_checkpointing=True,
        ... )
    """
    results = {
        "block_swapper": None,
        "attention_backend": False,
        "gradient_checkpointing": False,
    }
    
    # 1. Gradient Checkpointing (必须在其他优化前启用)
    if gradient_checkpointing:
        results["gradient_checkpointing"] = enable_gradient_checkpointing(transformer)
    
    # 2. Attention Backend
    if attention_backend:
        results["attention_backend"] = apply_attention_optimization(
            transformer, backend=attention_backend
        )
    
    # 3. Block Swapper
    if blocks_to_swap > 0:
        results["block_swapper"] = apply_block_swapper(
            transformer,
            blocks_to_swap=blocks_to_swap,
            device=device,
            verbose=verbose,
        )
    
    if verbose:
        logger.info("=" * 50)
        logger.info("[Optimizations] 应用结果:")
        logger.info(f"  - Gradient Checkpointing: {'✓' if results['gradient_checkpointing'] else '✗'}")
        logger.info(f"  - Attention Backend: {'✓' if results['attention_backend'] else '✗'}")
        logger.info(f"  - Block Swapper: {'✓' if results['block_swapper'] else '✗'}")
        logger.info("=" * 50)
    
    return results
