"""
💾 内存优化工具模块
提供块交换技术、激活检查点和内存管理功能

主要功能：
1. 动态内存块交换 (Block Swapping)
2. 激活值检查点管理 (Activation Checkpointing)
3. CPU-GPU内存交换 (CPU-GPU Memory Swapping)
4. 内存监控和优化 (Memory Monitoring)
"""

import torch
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """内存块数据结构"""
    tensor_id: int
    tensor: torch.Tensor
    priority: float
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, strategy: str = "conservative"):
        self.strategy = strategy
        self.pools = {
            "gpu": {},
            "cpu": {}
        }
        self.total_allocated = 0
        self.max_pool_size = 0
        
        # 根据策略设置池大小
        if strategy == "conservative":
            self.max_pool_size = 1024 * 1024 * 1024  # 1GB
        elif strategy == "aggressive":
            self.max_pool_size = 4 * 1024 * 1024 * 1024  # 4GB
        
    def allocate(self, size_bytes: int, device: str = "gpu") -> bool:
        """分配内存池空间"""
        if self.total_allocated + size_bytes <= self.max_pool_size:
            self.total_allocated += size_bytes
            return True
        return False
    
    def deallocate(self, size_bytes: int):
        """释放内存池空间"""
        self.total_allocated = max(0, self.total_allocated - size_bytes)


class BlockSwapManager:
    """块交换管理器"""
    
    def __init__(self, 
                 block_size: int = 512,
                 cpu_buffer_size_gb: float = 8.0,
                 swap_threshold: float = 0.7,
                 swap_frequency: int = 0,
                 smart_prefetch: bool = True,
                 swap_strategy: str = "priority",
                 compressed_swap: bool = False):
        
        self.block_size = block_size
        self.cpu_buffer_size = int(cpu_buffer_size_gb * 1024 * 1024 * 1024)  # 转换为字节
        self.swap_threshold = swap_threshold
        self.swap_frequency = swap_frequency
        self.smart_prefetch = smart_prefetch
        self.swap_strategy = swap_strategy
        self.compressed_swap = compressed_swap
        
        # 内存块管理
        self.gpu_blocks: Dict[int, MemoryBlock] = {}
        self.cpu_blocks: Dict[int, MemoryBlock] = {}
        self.block_counter = 0
        
        # 访问统计
        self.access_patterns = defaultdict(deque)
        self.swap_history = deque(maxlen=1000)
        
        # 内存池
        self.memory_pool = MemoryPool()
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        
        # 预取预测器
        if self.smart_prefetch:
            self.prefetch_queue = deque()
            self.prediction_model = self._init_prediction_model()
    
    def _init_prediction_model(self):
        """初始化简单的预取预测模型"""
        # 这里可以集成更复杂的预取算法
        # 目前使用基于访问频率的简单预测
        return {
            "hot_blocks": set(),
            "access_counts": defaultdict(int),
            "last_predictions": deque(maxlen=50)
        }
    
    def start_monitoring(self):
        """启动内存监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🔍 块交换内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("🛑 块交换内存监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._check_memory_usage()
                time.sleep(0.1)  # 100ms检查一次
            except Exception as e:
                logger.warning(f"内存监控错误: {e}")
    
    def _check_memory_usage(self):
        """检查GPU内存使用率并执行交换"""
        if not torch.cuda.is_available():
            return
        
        # 获取GPU内存使用率
        memory_info = torch.cuda.memory_stats(0)
        allocated = memory_info.get('allocated_bytes.all.current', 0)
        reserved = memory_info.get('reserved_bytes.all.current', 0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        # 计算使用率
        usage_ratio = max(allocated, reserved) / total
        
        if usage_ratio > self.swap_threshold:
            logger.debug(f"💾 GPU内存使用率 {usage_ratio:.2%} 超过阈值 {self.swap_threshold:.2%}，执行交换")
            self._perform_swap(usage_ratio - self.swap_threshold)
    
    def _perform_swap(self, excess_ratio: float):
        """执行内存交换"""
        # 确定需要交换的内存块数量
        target_blocks = int(len(self.gpu_blocks) * excess_ratio / self.swap_threshold)
        target_blocks = max(1, min(target_blocks, len(self.gpu_blocks) // 2))
        
        # 根据策略选择要交换的块
        blocks_to_swap = self._select_blocks_for_swap(target_blocks)
        
        for block_id in blocks_to_swap:
            if block_id in self.gpu_blocks:
                self._swap_out_block(block_id)
    
    def _select_blocks_for_swap(self, count: int) -> List[int]:
        """根据策略选择要交换的内存块"""
        if not self.gpu_blocks:
            return []
        
        if self.swap_strategy == "fifo":
            # 先进先出策略
            sorted_blocks = sorted(self.gpu_blocks.keys(), 
                                 key=lambda x: self.gpu_blocks[x].created_at)
        
        elif self.swap_strategy == "lru":
            # 最近最少使用策略
            sorted_blocks = sorted(self.gpu_blocks.keys(), 
                                 key=lambda x: self.gpu_blocks[x].last_accessed)
        
        elif self.swap_strategy == "priority":
            # 基于优先级的策略 (保留重要块)
            sorted_blocks = sorted(self.gpu_blocks.keys(), 
                                 key=lambda x: self.gpu_blocks[x].priority)
        
        else:
            sorted_blocks = list(self.gpu_blocks.keys())
        
        return sorted_blocks[:count]
    
    def _swap_out_block(self, block_id: int):
        """交换出内存块到CPU"""
        if block_id not in self.gpu_blocks:
            return
        
        block = self.gpu_blocks[block_id]
        
        # 检查CPU缓冲区空间
        current_cpu_usage = sum(b.size_bytes for b in self.cpu_blocks.values())
        if current_cpu_usage + block.size_bytes > self.cpu_buffer_size:
            # CPU缓冲区满，清理最旧的块
            self._evict_cpu_blocks()
        
        # 移动到CPU
        cpu_tensor = block.tensor.cpu().detach()
        self.cpu_blocks[block_id] = MemoryBlock(
            tensor_id=block.tensor_id,
            tensor=cpu_tensor,
            priority=block.priority,
            size_bytes=block.size_bytes,
            created_at=block.created_at,
            last_accessed=block.last_accessed,
            access_count=block.access_count
        )
        
        # 从GPU删除
        del self.gpu_blocks[block_id].tensor
        del self.gpu_blocks[block_id]
        
        # 记录交换历史
        self.swap_history.append({
            "block_id": block_id,
            "direction": "out",
            "timestamp": time.time(),
            "size": block.size_bytes
        })
        
        # 触发垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _swap_in_block(self, block_id: int):
        """交换入内存块到GPU"""
        if block_id not in self.cpu_blocks:
            return
        
        block = self.cpu_blocks[block_id]
        
        # 检查GPU空间 (使用配置的阈值)
        if torch.cuda.is_available():
            # 允许使用到 swap_threshold 的上限
            if torch.cuda.memory_reserved(0) > torch.cuda.get_device_properties(0).total_memory * self.swap_threshold:
                logger.warning(f"⚠️ GPU内存不足 (>{self.swap_threshold:.1%})，无法交换入块")
                return
        
        # 移动到GPU
        gpu_tensor = block.tensor.cuda().detach()
        self.gpu_blocks[block_id] = MemoryBlock(
            tensor_id=block.tensor_id,
            tensor=gpu_tensor,
            priority=block.priority,
            size_bytes=block.size_bytes,
            created_at=block.created_at,
            last_accessed=block.last_accessed,
            access_count=block.access_count
        )
        
        # 从CPU删除
        del self.cpu_blocks[block_id]
        
        # 记录交换历史
        self.swap_history.append({
            "block_id": block_id,
            "direction": "in",
            "timestamp": time.time(),
            "size": block.size_bytes
        })
    
    def _evict_cpu_blocks(self):
        """清理CPU块，腾出空间"""
        if not self.cpu_blocks:
            return
        
        # 按优先级清理CPU块
        sorted_blocks = sorted(self.cpu_blocks.keys(), 
                             key=lambda x: self.cpu_blocks[x].priority)
        
        # 清理10%或最少1个块
        evict_count = max(1, len(sorted_blocks) // 10)
        for block_id in sorted_blocks[:evict_count]:
            del self.cpu_blocks[block_id]
    
    def register_tensor(self, tensor: torch.Tensor, priority: float = 1.0) -> int:
        """注册张量到块管理器"""
        block_id = self.block_counter
        self.block_counter += 1
        
        # 计算块大小
        size_bytes = tensor.numel() * tensor.element_size()
        
        # 创建内存块
        memory_block = MemoryBlock(
            tensor_id=block_id,
            tensor=tensor,
            priority=priority,
            size_bytes=size_bytes,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        self.gpu_blocks[block_id] = memory_block
        
        logger.debug(f"📝 注册张量块 {block_id}，大小: {size_bytes / 1024 / 1024:.2f}MB")
        return block_id
    
    def update_tensor_access(self, block_id: int):
        """更新张量访问信息"""
        if block_id in self.gpu_blocks:
            self.gpu_blocks[block_id].update_access()
        elif block_id in self.cpu_blocks:
            self.cpu_blocks[block_id].update_access()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        stats = {
            "gpu_blocks": len(self.gpu_blocks),
            "cpu_blocks": len(self.cpu_blocks),
            "total_gpu_memory": sum(b.size_bytes for b in self.gpu_blocks.values()),
            "total_cpu_memory": sum(b.size_bytes for b in self.cpu_blocks.values()),
            "swap_operations": len(self.swap_history),
            "memory_pool_usage": self.memory_pool.total_allocated
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_usage"] = torch.cuda.memory_allocated(0)
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
            stats["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        
        return stats


class ActivationCheckpointManager:
    """激活检查点管理器"""
    
    def __init__(self, optimization_level: str = "basic"):
        self.optimization_level = optimization_level
        self.checkpoint_cache = {}
        self.computation_graph = {}
        
    @contextmanager
    def checkpoint_context(self, module_name: str):
        """检查点上下文管理器"""
        if self.optimization_level == "none":
            yield
            return
        
        start_time = time.time()
        
        try:
            if self.optimization_level == "aggressive":
                # 激进检查点：保存更多中间状态
                torch.cuda.empty_cache()
                gc.collect()
            
            yield
            
        finally:
            computation_time = time.time() - start_time
            logger.debug(f"🔄 模块 {module_name} 检查点计算耗时: {computation_time:.3f}s")
    
    def create_checkpoint(self, forward_fn: Callable, *args, **kwargs):
        """创建检查点"""
        if self.optimization_level == "none":
            return forward_fn(*args, **kwargs)
        
        if self.optimization_level == "basic":
            return torch.utils.checkpoint.checkpoint(forward_fn, *args, **kwargs)
        
        elif self.optimization_level == "aggressive":
            # 激进检查点：自定义重计算逻辑
            return torch.utils.checkpoint.checkpoint_sequential(
                forward_fn, 
                segments=2,  # 分段检查点
                *args, 
                **kwargs
            )
        
        return forward_fn(*args, **kwargs)


class MemoryOptimizer:
    """统一内存优化管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 获取 blocks_to_swap 参数
        self.blocks_to_swap = config.get('blocks_to_swap', 0)
        
        # 初始化各个组件
        self.block_swap = BlockSwapManager(
            block_size=config.get('memory_block_size', 512),
            cpu_buffer_size_gb=config.get('cpu_swap_buffer_size', 8.0),
            swap_threshold=config.get('swap_threshold', 0.7),
            swap_frequency=config.get('swap_frequency', 0),
            smart_prefetch=config.get('smart_prefetch', True),
            swap_strategy=config.get('swap_strategy', 'priority'),
            compressed_swap=config.get('compressed_swap', False)
        )
        
        self.checkpoint_manager = ActivationCheckpointManager(
            optimization_level=config.get('checkpoint_optimization', 'basic')
        )
        
        self.enabled = config.get('block_swap_enabled', True)
        
        # 性能统计
        self.stats = {
            "total_saves": 0,
            "total_swaps": 0,
            "avg_swap_time": 0.0,
            "memory_efficiency": 0.0
        }
    
    def start(self):
        """启动内存优化器"""
        if self.blocks_to_swap > 0:
            logger.info(f"💾 显存优化器已启动 (blocks_to_swap={self.blocks_to_swap})")
            # 根据 blocks_to_swap 设置更激进的清理阈值
            self.aggressive_cleanup = True
            self.cleanup_threshold = max(0.5, 0.8 - self.blocks_to_swap * 0.03)  # 每个 block 降低 3%
            logger.info(f"  清理阈值: {self.cleanup_threshold:.1%}")
        elif self.enabled and self.config.get('block_swap_enabled', False):
            self.block_swap.start_monitoring()
            self.aggressive_cleanup = False
            logger.info("💾 块交换内存优化器已启动")
        else:
            self.aggressive_cleanup = False
            logger.info("💾 内存优化器已就绪（Block Swap 已禁用）")
    
    def stop(self):
        """停止内存优化器"""
        if self.enabled:
            self.block_swap.stop_monitoring()
            logger.info("🛑 块交换内存优化器已停止")
    
    def optimize_training_step(self):
        """训练步骤内存优化"""
        if not self.enabled:
            return
        
        # 根据显存大小调整清理阈值
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0)
            usage = allocated / total_memory
            
            # 如果启用了 blocks_to_swap，使用更激进的清理策略
            if getattr(self, 'aggressive_cleanup', False):
                threshold = self.cleanup_threshold
            else:
                # 16GB 及以下: 90% 时开始清理
                # 24GB+: 95% 时清理
                threshold = 0.90 if total_gb < 20 else 0.95
            
            if usage > threshold:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 只有启用了 block_swap 才收集统计
        if self.config.get('block_swap_enabled', False):
            stats = self.block_swap.get_memory_stats()
            self._update_performance_stats(stats)
    
    def _update_performance_stats(self, stats: Dict[str, Any]):
        """更新性能统计"""
        total_blocks = stats["gpu_blocks"] + stats["cpu_blocks"]
        if total_blocks > 0:
            self.stats["memory_efficiency"] = (
                stats["gpu_blocks"] / total_blocks
            ) * 100
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        block_stats = self.block_swap.get_memory_stats()
        
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        
        return {
            "block_swap": block_stats,
            "checkpoint": {
                "optimization_level": self.checkpoint_manager.optimization_level,
                "cached_checkpoints": len(self.checkpoint_manager.checkpoint_cache)
            },
            "system_memory": {
                "total": system_memory.total,
                "available": system_memory.available,
                "percent": system_memory.percent
            },
            "performance": self.stats
        }