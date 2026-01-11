#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块

提供项目所有配置类的统一接口，避免配置分散和不一致
"""

from dataclasses import dataclass, field
from typing import Optional
import multiprocessing


# ============ 并行处理配置 ============

@dataclass
class ParallelConfig:
    """
    并行处理配置
    
    属性:
        max_workers: 最大工作线程数，0表示自动检测(CPU核心数-1)
        enable_parallel: 是否启用并行处理
        min_files_for_parallel: 启用并行的最小文件数，默认4
        chunk_size: 分块大小，默认1
    
    使用示例:
        >>> config = ParallelConfig(max_workers=4, min_files_for_parallel=2)
        >>> config.max_workers
        4
    """
    max_workers: int = 0  # 0 表示自动检测
    enable_parallel: bool = True
    min_files_for_parallel: int = 4
    chunk_size: int = 1
    
    def __post_init__(self):
        """初始化后自动计算最优workers数"""
        if self.max_workers == 0:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)


# ============ 进度显示配置 ============

@dataclass
class ProgressConfig:
    """进度显示配置"""
    show_progress: bool = True
    show_speed: bool = True
    show_eta: bool = True
    throttle_interval: float = 0.2  # 更新间隔(秒)
    width: int = 50  # 进度条宽度
    unit: str = "it"  # 单位
    unit_scale: bool = False
    unit_divisor: int = 1024


# ============ 优化配置 ============

@dataclass
class OptimizationConfig:
    """
    性能优化配置
    
    属性:
        quality_level: 质量级别 (1=fast, 2=balanced, 3=best)
        use_numba: 是否使用Numba加速
        use_simd: 是否使用SIMD向量化
        use_memory_pool: 是否使用内存池
        adaptive_optimization: 是否启用自适应优化
    
    质量级别说明:
        - 1 (fast): 快速模式，性能优先
        - 2 (balanced): 平衡模式，质量与性能兼顾
        - 3 (best): 最佳模式，质量优先
    """
    quality_level: int = 2  # 1=fast, 2=balanced, 3=best
    use_numba: bool = True
    use_simd: bool = True
    use_memory_pool: bool = True
    adaptive_optimization: bool = True  # 自适应优化
    
    def __post_init__(self):
        """验证配置"""
        if self.quality_level not in (1, 2, 3):
            raise ValueError("quality_level must be 1, 2, or 3")


# ============ GPU配置 ============

@dataclass
class GPUConfig:
    """GPU加速配置"""
    enable_gpu: bool = False
    device_id: int = 0
    fallback_to_cpu: bool = True  # GPU失败时回退到CPU
    
    def __post_init__(self):
        """自动检测GPU可用性"""
        if self.enable_gpu:
            try:
                import cupy as cp
                gpu_count = cp.cuda.runtime.getDeviceCount()
                if gpu_count == 0:
                    import warnings
                    warnings.warn("未检测到GPU，禁用GPU加速", RuntimeWarning)
                    self.enable_gpu = False
            except ImportError:
                import warnings
                warnings.warn("CuPy未安装，禁用GPU加速。安装方法: pip install cupy-cuda11x", ImportWarning)
                self.enable_gpu = False


# ============ 压缩配置 ============

@dataclass
class CompressionConfig:
    """压缩配置"""
    preset: str = "promax"  # fast, normal, max, promax - 默认 ProMax
    window_size: int = 4095
    k: int = 8  # 哈希长度
    max_bucket: int = 48
    lazy_match: int = 2
    
    def __post_init__(self):
        """根据preset自动调整参数"""
        if self.preset == "fast":
            self.max_bucket = 32
            self.lazy_match = 1
        elif self.preset == "max":
            self.max_bucket = 64
            self.lazy_match = 2
        elif self.preset == "promax":
            # ProMax 使用暴力搜索，不依赖这些参数
            self.max_bucket = 64
            self.lazy_match = 2
        elif self.preset != "normal":
            raise ValueError("preset must be 'fast', 'normal', 'max', or 'promax'")


# ============ 应用配置 ============

@dataclass
class AppConfig:
    """应用总配置"""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    
    # 通用配置
    verbose: bool = False
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    @classmethod
    def create_default(cls) -> 'AppConfig':
        """创建默认配置"""
        return cls()
    
    @classmethod
    def create_fast(cls) -> 'AppConfig':
        """创建快速模式配置"""
        return cls(
            optimization=OptimizationConfig(quality_level=1),
            compression=CompressionConfig(preset="fast"),
            parallel=ParallelConfig(min_files_for_parallel=2)
        )
    
    @classmethod
    def create_best(cls) -> 'AppConfig':
        """创建最佳质量配置"""
        return cls(
            optimization=OptimizationConfig(quality_level=3),
            compression=CompressionConfig(preset="max")
        )
    
    @classmethod
    def create_promax(cls) -> 'AppConfig':
        """创建ProMax最优压缩配置（暴力搜索）"""
        return cls(
            optimization=OptimizationConfig(quality_level=3),
            compression=CompressionConfig(preset="promax")
        )
    
    @classmethod
    def create_promax(cls) -> 'AppConfig':
        """创建ProMax最优压缩配置（暴力搜索）"""
        return cls(
            optimization=OptimizationConfig(quality_level=3),
            compression=CompressionConfig(preset="promax")
        )
        """创建GPU加速配置"""
        return cls(
            gpu=GPUConfig(enable_gpu=True),
            optimization=OptimizationConfig(quality_level=2)
        )
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = ["应用配置摘要:"]
        lines.append(f"  并行处理: {self.parallel.max_workers} workers, 最小文件数: {self.parallel.min_files_for_parallel}")
        lines.append(f"  进度显示: {'启用' if self.progress.show_progress else '禁用'}")
        lines.append(f"  优化级别: {self.optimization.quality_level} (1=快速, 2=平衡, 3=最佳)")
        lines.append(f"  GPU加速: {'启用' if self.gpu.enable_gpu else '禁用'}")
        lines.append(f"  压缩预设: {self.compression.preset}")
        return "\n".join(lines)


# ============ 辅助函数 ============

def get_optimal_workers(file_count: int, max_workers: Optional[int] = None) -> int:
    """根据文件数量获取最优工作线程数"""
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 文件数少于4个时不使用并行
    if file_count < 4:
        return 1
    
    # 限制最大workers数
    return min(max_workers, file_count, 8)


def validate_config(config: AppConfig) -> bool:
    """验证配置有效性"""
    try:
        # 验证质量级别
        if config.optimization.quality_level not in (1, 2, 3):
            return False
        
        # 验证压缩预设
        if config.compression.preset not in ('fast', 'normal', 'max', 'promax'):
            return False
        
        # 验证并行配置
        if config.parallel.max_workers < 0:
            return False
        
        return True
    except Exception:
        return False


# ============ 主入口（测试）============

if __name__ == "__main__":
    print("=" * 60)
    print("配置模块测试")
    print("=" * 60)
    
    # 测试默认配置
    print("\n1. 默认配置:")
    default_config = AppConfig.create_default()
    print(default_config.summary())
    
    # 测试快速模式
    print("\n2. 快速模式:")
    fast_config = AppConfig.create_fast()
    print(fast_config.summary())
    
    # 测试最佳质量模式
    print("\n3. 最佳质量模式:")
    best_config = AppConfig.create_best()
    print(best_config.summary())
    
    # 测试GPU模式
    print("\n4. GPU加速模式:")
    gpu_config = AppConfig.create_gpu()
    print(gpu_config.summary())
    
    # 测试配置验证
    print("\n5. 配置验证:")
    print(f"  默认配置有效: {validate_config(default_config)}")
    
    # 测试最优workers计算
    print("\n6. 最优workers计算:")
    for count in [1, 3, 5, 10, 20]:
        workers = get_optimal_workers(count)
        print(f"  {count} 个文件 → {workers} workers")
    
    print("\n" + "=" * 60)
    print("测试完成")
