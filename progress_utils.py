#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[call-arg] - 回调函数可能为 None
"""
进度条工具模块 - 统一的进度显示接口

功能介绍：
  提供统一的进度条接口，支持 GUI 和命令行两种显示模式
  
  主要组件：
  - ProgressCallback：统一进度回调管理器
  - ConsoleProgressBar：命令行进度条（不依赖 tqdm）
  - ProgressConfig：进度条配置

用法：
  命令行进度条：
    from progress_utils import ConsoleProgressBar
    with ConsoleProgressBar(100, desc="处理") as pbar:
        for i in range(100):
            pbar.update(1)
  
  GUI 进度回调：
    from progress_utils import ProgressCallback, ProgressConfig
    config = ProgressConfig(enable_file_progress=True)
    callback = ProgressCallback(
        file_progress_cb=my_file_callback,
        batch_progress_cb=my_batch_callback,
        config=config
    )

API 说明：
  ConsoleProgressBar(total, desc="", width=50, show_speed=True, show_eta=True)
    total: 总项数
    desc: 描述文本
    width: 进度条宽度
    show_speed: 是否显示速度
    show_eta: 是否显示 ETA
  
  ProgressCallback(file_progress_cb, batch_progress_cb, log_cb, config)
    file_progress_cb: 单文件进度回调 (done, total)
    batch_progress_cb: 批量进度回调 (processed, total, elapsed)
    log_cb: 日志回调
    config: ProgressConfig 配置对象
  
  ProgressConfig(
    enable_file_progress=True,
    enable_batch_progress=True,
    enable_console=True,
    progress_update_interval=0.1,
    console_width=50
  )

示例：
  # 命令行进度条
  from progress_utils import ConsoleProgressBar
  
  pbar = ConsoleProgressBar(100, desc="转换", show_speed=True)
  for i in range(100):
      pbar.update(1)
  pbar.close()
  
  # 使用上下文管理器
  with ConsoleProgressBar(50, desc="压缩") as pbar:
      for i in range(50):
          pbar.update(1)
  
  # GUI 进度回调
  from progress_utils import ProgressCallback
  
  def on_file_progress(done, total):
      print(f"文件进度: {done}/{total}")
  
  def on_batch_progress(processed, total, elapsed):
      print(f"批量进度: {processed}/{total}, 耗时: {elapsed:.1f}s")
  
  callback = ProgressCallback(
      file_progress_cb=on_file_progress,
      batch_progress_cb=on_batch_progress
  )
  callback.update_file_progress(50, 100)
  callback.update_batch_progress(10, 20, 5.5)

特性：
  - 自动节流更新，避免频繁刷新
  - 线程安全，支持多线程调用
  - 支持上下文管理器协议
  - 无外部依赖（命令行进度条）

依赖：
  必需：无（仅需 Python 标准库）
"""

import os
import sys
import time
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class ProgressConfig:
    """进度条配置"""
    enable_file_progress: bool = True
    enable_batch_progress: bool = True
    enable_console: bool = True
    progress_update_interval: float = 0.1  # 最小更新间隔（秒）
    console_width: int = 50  # 命令行进度条宽度


class ProgressCallback:
    """统一进度回调管理器"""
    
    def __init__(self, 
                 file_progress_cb: Optional[Callable[[int, int], None]] = None,
                 batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                 log_cb: Optional[Callable[[str], None]] = None,
                 config: Optional[ProgressConfig] = None):
        self.file_progress_cb = file_progress_cb
        self.batch_progress_cb = batch_progress_cb
        self.log_cb = log_cb
        self.config = config or ProgressConfig()
        self._last_file_update = 0
        self._last_batch_update = 0
        self._lock = threading.Lock()
    
    def update_file_progress(self, done: int, total: int) -> None:
        """更新单文件进度"""
        if not self.config.enable_file_progress or not self.file_progress_cb:
            return
            
        now = time.time()
        with self._lock:
            # 节流控制（但100%必须立即更新）
            if (now - self._last_file_update >= self.config.progress_update_interval) or (done >= total):
                self.file_progress_cb(done, total)
                self._last_file_update = now
    
    def update_batch_progress(self, processed: int, total: int, elapsed_time: float) -> None:
        """更新批量进度"""
        if not self.config.enable_batch_progress or not self.batch_progress_cb:
            return
            
        now = time.time()
        with self._lock:
            # 节流控制（但100%必须立即更新）
            if (now - self._last_batch_update >= self.config.progress_update_interval) or (processed >= total):
                self.batch_progress_cb(processed, total, elapsed_time)
                self._last_batch_update = now
    
    def log(self, message: str) -> None:
        """输出日志"""
        if self.log_cb:
            self.log_cb(message)


class ConsoleProgressBar:
    """
    命令行进度条 - 不依赖外部库
    
    提供在终端显示进度条的功能，支持速度和ETA显示。
    
    属性:
        total: 总项数
        desc: 描述文本
        width: 进度条宽度，默认50
        show_speed: 是否显示速度
        show_eta: 是否显示ETA
    
    使用示例:
        >>> with ConsoleProgressBar(100, desc="处理") as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)
        
        >>> pbar = ConsoleProgressBar(10, show_speed=True, show_eta=True)
        >>> for i in range(10):
        ...     pbar.update(1)
        >>> pbar.close()
    
    性能特性:
        - 自动节流更新，避免频繁刷新
        - 支持上下文管理器
        - 线程安全
    """
    
    def __init__(self, total: int, desc: str = "", width: int = 50, 
                 show_speed: bool = True, show_eta: bool = True):
        self.total = total
        self.desc = desc
        self.width = width
        self.show_speed = show_speed
        self.show_eta = show_eta
        self.current = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._closed = False
        self._last_update_time = self.start_time
        self._last_update_value = 0
    
    def update(self, n: int = 1) -> None:
        """更新进度"""
        if self._closed:
            return
            
        with self._lock:
            self.current += n
            now = time.time()
            
            # 限制更新频率（但100%必须立即更新）
            if (now - self._last_update_time < 0.1) and (self.current < self.total):
                return
                
            self._render(now)
            self._last_update_time = now
            self._last_update_value = self.current
    
    def set_description(self, desc: str) -> None:
        """设置描述"""
        self.desc = desc
    
    def _render(self, current_time: float) -> None:
        """渲染进度条"""
        if self.total <= 0:
            return
            
        progress = min(self.current / self.total, 1.0)
        elapsed = current_time - self.start_time
        
        # 计算速度
        speed = 0
        if elapsed > 0:
            speed = self.current / elapsed
        
        # 计算ETA
        eta = 0
        if self.current > 0 and speed > 0:
            remaining = self.total - self.current
            eta = remaining / speed
        
        # 构建进度条（使用ASCII字符，兼容GBK编码）
        filled = int(self.width * progress)
        bar = '=' * filled + '-' * (self.width - filled)
        
        # 格式化信息
        info_parts = []
        info_parts.append(f"{progress*100:5.1f}%")
        info_parts.append(f"[{self.current}/{self.total}]")
        
        if self.show_speed:
            info_parts.append(f"Speed: {self._format_speed(speed)}")
        
        if self.show_eta and eta > 0:
            info_parts.append(f"ETA: {self._format_time(eta)}")
        
        info_str = " ".join(info_parts)
        
        # 输出
        desc_str = f"{self.desc}: " if self.desc else ""
        sys.stdout.write(f"\r{desc_str}|{bar}| {info_str}")
        sys.stdout.flush()
    
    def _format_speed(self, speed: float) -> str:
        """格式化速度显示"""
        if speed < 1024:
            return f"{speed:.1f} B/s"
        elif speed < 1024 * 1024:
            return f"{speed/1024:.1f} KB/s"
        else:
            return f"{speed/(1024*1024):.1f} MB/s"
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def close(self) -> None:
        """关闭进度条"""
        if not self._closed:
            self._closed = True
            # 显示最终状态
            self._render(time.time())
            print()  # 换行
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BatchProgressManager:
    """
    批量处理进度管理器
    
    功能:
        - 统一管理单文件和批量进度
        - 自动节流GUI回调，避免界面卡顿
        - 支持CLI进度条和日志输出
        - 支持GUI和命令行同时显示（force_cli_progress=True）
    
    使用示例:
        >>> manager = BatchProgressManager(
        ...     total_files=10,
        ...     file_progress_cb=update_file_progress,
        ...     batch_progress_cb=update_batch_progress,
        ...     use_cli_progress=True
        ... )
        >>> 
        >>> for file in files:
        ...     manager.start_file(file)
        ...     # 处理文件，使用 manager.get_file_callback() 获取回调
        ...     process_file(file, progress_cb=manager.get_file_callback())
        ...     manager.finish_file(success=True)
        >>> 
        >>> manager.close()
    """
    
    def __init__(self,
                 total_files: int,
                 file_progress_cb: Optional[Callable[[int, int], None]] = None,
                 batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                 log_cb: Optional[Callable[[str], None]] = None,
                 use_cli_progress: bool = False,
                 cli_desc: str = "批量处理",
                 force_cli_progress: bool = False):
        """
        Args:
            total_files: 总文件数
            file_progress_cb: 单文件进度回调 (done, total)
            batch_progress_cb: 批量进度回调 (processed, total, elapsed)
            log_cb: 日志回调函数
            use_cli_progress: 是否使用CLI进度条
            cli_desc: CLI进度条描述
            force_cli_progress: 强制启用CLI进度条（即使GUI模式也显示）
        """
        self.total_files = total_files
        self.file_progress_cb = file_progress_cb
        self.batch_progress_cb = batch_progress_cb
        self.log_cb = log_cb
        
        self.processed_count = 0
        self.start_time = time.time()
        self.current_file = None
        
        # 节流控制 - GUI模式下避免过于频繁的回调
        self._last_file_update = 0
        self._last_batch_update = 0
        self._file_throttle_interval = 0.05  # 50ms - GUI优化
        self._batch_throttle_interval = 0.1  # 100ms
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 定时器线程 - 每0.1s强制更新批量进度时间
        self._timer_thread = None
        self._timer_running = False
        if batch_progress_cb:
            self._timer_running = True
            self._timer_thread = threading.Thread(target=self._time_update_loop, daemon=True)
            self._timer_thread.start()
        
        # CLI进度条 - GUI模式下也显示（如果 force_cli_progress=True）
        self.cli_pbar = None
        should_show_cli = use_cli_progress or (force_cli_progress and batch_progress_cb is not None)
        if should_show_cli:
            try:
                self.cli_pbar = ConsoleProgressBar(
                    total=total_files,
                    desc=cli_desc,
                    width=50,
                    show_speed=True,
                    show_eta=True
                )
            except Exception:
                pass
        
        # 初始化批量进度
        if self.batch_progress_cb:
            self.batch_progress_cb(0, total_files, 0.0)
    
    def _time_update_loop(self):
        """定时器线程：每0.1s强制更新批量进度的时间显示"""
        while self._timer_running:
            time.sleep(0.1)  # 每0.1秒更新一次
            if self._timer_running and self.batch_progress_cb:
                elapsed = time.time() - self.start_time
                # 直接更新，不经过节流检查（因为已经是0.1s间隔）
                self.batch_progress_cb(self.processed_count, self.total_files, elapsed)
    
    def get_file_callback(self) -> Optional[Callable[[int, int], None]]:
        """
        获取带节流优化的单文件进度回调
        
        Returns:
            节流后的进度回调函数
        """
        if not self.file_progress_cb:
            return None
        
        def _throttled_callback(done: int, total: int):
            """GUI优化:节流单文件进度回调"""
            now = time.time()
            with self._lock:
                # 只在间隔足够时才回调，或者已完成(done==total)
                if (now - self._last_file_update >= self._file_throttle_interval) or (done >= total):
                    self.file_progress_cb(done, total)
                    self._last_file_update = now
        
        return _throttled_callback
    
    def start_file(self, filename: str):
        """开始处理一个文件"""
        self.current_file = filename
        self._last_file_update = 0  # 重置节流计时器
    
    def finish_file(self, success: bool = True):
        """完成一个文件的处理"""
        with self._lock:
            self.processed_count += 1
            
            # 更新批量进度
            if self.cli_pbar:
                self.cli_pbar.update(1)
            
            # 批量进度由定时器线程自动更新，这里不需要重复调用
            # 只在最后一个文件完成时强制更新一次
            if self.batch_progress_cb and self.processed_count >= self.total_files:
                elapsed = time.time() - self.start_time
                self.batch_progress_cb(self.processed_count, self.total_files, elapsed)
    
    def log(self, message: str):
        """输出日志"""
        if self.log_cb:
            self.log_cb(message)
        elif not self.cli_pbar:  # CLI进度条模式下不输出日志
            print(message)
    
    def close(self):
        """关闭进度管理器"""
        # 停止定时器线程
        if self._timer_thread:
            self._timer_running = False
            self._timer_thread.join(timeout=0.5)  # 等待线程结束，最多0.5秒
        
        # 确保最终单文件进度为100%
        if self.file_progress_cb:
            self.file_progress_cb(100, 100)
        
        # 确保最终批量进度为100%
        if self.batch_progress_cb:
            elapsed = time.time() - self.start_time
            self.batch_progress_cb(self.total_files, self.total_files, elapsed)
        
        # 关闭CLI进度条
        if self.cli_pbar:
            self.cli_pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FileProgressTracker:
    """文件处理进度跟踪器"""
    
    def __init__(self, file_path: str, callback: Optional[ProgressCallback] = None):
        self.file_path = file_path
        self.callback = callback
        self.file_size = 0
        self.processed_size = 0
        self.start_time = time.time()
        
        try:
            self.file_size = os.path.getsize(file_path)
        except OSError:
            self.file_size = 0
    
    def update_processed_size(self, size: int) -> None:
        """更新已处理大小"""
        self.processed_size += size
        if self.callback and self.file_size > 0:
            self.callback.update_file_progress(self.processed_size, self.file_size)
    
    def get_progress(self) -> float:
        """获取当前进度百分比"""
        if self.file_size <= 0:
            return 0.0
        return min(self.processed_size / self.file_size * 100, 100.0)
    
    def get_speed(self) -> float:
        """获取处理速度（字节/秒）"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.processed_size / elapsed


class BatchProgressTracker:
    """批量处理进度跟踪器"""
    
    def __init__(self, total_files: int, callback: Optional[ProgressCallback] = None):
        self.total_files = total_files
        self.processed_files = 0
        self.succeeded_files = 0
        self.failed_files = 0
        self.callback = callback
        self.start_time = time.time()
    
    def update_file_completed(self, success: bool = True) -> None:
        """更新文件完成状态"""
        self.processed_files += 1
        if success:
            self.succeeded_files += 1
        else:
            self.failed_files += 1
        
        if self.callback:
            elapsed = time.time() - self.start_time
            self.callback.update_batch_progress(self.processed_files, self.total_files, elapsed)
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        elapsed = time.time() - self.start_time
        return {
            'total': self.total_files,
            'processed': self.processed_files,
            'succeeded': self.succeeded_files,
            'failed': self.failed_files,
            'elapsed_time': elapsed,
            'progress_percent': (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
        }


def format_file_size(size: int) -> str:
    """格式化文件大小显示"""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.1f} GB"


def format_speed(speed: float) -> str:
    """格式化速度显示"""
    if speed < 1024:
        return f"{speed:.1f} B/s"
    elif speed < 1024 * 1024:
        return f"{speed/1024:.1f} KB/s"
    else:
        return f"{speed/(1024*1024):.1f} MB/s"


# 全局默认实例
_default_progress_callback = ProgressCallback()


def get_default_callback() -> ProgressCallback:
    """获取默认进度回调实例"""
    return _default_progress_callback


def set_default_callback(callback: ProgressCallback) -> None:
    """设置默认进度回调实例"""
    global _default_progress_callback
    _default_progress_callback = callback


class DualProgressCallback:
    """
    双重进度回调包装器 - 同时支持GUI和命令行进度显示
    
    功能：
        - 将单个进度更新同时发送到GUI回调和命令行进度条
        - GUI运行时命令行也能看到进度
        - 自动管理命令行进度条的生命周期
    
    使用示例：
        >>> dual_cb = DualProgressCallback(
        ...     gui_callback=update_gui_progress,
        ...     total=100,
        ...     desc="处理文件"
        ... )
        >>> for i in range(100):
        ...     dual_cb(i, 100)  # 同时更新GUI和CLI
        >>> dual_cb.close()  # 关闭CLI进度条
    """
    
    def __init__(self,
                 gui_callback: Optional[Callable[[int, int], None]] = None,
                 total: int = 100,
                 desc: str = "处理",
                 enable_cli: bool = True):
        """
        Args:
            gui_callback: GUI进度回调函数 (done, total)
            total: 总进度值
            desc: 进度条描述
            enable_cli: 是否启用命令行进度条
        """
        self.gui_callback = gui_callback
        self.cli_pbar = None
        
        # 只在有GUI回调且启用CLI时创建命令行进度条
        if enable_cli and gui_callback is not None:
            try:
                self.cli_pbar = ConsoleProgressBar(
                    total=total,
                    desc=desc,
                    width=40,
                    show_speed=False,
                    show_eta=False
                )
            except Exception:
                pass
    
    def __call__(self, done: int, total: int) -> None:
        """更新进度 - 同时更新GUI和CLI"""
        # 更新GUI
        if self.gui_callback:
            self.gui_callback(done, total)
        
        # 更新CLI进度条
        if self.cli_pbar:
            self.cli_pbar.current = done
            self.cli_pbar.update(0)  # 强制刷新
    
    def close(self) -> None:
        """关闭命令行进度条"""
        if self.cli_pbar:
            self.cli_pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()