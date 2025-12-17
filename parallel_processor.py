#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[call-issue] - 动态函数调用
"""
并行处理模块 - 多线程批量处理能力

功能介绍：
  提供可配置的多线程批量处理能力
  
  主要组件：
  - parallel_process_files：并行处理文件列表
  - ProgressAggregator：进度聚合器（线程安全）
  - LogAggregator：日志聚合器（线程安全）
  - TaskController：任务控制器（暂停/停止/检查）
  - ParallelConfig：并行配置

用法：
  基本使用：
    from parallel_processor import parallel_process_files, ParallelConfig
    
    config = ParallelConfig(max_workers=4)
    results, failures = parallel_process_files(
        files=['file1.png', 'file2.png'],
        process_func=my_process_function,
        config=config
    )
  
  带进度回调：
    results, failures = parallel_process_files(
        files=file_list,
        process_func=convert_image,
        config=config,
        file_progress_cb=on_file_progress,
        batch_progress_cb=on_batch_progress,
        log_cb=on_log
    )

API 说明：
  parallel_process_files(files, process_func, config, **callbacks)
    files: 文件路径列表
    process_func: 处理函数，签名 func(file_path, **kwargs) -> result
    config: ParallelConfig 配置对象
    file_progress_cb: 单文件进度回调 (done, total)
    batch_progress_cb: 批量进度回调 (processed, total, elapsed)
    log_cb: 日志回调
    task_controller: TaskController 控制器
    **process_kwargs: 传递给 process_func 的额外参数
    
    返回: (results, failures)
      results: 成功结果列表
      failures: 失败列表 [(file_path, exception)]
  
  ParallelConfig(
    max_workers=0,              # 0=自动（CPU核心数-1）
    enable_parallel=True,       # 是否启用并行
    min_files_for_parallel=4    # 最少文件数才启用并行
  )
  
  TaskController()
    方法：
      reset() - 重置状态
      pause() - 暂停
      resume() - 继续
      stop() - 停止
      is_paused() -> bool
      is_stopped() -> bool
      wait_if_paused(timeout=0.1)
      check_stop() - 检查是否停止（抛出异常）

示例：
  # 基本示例
  from parallel_processor import parallel_process_files, ParallelConfig
  
  def process_image(file_path):
      # 处理逻辑
      return f"Processed: {file_path}"
  
  config = ParallelConfig(max_workers=4)
  files = ['img1.png', 'img2.png', 'img3.png']
  results, failures = parallel_process_files(files, process_image, config)
  
  print(f"成功: {len(results)}, 失败: {len(failures)}")
  
  # 带进度回调示例
  from parallel_processor import TaskController
  
  controller = TaskController()
  
  def on_batch_progress(processed, total, elapsed):
      print(f"进度: {processed}/{total}, 耗时: {elapsed:.1f}s")
  
  results, failures = parallel_process_files(
      files=file_list,
      process_func=process_image,
      batch_progress_cb=on_batch_progress,
      task_controller=controller
  )

优化特性：
  - 线程安全的进度更新
  - 自动节流回调，避免频繁更新
  - 支持实时聚合多个 worker 的进度
  - 智能选择并行/串行模式
  - 支持任务暂停/停止/检查

注意事项：
  - 优先使用 config.py 中的 ParallelConfig
  - 提供回退实现以保持向后兼容

依赖：
  必需：无（仅需 Python 标准库 concurrent.futures）
  推荐：config.py（统一配置管理）
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Any, Dict

# 导入统一配置（强制依赖 config.py）
from config import ParallelConfig


class ProgressAggregator:
    """
    进度聚合器 - 线程安全的进度收集与回调
    
    优化特性:
        - 线程安全的进度更新
        - 自动节流回调，避免频繁更新
        - 支持实时聚合多个worker的进度
        - 跟踪进度最快的文件（并行模式）
    """
    
    def __init__(self, total_files: int,
                 file_progress_cb: Optional[Callable[[int, int], None]] = None,
                 batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                 current_file_cb: Optional[Callable[[str], None]] = None):
        self.total_files = total_files
        self.file_progress_cb = file_progress_cb
        self.batch_progress_cb = batch_progress_cb
        self.current_file_cb = current_file_cb  # 新增：当前文件名回调
        self.start_time = time.time()
        
        # 线程安全状态
        self._lock = threading.Lock()
        self._processed = 0
        self._current_file_progress: Dict[int, tuple] = {}  # {worker_id: (done, total, file_name)}
        self._fastest_file = ''  # 进度最快的文件
        self._fastest_progress = 0.0  # 最快进度
        
        # 节流控制
        self._last_batch_update = 0.0
        self._batch_update_interval = 0.3
        self._last_file_update = 0.0
        self._file_update_interval = 0.1
    
    def update_file_progress(self, worker_id: int, done: int, total: int, file_name: str = ''):
        """
        更新单个文件的进度
        
        支持多线程并发调用，自动聚合所有worker的进度
        并跟踪进度最快的文件
        """
        now = time.time()
        should_update = False
        update_done = done
        update_total = total
        
        with self._lock:
            self._current_file_progress[worker_id] = (done, total, file_name)
            
            # 计算当前进度百分比
            current_pct = (done / total * 100.0) if total > 0 else 0.0
            
            # 更新进度最快的文件（只在进度真正更快时才更新）
            if current_pct > self._fastest_progress:
                self._fastest_progress = current_pct
                self._fastest_file = file_name
                
                # 通知当前文件名变化
                if self.current_file_cb and file_name:
                    self.current_file_cb(file_name)
                
                # 使用最快文件的进度
                update_done = done
                update_total = total
            elif file_name == self._fastest_file:
                # 如果是当前最快文件的进度更新，则同步更新
                self._fastest_progress = current_pct
                update_done = done
                update_total = total
            else:
                # 其他文件的进度不更新显示，保持显示最快文件的进度
                update_done = int(self._fastest_progress * total / 100.0)
                update_total = total
            
            # 节流控制（但100%必须立即更新）
            if (now - self._last_file_update >= self._file_update_interval) or (current_pct >= 100.0):
                should_update = True
                self._last_file_update = now
        
        # 触发文件进度回调（显示进度最快的文件）
        if should_update and self.file_progress_cb:
            self.file_progress_cb(update_done, update_total)
    
    def _find_fastest_file(self):
        """
        查找当前进度最快的文件（内部方法，需要在锁内调用）
        
        返回: (fastest_file_name, fastest_progress_pct)
        """
        if not self._current_file_progress:
            return '', 0.0
        
        fastest_file = ''
        fastest_pct = 0.0
        
        for worker_id, (done, total, file_name) in self._current_file_progress.items():
            if total > 0:
                pct = done / total * 100.0
                if pct > fastest_pct:
                    fastest_pct = pct
                    fastest_file = file_name
        
        return fastest_file, fastest_pct
    
    def mark_file_completed(self, worker_id: int):
        """标记一个文件处理完成"""
        with self._lock:
            self._processed += 1
            processed = self._processed
            now = time.time()
            
            # 从进度字典中移除已完成的worker
            if worker_id in self._current_file_progress:
                completed_file = self._current_file_progress[worker_id][2]
                del self._current_file_progress[worker_id]
                
                # 如果完成的是当前最快文件，需要重新查找下一个最快的文件
                if completed_file == self._fastest_file:
                    self._fastest_file, self._fastest_progress = self._find_fastest_file()
                    
                    # 通知文件名变化
                    if self.current_file_cb and self._fastest_file:
                        self.current_file_cb(self._fastest_file)
            
            # 节流批量进度回调（但最后一个文件完成时必须立即更新）
            if self.batch_progress_cb:
                elapsed = now - self.start_time
                # 立即更新：达到节流间隔 或 所有文件处理完成
                if (now - self._last_batch_update >= self._batch_update_interval) or (processed >= self.total_files):
                    self.batch_progress_cb(processed, self.total_files, elapsed)
                    self._last_batch_update = now
    
    def finalize(self):
        """最终更新（确保100%）"""
        # 确保单文件进度为100%
        if self.file_progress_cb:
            self.file_progress_cb(100, 100)
        
        # 确保批量进度为100%
        if self.batch_progress_cb:
            elapsed = time.time() - self.start_time
            self.batch_progress_cb(self.total_files, self.total_files, elapsed)


class LogAggregator:
    """
    日志聚合器 - 线程安全的日志收集
    
    特性:
        - 线程安全的日志输出
        - 自动序列化多线程日志
    """
    
    def __init__(self, log_cb: Optional[Callable[[str], None]] = None):
        self.log_cb = log_cb
        self._lock = threading.Lock()
    
    def log(self, message: str):
        """线程安全的日志输出"""
        if self.log_cb:
            with self._lock:
                self.log_cb(message)


class TaskController:
    """任务控制器 - 支持暂停/停止/检查（使用条件变量优化）"""
    
    def __init__(self):
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self._state_condition = threading.Condition()
    
    def reset(self):
        """重置状态"""
        with self._state_condition:
            self.pause_event.clear()
            self.stop_event.clear()
            self._state_condition.notify_all()
    
    def pause(self):
        """暂停"""
        self.pause_event.set()
    
    def resume(self):
        """继续"""
        with self._state_condition:
            self.pause_event.clear()
            self._state_condition.notify_all()  # 唤醒所有等待线程
    
    def stop(self):
        """停止"""
        with self._state_condition:
            self.stop_event.set()
            self.pause_event.clear()  # 停止时也要清除暂停状态
            self._state_condition.notify_all()  # 唤醒所有等待线程
    
    def is_paused(self) -> bool:
        return self.pause_event.is_set()
    
    def is_stopped(self) -> bool:
        return self.stop_event.is_set()
    
    def wait_if_paused(self, timeout: float = 0.1):
        """如果暂停则等待（使用条件变量阻塞，零CPU消耗）"""
        with self._state_condition:
            while self.pause_event.is_set() and not self.stop_event.is_set():
                # 使用条件变量阻塞等待，而非忙等待
                self._state_condition.wait(timeout)
    
    def check_stop(self):
        """检查是否停止（抛出异常）"""
        if self.stop_event.is_set():
            raise UserStopException("用户停止处理")


class UserStopException(Exception):
    """用户停止异常"""
    pass


def parallel_process_files(
    files: List[str],
    process_func: Callable[[str, Any], Any],
    config: Optional[ParallelConfig] = None,
    file_progress_cb: Optional[Callable[[int, int], None]] = None,
    batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    task_controller: Optional[TaskController] = None,
    **process_kwargs
) -> tuple[List[Any], List[tuple[str, Exception]]]:
    """
    并行处理文件列表
    
    参数：
        files: 文件路径列表
        process_func: 处理函数，签名为 func(file_path, **kwargs) -> result
        config: 并行配置
        file_progress_cb: 单文件进度回调
        batch_progress_cb: 批量进度回调
        log_cb: 日志回调
        task_controller: 任务控制器
        **process_kwargs: 传递给 process_func 的额外参数
    
    返回：
        (成功结果列表, 失败列表[(文件路径, 异常)])
    """
    if config is None:
        config = ParallelConfig()
    
    total = len(files)
    
    # 判断是否启用并行
    use_parallel = (
        config.enable_parallel and 
        total >= config.min_files_for_parallel and
        config.max_workers != 1
    )
    
    # 确定工作线程数
    if config.max_workers <= 0:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        max_workers = config.max_workers
    
    # 创建聚合器（支持当前文件回调）
    current_file_cb = process_kwargs.get('current_file_cb', None)
    progress_agg = ProgressAggregator(total, file_progress_cb, batch_progress_cb, current_file_cb)
    log_agg = LogAggregator(log_cb)
    
    # 初始化批量进度
    if batch_progress_cb:
        batch_progress_cb(0, total, 0.0)
    
    results = []
    errors = []
    
    def _process_wrapper(file_path: str, worker_id: int):
        """包装处理函数，添加进度和日志"""
        try:
            # 检查中断
            if task_controller:
                task_controller.wait_if_paused()
                task_controller.check_stop()
            
            # 提取文件名（用于显示）
            import os
            file_name = os.path.basename(file_path)
            
            # 创建单文件进度回调
            def _file_cb(done, total):
                progress_agg.update_file_progress(worker_id, done, total, file_name)
            
            # 执行处理（注入进度回调）
            kwargs = process_kwargs.copy()
            if 'file_progress_cb' in process_func.__code__.co_varnames:
                kwargs['file_progress_cb'] = _file_cb
            if 'log_cb' in process_func.__code__.co_varnames:
                kwargs['log_cb'] = log_agg.log
            
            result = process_func(file_path, **kwargs)
            
            # 标记完成
            progress_agg.mark_file_completed(worker_id)
            return (file_path, result, None)
            
        except Exception as e:
            progress_agg.mark_file_completed(worker_id)
            return (file_path, None, e)
    
    if use_parallel:
        # 并行处理
        if log_cb:
            log_cb(f"启用并行处理,线程数: {max_workers}")
        
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            # 提交所有任务
            futures = {
                executor.submit(_process_wrapper, file_path, i): (file_path, i)
                for i, file_path in enumerate(files)
            }
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    # 添加超时保护（5分钟）
                    file_path, result, error = future.result(timeout=300)
                    if error:
                        errors.append((file_path, error))
                        if log_cb:
                            log_cb(f"ERROR 处理失败 {file_path}: {error}")
                    else:
                        results.append(result)
                except Exception as e:
                    file_path, _ = futures[future]
                    errors.append((file_path, e))
                    if log_cb:
                        log_cb(f"ERROR 处理失败 {file_path}: {e}")
                
                # 检查中断
                if task_controller and task_controller.is_stopped():
                    # 立即停止，不等待未完成的任务
                    if log_cb:
                        log_cb("用户停止处理，正在取消剩余任务...")
                    break
        finally:
            # 确保executor正确关闭
            # 如果用户停止，不等待未完成任务；否则等待所有任务完成
            if task_controller and task_controller.is_stopped():
                # 立即关闭，取消未开始的任务
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python < 3.9 不支持 cancel_futures
                    executor.shutdown(wait=False)
            else:
                # 正常完成，等待所有任务
                executor.shutdown(wait=True)
    else:
        # 串行处理
        for i, file_path in enumerate(files):
            file_path_result, result, error = _process_wrapper(file_path, i)
            if error:
                errors.append((file_path, error))
                if log_cb:
                    log_cb(f"ERROR 处理失败 {file_path}: {error}")
            else:
                results.append(result)
            
            # 检查中断
            if task_controller:
                try:
                    task_controller.wait_if_paused()
                    task_controller.check_stop()
                except UserStopException:
                    break
    
    # 最终更新
    progress_agg.finalize()
    
    return results, errors


def get_optimal_workers(file_count: int, config: Optional[ParallelConfig] = None, io_bound: bool = True) -> int:
    """
    获取最优工作线程数
    
    Args:
        file_count: 文件数量
        config: 并行配置
        io_bound: 是否为I/O密集型任务（图像解码为I/O密集，压缩为CPU密集）
    
    Returns:
        最优线程数
    
    算法说明:
        - I/O密集型: 可超额分配（CPU核心数 * 2）
        - CPU密集型: 略小于CPU核心数（避免上下文切换）
        - 小批量: 限制线程数避免开销
    """
    if config is None:
        config = ParallelConfig()
    
    if not config.enable_parallel or file_count < config.min_files_for_parallel:
        return 1
    
    if config.max_workers > 0:
        return config.max_workers
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # 动态调整策略（改进算法）
    if io_bound:
        # I/O密集型：可超额分配
        max_theoretical = min(cpu_count * 2, 32)  # 最多32线程
    else:
        # CPU密集型：接近CPU核心数
        max_theoretical = max(1, cpu_count - 1)
    
    # 根据任务数量自适应
    if file_count < 4:
        return 1
    elif file_count < 10:
        return min(2, max_theoretical)
    elif file_count < 50:
        return min(max(4, cpu_count // 2), max_theoretical)
    else:
        return max_theoretical
