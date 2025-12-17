#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Softpal PGD Toolkit - PNG⇔PGD 互转工具（现代化 UI 版）

功能介绍：
  支持 PNG 与 PGD 格式的双向转换，支持多种 PGD 变体格式：
  - GE 格式：支持类型 1/2/3 压缩方法
  - Others 格式：00_C、11_C、PGD/TGA、PGD3（增量叠加）
  - 自动类型检测：智能识别 PGD 文件类型

现代化特性：
  1. 🎨 使用 CustomTkinter 构建现代化界面
  2. 🌓 支持深色/浅色主题切换
  3. 📊 优化的进度显示和状态反馈
  4. ⚡ GPU 加速和并行处理支持
  5. 🖱️ 增强的拖放体验
  6. 💾 智能布局和响应式设计

用法：
  图形界面模式：
    python main_optimized.py
    python main_optimized.py gui
  
  命令行模式（PNG 转 PGD）：
    python main_optimized.py png2pgd --in <input.png> --out <output.pgd> --fmt <格式> [options]

命令行参数：
  png2pgd 子命令：
    --in <path>        输入 PNG 文件路径（必需）
    --out <path>       输出 PGD 文件路径（必需）
    --fmt <format>     目标 PGD 格式（必需）
                       可选值：11_C, 00_C, TGA, PGD3
    --preset <level>   压缩预设（默认：normal）
                       可选值：fast, normal, max
                       适用于：00_C, 11_C, PGD3 格式
    --offset <x,y>     像素偏移，格式：x,y（默认：0,0）
    --template <path>  PGD3 基准 GE 文件（PGD3 格式时必需）

示例：
  # 启动 GUI
  python main_optimized.py
  
  # 命令行转换：PNG → PGD/11_C（快速预设）
  python main_optimized.py png2pgd --in input.png --out output.pgd --fmt 11_C --preset fast
  
  # 命令行转换：PNG → PGD3（需要模板）
  python main_optimized.py png2pgd --in input.png --out output.pgd --fmt PGD3 --template base.pgd
  
  # 命令行转换：PNG → PGD/00_C（最佳压缩）
  python main_optimized.py png2pgd --in input.png --out output.pgd --fmt 00_C --preset max

依赖：
  必需：numpy, opencv-python, pillow, xxhash
  可选：numba, tqdm, tkinterdnd2
  GPU 加速：cupy-cuda11x 或 cupy-cuda12x
"""

import os
import sys
import time
import threading
import argparse
import struct  # 新增：用于 PGD 类型检测
from typing import Optional, Tuple, List, Callable, Any
from functools import wraps

# --------- 延迟加载配置 ----------
# 全局变量用于存储导入状态
_DND_AVAILABLE = None
_DND_FILES = None
_TkinterDnD = None

_PIL_Image = None

# --------- 延迟加载函数 ----------
def _ensure_tkdnd():
    """确保 tkinterdnd2 已加载"""
    global _DND_AVAILABLE, _DND_FILES, _TkinterDnD
    if _DND_AVAILABLE is None:
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            _DND_AVAILABLE = True
            _DND_FILES = DND_FILES
            _TkinterDnD = TkinterDnD
        except ImportError:
            _DND_AVAILABLE = False
    return _DND_AVAILABLE

def _ensure_pil():
    """确保 Pillow 已加载"""
    global _PIL_Image
    if _PIL_Image is None:
        try:
            from PIL import Image
            _PIL_Image = Image
        except ImportError:
            print("错误: 需要安装 Pillow (pip install pillow)")
            sys.exit(1)
    return _PIL_Image

# --------- GUI 库导入（延迟加载） ----------
# CustomTkinter 在 launch_gui() 中延迟加载以提升启动速度
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, colorchooser
except ImportError:
    print("错误: 需要安装 Tkinter")
    sys.exit(1)

# CustomTkinter 延迟加载变量
_ctk = None
def _ensure_ctk():
    """延迟加载 CustomTkinter"""
    global _ctk
    if _ctk is None:
        try:
            import customtkinter as ctk
            _ctk = ctk
            # 配置主题（仅在首次加载时）
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
        except ImportError as e:
            print(f"错误: 缺少必要的 GUI 库: {e}")
            print("请安装: pip install customtkinter")
            sys.exit(1)
    return _ctk

# --------- 核心模块导入 ----------
# 进度和并行处理工具
try:
    from progress_utils import ProgressCallback, ProgressConfig
    PROGRESS_UTILS_AVAILABLE = True
except ImportError:
    PROGRESS_UTILS_AVAILABLE = False
    print("警告: 未找到 progress_utils.py 模块")

try:
    from parallel_processor import parallel_process_files, ParallelConfig, TaskController as ParallelTaskController
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("警告: 未找到 parallel_processor.py 模块")

# GE 系列
try:
    from pgd2png_ge import pgd2png_batch, add_log_listener as add_pgd_listener, remove_log_listener as remove_pgd_listener
except ImportError:
    pgd2png_batch = None
    def add_pgd_listener(fn): pass
    def remove_pgd_listener(fn): pass
    print("警告: 找不到 pgd2png_ge.py 模块")

try:
    from png2pgd_ge import png2pgd_batch, _parse_rgb_text, add_log_listener as add_png_listener, remove_log_listener as remove_png_listener
except ImportError:
    png2pgd_batch = None
    _parse_rgb_text = lambda s: (255,255,255)
    def add_png_listener(fn): pass
    def remove_png_listener(fn): pass
    print("警告: 找不到 png2pgd_ge.py 模块")

# Others 系列
try:
    import pgd2png_others as _pgd2png_oth
except ImportError:
    _pgd2png_oth = None
    print("警告: 未找到 pgd2png_others.py")

# 类型检测模块（统一接口）
try:
    from pgd_type_detector import detect_pgd_type, is_ge_format, is_others_format, get_format_description
    TYPE_DETECTOR_AVAILABLE = True
except ImportError:
    TYPE_DETECTOR_AVAILABLE = False
    import warnings
    warnings.warn("缺少 pgd_type_detector.py 模块，无法进行类型检测。请确保所有模块完整", ImportWarning)
    sys.exit(1)

try:
    import png2pgd_others as _png2pgd_oth
except ImportError:
    _png2pgd_oth = None
    print("警告: 未找到 png2pgd_others.py")

# 优化模块
try:
    from pgd_optimizer import OptimizedEncoder
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    OptimizedEncoder = None
    print("警告: 未找到 pgd_optimizer.py 模块，将使用标准编码器")

# GPU 加速模块（可选）
try:
    from pgd_gpu_accelerator import GPUEncoder, CUPY_AVAILABLE, GPU_COUNT
    GPU_ACCELERATOR_AVAILABLE = True
except ImportError:
    GPU_ACCELERATOR_AVAILABLE = False
    GPUEncoder = None
    CUPY_AVAILABLE = False
    GPU_COUNT = 0
    print("提示: 未找到 pgd_gpu_accelerator.py 模块，GPU加速被禁用（可选）")

# --------- 优化工具类 ----------
class ThrottledUpdater:
    """限制调用频率的装饰器（默认 200ms）"""
    def __init__(self, min_interval: float = 0.2):
        self.min_interval = min_interval
        self.last_update = 0
        self._lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 快速路径检查（无锁），减少锁竞争
            now = time.time()
            if now - self.last_update < self.min_interval:
                return
            
            with self._lock:
                # 双重检查，避免在获取锁期间已被更新
                now = time.time()
                if now - self.last_update >= self.min_interval:
                    func(*args, **kwargs)
                    self.last_update = now
        return wrapper

class BufferedLogListener:
    """缓冲日志监听器（适配 CustomTkinter）"""
    def __init__(self, text_widget, flush_interval: float = 0.4, max_buffer: int = 50):
        self.text_widget = text_widget
        self.flush_interval = flush_interval
        self.max_buffer = max_buffer
        self.buffer = []
        self.last_flush = time.time()
        self._lock = threading.Lock()
        self._flush_scheduled = False
    
    def __call__(self, line: str):
        """线程安全的日志添加"""
        with self._lock:
            self.buffer.append(line)
            now = time.time()
            
            # 满足任一条件则刷新
            if len(self.buffer) >= self.max_buffer or now - self.last_flush > self.flush_interval:
                self._schedule_flush()
    
    def _schedule_flush(self):
        """在主线程中安排刷新"""
        if not self._flush_scheduled and self.buffer:
            self._flush_scheduled = True
            try:
                self.text_widget.after(10, self._flush)
            except Exception:
                pass
    
    def _flush(self):
        """执行实际刷新（必须在主线程调用，优化性能）"""
        with self._lock:
            if not self.buffer:
                self._flush_scheduled = False
                return
            
            # 使用 join 比循环拼接快
            text_block = "\n".join(self.buffer) + "\n"
            buffer_len = len(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            self._flush_scheduled = False
        
        # 在锁外执行 UI 更新，减少锁持有时间
        try:
            self.text_widget.configure(state="normal")
            # 批量插入比逐行插入快得多
            self.text_widget.insert('end', text_block)
            # 只在缓冲区较大时才滚动到底部，减少滚动开销
            if buffer_len >= 10:
                self.text_widget.see('end')
            self.text_widget.configure(state="disabled")
        except Exception:
            pass

class OptimizedTaskController:
    """优化的任务控制器（使用条件变量）"""
    def __init__(self):
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.state_condition = threading.Condition()
    
    def reset(self):
        with self.state_condition:
            self.pause_event.clear()
            self.stop_event.clear()
            self.state_condition.notify_all()
    
    def pause(self):
        self.pause_event.set()
    
    def resume(self):
        with self.state_condition:
            self.pause_event.clear()
            self.state_condition.notify_all()
    
    def stop(self):
        with self.state_condition:
            self.stop_event.set()
            self.pause_event.clear()
            self.state_condition.notify_all()
    
    def is_paused(self) -> bool:
        return self.pause_event.is_set()
    
    def is_stopped(self) -> bool:
        return self.stop_event.is_set()
    
    def wait_if_paused(self, timeout: float = 0.1):
        """如果暂停则等待（兼容parallel_processor）"""
        with self.state_condition:
            while self.pause_event.is_set() and not self.stop_event.is_set():
                self.state_condition.wait(timeout)
    
    def check_stop(self):
        """检查是否停止（兼容parallel_processor）"""
        if self.stop_event.is_set():
            raise UserStopException("用户停止处理")
    
    def check_state(self):
        """阻塞等待而非忙等待（旧API，与check_stop功能重复）"""
        with self.state_condition:
            while self.pause_event.is_set() and not self.stop_event.is_set():
                self.state_condition.wait(0.1)
            if self.stop_event.is_set():
                raise UserStopException("用户停止处理")

class UserStopException(Exception):
    """用户停止异常"""
    pass

# --------- 辅助函数 ----------
def _rgb_text_from_bgr(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"{r},{g},{b}"

def _hex_from_bgr(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def _set_path_from_dialog(var: tk.StringVar, path: str):
    """从对话框设置路径（处理空值）"""
    if path:
        var.set(os.path.normpath(path))

def _install_drop_target(entry_widget, var: tk.StringVar):
    """安装拖放支持（延迟加载，支持 CustomTkinter）"""
    if not _ensure_tkdnd():
        return
    
    def _drop(event):
        try:
            data = event.data
            if not data:
                return
            # 处理可能的路径列表（使用 tk.splitlist）
            files = entry_widget.tk.splitlist(data)
            if files:
                path = files[0]
                var.set(os.path.normpath(path))
        except Exception:
            pass
    
    try:
        # 获取 CustomTkinter Entry 的内部 tk Entry widget
        if hasattr(entry_widget, '_entry'):
            tk_entry = entry_widget._entry
        else:
            tk_entry = entry_widget
        
        # 注册拖放目标
        if hasattr(tk_entry, 'drop_target_register'):
            tk_entry.drop_target_register(_DND_FILES)
            tk_entry.dnd_bind('<<Drop>>', _drop)
    except Exception:
        # 静默失败，不影响主程序
        pass

def _find_files_fast(path: str, exts: Tuple[str, ...], recursive: bool) -> List[str]:
    """
    使用 os.scandir 的高效文件查找（比 os.walk 快 2-3 倍）
    """
    if os.path.isfile(path):
        return [path] if path.lower().endswith(exts) else []
    
    if not os.path.isdir(path):
        return []
    
    results = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file() and entry.name.lower().endswith(exts):
                        results.append(entry.path)
                    elif recursive and entry.is_dir():
                        # 递归处理子目录
                        results.extend(_find_files_fast(entry.path, exts, recursive))
                except OSError:
                    # 跳过无法访问的条目
                    continue
    except (PermissionError, OSError):
        # 处理无权限访问的目录
        pass
    
    return sorted(results)

# 使用统一的类型检测接口（pgd_type_detector.py）
_detect_pgd_type = detect_pgd_type

def format_speed(speed: float) -> str:
    """格式化速度显示"""
    if speed < 1024:
        return f"{speed:.1f} B/s"
    elif speed < 1024 * 1024:
        return f"{speed/1024:.1f} KB/s"
    else:
        return f"{speed/1024/1024:.1f} MB/s"

# --------- Others 系列包装函数 ----------
def others_pgd2png_batch(in_path: str, out_path: Optional[str] = None, recursive: bool = False,
                         file_progress_cb: Optional[Callable[[int, int], None]] = None,
                         batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                         log_cb: Optional[Callable[[str], None]] = None):
    """PGD → PNG（Others 格式）"""
    if _pgd2png_oth is None:
        raise RuntimeError("未找到 pgd2png_others.py")
    
    files = _find_files_fast(in_path, (".pgd",), recursive)
    if not files:
        raise ValueError(f"未找到 PGD 文件：{in_path}")
    
    is_batch = len(files) > 1 or os.path.isdir(in_path)
    if is_batch and not out_path:
        raise ValueError("批量处理时必须指定输出文件夹")
    
    total = len(files)
    t0 = time.time()
    if batch_progress_cb:
        batch_progress_cb(0, total, 0.0)
    
    for i, src in enumerate(files, 1):
        file_start_time = time.time()
        file_size = os.path.getsize(src)
        try:
            if file_progress_cb:
                file_progress_cb(0, 1)
            
            if is_batch:
                assert out_path is not None, "批量处理时必须指定输出文件夹"
                os.makedirs(out_path, exist_ok=True)
                dst = os.path.join(out_path, os.path.splitext(os.path.basename(src))[0] + ".png")
            else:
                dst = out_path or os.path.splitext(src)[0] + ".png"
            
            _pgd2png_oth.pgd_to_png(src, dst)
            output_size = os.path.getsize(dst)
            
            if log_cb:
                pgd_type = _detect_pgd_type(src)
                elapsed = time.time() - file_start_time
                compression_ratio = output_size / file_size * 100 if file_size > 0 else 0
                log_cb(f"OK 导出 PNG：{dst} (类型: {pgd_type}, 用时: {elapsed:.2f}s, 压缩率: {compression_ratio:.1f}%)")
            if file_progress_cb:
                file_progress_cb(1, 1)
        except Exception as e:
            if log_cb:
                log_cb(f"ERROR 处理失败 {src}: {e}")
        
        if batch_progress_cb:
            batch_progress_cb(i, total, time.time() - t0)

def others_png2pgd_batch(in_path: str, target_fmt: str, out_path: Optional[str] = None, tmpl_path: Optional[str] = None,
                         base_ge_path: Optional[str] = None, use_template_for_base: bool = True,
                         preset: str = "normal", fill_bgr: Tuple[int, int, int] = (255, 255, 255), recursive: bool = False,
                         file_progress_cb: Optional[Callable[[int, int], None]] = None,
                         batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                         log_cb: Optional[Callable[[str], None]] = None,
                         quality_level: int = 2,
                         use_gpu: bool = False):
    """PNG → PGD（Others 格式）"""
    if _png2pgd_oth is None:
        raise RuntimeError("未找到 png2pgd_others.py")
    
    Image = _ensure_pil()
    files = _find_files_fast(in_path, (".png",), recursive)
    
    if not files:
        raise ValueError(f"未找到 PNG 文件：{in_path}")
    
    is_batch = len(files) > 1 or os.path.isdir(in_path)
    if is_batch and not out_path:
        raise ValueError("批量处理时必须指定输出文件夹")
    
    total = len(files)
    t0 = time.time()
    if batch_progress_cb:
        batch_progress_cb(0, total, 0.0)
    
    for i, src in enumerate(files, 1):
        file_start_time = time.time()
        file_size = os.path.getsize(src)
        try:
            if file_progress_cb:
                file_progress_cb(0, 1)
            if is_batch:
                if out_path is None:
                    raise ValueError("批量处理时必须指定输出文件夹")
                os.makedirs(out_path, exist_ok=True)
                # 默认使用小写 .pgd 扩展名
                dst = os.path.join(out_path, os.path.splitext(os.path.basename(src))[0] + ".pgd")
            else:
                dst = out_path or os.path.splitext(src)[0] + ".pgd"
            
            # 确定格式
            current_fmt = target_fmt
            actual_tmpl_file = None  # 用于保存实际找到的模板文件路径
            output_ext = ".pgd"  # 默认输出扩展名
            
            # 如果指定了模板，从模板获取类型
            if tmpl_path and current_fmt == '自动':
                actual_tmpl_for_detect = tmpl_path
                # 如果模板路径是目录，在其中查找与PNG同名的模板文件
                if os.path.isdir(tmpl_path):
                    png_stem = os.path.splitext(os.path.basename(src))[0]
                    cand1 = os.path.join(tmpl_path, png_stem + '.pgd3')
                    cand2 = os.path.join(tmpl_path, png_stem + '.pgd')
                    cand3 = os.path.join(tmpl_path, png_stem + '.PGD')
                    if os.path.isfile(cand1):
                        actual_tmpl_for_detect = cand1
                    elif os.path.isfile(cand2):
                        actual_tmpl_for_detect = cand2
                    elif os.path.isfile(cand3):
                        actual_tmpl_for_detect = cand3
                    else:
                        raise FileNotFoundError(f"在模板目录 {tmpl_path} 中未找到 {png_stem}.pgd3/.pgd/.PGD")
                
                if os.path.isfile(actual_tmpl_for_detect):
                    template_type = _detect_pgd_type(actual_tmpl_for_detect)
                    current_fmt = template_type
                    actual_tmpl_file = actual_tmpl_for_detect  # 保存实际模板文件路径
                    # 从模板文件名获取扩展名（保持大小写）
                    output_ext = os.path.splitext(actual_tmpl_for_detect)[1]
                    if not output_ext:  # 如果没有扩展名，使用默认
                        output_ext = ".pgd"
                    # 更新输出路径以使用正确的扩展名
                    if is_batch:
                        if out_path is None:
                            raise ValueError("批量处理时必须指定输出文件夹")
                        dst = os.path.join(out_path, os.path.splitext(os.path.basename(src))[0] + output_ext)
                    else:
                        dst = out_path or os.path.splitext(src)[0] + output_ext
                    if log_cb:
                        log_cb(f"从模板自动识别 PGD 类型: {current_fmt}")
                else:
                    raise ValueError(f"模板文件不存在: {actual_tmpl_for_detect}")
            elif current_fmt == '自动':
                raise ValueError("自动识别 PGD 类型需要指定模板文件或文件夹")
            
            # 写不同格式
            pgd_type_str = current_fmt
            if current_fmt == "11_C":
                _png2pgd_oth._write_11c_from_png(src, dst, (0,0), preset=preset)
            elif current_fmt == "00_C":
                _png2pgd_oth._write_00c_from_png(src, dst, (0,0), preset=preset)
            elif current_fmt == "TGA":
                _png2pgd_oth._write_pgd_tga_from_png(src, dst, (0,0), preset=preset)
            elif current_fmt == "PGD3":
                base_ge = base_ge_path
                # 在自动模式下，使用之前找到的模板文件
                if actual_tmpl_file:
                    # 从自动识别阶段保存的模板文件中读取基准 PGD 名称
                    pgd3_info = _png2pgd_oth.read_pgd3_header(actual_tmpl_file)
                    base_name = pgd3_info['basename']
                    # 在模板 PGD 所在目录查找基准 GE
                    base_dir = os.path.dirname(os.path.abspath(actual_tmpl_file))
                    base_ge = os.path.join(base_dir, base_name)
                    if log_cb:
                        log_cb(f"从模板读取基准 PGD: {base_name}，在目录 {base_dir} 中查找")
                elif use_template_for_base and tmpl_path:
                    # 非自动模式，按照原逻辑处理
                    # 如果模板路径是目录，则在该目录中查找与PNG同名的模板文件
                    actual_tmpl_path = tmpl_path
                    if os.path.isdir(tmpl_path):
                        png_stem = os.path.splitext(os.path.basename(src))[0]
                        # 尝试 .pgd3 和 .pgd 扩展名
                        cand1 = os.path.join(tmpl_path, png_stem + '.pgd3')
                        cand2 = os.path.join(tmpl_path, png_stem + '.pgd')
                        if os.path.isfile(cand1):
                            actual_tmpl_path = cand1
                            if log_cb:
                                log_cb(f"在模板目录中找到: {os.path.basename(actual_tmpl_path)}")
                        elif os.path.isfile(cand2):
                            actual_tmpl_path = cand2
                            if log_cb:
                                log_cb(f"在模板目录中找到: {os.path.basename(actual_tmpl_path)}")
                        else:
                            raise FileNotFoundError(f"在模板目录 {tmpl_path} 中未找到 {png_stem}.pgd3 或 {png_stem}.pgd")
                    elif not os.path.isfile(actual_tmpl_path):
                        raise ValueError(f"模板 PGD 文件不存在：{actual_tmpl_path}")
                    
                    # 从模板读取基准 PGD 名称
                    pgd3_info = _png2pgd_oth.read_pgd3_header(actual_tmpl_path)
                    base_name = pgd3_info['basename']
                    # 在模板 PGD 所在目录查找基准 GE
                    base_dir = os.path.dirname(os.path.abspath(actual_tmpl_path))
                    base_ge = os.path.join(base_dir, base_name)
                    if log_cb:
                        log_cb(f"从模板读取基准 PGD: {base_name}，在目录 {base_dir} 中查找")
                
                if not base_ge or not os.path.isfile(base_ge):
                    raise ValueError(f"PGD3 需要基准 PGD/GE 文件: {base_ge}")
                
                _png2pgd_oth.png_to_pgd3(src, base_ge=base_ge, out_path=dst, preset=preset)
            elif current_fmt == "GE":
                # 交由 png2pgd_ge 处理
                if png2pgd_batch is None:
                    raise RuntimeError("未找到 png2pgd_ge.py 模块")
                
                # 使用 GE 模块处理单个文件
                # 从模板获取压缩类型（默认使用类型3）
                ge_ctype = 3
                ge_template = None
                
                # 优先使用自动模式下找到的模板文件
                if actual_tmpl_file:
                    ge_template = actual_tmpl_file
                elif tmpl_path and os.path.isfile(tmpl_path):
                    ge_template = tmpl_path
                
                if ge_template:
                    try:
                        # 尝试从模板读取压缩类型
                        from png2pgd_ge import read_pgd_header
                        hdr = read_pgd_header(ge_template)
                        ge_ctype = hdr.compr_method
                        if log_cb:
                            log_cb(f"从模板读取 GE 压缩类型: {ge_ctype}")
                    except Exception as e:
                        if log_cb:
                            log_cb(f"WARNING: 无法读取模板头信息: {e}，使用默认类型 3")
                
                # 调用 GE 的单文件处理函数
                from png2pgd_ge import png2pgd_single
                png2pgd_single(src, ge_ctype, dst, ge_template, preset, fill_bgr,
                              quality_level=quality_level, use_gpu=use_gpu)
                pgd_type_str = f"GE (类型 {ge_ctype})"
            else:
                raise ValueError(f"未知 Others 目标格式：{current_fmt}")
            
            output_size = os.path.getsize(dst)
            
            if log_cb:
                elapsed = time.time() - file_start_time
                compression_ratio = output_size / file_size * 100 if file_size > 0 else 0
                log_cb(f"OK 生成 PGD：{dst} (类型: {pgd_type_str}, 用时: {elapsed:.2f}s, 压缩率: {compression_ratio:.1f}%)")
            if file_progress_cb:
                file_progress_cb(1, 1)
        except Exception as e:
            if log_cb:
                log_cb(f"ERROR 处理失败 {src}: {e}")
        
        if batch_progress_cb:
            batch_progress_cb(i, total, time.time() - t0)

# --------- GUI 主程序 ----------
def _build_init_log_text():
    """延迟构建初始日志文本（避免启动时执行，进一步优化）"""
    # 使用更简洁的初始文本，减少字符串拼接开销
    base_text = """================================================================================
        Softpal PGD Toolkit — PNG ⇔ PGD 互转工具        
================================================================================

【支持的格式】
  ■ GE 系列：类型 1/2/3 压缩
  ■ Others 系列：00_C、11_C、TGA、PGD3

【核心功能】
  ✓ 自动类型检测  ✓ 双向转换  ✓ 批量处理  ✓ 拖放支持  ✓ 实时进度"""
    
    # 动态添加 GPU 信息（延迟评估）
    gpu_info = f"  ✓ GPU 加速：检测到 {GPU_COUNT} 个 CUDA 设备可用" if CUPY_AVAILABLE else "  • GPU 加速：未检测到 GPU（仅 CPU 模式）"
    
    return f"""{base_text}
{gpu_info}

【优化特性】
  • 并行处理  • 内存优化  • 进度节流  • 现代化 UI

【使用步骤】
  1. 选择转换模式（PGD → PNG 或 PNG → PGD）
  2. 选择 PGD 类型（推荐'自动'）
  3. 配置参数并选择文件
  4. 点击'开始处理'

【提示】拖放支持 | 主题切换 | 日志级别控制

================================================================================
>> 等待用户操作...

"""

def launch_gui():
    """启动现代化 GUI 界面（CustomTkinter）"""
    # 延迟加载 CustomTkinter（提升启动速度）
    ctk = _ensure_ctk()
    
    # 创建 CustomTkinter 主窗口（集成 TkinterDnD 支持）
    if _ensure_tkdnd():
        # 继承 ctk.CTk 和 DnDWrapper 以支持拖放
        class ModernApp(ctk.CTk, _TkinterDnD.DnDWrapper):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.TkdndVersion = _TkinterDnD._require(self)  # type: ignore[attr-defined]
        root = ModernApp()
    else:
        root = ctk.CTk()
    
    root.title("🎨 Softpal PGD Toolkit")
    root.geometry("1100x900")
    root.resizable(True, True)
    
    # 设置窗口最小尺寸
    root.minsize(900, 700)
    
    # 设置窗口图标（如果有的话）
    try:
        # 可以添加图标设置
        pass
    except:
        pass

    # 批量创建状态变量（减少调用开销）
    vars_dict = {
        'mode': tk.StringVar(value='pgd2png'),
        'pgd_type': tk.StringVar(value='自动'),
        'ctype': tk.StringVar(value='3'),
        'preset': tk.StringVar(value='max'),  # 默认max压缩预设
        'in_path': tk.StringVar(),
        'out_path': tk.StringVar(),
        'tmpl_path': tk.StringVar(),
        'base_ge_path': tk.StringVar(),
        'fill_color_text': tk.StringVar(value="255,255,255"),
        'recursive_var': tk.BooleanVar(value=False),
        'log_level_var': tk.StringVar(value="普通"),
        'use_template_for_base': tk.BooleanVar(value=True),
        'quality_level': tk.StringVar(value='3 (best)'),  # 默认3 (best)质量级别
        'use_gpu': tk.BooleanVar(value=CUPY_AVAILABLE and GPU_COUNT > 0)  # 如果GPU可用则默认启用
    }
    
    # 解包变量（便于后续使用）
    mode = vars_dict['mode']
    pgd_type = vars_dict['pgd_type']
    ctype = vars_dict['ctype']
    preset = vars_dict['preset']
    in_path = vars_dict['in_path']
    out_path = vars_dict['out_path']
    tmpl_path = vars_dict['tmpl_path']
    base_ge_path = vars_dict['base_ge_path']
    fill_color_text = vars_dict['fill_color_text']
    recursive_var = vars_dict['recursive_var']
    log_level_var = vars_dict['log_level_var']
    use_template_for_base = vars_dict['use_template_for_base']
    quality_level = vars_dict['quality_level']
    use_gpu = vars_dict['use_gpu']
    
    # 任务控制器（优化版）
    task_controller = OptimizedTaskController()
    
    # 主容器框架
    main_container = ctk.CTkFrame(root, fg_color="transparent")
    main_container.pack(fill='both', expand=True, padx=15, pady=15)
    main_container.grid_columnconfigure(0, weight=3)  # 左侧控制面板（更宽）
    main_container.grid_columnconfigure(1, weight=2)  # 右侧日志区域
    main_container.grid_rowconfigure(1, weight=1)
    
    # 顶部标题栏（横跨两列）
    title_frame = ctk.CTkFrame(main_container, fg_color=("#E3F2FD", "#1a1a2e"), corner_radius=12, border_width=2, border_color=("#90CAF9", "#42A5F5"))
    title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 15))
    
    title_label = ctk.CTkLabel(
        title_frame,
        text="🎨 Softpal PGD Toolkit - PNG ⇔ PGD 互转工具",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color=("#1565C0", "#64B5F6")
    )
    title_label.pack(pady=12)
    
    # 左侧：主工作区域（控制面板）- 分为上下两部分
    left_container = ctk.CTkFrame(main_container, fg_color="transparent")
    left_container.grid(row=1, column=0, sticky="nsew", padx=(5, 8))
    left_container.grid_rowconfigure(0, weight=1)  # 设置区可滚动
    left_container.grid_rowconfigure(1, weight=0)  # 按钮区固定
    left_container.grid_columnconfigure(0, weight=1)
    
    # 上部：可滚动的设置区域
    work_frame = ctk.CTkScrollableFrame(left_container, corner_radius=12, border_width=1, border_color=("#BDBDBD", "#424242"))
    work_frame.grid(row=0, column=0, sticky="nsew")
    work_frame.grid_columnconfigure(1, weight=1)
    
    # 创建主题切换按钮（右上角，增强视觉效果）
    def toggle_theme():
        current = ctk.get_appearance_mode()
        new_mode = "Dark" if current == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
        theme_btn.configure(text="🌞" if new_mode == "Dark" else "🌙")
    
    theme_btn = ctk.CTkButton(
        title_frame,
        text="🌙",
        width=45,
        height=35,
        corner_radius=20,
        font=ctk.CTkFont(size=18),
        fg_color=("#FF9800", "#FFB74D"),
        hover_color=("#F57C00", "#FFA726"),
        command=toggle_theme
    )
    theme_btn.place(relx=0.97, rely=0.5, anchor="e")
    
    # 第1行：模式选择（改为独立两行布局）
    row = 0
    mode_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    mode_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(10, 5), padx=10)
    
    ctk.CTkLabel(mode_label_frame, text="🔄 转换模式", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    row += 1
    mode_segment_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    mode_segment_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 8), padx=10)
    
    mode_segment = ctk.CTkSegmentedButton(
        mode_segment_frame,
        values=['PGD → PNG', 'PNG → PGD'],
        command=lambda v: mode.set('pgd2png' if v == 'PGD → PNG' else 'png2pgd'),
        font=ctk.CTkFont(size=13)
    )
    mode_segment.pack(fill='x', padx=5)
    mode_segment.set('PGD → PNG')
    
    # 同步 mode 变量到 segment button
    def sync_mode_display(*args):
        current_mode = mode.get()
        if current_mode == 'pgd2png':
            mode_segment.set('PGD → PNG')
        else:
            mode_segment.set('PNG → PGD')
    
    mode.trace_add('write', sync_mode_display)
    
    # PGD类型选择（独立一行）
    row += 1
    pgd_type_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    pgd_type_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=5, padx=10)
    
    ctk.CTkLabel(pgd_type_frame, text="📄 PGD类型", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left', padx=(0, 10))
    cb_pgd_type = ctk.CTkComboBox(
        pgd_type_frame,
        variable=pgd_type,
        values=['自动', 'GE', '11_C', '00_C', 'TGA', 'PGD3'],
        width=150,
        state='readonly',
        font=ctk.CTkFont(size=13)
    )
    cb_pgd_type.pack(side='left', fill='x', expand=True, padx=5)
    
    # 第2行：压缩设置（独立两行）
    row += 1
    compress_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    compress_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    ctk.CTkLabel(compress_label_frame, text="⚙️ 压缩设置", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    row += 1
    compress_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    compress_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    compress_frame.grid_columnconfigure(1, weight=1)
    compress_frame.grid_columnconfigure(3, weight=1)
    
    ctk.CTkLabel(compress_frame, text="类型:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky='w', padx=(0, 5))
    cb_ctype = ctk.CTkComboBox(
        compress_frame,
        variable=ctype,
        values=['1', '2', '3'],
        width=80,
        state='disabled',
        font=ctk.CTkFont(size=12)
    )
    cb_ctype.grid(row=0, column=1, sticky='ew', padx=(0, 15))
    
    ctk.CTkLabel(compress_frame, text="预设:", font=ctk.CTkFont(size=12)).grid(row=0, column=2, sticky='w', padx=(0, 5))
    cb_preset = ctk.CTkComboBox(
        compress_frame,
        variable=preset,
        values=['fast', 'normal', 'max'],
        width=120,
        state='readonly',
        font=ctk.CTkFont(size=12)
    )
    cb_preset.grid(row=0, column=3, sticky='ew')
    
    # 第3行：质量级别和GPU加速（独立两行）
    row += 1
    quality_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    quality_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    ctk.CTkLabel(quality_label_frame, text="💪 优化选项", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    row += 1
    quality_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    quality_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    quality_frame.grid_columnconfigure(1, weight=1)
    
    ctk.CTkLabel(quality_frame, text="质量:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky='w', padx=(0, 5))
    quality_values = ['1 (fast)', '2 (balanced)', '3 (best)'] if OPTIMIZER_AVAILABLE else ['2 (std)']
    cb_quality = ctk.CTkComboBox(
        quality_frame,
        variable=quality_level,
        values=quality_values,
        width=140,
        state='readonly' if OPTIMIZER_AVAILABLE else 'disabled',
        font=ctk.CTkFont(size=12)
    )
    cb_quality.grid(row=0, column=1, sticky='ew', padx=(0, 8))
    
    # 优化模块状态指示
    opt_status_text = "✓ 3阶段" if OPTIMIZER_AVAILABLE else "✗ 未启用"
    opt_status_color = ("#4CAF50", "#66BB6A") if OPTIMIZER_AVAILABLE else ("gray", "gray")
    ctk.CTkLabel(
        quality_frame,
        text=opt_status_text,
        text_color=opt_status_color,
        font=ctk.CTkFont(size=11)
    ).grid(row=0, column=2, sticky='w', padx=(0, 15))
    
    # GPU 加速选项
    cb_gpu = ctk.CTkCheckBox(
        quality_frame,
        text="🚀 GPU加速",
        variable=use_gpu,
        font=ctk.CTkFont(size=12, weight="bold")
    )
    cb_gpu.grid(row=0, column=3, sticky='w', padx=(0, 8))
    if not CUPY_AVAILABLE:
        cb_gpu.configure(state='disabled')
        use_gpu.set(False)
    
    # GPU 状态显示
    gpu_status_text = f"✓ {GPU_COUNT}GPU" if CUPY_AVAILABLE else "✗ 不可用"
    gpu_status_color = ("#4CAF50", "#66BB6A") if CUPY_AVAILABLE else ("gray", "gray")
    ctk.CTkLabel(
        quality_frame,
        text=gpu_status_text,
        text_color=gpu_status_color,
        font=ctk.CTkFont(size=11)
    ).grid(row=0, column=4, sticky='w')
    
    # 第4行：递归选项
    row += 1
    option_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    option_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=5, padx=10)
    
    ctk.CTkCheckBox(
        option_frame,
        text="📂 递归处理子文件夹",
        variable=recursive_var,
        font=ctk.CTkFont(size=13)
    ).pack(side='left')
    
    # 第5行：输入路径（优化布局）
    row += 1
    input_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    input_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    ctk.CTkLabel(input_label_frame, text="📥 输入路径", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    row += 1
    path_frame_in = ctk.CTkFrame(work_frame, fg_color="transparent")
    path_frame_in.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    path_frame_in.grid_columnconfigure(0, weight=1)
    
    ent_in = ctk.CTkEntry(path_frame_in, textvariable=in_path)
    ent_in.grid(row=0, column=0, sticky='ew', padx=(0, 8))
    
    btn_in_file = ctk.CTkButton(path_frame_in, text="📄 文件", width=70,
                                command=lambda: _set_path_from_dialog(in_path, filedialog.askopenfilename(
                                    filetypes=(('Images/PGD', '*.png;*.pgd'), ('All', '*.*')))))
    btn_in_file.grid(row=0, column=1, padx=(0, 4))
    
    btn_in_dir = ctk.CTkButton(path_frame_in, text="📁 文件夹", width=80,
                               command=lambda: _set_path_from_dialog(in_path, filedialog.askdirectory()))
    btn_in_dir.grid(row=0, column=2)
    
    # 第6行：输出路径（优化布局）
    row += 1
    output_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    output_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    ctk.CTkLabel(output_label_frame, text="📤 输出路径", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    row += 1
    path_frame_out = ctk.CTkFrame(work_frame, fg_color="transparent")
    path_frame_out.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    path_frame_out.grid_columnconfigure(0, weight=1)
    
    ent_out = ctk.CTkEntry(path_frame_out, textvariable=out_path)
    ent_out.grid(row=0, column=0, sticky='ew', padx=(0, 8))
    
    btn_out_file = ctk.CTkButton(path_frame_out, text="📄 文件", width=70,
                                 command=lambda: _set_path_from_dialog(out_path, filedialog.asksaveasfilename(
                                     defaultextension='', filetypes=(('PNG/PGD', '*.png;*.pgd'), ('All', '*.*')))))
    btn_out_file.grid(row=0, column=1, padx=(0, 4))
    
    btn_out_dir = ctk.CTkButton(path_frame_out, text="📁 文件夹", width=80,
                                command=lambda: _set_path_from_dialog(out_path, filedialog.askdirectory()))
    btn_out_dir.grid(row=0, column=2)
    
    # 第7行：模板 PGD（优化布局）
    row += 1
    tmpl_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    tmpl_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    lbl_tmpl = ctk.CTkLabel(tmpl_label_frame, text="📋 模板 PGD", font=ctk.CTkFont(size=14, weight="bold"),
                          text_color=("gray", "gray"))
    lbl_tmpl.pack(side='left')
    
    row += 1
    path_frame_tmpl = ctk.CTkFrame(work_frame, fg_color="transparent")
    path_frame_tmpl.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    path_frame_tmpl.grid_columnconfigure(0, weight=1)
    
    ent_tmpl = ctk.CTkEntry(path_frame_tmpl, textvariable=tmpl_path)
    ent_tmpl.grid(row=0, column=0, sticky='ew', padx=(0, 8))
    
    btn_tmpl_file = ctk.CTkButton(path_frame_tmpl, text="📄 文件", width=70,
                                  command=lambda: _set_path_from_dialog(tmpl_path, filedialog.askopenfilename(
                                      filetypes=(('PGD', '*.pgd;*.pgd3'), ('All', '*.*')))))
    btn_tmpl_file.grid(row=0, column=1, padx=(0, 4))
    
    btn_tmpl_dir = ctk.CTkButton(path_frame_tmpl, text="📁 文件夹", width=80,
                                 command=lambda: _set_path_from_dialog(tmpl_path, filedialog.askdirectory()))
    btn_tmpl_dir.grid(row=0, column=2)
    
    # 第8行：基准 PGD/GE 文件（优化布局）
    row += 1
    base_label_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    base_label_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(8, 3), padx=10)
    lbl_base_ge = ctk.CTkLabel(base_label_frame, text="🎯 基准 PGD/GE", font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=("gray", "gray"))
    lbl_base_ge.pack(side='left')
    
    row += 1
    path_frame_base = ctk.CTkFrame(work_frame, fg_color="transparent")
    path_frame_base.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(0, 5), padx=10)
    path_frame_base.grid_columnconfigure(0, weight=1)
    
    ent_base_ge = ctk.CTkEntry(path_frame_base, textvariable=base_ge_path)
    ent_base_ge.grid(row=0, column=0, sticky='ew', padx=(0, 8))
    
    btn_base_ge_file = ctk.CTkButton(path_frame_base, text="📄 文件", width=70,
                                     command=lambda: _set_path_from_dialog(base_ge_path, filedialog.askopenfilename(
                                         filetypes=(('PGD', '*.pgd'), ('All', '*.*')))))
    btn_base_ge_file.grid(row=0, column=1)
    
    # 第9行：PGD3选项
    row += 1
    pgd3_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    pgd3_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=5, padx=10)
    
    ctk.CTkLabel(pgd3_frame, text="⚙️ PGD3选项", font=ctk.CTkFont(size=13, weight="bold")).pack(side='left', padx=(0, 15))
    
    pgd3_opts_frame = ctk.CTkFrame(pgd3_frame, fg_color="transparent")
    pgd3_opts_frame.pack(side='left')
    
    ctk.CTkRadioButton(pgd3_opts_frame, text='使用模板自动查找基准', variable=use_template_for_base, value=True,
                      font=ctk.CTkFont(size=12)).pack(side='left', padx=5)
    ctk.CTkRadioButton(pgd3_opts_frame, text='手动指定基准', variable=use_template_for_base, value=False,
                      font=ctk.CTkFont(size=12)).pack(side='left', padx=5)
    
    # 第10行：透明填充色
    row += 1
    fill_frame = ctk.CTkFrame(work_frame, fg_color="transparent")
    fill_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=5, padx=10)
    
    lbl_fill = ctk.CTkLabel(fill_frame, text="🎨 透明填充色", font=ctk.CTkFont(size=13, weight="bold"))
    lbl_fill.pack(side='left', padx=(0, 10))
    
    ent_fill = ctk.CTkEntry(fill_frame, textvariable=fill_color_text, width=120)
    ent_fill.pack(side='left', padx=5)
    
    fill_preview = tk.Canvas(fill_frame, width=28, height=22, highlightthickness=1, highlightbackground="#ccc")
    fill_preview.pack(side='left', padx=5)
    
    def _update_fill_preview():
        nonlocal fill_bgr_current
        try:
            fill_bgr_current = _parse_rgb_text(fill_color_text.get())
            hexv = _hex_from_bgr(fill_bgr_current)
            fill_preview.delete("all")
            fill_preview.create_rectangle(2, 2, 26, 20, outline="", fill=hexv)
        except Exception:
            fill_preview.delete("all")
            for i in range(3):
                for j in range(2):
                    fill_preview.create_rectangle(2 + i * 8, 2 + j * 8, 10 + i * 8, 10 + j * 8,
                                                  outline="#ddd", fill="#eee" if (i + j) % 2 == 0 else "#ddd")
    
    def pick_color():
        nonlocal fill_bgr_current
        color = colorchooser.askcolor(title="选择填充色", initialcolor=f"#{fill_bgr_current[2]:02x}{fill_bgr_current[1]:02x}{fill_bgr_current[0]:02x}")
        if color[0]:
            r, g, b = [int(round(c)) for c in color[0]]
            fill_color_text.set(f"{r},{g},{b}")
    
    btn_color = ctk.CTkButton(fill_frame, text="🎨 选色", width=80, command=pick_color)
    btn_color.pack(side='left', padx=5)
    
    fill_bgr_current = (255, 255, 255)
    _update_fill_preview()
    fill_color_text.trace_add('write', lambda *_: _update_fill_preview())
    
    # 右侧：日志区域（分为上下两部分）
    log_container = ctk.CTkFrame(main_container, fg_color="transparent")
    log_container.grid(row=1, column=1, sticky="nsew", padx=(8, 5))
    log_container.grid_rowconfigure(0, weight=1)  # 日志框可扩展
    log_container.grid_rowconfigure(1, weight=0)  # 进度区固定
    log_container.grid_columnconfigure(0, weight=1)
    
    # 上部：日志框
    log_frame_top = ctk.CTkFrame(log_container, corner_radius=12, border_width=1, border_color=("#BDBDBD", "#424242"))
    log_frame_top.grid(row=0, column=0, sticky="nsew")
    log_frame_top.grid_rowconfigure(1, weight=1)
    log_frame_top.grid_columnconfigure(0, weight=1)
    
    # 日志头部
    log_header = ctk.CTkFrame(log_frame_top, fg_color="transparent")
    log_header.grid(row=0, column=0, sticky='ew', padx=12, pady=(12, 8))
    
    ctk.CTkLabel(log_header, text="📝 处理日志", font=ctk.CTkFont(size=14, weight="bold")).pack(side='left')
    
    # 日志控制按钮
    log_control_frame = ctk.CTkFrame(log_header, fg_color="transparent")
    log_control_frame.pack(side='right')
    
    ctk.CTkLabel(log_control_frame, text="📄 级别:", font=ctk.CTkFont(size=11)).pack(side='left', padx=(0, 5))
    cb_loglevel = ctk.CTkComboBox(log_control_frame, variable=log_level_var, values=['简洁', '普通', '详细'], 
                                 width=90, state='readonly', font=ctk.CTkFont(size=11))
    cb_loglevel.pack(side='left', padx=3)
    
    btn_clear = ctk.CTkButton(log_control_frame, text='🗑️ 清空', width=80, height=28,
                             corner_radius=6,
                             fg_color=("#757575", "#616161"),
                             hover_color=("#616161", "#424242"),
                             font=ctk.CTkFont(size=11),
                             command=lambda: (txt_log.configure(state="normal"), txt_log.delete('1.0', 'end'), txt_log.configure(state="disabled")))
    btn_clear.pack(side='left', padx=(8, 0))
    
    # 日志文本框
    txt_log = ctk.CTkTextbox(log_frame_top, wrap='word', 
                             font=ctk.CTkFont(family="Consolas", size=11),
                             corner_radius=8)
    txt_log.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 12))
    
    # 延迟插入初始文本（在窗口显示后异步加载，提升启动速度）
    def _load_init_text():
        try:
            txt_log.configure(state="normal")
            txt_log.insert('end', _build_init_log_text())
            txt_log.configure(state="disabled")
        except Exception:
            pass
    
    # 使用 after_idle 在窗口完全显示后再加载文本
    root.after_idle(_load_init_text)
    
    # 下部：进度和统计信息区（固定）
    progress_container = ctk.CTkFrame(log_container, corner_radius=10, fg_color=("#F5F5F5", "#2b2b2b"))
    progress_container.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    progress_container.grid_columnconfigure(0, weight=1)
    
    # 统计信息
    stats_frame = ctk.CTkFrame(progress_container, fg_color=("#E8F5E9", "#1B5E20"), corner_radius=8)
    stats_frame.grid(row=0, column=0, sticky='ew', pady=(12, 8), padx=12)
    
    ctk.CTkLabel(stats_frame, text="📊 统计信息", font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("#2E7D32", "#81C784")).pack(side='left', padx=12, pady=8)
    lbl_total_stats = ctk.CTkLabel(stats_frame, text="总文件: 0 | 成功: 0 | 失败: 0", 
                                   text_color=("#1B5E20", "#A5D6A7"), font=ctk.CTkFont(size=12, weight="bold"))
    lbl_total_stats.pack(side='left', padx=(15, 12), pady=8)
    
    # 单文件进度
    file_pb_frame = ctk.CTkFrame(progress_container, fg_color="transparent")
    file_pb_frame.grid(row=1, column=0, sticky='ew', pady=(0, 5), padx=12)
    file_pb_frame.grid_columnconfigure(0, weight=1)
    
    ctk.CTkLabel(file_pb_frame, text="📊 单文件进度", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky='w', pady=(0, 4))
    
    pb_file = ctk.CTkProgressBar(file_pb_frame, mode='determinate', height=16, corner_radius=8,
                                 progress_color=("#4CAF50", "#66BB6A"))
    pb_file.grid(row=1, column=0, sticky='ew', pady=(0, 4))
    pb_file.set(0)
    
    file_info_frame = ctk.CTkFrame(file_pb_frame, fg_color="transparent")
    file_info_frame.grid(row=2, column=0, sticky='ew')
    file_info_frame.grid_columnconfigure(1, weight=1)
    
    lbl_file_pct = ctk.CTkLabel(file_info_frame, text="0%", width=50, font=ctk.CTkFont(size=11, weight="bold"),
                                text_color=("#2E7D32", "#81C784"))
    lbl_file_pct.grid(row=0, column=0, sticky='w')
    lbl_current_file = ctk.CTkLabel(file_info_frame, text="等待处理...", anchor='w', 
                                    font=ctk.CTkFont(size=10), text_color=("#616161", "#BDBDBD"))
    lbl_current_file.grid(row=0, column=1, sticky='ew', padx=(10, 0))
    
    # 批量进度
    batch_pb_frame = ctk.CTkFrame(progress_container, fg_color="transparent")
    batch_pb_frame.grid(row=2, column=0, sticky='ew', pady=(5, 12), padx=12)
    batch_pb_frame.grid_columnconfigure(0, weight=1)
    
    ctk.CTkLabel(batch_pb_frame, text="📈 批量进度", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky='w', pady=(0, 4))
    
    pb_batch = ctk.CTkProgressBar(batch_pb_frame, mode='determinate', height=16, corner_radius=8,
                                  progress_color=("#2196F3", "#64B5F6"))
    pb_batch.grid(row=1, column=0, sticky='ew', pady=(0, 4))
    pb_batch.set(0)
    
    batch_info_frame = ctk.CTkFrame(batch_pb_frame, fg_color="transparent")
    batch_info_frame.grid(row=2, column=0, sticky='ew')
    batch_info_frame.grid_columnconfigure(0, weight=1)
    batch_info_frame.grid_columnconfigure(1, weight=1)
    
    lbl_batch_stat = ctk.CTkLabel(batch_info_frame, text="0/0 (0.0%)", 
                                  font=ctk.CTkFont(size=11, weight="bold"),
                                  text_color=("#1565C0", "#90CAF9"))
    lbl_batch_stat.grid(row=0, column=0, sticky='w')
    lbl_batch_time = ctk.CTkLabel(batch_info_frame, text="已用时间: 0.0s", 
                                  font=ctk.CTkFont(size=11),
                                  text_color=("#616161", "#BDBDBD"))
    lbl_batch_time.grid(row=0, column=1, sticky='e')
    
    # 底部：控制按钮区（固定位置）
    control_panel = ctk.CTkFrame(left_container, corner_radius=10, fg_color=("#F5F5F5", "#2b2b2b"))
    control_panel.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    control_panel.grid_columnconfigure(0, weight=1)
    
    # 按钮标签
    ctk.CTkLabel(control_panel, text="⚙️ 任务控制", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, sticky='w', padx=12, pady=(10, 8))
    
    # 开始按钮
    btn_run = ctk.CTkButton(control_panel, text='▶️ 开始处理', height=45, 
                           corner_radius=10,
                           font=ctk.CTkFont(size=16, weight="bold"), 
                           fg_color=("#4CAF50", "#388E3C"),
                           hover_color=("#388E3C", "#2E7D32"))
    btn_run.grid(row=1, column=0, sticky='ew', padx=12, pady=(0, 8))
    
    # 暂停和停止按钮横向排列
    control_subframe = ctk.CTkFrame(control_panel, fg_color="transparent")
    control_subframe.grid(row=2, column=0, sticky='ew', padx=12, pady=(0, 12))
    control_subframe.grid_columnconfigure(0, weight=1)
    control_subframe.grid_columnconfigure(1, weight=1)
    
    btn_pause = ctk.CTkButton(control_subframe, text='⏸️ 暂停', height=38, state='disabled',
                             corner_radius=10,
                             font=ctk.CTkFont(size=14), 
                             fg_color=("#FF9800", "#F57C00"),
                             hover_color=("#F57C00", "#E65100"))
    btn_pause.grid(row=0, column=0, sticky='ew', padx=(0, 4))
    
    btn_stop = ctk.CTkButton(control_subframe, text='⏹️ 停止', height=38, state='disabled',
                            corner_radius=10,
                            font=ctk.CTkFont(size=14), 
                            fg_color=("#F44336", "#D32F2F"),
                            hover_color=("#D32F2F", "#C62828"))
    btn_stop.grid(row=0, column=1, sticky='ew', padx=(4, 0))

    # 安装拖放支持（立即初始化，因为已经使用 ModernApp 类）
    _install_drop_target(ent_in, in_path)
    _install_drop_target(ent_out, out_path)
    _install_drop_target(ent_tmpl, tmpl_path)
    _install_drop_target(ent_base_ge, base_ge_path)
    
    # 状态跟踪
    state = {
        'file_total': 0, 'file_last_t': 0.0, 'file_last_done': 0,
        'file_start_t': 0.0, 'batch_start_t': 0.0, 'timer_running': False,
        'total_files': 0, 'processed_files': 0, 'current_file': '',
        'paused_elapsed': 0.0, 'pause_start_t': 0.0  # 暂停时累计时间
    }
    succeeded_files: List[str] = []
    failed_files: List[str] = []
    
    # 创建统一的进度回调管理器（遵循规范）
    if PROGRESS_UTILS_AVAILABLE:
        from progress_utils import ProgressCallback, ProgressConfig
        
        progress_config = ProgressConfig(
            progress_update_interval=0.15  # 单文件节流间隔
        )
        
        def _file_progress_wrapper(done: int, total: int):
            """ProgressCallback 的单文件进度包装"""
            _progress_file_cb(done, total)
        
        def _batch_progress_wrapper(processed: int, total: int, elapsed: float):
            """ProgressCallback 的批量进度包装"""
            _progress_batch_cb(processed, total, elapsed)
        
        progress_callback = ProgressCallback(
            file_progress_cb=_file_progress_wrapper,
            batch_progress_cb=_batch_progress_wrapper,
            config=progress_config
        )
        
        # 使用 ProgressCallback 的节流方法
        throttled_file_cb = progress_callback.update_file_progress
        throttled_batch_cb = progress_callback.update_batch_progress
    else:
        # 回退：使用内置 ThrottledUpdater
        throttled_file_cb = ThrottledUpdater(min_interval=0.15)(lambda *args: _progress_file_cb(*args))
        throttled_batch_cb = ThrottledUpdater(min_interval=0.3)(lambda *args: _progress_batch_cb(*args))
    
    def _progress_file_cb(done: int, total: int):
        """单文件进度回调（适配 CustomTkinter）"""
        task_controller.check_state()
        
        if state['file_total'] != total:
            state['file_total'] = total
            state['file_last_done'] = 0
            state['file_last_t'] = time.time()
            state['file_start_t'] = state['file_last_t']
        
        # 确保进度条正确显示到0-100%
        pct = 0.0 if total <= 0 else min(done / total, 1.0)
        pb_file.set(pct)
        lbl_file_pct.configure(text=f"{pct*100:.1f}%")
        
        # 强制刷新界面，确保100%显示
        if done >= total:
            root.update_idletasks()
    
    def _current_file_cb(file_name: str):
        """当前文件名回调（并行模式下显示进度最快的文件）"""
        state['current_file'] = file_name
        lbl_current_file.configure(text=f"正在处理: {file_name}")
    
    def _progress_batch_cb(processed: int, total: int, elapsed: float):
        """批量进度回调（适配 CustomTkinter）"""
        task_controller.check_state()
        
        # 确保进度条正确显示到0-100%
        pct = 0.0 if total <= 0 else min(processed / total, 1.0)
        pb_batch.set(pct)
        lbl_batch_stat.configure(text=f"{processed}/{total} ({pct*100:.1f}%)")
        state['total_files'] = total
        state['processed_files'] = processed
        
        # 强制刷新界面，确保100%显示
        if processed >= total:
            root.update_idletasks()
    
    def _tick_elapsed():
        """更新已用时间 - 修复暂停时计时问题"""
        if not state['timer_running']:
            return
        
        # 计算实际用时：当前时间 - 开始时间 + 暂停累计时间
        current_elapsed = time.time() - state['batch_start_t']
        total_elapsed = max(0.0, state['paused_elapsed'] + current_elapsed)
        lbl_batch_time.configure(text=f"已用时间: {total_elapsed:.1f}s")
        root.after(100, _tick_elapsed)
    
    # 日志监听器（带缓冲）
    buffered_log = BufferedLogListener(txt_log, flush_interval=0.2, max_buffer=20)
    
    def update_stats_label():
        """更新统计信息标签"""
        total = state.get('total_files', 0)
        success_count = len(succeeded_files)
        failed_count = len(failed_files)
        lbl_total_stats.configure(text=f"总文件: {total} | 成功: {success_count} | 失败: {failed_count}")
    
    def gui_log_listener(line: str):
        """处理日志行并更新统计"""
        level = log_level_var.get()
        show = False
        
        if "ERROR " in line:
            show = True
        elif level == "详细":
            show = True
        elif level == "普通":
            if "OK " in line or "WARNING" in line:
                show = True
        
        if show:
            buffered_log(line)
        
        # 更新统计信息
        if "OK 导出 PNG：" in line or "OK 生成 PGD：" in line:
            try:
                if "导出 PNG" in line:
                    path = line.split("OK 导出 PNG：", 1)[1].split("(")[0].strip()
                else:
                    path = line.split("OK 生成 PGD：", 1)[1].split("(")[0].strip()
                succeeded_files.append(path)
                update_stats_label()
            except Exception:
                pass
        elif "ERROR 处理失败 " in line:
            try:
                after_error = line.split("ERROR 处理失败 ", 1)[1]
                if ".PGD:" in after_error:
                    path = after_error.split(".PGD:", 1)[0] + ".PGD"
                elif ".PNG:" in after_error:
                    path = after_error.split(".PNG:", 1)[0] + ".PNG"
                else:
                    path = after_error.rsplit(":", 1)[0].strip()
                failed_files.append(path)
                update_stats_label()
            except Exception:
                pass
    
    def run_task():
        """后台任务执行 - 修复了未定义变量 bug"""
        try:
            # 重置状态（适配 CustomTkinter）
            pb_file.set(0)
            pb_batch.set(0)
            
            lbl_file_pct.configure(text="0%")
            lbl_current_file.configure(text="等待处理...")
            lbl_batch_stat.configure(text="0/0 (0.0%)")
            lbl_batch_time.configure(text="已用时间: 0.0s")
            lbl_total_stats.configure(text="总文件: 0 | 成功: 0 | 失败: 0")
            
            succeeded_files.clear()
            failed_files.clear()
            state.update({
                'file_total': 0, 'file_last_t': 0.0, 'file_last_done': 0,
                'file_start_t': 0.0, 'batch_start_t': 0.0, 'timer_running': False,
                'total_files': 0, 'processed_files': 0, 'current_file': '',
                'paused_elapsed': 0.0, 'pause_start_t': 0.0
            })
            task_controller.reset()
            
            # 解析填充色
            nonlocal fill_bgr_current
            try:
                fill_bgr_current = _parse_rgb_text(fill_color_text.get())
            except Exception as e:
                buffered_log(f"WARNING 填充色解析失败：{e}，将使用默认 255,255,255")
                fill_bgr_current = (255, 255, 255)
            _update_fill_preview()
            
            # 参数验证
            m = mode.get()
            ptype = pgd_type.get()
            inp = in_path.get().strip()
            outp = out_path.get().strip() or None
            tmpl = tmpl_path.get().strip() or None
            base_ge = base_ge_path.get().strip() or None
            use_tmpl_base = use_template_for_base.get()
            
            if not inp:
                raise ValueError("请选择输入路径")
            
            # 挂载日志监听
            add_pgd_listener(gui_log_listener)
            add_png_listener(gui_log_listener)
            
            # 启动计时器
            state['batch_start_t'] = time.time()
            state['timer_running'] = True
            _tick_elapsed()
            
            # 执行转换 - 优化版：缓存类型检测结果 + 并行处理支持
            if m == 'pgd2png':
                buffered_log(f"开始处理：PGD → PNG（自动识别类型）")
                
                # 获取所有 PGD 文件
                files = _find_files_fast(inp, (".pgd",), recursive=recursive_var.get())
                
                # 统计文件数和各类型的分布
                total_files = len(files)
                type_stats = {'GE': 0, 'PGD/00_C': 0, 'PGD/11_C': 0, 'PGD/TGA': 0, 'PGD3': 0, 'UNKNOWN': 0}
                
                # 初始化批量进度（GUI规范要求）
                throttled_batch_cb(0, total_files, 0.0)
                
                # 预先检测所有文件类型（快速模式）并缓存结果
                buffered_log(f"扫描 {total_files} 个文件...")
                type_cache = {}  # 缓存类型检测结果
                for src in files:
                    detected_type = _detect_pgd_type(src)
                    type_cache[src] = detected_type
                    type_stats[detected_type] = type_stats.get(detected_type, 0) + 1
                
                # 输出类型分布
                type_summary = ", ".join([f"{k}: {v}" for k, v in type_stats.items() if v > 0])
                buffered_log(f"类型分布: {type_summary}")
                
                # 判断是否使用并行处理
                use_parallel = PARALLEL_AVAILABLE and total_files >= 4
                
                if use_parallel:
                    # 并行处理模式
                    from parallel_processor import parallel_process_files, ParallelConfig
                    from config import ParallelConfig as ConfigParallelConfig
                    
                    buffered_log(f"启用并行处理模式（文件数: {total_files}）")
                    
                    # 创建并行配置
                    parallel_config = ConfigParallelConfig(
                        max_workers=0,  # 自动检测
                        enable_parallel=True,
                        min_files_for_parallel=4
                    )
                    
                    # 定义单文件处理函数
                    def process_single_pgd(file_path: str, file_progress_cb=None, log_cb=None, **kwargs):
                        """单文件处理函数（并行调用）"""
                        try:
                            detected_type = type_cache.get(file_path, 'UNKNOWN')
                            file_size = os.path.getsize(file_path)
                            
                            if detected_type == 'GE':
                                dst = os.path.join(outp, os.path.splitext(os.path.basename(file_path))[0] + ".png") if outp else os.path.splitext(file_path)[0] + ".png"
                                from pgd2png_ge import pgd_to_png as ge_pgd_to_png
                                ge_pgd_to_png(file_path, dst, progress_cb=file_progress_cb)
                                if log_cb:
                                    log_cb(f"OK 导出 PNG：{dst} (类型: GE, 大小: {file_size} bytes)")
                                return dst
                            elif detected_type in ['PGD/00_C', 'PGD/11_C', 'PGD/TGA', 'PGD3']:
                                if _pgd2png_oth:
                                    dst = os.path.join(outp, os.path.splitext(os.path.basename(file_path))[0] + ".png") if outp else os.path.splitext(file_path)[0] + ".png"
                                    _pgd2png_oth.pgd_to_png(file_path, dst, progress_cb=file_progress_cb)
                                    if log_cb:
                                        log_cb(f"OK 导出 PNG：{dst} (类型: {detected_type}, 大小: {file_size} bytes)")
                                    return dst
                                else:
                                    raise RuntimeError(f"未找到 pgd2png_others.py 模块，无法处理 {detected_type}")
                            else:
                                raise ValueError(f"无法识别的 PGD 类型: {detected_type}")
                        except Exception as e:
                            if log_cb:
                                log_cb(f"ERROR 处理失败 {file_path}: {e}")
                            raise
                    
                    # 执行并行处理（直接使用task_controller）
                    from parallel_processor import parallel_process_files, ParallelConfig
                    
                    # OptimizedTaskController与TaskController API兼容
                    results, errors = parallel_process_files(
                        files=files,
                        process_func=process_single_pgd,
                        config=parallel_config,
                        file_progress_cb=throttled_file_cb,
                        batch_progress_cb=throttled_batch_cb,
                        log_cb=gui_log_listener,
                        task_controller=task_controller,  # type: ignore[arg-type]
                        current_file_cb=_current_file_cb  # 新增：显示进度最快的文件
                    )
                    
                    # 注意：不再手动添加到 succeeded_files，
                    # 因为 gui_log_listener 已经自动监听 "OK" 日志并添加
                    # 避免重复计数
                    
                    # 处理完成后强制设置进度为100%
                    throttled_file_cb(100, 100)
                    throttled_batch_cb(total_files, total_files, time.time() - state['batch_start_t'])
                    
                    update_stats_label()
                    lbl_current_file.configure(text="处理完成")
                    
                else:
                    # 串行处理模式（文件数少或并行不可用）
                    if total_files < 4:
                        buffered_log(f"文件数较少，使用串行处理模式")
                    else:
                        buffered_log(f"并行处理不可用，使用串行模式")
                    
                    # 优化的循环：使用缓存的类型检测结果
                    for i, src in enumerate(files, 1):
                        try:
                            # 更新当前文件名显示
                            filename = os.path.basename(src)
                            state['current_file'] = filename
                            lbl_current_file.configure(text=f"正在处理: {filename}")
                            root.update_idletasks()
                            
                            file_start_time = time.time()
                            file_size = os.path.getsize(src)
                            
                            # 使用缓存的类型检测结果
                            detected_type = type_cache.get(src, 'UNKNOWN')
                            
                            if detected_type == 'GE':
                                # 使用 GE 模块处理，传递单文件进度回调
                                dst = os.path.join(outp, os.path.splitext(os.path.basename(src))[0] + ".png") if outp else os.path.splitext(src)[0] + ".png"
                                from pgd2png_ge import pgd_to_png as ge_pgd_to_png
                                ge_pgd_to_png(src, dst, progress_cb=throttled_file_cb)
                                buffered_log(f"OK 导出 PNG：{dst} (类型: GE, 大小: {file_size} bytes)")
                            elif detected_type in ['PGD/00_C', 'PGD/11_C', 'PGD/TGA', 'PGD3']:
                                # 使用 Others 模块处理，传递单文件进度回调
                                if _pgd2png_oth:
                                    dst = os.path.join(outp, os.path.splitext(os.path.basename(src))[0] + ".png") if outp else os.path.splitext(src)[0] + ".png"
                                    _pgd2png_oth.pgd_to_png(src, dst, progress_cb=throttled_file_cb)
                                    buffered_log(f"OK 导出 PNG：{dst} (类型: {detected_type}, 大小: {file_size} bytes)")
                                else:
                                    raise RuntimeError(f"未找到 pgd2png_others.py 模块，无法处理 {detected_type}")
                            else:
                                raise ValueError(f"无法识别的 PGD 类型: {detected_type}")
                            
                            # 确保单文件进度条显示100%
                            throttled_file_cb(100, 100)
                            
                            # 更新进度 - 使用节流回调
                            throttled_batch_cb(i, total_files, time.time() - state['batch_start_t'])
                            
                            # 更新统计
                            state['processed_files'] = i
                            state['total_files'] = total_files
                            
                        except Exception as e:
                            buffered_log(f"ERROR 处理失败 {src}: {e}")
                            failed_files.append(src)
                            update_stats_label()
                    
                    # 最终更新批量进度到100%
                    throttled_batch_cb(total_files, total_files, time.time() - state['batch_start_t'])
                    lbl_current_file.configure(text="处理完成")
                    root.update_idletasks()
                
            else:
                # PNG→PGD 模式 - 优化版：支持并行处理
                pre = preset.get()
                
                # 读取优化选项
                quality = int(quality_level.get().split()[0])  # 从 "1 (fast)" 中提取 "1"
                enable_gpu = use_gpu.get() and CUPY_AVAILABLE
                
                # 验证 GPU 可用性
                if enable_gpu and GPU_COUNT == 0:
                    buffered_log("WARNING: GPU加速被启用但未检测到GPU设备，将回退到CPU模式")
                    enable_gpu = False
                
                # 获取所有 PNG 文件
                files = _find_files_fast(inp, (".png",), recursive=recursive_var.get())
                if not files:
                    raise ValueError(f"未找到 PNG 文件：{inp}")
                
                total_files = len(files)
                
                # 判断是否使用并行处理
                use_parallel = PARALLEL_AVAILABLE and total_files >= 4
                
                typ = None  # 初始化 typ 变量，防止未定义错误
                if ptype == 'GE':
                    typ = int(ctype.get())
                    buffered_log(f"开始处理：PNG → PGD（GE 类型 {typ}）")
                    if typ == 2:
                        buffered_log(f"提示：类型 2 不支持透明，将与填充色 {_rgb_text_from_bgr(fill_bgr_current)} 混合")
                    
                    # 显示优化状态
                    if OPTIMIZER_AVAILABLE:
                        buffered_log(f"优化级别：{quality} ({'快速' if quality == 1 else '平衡' if quality == 2 else '最佳'})")
                    if enable_gpu:
                        buffered_log(f"GPU加速：已启用 ({GPU_COUNT} GPU)")
                    
                    if use_parallel:
                        # 并行处理模式
                        from parallel_processor import parallel_process_files, TaskController as ParallelTaskCtrl
                        from config import ParallelConfig as ConfigParallelConfig
                        
                        buffered_log(f"启用并行处理模式（文件数: {total_files}）")
                        
                        parallel_config = ConfigParallelConfig(
                            max_workers=0,
                            enable_parallel=True,
                            min_files_for_parallel=4
                        )
                        
                        def process_single_png_ge(file_path: str, file_progress_cb=None, log_cb=None, **kwargs):
                            """GE 单文件处理函数"""
                            try:
                                dst = os.path.join(outp, os.path.splitext(os.path.basename(file_path))[0] + ".pgd") if outp else os.path.splitext(file_path)[0] + ".pgd"
                                from png2pgd_ge import png2pgd_single
                                png2pgd_single(file_path, typ, dst, tmpl, pre, fill_bgr_current,
                                              progress_cb=file_progress_cb,
                                              quality_level=quality,
                                              use_gpu=enable_gpu)
                                if log_cb:
                                    log_cb(f"OK 生成 PGD：{dst} (类型: GE {typ})")
                                return dst
                            except Exception as e:
                                if log_cb:
                                    log_cb(f"ERROR 处理失败 {file_path}: {e}")
                                raise
                        
                        # 直接使用task_controller
                        results, errors = parallel_process_files(
                            files=files,
                            process_func=process_single_png_ge,
                            config=parallel_config,
                            file_progress_cb=throttled_file_cb,
                            batch_progress_cb=throttled_batch_cb,
                            log_cb=gui_log_listener,
                            task_controller=task_controller,  # type: ignore[arg-type]
                            current_file_cb=_current_file_cb  # 新增：显示进度最快的文件
                        )
                        
                        # 注意：不再手动添加到 succeeded_files，
                        # 因为 gui_log_listener 已经自动监听 "OK" 日志并添加
                        # 避免重复计数
                        
                        # 处理完成后强制设置进度为100%
                        throttled_file_cb(100, 100)
                        throttled_batch_cb(total_files, total_files, time.time() - state['batch_start_t'])
                        lbl_current_file.configure(text="处理完成")
                        
                    else:
                        # 串行处理模式（使用原有批处理函数）
                        if total_files < 4:
                            buffered_log(f"文件数较少，使用串行处理模式")
                        else:
                            buffered_log(f"并行处理不可用，使用串行模式")
                        
                        if png2pgd_batch:
                            png2pgd_batch(inp, typ, outp, tmpl, pre, fill_bgr=fill_bgr_current,
                                          recursive=recursive_var.get(),
                                          file_progress_cb=throttled_file_cb,
                                          batch_progress_cb=throttled_batch_cb,
                                          quality_level=quality,
                                          use_gpu=enable_gpu)
                else:
                    # Others 类型（包括自动模式）
                    buffered_log(f"开始处理：PNG → PGD（{ptype}）")
                    if ptype == 'PGD3':
                        # PGD3 有三种模式：
                        # 1. 使用模板自动查找：tmpl_path指定模板PGD，从中读取basename
                        # 2. 直接指定基准：base_ge_path直接指定基准GE文件
                        # 3. 自动模式：不提供任何参数，在PNG目录查找同名.pgd3/.pgd
                        if use_tmpl_base:
                            if tmpl:
                                buffered_log(f"PGD3模式1：使用模板 {tmpl} 自动查找基准GE")
                            else:
                                buffered_log(f"PGD3模式3：自动在PNG目录查找同名模板PGD")
                        else:
                            if base_ge:
                                buffered_log(f"PGD3模式2：直接使用基准GE文件 {base_ge}")
                            else:
                                raise ValueError("PGD3 需要指定基准 PGD/GE 文件（请取消勾选'使用模板自动查找基准'并指定基准文件）")
                    
                    # 显示优化状态
                    if OPTIMIZER_AVAILABLE:
                        buffered_log(f"优化级别：{quality} ({'快速' if quality == 1 else '平衡' if quality == 2 else '最佳'})")
                    if enable_gpu:
                        buffered_log(f"GPU加速：已启用 ({GPU_COUNT} GPU)")
                    
                    # 自动模式：需要先检测所有文件的模板类型
                    if ptype == '自动':
                        if not tmpl:
                            raise ValueError("自动模式需要指定模板文件夹")
                        
                        buffered_log(f"自动模式：扫描 {total_files} 个文件的模板...")
                        
                        # 预先检测所有 PNG 文件的目标类型（通过模板匹配）
                        type_cache = {}  # 缓存: {png_path: (target_type, template_file)}
                        type_stats = {}  # 统计: {type: count}
                        
                        for png_file in files:
                            png_stem = os.path.splitext(os.path.basename(png_file))[0]
                            
                            # 在模板目录中查找同名模板
                            actual_tmpl = None
                            if os.path.isdir(tmpl):
                                for ext in ['.pgd3', '.pgd', '.PGD']:
                                    cand = os.path.join(tmpl, png_stem + ext)
                                    if os.path.isfile(cand):
                                        actual_tmpl = cand
                                        break
                            elif os.path.isfile(tmpl):
                                actual_tmpl = tmpl  # 单一模板文件
                            
                            if actual_tmpl:
                                # 检测模板类型
                                template_type = _detect_pgd_type(actual_tmpl)
                                type_cache[png_file] = (template_type, actual_tmpl)
                                type_stats[template_type] = type_stats.get(template_type, 0) + 1
                            else:
                                # 没有找到模板，标记为 UNKNOWN
                                type_cache[png_file] = ('UNKNOWN', None)
                                type_stats['UNKNOWN'] = type_stats.get('UNKNOWN', 0) + 1
                        
                        # 输出类型分布
                        type_summary = ", ".join([f"{k}: {v}" for k, v in type_stats.items() if v > 0])
                        buffered_log(f"目标类型分布: {type_summary}")
                        
                        # 判断是否使用并行处理
                        if use_parallel:
                            # 并行处理模式
                            from parallel_processor import parallel_process_files, TaskController as ParallelTaskCtrl
                            from config import ParallelConfig as ConfigParallelConfig
                            
                            buffered_log(f"启用并行处理模式（文件数: {total_files}）")
                            
                            parallel_config = ConfigParallelConfig(
                                max_workers=0,
                                enable_parallel=True,
                                min_files_for_parallel=4
                            )
                            
                            def process_single_png_auto(file_path: str, file_progress_cb=None, log_cb=None, **kwargs):
                                """自动模式单文件处理函数"""
                                try:
                                    target_type, template_file = type_cache.get(file_path, ('UNKNOWN', None))
                                    
                                    if target_type == 'UNKNOWN' or not template_file:
                                        raise ValueError(f"未找到模板: {os.path.basename(file_path)}")
                                    
                                    # 根据模板类型确定输出扩展名
                                    output_ext = os.path.splitext(template_file)[1] or '.pgd'
                                    dst = os.path.join(outp, os.path.splitext(os.path.basename(file_path))[0] + output_ext) if outp else os.path.splitext(file_path)[0] + output_ext
                                    
                                    if _png2pgd_oth is None:
                                        raise RuntimeError("未找到 png2pgd_others.py 模块")
                                    
                                    # 根据检测到的类型调用相应的编码器
                                    if target_type == "PGD/11_C":
                                        _png2pgd_oth._write_11c_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                    elif target_type == "PGD/00_C":
                                        _png2pgd_oth._write_00c_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                    elif target_type == "PGD/TGA":
                                        _png2pgd_oth._write_pgd_tga_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                    elif target_type == "PGD3":
                                        # PGD3 需要从模板读取基准 GE
                                        pgd3_info = _png2pgd_oth.read_pgd3_header(template_file)
                                        base_name = pgd3_info['basename']
                                        # 在模板 PGD 所在目录查找基准 GE
                                        base_dir = os.path.dirname(os.path.abspath(template_file))
                                        base_ge_file = os.path.join(base_dir, base_name)
                                        if not os.path.isfile(base_ge_file):
                                            raise ValueError(f"PGD3 基准文件不存在: {base_ge_file}")
                                        _png2pgd_oth.png_to_pgd3(file_path, base_ge=base_ge_file, out_path=dst, preset=pre)
                                    elif target_type == "GE":
                                        # GE 格式：使用 png2pgd_ge 模块
                                        from png2pgd_ge import png2pgd_single
                                        # 使用用户界面选择的压缩类型，而不是硬编码为3
                                        user_selected_ctype = int(ctype.get()) if ctype.get() in ['1', '2', '3'] else 3
                                        if log_cb:
                                            log_cb(f"用户选择的压缩类型: GE 类型 {user_selected_ctype}")
                                        png2pgd_single(file_path, user_selected_ctype, dst, template_file, pre, fill_bgr_current,
                                                      progress_cb=file_progress_cb,
                                                      quality_level=quality,
                                                      use_gpu=enable_gpu)
                                    else:
                                        raise ValueError(f"不支持的目标类型: {target_type}")
                                    
                                    if log_cb:
                                        log_cb(f"OK 生成 PGD：{dst} (类型: {target_type})")
                                    return dst
                                except Exception as e:
                                    if log_cb:
                                        log_cb(f"ERROR 处理失败 {file_path}: {e}")
                                    raise
                            
                            # 直接使用task_controller
                            results, errors = parallel_process_files(
                                files=files,
                                process_func=process_single_png_auto,
                                config=parallel_config,
                                file_progress_cb=throttled_file_cb,
                                batch_progress_cb=throttled_batch_cb,
                                log_cb=gui_log_listener,
                                task_controller=task_controller,  # type: ignore[arg-type]
                                current_file_cb=_current_file_cb  # 新增：显示进度最快的文件
                            )
                            
                            # 注意：不再手动添加到 succeeded_files，
                            # 因为 gui_log_listener 已经自动监听 "OK" 日志并添加
                            
                            # 处理完成后强制设置进度为100%
                            throttled_file_cb(100, 100)
                            throttled_batch_cb(total_files, total_files, time.time() - state['batch_start_t'])
                            lbl_current_file.configure(text="处理完成")
                        else:
                            # 串行处理模式
                            buffered_log(f"文件数较少，使用串行处理模式")
                            
                            # 使用原有的 others_png2pgd_batch 函数
                            others_png2pgd_batch(inp, ptype, outp, tmpl, base_ge, use_tmpl_base, pre, fill_bgr=fill_bgr_current,
                                                 recursive=recursive_var.get(),
                                                 file_progress_cb=throttled_file_cb,
                                                 batch_progress_cb=throttled_batch_cb,
                                                 log_cb=gui_log_listener,
                                                 quality_level=quality,
                                                 use_gpu=enable_gpu)
                    
                    # Others 类型也支持并行处理
                    elif use_parallel and ptype != 'PGD3':  # PGD3 因为需要特殊处理，暂不支持并行
                        from parallel_processor import parallel_process_files, TaskController as ParallelTaskCtrl
                        from config import ParallelConfig as ConfigParallelConfig
                        
                        buffered_log(f"启用并行处理模式（文件数: {total_files}）")
                        
                        parallel_config = ConfigParallelConfig(
                            max_workers=0,
                            enable_parallel=True,
                            min_files_for_parallel=4
                        )
                        
                        def process_single_png_others(file_path: str, file_progress_cb=None, log_cb=None, **kwargs):
                            """Others 单文件处理函数"""
                            try:
                                dst = os.path.join(outp, os.path.splitext(os.path.basename(file_path))[0] + ".pgd") if outp else os.path.splitext(file_path)[0] + ".pgd"
                                
                                if _png2pgd_oth is None:
                                    raise RuntimeError("未找到 png2pgd_others.py 模块")
                                
                                if ptype == "11_C":
                                    _png2pgd_oth._write_11c_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                elif ptype == "00_C":
                                    _png2pgd_oth._write_00c_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                elif ptype == "TGA":
                                    _png2pgd_oth._write_pgd_tga_from_png(file_path, dst, (0,0), preset=pre, progress_cb=file_progress_cb)
                                elif ptype == "GE":
                                    from png2pgd_ge import png2pgd_single
                                    # 使用用户界面选择的压缩类型，而不是硬编码为3
                                    user_selected_ctype = int(ctype.get()) if ctype.get() in ['1', '2', '3'] else 3
                                    png2pgd_single(file_path, user_selected_ctype, dst, tmpl, pre, fill_bgr_current,
                                                  progress_cb=file_progress_cb,
                                                  quality_level=quality,
                                                  use_gpu=enable_gpu)
                                else:
                                    raise ValueError(f"未知 Others 目标格式：{ptype}")
                                
                                if log_cb:
                                    log_cb(f"OK 生成 PGD：{dst} (类型: {ptype})")
                                return dst
                            except Exception as e:
                                if log_cb:
                                    log_cb(f"ERROR 处理失败 {file_path}: {e}")
                                raise
                        
                        # 直接使用task_controller
                        results, errors = parallel_process_files(
                            files=files,
                            process_func=process_single_png_others,
                            config=parallel_config,
                            file_progress_cb=throttled_file_cb,
                            batch_progress_cb=throttled_batch_cb,
                            log_cb=gui_log_listener,
                            task_controller=task_controller,  # type: ignore[arg-type]
                            current_file_cb=_current_file_cb  # 新增：显示进度最快的文件
                        )
                        
                        # 注意：不再手动添加到 succeeded_files，
                        # 因为 gui_log_listener 已经自动监听 "OK" 日志并添加
                        
                        # 处理完成后强制设置进度为100%
                        throttled_file_cb(100, 100)
                        throttled_batch_cb(total_files, total_files, time.time() - state['batch_start_t'])
                        lbl_current_file.configure(text="处理完成")
                    else:
                        # 串行处理模式（使用原有批处理函数）
                        if ptype == 'PGD3' or total_files < 4:
                            if ptype == 'PGD3':
                                buffered_log(f"PGD3 格式暂不支持并行，使用串行模式")
                            else:
                                buffered_log(f"文件数较少，使用串行处理模式")
                        else:
                            buffered_log(f"并行处理不可用，使用串行模式")
                        
                        others_png2pgd_batch(inp, ptype, outp, tmpl, base_ge, use_tmpl_base, pre, fill_bgr=fill_bgr_current,
                                             recursive=recursive_var.get(),
                                             file_progress_cb=throttled_file_cb,
                                             batch_progress_cb=throttled_batch_cb,
                                             log_cb=gui_log_listener,
                                             quality_level=quality,
                                             use_gpu=enable_gpu)
            
            # 完成处理
            buffered_log("\n" + "=" * 60 + "\n")
            if task_controller.is_stopped():
                buffered_log(f"处理已停止。成功: {len(succeeded_files)}, 失败: {len(failed_files)}")
            else:
                buffered_log(f"处理完成。共 {len(succeeded_files)} 个文件成功, {len(failed_files)} 个失败")
            
            if failed_files:
                buffered_log("\n失败文件列表：\n")
                for p in failed_files:
                    buffered_log(f" - {p}")
            elif not task_controller.is_stopped():
                buffered_log("\n全部文件处理成功。")
            
            update_stats_label()
        except UserStopException:
            buffered_log("\n处理已停止")
        except Exception as e:
            buffered_log(f"ERROR {str(e)}")
            messagebox.showerror("错误", str(e))
        finally:
            state['timer_running'] = False
            btn_run.configure(state='normal')
            btn_pause.configure(state='disabled', text='⏸️ 暂停')
            btn_stop.configure(state='disabled')
            
            # 确保刷新剩余日志
            buffered_log._flush()
            
            # 卸载监听
            remove_pgd_listener(gui_log_listener)
            remove_png_listener(gui_log_listener)
    
    def on_run():
        """开始处理按钮回调"""
        if not in_path.get():
            messagebox.showwarning("警告", "请选择输入路径")
            return
        
        ptype = pgd_type.get()
        # PGD3 验证逻辑：有三种模式，至少需要一种
        if mode.get() == 'png2pgd' and ptype == 'PGD3':
            has_template = bool(tmpl_path.get().strip())
            has_base_ge = bool(base_ge_path.get().strip())
            use_tmpl = use_template_for_base.get()
            
            # 模式1：使用模板 + 有模板路径
            # 模式2：不使用模板 + 有基准GE路径
            # 模式3：使用模板 + 无模板路径（自动查找）
            if use_tmpl:
                # 模式1和3都可以，不需要验证
                pass
            else:
                # 模式2：必须有基准GE
                if not has_base_ge:
                    messagebox.showerror("错误", 
                        "PGD3 模式错误：\n"
                        "已取消勾选'使用模板自动查找基准'，但未指定'基准 PGD/GE 文件'\n\n"
                        "请选择以下之一：\n"
                        "1. 勾选'使用模板自动查找基准'（推荐）\n"
                        "2. 指定'基准 PGD/GE 文件'")
                    return
        
        btn_run.configure(state='disabled')
        btn_pause.configure(state='normal')
        btn_stop.configure(state='normal')
        task_controller.reset()
        
        # 在新线程中运行任务
        threading.Thread(target=run_task, daemon=True).start()
    
    def on_pause():
        """暂停/继续按钮回调 - 修复计时器问题"""
        if task_controller.is_paused():
            # 继续执行
            task_controller.resume()
            btn_pause.configure(text='⏸️ 暂停')
            
            # 重新启动计时器，调整起始时间
            if state.get('pause_start_t', 0) > 0:
                # 累加暂停期间的时间
                pause_duration = time.time() - state['pause_start_t']
                state['batch_start_t'] += pause_duration
                state['pause_start_t'] = 0.0
            
            state['timer_running'] = True
            _tick_elapsed()
            buffered_log("继续处理...")
        else:
            # 暂停执行
            task_controller.pause()
            btn_pause.configure(text='▶️ 继续')
            
            # 停止计时器，记录暂停时间
            state['timer_running'] = False
            state['pause_start_t'] = time.time()
            
            buffered_log("暂停处理...")
    
    def on_stop():
        """停止按钮回调"""
        if messagebox.askyesno("确认", "确定要停止处理吗？"):
            task_controller.stop()
            btn_stop.configure(state='disabled')
            btn_pause.configure(state='disabled')
            buffered_log("正在停止处理...")
    
    # 绑定按钮命令
    btn_run.configure(command=on_run)
    btn_pause.configure(command=on_pause)
    btn_stop.configure(command=on_stop)
    
    # 模式变化回调
    def on_mode_or_type_change(*_):
        """根据模式启用/禁用控件（适配 CustomTkinter）"""
        is_png2pgd = mode.get() == 'png2pgd'
        ptype = pgd_type.get()
        
        # 1) 压缩类型：仅在 PNG→PGD 且 GE 类型可选
        if is_png2pgd and ptype == 'GE':
            # 先确保有有效值（仅在无效时重置，不强制刷新显示）
            current_ctype = ctype.get()
            if current_ctype not in ['1', '2', '3']:
                ctype.set('3')
            
            # 启用控件（不使用 set() 强制刷新，避免值被重置）
            cb_ctype.configure(state='readonly')
        else:
            cb_ctype.configure(state='disabled')
        
        # 2) 压缩预设：在 PNG→PGD 模式下且非TGA类型时启用，在 PGD→PNG 模式下也保持启用
        if (is_png2pgd and ptype != 'TGA') or not is_png2pgd:
            cb_preset.configure(state='readonly')
        else:
            cb_preset.configure(state='disabled')
        
        # 3) PGD3 控件组：仅在 PNG→PGD 且 PGD3 类型时启用
        is_pgd3 = is_png2pgd and ptype == 'PGD3'
        state_pgd3 = 'normal' if is_pgd3 else 'disabled'
        lbl_base_ge.configure(state=state_pgd3)
        ent_base_ge.configure(state=state_pgd3)
        btn_base_ge_file.configure(state=state_pgd3)
        for child in pgd3_opts_frame.winfo_children():
            if hasattr(child, 'configure'):
                child.configure(state=state_pgd3)
        
        # 根据模式动态调整模板输入框状态
        if is_png2pgd:
            # PNG → PGD：启用模板输入框
            ent_tmpl.configure(state='normal')
            btn_tmpl_file.configure(state='normal')
            btn_tmpl_dir.configure(state='normal')
        else:
            # PGD → PNG：禁用模板输入框
            ent_tmpl.configure(state='disabled')
            btn_tmpl_file.configure(state='disabled')
            btn_tmpl_dir.configure(state='disabled')
        
        # 5) 模板 PGD 的提示文本
        if is_pgd3:
            lbl_tmpl.configure(text="📋 模板 PGD（原来的 PGD3）")
        else:
            lbl_tmpl.configure(text="📋 模板 PGD")
        
        # 根据模式和PGD类型调整模板标签颜色
        if is_png2pgd:
            # PNG → PGD时，标签颜色根据PGD类型动态调整
            lbl_tmpl.configure(text_color=("#1565C0", "#64B5F6") if is_pgd3 else ("#000000", "#000000"))
            lbl_base_ge.configure(text_color=("#1565C0", "#64B5F6") if is_pgd3 else ("gray", "gray"))
        else:
            # PGD → PNG时，标签应显示为灰色（禁用状态）
            lbl_tmpl.configure(text_color=("gray", "gray"))
            lbl_base_ge.configure(text_color=("gray", "gray"))
        
        # 6) 更新填充色状态（调用独立函数）
        update_fill_color_state()
    
    def update_fill_color_state(*_):
        """更新透明填充色控件状态（独立函数，避免与主回调冲突）"""
        is_png2pgd = mode.get() == 'png2pgd'
        ptype = pgd_type.get()
        ctype_val = ctype.get()
        
        # 透明填充色：仅在 PNG→PGD, GE 类型, type=2 时启用
        is_type2 = is_png2pgd and ptype == 'GE' and ctype_val == '2'
        state_fill = 'normal' if is_type2 else 'disabled'
        lbl_fill.configure(state=state_fill)
        ent_fill.configure(state=state_fill)
        # fill_preview 是 tk.Canvas，不需要修改
        btn_color.configure(state=state_fill)
    
    # 绑定变量变化
    # mode 和 pgd_type：影响所有控件状态
    mode.trace_add('write', on_mode_or_type_change)
    pgd_type.trace_add('write', on_mode_or_type_change)
    # ctype：仅影响填充色控件状态（独立监听，避免循环触发ComboBox显示问题）
    ctype.trace_add('write', update_fill_color_state)
    # 初始化控件状态
    on_mode_or_type_change()
    
    # 启动主循环
    root.mainloop()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Softpal PGD Toolkit（GUI/CLI）")
    sub = subparser = parser.add_subparsers(dest="cmd")
    
    # gui
    sp_gui = sub.add_parser("gui", help="启动图形界面")
    sp_gui.set_defaults(func=lambda args: launch_gui())

    # png2pgd
    sp_p2p = sub.add_parser("png2pgd", help="PNG 转 PGD（命令行）")
    sp_p2p.add_argument("--in", dest="inp", required=True, help="输入 PNG 路径")
    sp_p2p.add_argument("--out", dest="out", required=True, help="输出 PGD 路径")
    sp_p2p.add_argument("--fmt", dest="fmt", required=True, choices=["11_C","00_C","TGA","PGD3"], help="目标 PGD 格式")
    sp_p2p.add_argument("--preset", dest="preset", default="normal", choices=["fast","normal","max"], help="压缩预设（适用于 00_C/11_C/PGD3）")
    sp_p2p.add_argument("--offset", dest="offset", default="0,0", help="像素偏移: x,y（默认 0,0）")
    sp_p2p.add_argument("--template", dest="tmpl", default=None, help="PGD3 基准 GE（部分工程需要）")

    def _cli_png2pgd(args):
        from png2pgd_others import _write_11c_from_png, _write_00c_from_png, _write_pgd_tga_from_png, png_to_pgd3
        # 偏移解析
        try:
            ox, oy = args.offset.split(",")
            offset = (int(ox), int(oy))
        except Exception:
            raise SystemExit("offset 格式应为 x,y 例如 0,0")
        pre = args.preset

        src = args.inp
        dst = args.out
        fmt = args.fmt

        if fmt == "11_C":
            _write_11c_from_png(src, dst, offset, preset=pre)
        elif fmt == "00_C":
            _write_00c_from_png(src, dst, offset, preset=pre)
        elif fmt == "TGA":
            _write_pgd_tga_from_png(src, dst, offset, preset=pre)
        elif fmt == "PGD3":
            # 与 GUI 保持一致：要求提供模板，但内部目前未使用该路径
            if not args.tmpl:
                raise SystemExit("PGD3 需要提供 --template 基准 GE")
            png_to_pgd3(src, base_ge=None, out_path=dst, preset=pre)
        else:
            raise SystemExit(f"未知格式: {fmt}")

        print(f"完成：{dst}")

    sp_p2p.set_defaults(func=_cli_png2pgd)

    # 无子命令时默认启动 GUI
    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        return launch_gui()
    return args.func(args)

if __name__ == '__main__':
    main()