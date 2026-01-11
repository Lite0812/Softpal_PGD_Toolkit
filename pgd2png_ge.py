#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[arg-type] - bytearray/bytes 兼容性问题
"""
PGD2PNG (GE 格式) - 将 GE 格式 PGD 文件转换为 PNG

功能介绍：
  支持将 GE 格式的 PGD 文件解码为 PNG 图像
  - 类型 1：BGRA 分平面压缩（无损，支持透明）
  - 类型 2：近似 YUV 4:2:0 压缩（有损，不支持透明）
  - 类型 3：行内差分压缩（无损，支持透明）
  支持单文件和批量处理，支持进度回调

用法：
  单文件转换：
    python pgd2png_ge.py <input.pgd> [output.png]
  
  批量转换：
    python pgd2png_ge.py <input_folder> <output_folder> [--recursive]

命令行参数：
  input          输入 PGD 文件或文件夹路径
  output         输出 PNG 文件或文件夹路径（可选）
                 单文件时默认为同名 .png
                 批量处理时默认为输入目录
  --recursive    递归处理子文件夹（仅批量模式）

示例：
  # 单文件转换
  python pgd2png_ge.py image.pgd
  python pgd2png_ge.py image.pgd output.png
  
  # 批量转换
  python pgd2png_ge.py input_folder output_folder
  python pgd2png_ge.py input_folder output_folder --recursive

依赖：
  必需：numpy, pillow
  可选：progress_utils (进度条支持)
"""

import os
import sys
import struct
import time
import argparse
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

# 导入进度工具
try:
    from progress_utils import ConsoleProgressBar
    PROGRESS_UTILS_AVAILABLE = True
except ImportError:
    PROGRESS_UTILS_AVAILABLE = False

# 导入 Numba 加速器
try:
    from pgd_numba_accelerator import (
        is_accelerator_available,
        optimize_decompress,
        optimize_yuv_decode,
        NUMBA_AVAILABLE as ACCELERATOR_AVAILABLE
    )
except ImportError:
    ACCELERATOR_AVAILABLE = False
    def is_accelerator_available(): return False
    optimize_decompress = None
    optimize_yuv_decode = None

try:
    import numpy as np
except ImportError:
    print("错误: 需要安装 NumPy: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("错误: 需要安装 Pillow: pip install pillow")
    sys.exit(1)

# ------------------ 日志系统 ------------------
_LOG_LISTENERS: List[Callable[[str], None]] = []

def add_log_listener(fn: Callable[[str], None]) -> None:
    """添加日志监听器（供GUI使用）"""
    if fn not in _LOG_LISTENERS:
        _LOG_LISTENERS.append(fn)

def remove_log_listener(fn: Callable[[str], None]) -> None:
    """移除日志监听器"""
    try:
        _LOG_LISTENERS.remove(fn)
    except ValueError:
        pass

def log(msg: str):
    """日志函数，会同时输出到控制台和所有监听器"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    for fn in list(_LOG_LISTENERS):
        try:
            fn(line)
        except Exception:
            pass

# ------------------ 核心解码函数 ------------------
def _u16(b, off=0): return struct.unpack_from('<H', b, off)[0]
def _u32(b, off=0): return struct.unpack_from('<I', b, off)[0]

def _clamp_u8(v: int) -> int:
    if v < 0: return 0
    if v > 255: return 255
    return v

def _overlap_copy(dst: bytearray, dst_pos: int, src_pos: int, count: int):
    for i in range(count):
        dst[dst_pos + i] = dst[src_pos + i]

def _decompress_ge_lz_mem(comp: bytes, out_len: int,
                          progress_cb: Optional[Callable[[int, int], None]] = None) -> bytearray:
    """
    GE-LZ解压，支持进度回调
    
    智能版本选择：
    - Numba 可用：使用加速版本（3-5倍快）+ 阶段性进度
    - Numba 不可用：使用 Python 版本 + 详细进度
    
    Args:
        comp: 压缩数据
        out_len: 解压后长度
        progress_cb: 进度回调函数 (done, total)，范围0-100
    """
    # 优先使用 Numba 加速版本
    if ACCELERATOR_AVAILABLE and optimize_decompress is not None:
        result = optimize_decompress(comp, out_len, progress_cb)
        return bytearray(result)
    
    # Python 回退版本（详细进度）
    out = bytearray(out_len)
    dst = 0
    idx = 0
    n = len(comp)
    if n == 0:
        return out
    ctl = 2
    
    # 进度报告（每处理2%数据就更新）
    last_reported_pct = -1
    
    def report_progress():
        """报告解压进度（0-100%），每2%更新一次"""
        nonlocal last_reported_pct
        if progress_cb and out_len > 0:
            current_pct = int(dst * 100 / out_len)
            # 每2%更新一次，避免重复更新
            if current_pct >= last_reported_pct + 2 or (current_pct == 0 and last_reported_pct == -1):
                progress_cb(current_pct, 100)
                last_reported_pct = current_pct
    
    while dst < out_len:
        # 每次循环都检查并报告进度（函数内部会控制更新频率）
        report_progress()
        
        ctl >>= 1
        if ctl == 1:
            if idx >= n:
                raise ValueError("压缩流错误：缺少控制字节")
            ctl = comp[idx] | 0x100
            idx += 1

        if ctl & 1:
            if idx + 2 > n:
                raise ValueError("压缩流错误：复制项长度不足")
            lo = comp[idx]; hi = comp[idx+1]; idx += 2
            offset = (hi << 8) | lo
            count = (offset & 7)
            if (offset & 8) == 0:
                if idx >= n:
                    raise ValueError("压缩流错误：长复制项缺少附加字节")
                count = (count << 8) | comp[idx]; idx += 1
            count += 4
            offset >>= 4
            src_pos = dst - offset
            _overlap_copy(out, dst, src_pos, count)
            dst += count
        else:
            if idx >= n:
                raise ValueError("压缩流错误：literal 段长度缺失")
            count = comp[idx]; idx += 1
            if idx + count > n:
                raise ValueError("压缩流错误：literal 段数据不足")
            out[dst:dst+count] = comp[idx:idx+count]
            idx += count
            dst += count
    
    # 确保最终进度为100%
    if progress_cb:
        progress_cb(100, 100)
    
    return out

def _postprocess_method1(unpacked: bytes, width: int, height: int) -> Tuple[bytes, str]:
    plane_size = width * height
    a = np.frombuffer(unpacked[0:plane_size], dtype=np.uint8)
    r = np.frombuffer(unpacked[plane_size:2*plane_size], dtype=np.uint8)
    g = np.frombuffer(unpacked[2*plane_size:3*plane_size], dtype=np.uint8)
    b = np.frombuffer(unpacked[3*plane_size:4*plane_size], dtype=np.uint8)
    out = np.empty((plane_size, 4), dtype=np.uint8)
    out[:,0] = b
    out[:,1] = g
    out[:,2] = r
    out[:,3] = a
    return out.tobytes(), 'BGRA'

def _postprocess_method2(unpacked: bytes, width: int, height: int) -> Tuple[bytes, str]:
    """
    YUV 4:2:0 解码
    
    智能版本选择：
    - Numba 可用：使用加速版本（5-10倍快）
    - Numba 不可用：使用 Python 版本
    """
    # 优先使用 Numba 加速版本
    if ACCELERATOR_AVAILABLE and optimize_yuv_decode is not None:
        result = optimize_yuv_decode(unpacked, width, height)
        return result, 'BGR'
    
    # Python 回退版本
    stride = width * 3
    out = bytearray(stride * height)
    seg = (width * height) // 4
    src0 = 0
    src1 = seg
    src2 = seg + src1
    points = (0, 1, width, width + 1)
    dst = 0
    for _y in range(height // 2):
        for _x in range(width // 2):
            i0 = unpacked[src0]; src0 += 1
            i1 = unpacked[src1]; src1 += 1
            if i0 >= 128: i0 -= 256
            if i1 >= 128: i1 -= 256
            b = 226 * i0
            g = -43 * i0 - 89 * i1
            r = 179 * i1
            for off in points:
                base = unpacked[src2 + off] << 7
                px = dst + 3 * off
                out[px + 0] = _clamp_u8((base + b) >> 7)
                out[px + 1] = _clamp_u8((base + g) >> 7)
                out[px + 2] = _clamp_u8((base + r) >> 7)
            src2 += 2
            dst  += 6
        src2 += width
        dst  += stride
    return bytes(out), 'BGR'

def _postprocess_pal(input_bytes: bytes, width: int, height: int, pixel_size: int) -> bytes:
    stride = width * pixel_size
    out = bytearray(height * stride)
    ctl_pos = 0
    src = height
    dst = 0
    while ctl_pos < height:
        c = input_bytes[ctl_pos]
        ctl_pos += 1
        if c & 1:
            out[dst:dst+pixel_size] = input_bytes[src:src+pixel_size]
            src += pixel_size; dst += pixel_size
            count = stride - pixel_size
            prev = dst - pixel_size
            for _ in range(count):
                out[dst] = (out[prev] - input_bytes[src]) & 0xFF
                dst += 1; prev += 1; src += 1
        elif c & 2:
            prev = dst - stride
            for _ in range(stride):
                out[dst] = (out[prev] - input_bytes[src]) & 0xFF
                dst += 1; prev += 1; src += 1
        else:
            out[dst:dst+pixel_size] = input_bytes[src:src+pixel_size]
            dst += pixel_size; src += pixel_size
            prev = dst - stride
            count = stride - pixel_size
            for _ in range(count):
                avg = (out[prev] + out[dst - pixel_size]) // 2
                out[dst] = (avg - input_bytes[src]) & 0xFF
                dst += 1; prev += 1; src += 1
    return bytes(out)

def _postprocess_method3(unpacked: bytes) -> Tuple[bytes, str, int, int]:
    bpp   = _u16(unpacked, 2)
    width = _u16(unpacked, 4)
    height= _u16(unpacked, 6)
    if bpp not in (24, 32):
        raise ValueError(f"Unsupported bpp in method3: {bpp}")
    body = unpacked[8:]
    pixel_size = bpp // 8
    raw = _postprocess_pal(body, width, height, pixel_size)
    mode = 'BGR' if bpp == 24 else 'BGRA'
    return raw, mode, width, height

def load_pgd(path: str) -> dict:
    with open(path, 'rb') as f:
        header = f.read(0x20)
        if len(header) < 0x20:
            raise ValueError("文件过短或不是 PGD32")
        magic2 = header[0:2]
        if magic2 == b'PG':
            raise ValueError(f"检测到PG格式文件（非标准GE格式）：{path} - 请确认文件来源或修改代码以支持此格式")
        if magic2 != b'GE':
            raise ValueError(f"非 GE/PGD32 文件：magic={magic2!r}")
        hdr_size = _u16(header, 2)
        if hdr_size != 0x20:
            raise ValueError(f"不支持的头大小: {hdr_size}")
        orig_x   = _u32(header, 4)
        orig_y   = _u32(header, 8)
        width    = _u32(header, 12)
        height   = _u32(header, 16)
        method   = _u16(header, 28)
        info = f.read(8)
        if len(info) < 8:
            raise ValueError("PGD 信息区不完整")
        uncompr_len = _u32(info, 0)
        compr_len   = _u32(info, 4)
        comp_data = f.read(compr_len)
        if len(comp_data) != compr_len:
            raise ValueError("压缩数据长度不匹配")
        decomp = _decompress_ge_lz_mem(comp_data, uncompr_len)
        return {
            'width': width,
            'height': height,
            'method': method,
            'unpacked': bytes(decomp),
        }

def pgd_to_png(pgd_path: str, out_path: Optional[str] = None,
               progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """
    将PGD(GE格式)转换为PNG，支持进度回调
    
    参数：
        pgd_path: 输入PGD文件路径
        out_path: 输出PNG文件路径
        progress_cb: 进度回调函数 (done, total)
    """
    def update_progress(step: int):
        """更新进度（0-100）"""
        if progress_cb:
            progress_cb(step, 100)
    
    update_progress(0)  # 0%: 开始
    
    with open(pgd_path, 'rb') as f:
        update_progress(5)  # 5%: 读取头部
        header = f.read(0x20)
        if len(header) < 0x20:
            raise ValueError("文件过短或不是 PGD32")
        magic2 = header[0:2]
        if magic2 == b'PG':
            raise ValueError(f"检测到PG格式文件（非标准GE格式）：{pgd_path} - 请确认文件来源或修改代码以支持此格式")
        if magic2 != b'GE':
            raise ValueError(f"非 GE/PGD32 文件：magic={magic2!r}")
        hdr_size = _u16(header, 2)
        if hdr_size != 0x20:
            raise ValueError(f"不支持的头大小: {hdr_size}")
        
        update_progress(10)  # 10%: 解析头部信息
        orig_x   = _u32(header, 4)
        orig_y   = _u32(header, 8)
        width    = _u32(header, 12)
        height   = _u32(header, 16)
        method   = _u16(header, 28)
        
        update_progress(15)  # 15%: 读取压缩信息
        info = f.read(8)
        if len(info) < 8:
            raise ValueError("PGD 信息区不完整")
        uncompr_len = _u32(info, 0)
        compr_len   = _u32(info, 4)
        
        update_progress(20)  # 20%: 读取压缩数据
        comp_data = f.read(compr_len)
        if len(comp_data) != compr_len:
            raise ValueError("压缩数据长度不匹配")
    
    # 解压阶段 (30-60%)，嵌套进度回调
    def decompress_progress(done: int, total: int):
        """ 将解压进度0-100%映射到30-60% """
        overall_pct = 30 + int(done * 0.30)  # 30 + done * (60-30)/100
        update_progress(overall_pct)
    
    update_progress(30)  # 30%: 开始觨压
    decomp = _decompress_ge_lz_mem(comp_data, uncompr_len, progress_cb=decompress_progress)
    
    update_progress(60)  # 60%: 解压完成，后处理
    if method == 1:
        raw, raw_mode = _postprocess_method1(decomp, width, height)
        update_progress(75)  # 75%: 后处理完成
        pil = Image.frombytes('RGBA', (width, height), raw, 'raw', raw_mode)
    elif method == 2:
        raw, raw_mode = _postprocess_method2(decomp, width, height)
        update_progress(75)  # 75%: 后处理完成
        pil = Image.frombytes('RGB', (width, height), raw, 'raw', raw_mode)
    elif method == 3:
        raw, raw_mode, w2, h2 = _postprocess_method3(decomp)
        update_progress(75)  # 75%: 后处理完成
        pil = (Image.frombytes('RGBA', (w2, h2), raw, 'raw', raw_mode)
               if raw_mode == 'BGRA'
               else Image.frombytes('RGB', (w2, h2), raw, 'raw', raw_mode))
    else:
        raise ValueError(f"不支持的 GE 压缩方法: {method}")
    
    update_progress(85)  # 85%: 创建图像完成
    if out_path is None:
        base, _ = os.path.splitext(pgd_path)
        out_path = base + '.png'
    
    update_progress(90)  # 90%: 保存PNG
    pil.save(out_path, 'PNG')
    
    update_progress(100)  # 100%: 完成
    return out_path

# ------------------ 工具函数 ------------------
def find_files(path: str, extensions: Tuple[str, ...], recursive: bool = True) -> List[str]:
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        if recursive:
            for root, _, filenames in os.walk(path):
                for f in filenames:
                    lf = f.lower()
                    if any(lf.endswith(ext) for ext in extensions):
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(path):
                full = os.path.join(path, f)
                if os.path.isfile(full):
                    lf = f.lower()
                    if any(lf.endswith(ext) for ext in extensions):
                        files.append(full)
        return sorted(files)
    return []

def _nt_longpath(p: str) -> str:
    p = os.path.abspath(p)
    if os.name == 'nt':
        if not p.startswith('\\\\?\\'):
            p = '\\\\?\\' + p
    return p

@dataclass
class ProcessingStats:
    total_input_size: int = 0
    total_output_size: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    file_count: int = 0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def avg_speed(self) -> float:
        return self.total_input_size / self.total_time if self.total_time > 0 else 0

    @property
    def compression_ratio(self) -> float:
        return (self.total_output_size / self.total_input_size * 100) if self.total_input_size > 0 else 0

def format_size(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def format_speed(speed: float) -> str:
    if speed < 1024:
        return f"{speed:.1f} B/s"
    elif speed < 1024*1024:
        return f"{speed/1024:.1f} KB/s"
    else:
        return f"{speed/(1024*1024):.1f} MB/s"

def pgd2png_batch(in_pgd: str, out_png: Optional[str] = None, recursive: bool = False,
                 file_progress_cb: Optional[Callable[[int, int], None]] = None,
                 batch_progress_cb: Optional[Callable[[int, int, float], None]] = None):
    """
    批量处理PGD到PNG，支持进度回调
    
    使用BatchProgressManager统一管理进度，GUI模式下自动节流优化
    """
    input_files = find_files(in_pgd, ('.pgd',), recursive=recursive)
    if not input_files:
        raise ValueError(f"未找到PGD文件：{in_pgd}")
    is_batch = len(input_files) > 1 or os.path.isdir(in_pgd)
    if is_batch and not out_png:
        raise ValueError("批量处理时必须指定输出文件夹")
    
    stats = ProcessingStats()
    stats.start_time = time.time()
    
    # 使用BatchProgressManager统一管理进度
    use_cli = PROGRESS_UTILS_AVAILABLE and batch_progress_cb is None and len(input_files) > 1
    
    try:
        from progress_utils import BatchProgressManager
        progress_manager = BatchProgressManager(
            total_files=len(input_files),
            file_progress_cb=file_progress_cb,
            batch_progress_cb=batch_progress_cb,
            use_cli_progress=use_cli,
            cli_desc="PGD→PNG",
            force_cli_progress=True  # GUI模式下也显示命令行进度
        )
    except ImportError:
        # 回退到原始实现
        progress_manager = None
        if batch_progress_cb:
            batch_progress_cb(0, len(input_files), 0.0)
    
    for i, pgd_path in enumerate(input_files):
        try:
            if is_batch:
                base_name = os.path.splitext(os.path.basename(pgd_path))[0]
            if out_png and os.path.isdir(out_png):
                output_path = os.path.join(out_png, base_name + ".png")
            else:
                output_path = out_png or os.path.splitext(pgd_path)[0] + ".png"
            
            # 使用进度管理器的节流回调
            if progress_manager:
                progress_manager.start_file(pgd_path)
                file_cb = progress_manager.get_file_callback()
            else:
                file_cb = file_progress_cb
            
            # 传递进度回调给pgd_to_png
            result_path = pgd_to_png(pgd_path, output_path, progress_cb=file_cb)
            
            stats.total_input_size += os.path.getsize(pgd_path)
            stats.total_output_size += os.path.getsize(result_path)
            stats.file_count += 1
            
            if progress_manager:
                progress_manager.finish_file(success=True)
                progress_manager.log(f"OK 导出 PNG：{result_path}")
            else:
                log(f"OK 导出 PNG：{result_path}")
                if batch_progress_cb:
                    batch_progress_cb(i + 1, len(input_files), time.time() - stats.start_time)
                    
        except Exception as e:
            error_msg = f"ERROR 处理失败 {pgd_path}: {str(e)}"
            if progress_manager:
                progress_manager.log(error_msg)
                progress_manager.finish_file(success=False)
            else:
                log(error_msg)
            if not is_batch:
                raise
    
    # 关闭进度管理器
    if progress_manager:
        progress_manager.close()
    
    stats.end_time = time.time()
    print("="*60)
    print(f"处理完成。共 {stats.file_count} 个文件")
    print(f"总压缩率：{stats.compression_ratio:.1f}%")
    print(f"总用时：{stats.total_time:.2f} 秒")
    print(f"平均速度：{format_speed(stats.avg_speed)}")

def main():
    parser = argparse.ArgumentParser(description="PGD → PNG 转换工具")
    parser.add_argument('input', help='输入 .pgd 文件或文件夹')
    parser.add_argument('output', nargs='?', default=None, help='输出 .png 文件或文件夹')
    parser.add_argument('--recursive', action='store_true', default=False, help='递归子文件夹')
    args = parser.parse_args()
    try:
        pgd2png_batch(args.input, args.output, recursive=args.recursive)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()