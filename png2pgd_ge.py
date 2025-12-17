#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[arg-type,possibly-unbound] - memoryview/bytes 兼容性和可选依赖
"""
PNG2PGD (GE 格式) - 将 PNG 转换为 GE 格式 PGD

功能介绍：
  支持将 PNG 图像编码为 GE 格式 PGD 文件
  - 类型 1：BGRA 分平面压缩（无损，支持透明）
  - 类型 2：近似 YUV 4:2:0 压缩（有损，不支持透明）
  - 类型 3：行内差分压缩（无损，支持透明）
  支持单文件和批量处理，支持压缩预设和优化等级

用法：
  单文件转换：
    python png2pgd_ge.py -m <类型> <input.png> [output.pgd] [options]
  
  批量转换：
    python png2pgd_ge.py -m <类型> <input_folder> <output_folder> [options]

命令行参数：
  -m, --method <type>    压缩类型（必需）
                         可选值：1, 2, 3
  input                  输入 PNG 文件或文件夹路径
  output                 输出 PGD 文件或文件夹路径（可选）
  --preset <level>       压缩预设（默认：normal）
                         可选值：fast, normal, max
  --template <path>      模板 PGD 文件路径（用于复制元数据）
  --recursive            递归处理子文件夹（批量模式）
  --quality <1|2|3>      优化等级（默认：2）
                         1=fast, 2=balanced, 3=best
                         需要 pgd_optimizer.py 模块
  --gpu                  启用 GPU 加速（需要 cupy）

示例：
  # 单文件转换（类型 1）
  python png2pgd_ge.py -m 1 input.png
  python png2pgd_ge.py -m 1 input.png output.pgd
  
  # 单文件转换（类型 3，最佳压缩）
  python png2pgd_ge.py -m 3 input.png --preset max
  
  # 批量转换（类型 2，递归）
  python png2pgd_ge.py -m 2 input_folder output_folder --recursive
  
  # 使用模板和优化
  python png2pgd_ge.py -m 3 input.png --template base.pgd --quality 3
  
  # 启用 GPU 加速
  python png2pgd_ge.py -m 1 input.png --gpu --quality 3

依赖：
  必需：numpy, opencv-python, xxhash
  可选：numba (加速), progress_utils (进度条)
  GPU 加速：cupy-cuda11x 或 cupy-cuda12x
"""

import os
import sys
import struct
import time
import argparse
from typing import Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
from collections import defaultdict, deque

# 导入进度工具
try:
    from progress_utils import ConsoleProgressBar
    PROGRESS_UTILS_AVAILABLE = True
except ImportError:
    PROGRESS_UTILS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    print("错误: 需要安装 NumPy: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("错误: 需要安装 OpenCV: pip install opencv-python")
    sys.exit(1)

try:
    import xxhash
    _FAST_HASH = getattr(xxhash, "xxh3_64_intdigest", xxhash.xxh64_intdigest)
except ImportError:
    print("错误: 需要安装 xxhash: pip install xxhash")
    sys.exit(1)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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

# ------------------ 核心编码函数 ------------------
@dataclass
class Pgd32Header:
    magic: bytes
    sizeof_header: int
    orig_x: int
    orig_y: int
    width: int
    height: int
    orig_width: int
    orig_height: int
    compr_method: int
    unknown: int

@dataclass
class Pgd32Info:
    uncomprlen: int
    comprlen: int

def _nt_longpath(p: str) -> str:
    p = os.path.abspath(p)
    if os.name == 'nt':
        if not p.startswith('\\\\?\\'):
            p = '\\\\?\\' + p
    return p

def clamp_u16(x: int) -> int:
    return 0 if x < 0 else (0xFFFF if x > 0xFFFF else x)

def pack_HHHH(a: int, b: int, c: int, d: int) -> bytes:
    return struct.pack("<HHHH", clamp_u16(a), clamp_u16(b), clamp_u16(c), clamp_u16(d))

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

def imread_any(path: str, flags=cv2.IMREAD_UNCHANGED):
    p = _nt_longpath(path)
    try:
        with open(p, 'rb') as f:
            data = f.read()
    except Exception:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, flags)

def write_bytes(path: str, data: bytes):
    p = _nt_longpath(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, 'wb') as f:
        f.write(data)

def read_pgd_header(path: str) -> Pgd32Header:
    p = _nt_longpath(path)
    with open(p, 'rb') as f:
        h = f.read(32)
    if len(h) != 32:
        raise ValueError("PGD 文件头不完整")
    magic, sizeof_header, orig_x, orig_y, width, height, orig_width, orig_height, compr_method, unknown = struct.unpack(
        "<2sHiiIIIIHH", h
    )
    return Pgd32Header(magic, sizeof_header, orig_x, orig_y, width, height, orig_width, orig_height, compr_method, unknown)

def write_pgd_header(fp, hdr: Pgd32Header):
    fp.write(struct.pack(
        "<2sHiiIIIIHH",
        hdr.magic, hdr.sizeof_header, hdr.orig_x, hdr.orig_y,
        hdr.width, hdr.height, hdr.orig_width, hdr.orig_height,
        hdr.compr_method, hdr.unknown
    ))

class FastMatcher:
    def __init__(self, window: int = 4095, k: int = 8, max_bucket: int = 48):
        self.window = window
        self.k = k
        self.max_bucket = max_bucket
        self.table: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_bucket))
        self.mv: Optional[memoryview] = None
        self.n = 0
        self.npbuf: Optional[np.ndarray] = None

    def bind(self, data: Union[bytes, bytearray, memoryview]):
        self.mv = memoryview(data)
        self.n = len(self.mv)
        self.npbuf = np.frombuffer(self.mv, dtype=np.uint8)
        self.table.clear()

    def _h(self, pos: int) -> int:
        if self.mv is None or pos < 0 or pos + self.k > self.n:
            return -1
        return _FAST_HASH(self.mv[pos:pos+self.k])

    def feed(self, pos: int):
        h = self._h(pos)
        if h != -1:
            self.table[h].append(pos)

    def _lcp_len(self, a: int, b: int, limit: int) -> int:
        if limit <= 0:
            return 0
        if NUMBA_AVAILABLE and self.npbuf is not None:
            return self._lcp_len_numba(self.npbuf, a, b, limit)
        mv = self.mv
        i = 0
        while i + 16 <= limit and mv[a+i:a+i+16] == mv[b+i:b+i+16]:
            i += 16
        while i + 4 <= limit and mv[a+i:a+i+4] == mv[b+i:b+i+4]:
            i += 4
        while i < limit and mv[a+i] == mv[b+i]:
            i += 1
        return i

    if NUMBA_AVAILABLE:
        @staticmethod
        @njit(cache=True, fastmath=True)
        def _lcp_len_numba(buf: np.ndarray, a: int, b: int, limit: int) -> int:
            i = 0
            while i + 16 <= limit:
                ok = True
                for j in range(16):
                    if buf[a+i+j] != buf[b+i+j]:
                        ok = False
                        break
                if not ok:
                    break
                i += 16
            while i + 4 <= limit:
                if (buf[a+i] != buf[b+i] or
                    buf[a+i+1] != buf[b+i+1] or
                    buf[a+i+2] != buf[b+i+2] or
                    buf[a+i+3] != buf[b+i+3]):
                    break
                i += 4
            while i < limit and buf[a+i] == buf[b+i]:
                i += 1
            return i

    def find(self, pos: int, max_len: int) -> Tuple[int,int]:
        if self.mv is None or pos + 4 > self.n:
            return (0, 0)
        h = self._h(pos)
        if h == -1:
            return (0, 0)
        bucket = self.table.get(h)
        if not bucket:
            return (0, 0)
        window_start = max(0, pos - self.window)
        best_off, best_len = 0, 0
        n = self.n
        for cand in reversed(bucket):
            if cand < window_start:
                break
            limit = min(max_len, pos - cand, n - pos)
            if limit <= best_len:
                continue
            l = self._lcp_len(pos, cand, limit)
            if l >= 4 and l > best_len:
                best_len = l
                best_off = pos - cand
                if best_len >= max_len:
                    break
        return (best_off, best_len) if best_len >= 4 else (0, 0)

def _pack_copy_short(offset: int, length_minus_4: int) -> bytes:
    packed = (offset << 4) | ((length_minus_4 & 0x7) | 0x8)
    return struct.pack("<H", packed & 0xFFFF)

def _pack_copy_long(offset: int, length_minus_4: int) -> bytes:
    packed16 = (offset << 4) | ((length_minus_4 >> 8) & 0x7)
    return struct.pack("<H", packed16 & 0xFFFF) + bytes([length_minus_4 & 0xFF])

def ge_pre_compress(data: bytes, preset: str = "fast", progress_cb: Optional[Callable[[int, int], None]] = None) -> bytes:
    """
    压缩数据，支持进度回调
    
    使用look-behind LZ压缩算法，通过哈希表加速匹配查找。
    支持惰性匹配优化，在normal和max模式下启用。
    
    Args:
        data: 要压缩的原始数据
        preset: 压缩预设
            - 'fast': 快速 (k=8, bucket=32, lazy=1)
            - 'normal': 标准 (k=8, bucket=48, lazy=2) [默认]
            - 'max': 最大 (k=8, bucket=64, lazy=2)
        progress_cb: 进度回调函数 (done, total)
    
    Returns:
        bytes: 压缩后的数据
    
    性能特性:
        - 使用xxhash进行快速哈希计算
        - 内存视图避免数据拷贝
        - 可选Numba JIT加速LCP计算
        - 16字节块匹配加速
    """
    n = len(data)
    if n == 0:
        return b"\x00"
    if preset == "fast":
        k, bucket, lazy = 8, 32, 1
    elif preset == "normal":
        k, bucket, lazy = 8, 48, 2
    elif preset == "max":
        k, bucket, lazy = 8, 64, 2
    else:
        k, bucket, lazy = 8, 48, 2
    matcher = FastMatcher(window=4095, k=k, max_bucket=bucket)
    matcher.bind(data)
    compr = bytearray()
    pos = 0
    max_raw = 255
    max_len = 273
    mv = memoryview(data)
    
    # 命令行进度条
    pbar = None
    if PROGRESS_UTILS_AVAILABLE and progress_cb is None:
        try:
            pbar = ConsoleProgressBar(total=n, desc="压缩", width=50, show_speed=True, show_eta=False)
        except Exception:
            pass

    def _emit_block(flag: int, payload: bytearray):
        compr.append(flag & 0xFF)
        compr.extend(payload)

    last_progress = 0
    while pos < n:
        flag = 0
        payload = bytearray()
        blocks = 0
        start_pos = pos
        while blocks < 8 and pos < n:
            off, length = matcher.find(pos, max_len)
            if length >= 4:
                best_len, best_off = length, off
                for look in (1, 2):
                    if pos + look < n:
                        o2, l2 = matcher.find(pos + look, max_len)
                        if l2 > best_len + look:
                            best_len, best_off = l2, o2
                            length = 0
                            break
            if length >= 4:
                flag |= (1 << blocks)
                off = min(off, 4095)
                l4 = length - 4
                if length <= 11:
                    payload += _pack_copy_short(off, l4)
                else:
                    payload += _pack_copy_long(off, l4)
                end = pos + length
                feed_end = max(pos, end - k + 1)
                while pos < feed_end:
                    matcher.feed(pos)
                    pos += 1
                pos = end
            else:
                start = pos
                run = 1
                matcher.feed(pos)
                pos += 1
                while run < max_raw and pos < n:
                    o2, l2 = matcher.find(pos, max_len)
                    if l2 >= 4:
                        break
                    matcher.feed(pos)
                    pos += 1
                    run += 1
                payload.append(run)
                payload += mv[start:start+run]
            blocks += 1
        _emit_block(flag, payload)
        
        # 更新进度
        if progress_cb:
            progress_cb(min(pos, n), n)
        elif pbar:
            pbar.update(pos - start_pos)
            
        # 更新GUI进度（粗略）
        current_progress = int(min(pos, n) / n * 100)
        if current_progress != last_progress:
            last_progress = current_progress
    
    if pbar:
        pbar.close()
    return bytes(compr)

_Kb = 226/128.0
_Kr = 179/128.0
_KgU = -43/128.0
_KgV = -89/128.0

def alpha_blend_with_color(bgra: np.ndarray, fill_bgr: Tuple[int,int,int]=(255,255,255)) -> np.ndarray:
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    if bgra.shape[2] == 3:
        return bgra
    b, g, r, a = cv2.split(bgra)
    alpha = a.astype(np.float32) / 255.0
    inv_alpha = 1.0 - alpha
    fb, fg, fr = [np.float32(c) for c in fill_bgr]
    b = (b.astype(np.float32) * alpha + fb * inv_alpha).astype(np.uint8)
    g = (g.astype(np.float32) * alpha + fg * inv_alpha).astype(np.uint8)
    r = (r.astype(np.float32) * alpha + fr * inv_alpha).astype(np.uint8)
    return cv2.merge([b, g, r])

def ge1_encode_from_bgra(bgra: np.ndarray) -> bytes:
    h, w, c = bgra.shape
    assert c == 4
    plane = w * h
    out = np.empty(plane * 4, dtype=np.uint8)
    B = bgra[:,:,0].reshape(-1)
    G = bgra[:,:,1].reshape(-1)
    R = bgra[:,:,2].reshape(-1)
    A = bgra[:,:,3].reshape(-1)
    out[0:plane] = A
    out[plane:2*plane] = R
    out[2*plane:3*plane] = G
    out[3*plane:4*plane] = B
    return out.tobytes()

def ge2_encode_from_bgr(bgr: np.ndarray) -> bytes:
    h, w, c = bgr.shape
    if (w % 2) or (h % 2):
        raise ValueError("压缩类型 2 编码要求偶数尺寸")
    B = bgr[:,:,0].astype(np.float32)
    G = bgr[:,:,1].astype(np.float32)
    R = bgr[:,:,2].astype(np.float32)
    Y = (0.114*B + 0.587*G + 0.299*R).round()
    U_est = ((B - Y) / _Kb).reshape(h//2, 2, w//2, 2).mean(axis=(1,3))
    V_est = ((R - Y) / _Kr).reshape(h//2, 2, w//2, 2).mean(axis=(1,3))
    G_pred = Y + (_KgU * U_est.repeat(2, axis=0).repeat(2, axis=1) + _KgV * V_est.repeat(2, axis=0).repeat(2, axis=1))
    Y += (G - G_pred) * 0.25
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    U = np.clip(np.round(U_est), -128, 127).astype(np.int8)
    V = np.clip(np.round(V_est), -128, 127).astype(np.int8)
    out = bytearray()
    out += U.tobytes(order="C")
    out += V.tobytes(order="C")
    out += Y.tobytes(order="C")
    return bytes(out)

def ge3_encode(img: np.ndarray, bpp: int) -> bytes:
    h, w = img.shape[:2]
    ch = 4 if bpp == 32 else 3
    assert img.shape[2] == ch
    flags = np.ones((h,), dtype=np.uint8)
    out = bytearray()
    out += pack_HHHH(7, bpp, w, h)
    out += flags.tobytes()
    for y in range(h):
        row = img[y]
        out += row[0].tobytes()
        dif = (row[:-1].astype(np.int16) - row[1:].astype(np.int16)) & 0xFF
        out += dif.astype(np.uint8).tobytes()
    return bytes(out)

def read_png_rgba(path: str, force_bpp: Optional[int] = None) -> Tuple[np.ndarray, int]:
    im = imread_any(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"无法读取 PNG：{path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    elif im.shape[2] == 4:
        pass
    else:
        raise ValueError("不支持的 PNG 通道数")
    if force_bpp == 24:
        bgr = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return bgr, 24
    return im, 32

def find_template_for_image(image_path: str, template_folder: str) -> Optional[str]:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    template_path = os.path.join(template_folder, base_name + '.pgd')
    return template_path if os.path.exists(template_path) else None

def png2pgd_single(png_path: str, ctype: int, out_pgd: Optional[str], 
                   template_pgd: Optional[str], preset: str, fill_bgr: Tuple[int,int,int],
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   quality_level: int = 2,
                   use_gpu: bool = False) -> str:
    """
    转换单个PNG到PGD，支持进度回调、优化级别和GPU加速
    
    Args:
        png_path: 输入PNG文件路径
        ctype: 压缩类型 (1=BGRA分平面, 2=YUV 4:2:0, 3=行内差分)
        out_pgd: 输出PGD文件路径，None时自动生成
        template_pgd: 模板PGD文件或文件夹路径，用于获取元数据
        preset: 压缩预设 ('fast', 'normal', 'max')
        fill_bgr: 类型2透明混合填充色 (B,G,R)
        progress_cb: 进度回调函数 (done, total)，范围0-100
        quality_level: 质量级别 (1=快速, 2=平衡, 3=最佳)
        use_gpu: 是否启用GPU加速
    
    Returns:
        str: 输出PGD文件的路径
    
    Raises:
        ValueError: 当参数无效或文件无法处理时
        FileNotFoundError: 当输入文件不存在时
    
    Example:
        >>> png2pgd_single('input.png', 2, 'output.pgd', None, 'normal', (255,255,255))
        'output.pgd'
    """
    if ctype == 2:
        test_im = imread_any(png_path, cv2.IMREAD_UNCHANGED)
        if test_im is not None and test_im.ndim == 3 and test_im.shape[2] == 4:
            log(f"WARNING {png_path} 包含透明通道，类型2不支持透明，将与填充色混合")
    
    current_template = None
    if template_pgd and os.path.isdir(template_pgd):
        current_template = find_template_for_image(png_path, template_pgd)
        if not current_template:
            log(f"WARNING 未找到模板PGD for {png_path}，使用默认参数")
    elif template_pgd and os.path.isfile(template_pgd):
        current_template = template_pgd

    if current_template and os.path.exists(current_template):
        base = read_pgd_header(current_template)
        width, height = base.width, base.height
        im = imread_any(png_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError(f"无法读取 PNG：{png_path}")
        interp = cv2.INTER_AREA if (im.shape[1] > width or im.shape[0] > height) else cv2.INTER_LANCZOS4
        if (im.shape[1], im.shape[0]) != (width, height):
            im = cv2.resize(im, (width, height), interpolation=interp)
        
        if ctype == 2:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            bgr = alpha_blend_with_color(im, fill_bgr)
            if (width % 2) or (height % 2):
                new_w = (width + 1) & ~1
                new_h = (height + 1) & ~1
                pad_right = new_w - width
                pad_bottom = new_h - height
                bgr = cv2.copyMakeBorder(bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
                width, height = new_w, new_h
            # 使用优化版本（如果可用）或GPU加速
            pre = None  # 初始化，防止未定义变量错误
            
            if use_gpu:
                try:
                    from pgd_gpu_accelerator import GPUEncoder
                    gpu_enc = GPUEncoder(device_id=0)
                    pre = gpu_enc.encode_ge2_gpu(bgr)
                    log("✓ 使用 GPU 加速编码")
                except Exception as e:
                    log(f"⚠️ GPU加速失败: {e}，回退到CPU")
                    # GPU加速失败，回退到CPU优化版本
                    try:
                        from pgd_optimizer import ge2_encode_optimized
                        pre = ge2_encode_optimized(bgr, quality_level=quality_level)
                    except ImportError:
                        pre = ge2_encode_from_bgr(bgr)
            elif quality_level != 2:
                # 使用优化版本
                try:
                    from pgd_optimizer import ge2_encode_optimized
                    pre = ge2_encode_optimized(bgr, quality_level=quality_level)
                except ImportError:
                    pre = ge2_encode_from_bgr(bgr)
            else:
                # 使用标准版本
                pre = ge2_encode_from_bgr(bgr)
            
            # 最终检查：确保 pre 已定义
            if pre is None:
                log("WARNING: 所有编码尝试失败，使用标准编码器")
                pre = ge2_encode_from_bgr(bgr)
            unlen = (width*height)//4 + (width*height)//4 + width*height
        elif ctype == 1:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            # GE1 暂时不支持优化，使用标准版本
            pre = ge1_encode_from_bgra(im)
            unlen = width*height*4
        elif ctype == 3:
            if im.ndim == 2:
                bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                pre = ge3_encode(bgr, 24)
            elif im.shape[2] == 3:
                pre = ge3_encode(im, 24)
            elif im.shape[2] == 4:
                pre = ge3_encode(im, 32)
            else:
                raise ValueError("不支持的 PNG 通道数")
            unlen = len(pre)
        else:
            raise ValueError("压缩类型必须为 1/2/3")
    else:
        im_rgba, _ = read_png_rgba(png_path)
        h, w = im_rgba.shape[:2]
        if ctype == 1:
            # GE1 不支持优化，使用标准版本
            pre = ge1_encode_from_bgra(im_rgba)
            unlen = w*h*4
            width, height = w, h
        elif ctype == 2:
            bgr = alpha_blend_with_color(im_rgba, fill_bgr)
            if (w % 2) or (h % 2):
                new_w = (w + 1) & ~1
                new_h = (h + 1) & ~1
                pad_right = new_w - w
                pad_bottom = new_h - h
                bgr = cv2.copyMakeBorder(bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
                w, h = new_w, new_h
            # 使用优化版本（如果可用）或GPU加速
            pre = None  # 初始化，防止未定义变量错误
            
            if use_gpu:
                try:
                    from pgd_gpu_accelerator import GPUEncoder
                    gpu_enc = GPUEncoder(device_id=0)
                    pre = gpu_enc.encode_ge2_gpu(bgr)
                    log("✓ 使用 GPU 加速编码")
                except Exception as e:
                    log(f"⚠️ GPU加速失败: {e}，回退到CPU")
                    try:
                        from pgd_optimizer import ge2_encode_optimized
                        pre = ge2_encode_optimized(bgr, quality_level=quality_level)
                    except ImportError:
                        pre = ge2_encode_from_bgr(bgr)
            elif quality_level != 2:
                try:
                    from pgd_optimizer import ge2_encode_optimized
                    pre = ge2_encode_optimized(bgr, quality_level=quality_level)
                except ImportError:
                    pre = ge2_encode_from_bgr(bgr)
            else:
                pre = ge2_encode_from_bgr(bgr)
            
            # 最终检查：确保 pre 已定义
            if pre is None:
                log("WARNING: 所有编码尝试失败，使用标准编码器")
                pre = ge2_encode_from_bgr(bgr)
            unlen = (w*h)//4 + (w*h)//4 + w*h
            width, height = w, h
        elif ctype == 3:
            if im_rgba.shape[2] == 4:
                pre = ge3_encode(im_rgba, 32)
                width, height = w, h
            else:
                bgr = cv2.cvtColor(im_rgba, cv2.COLOR_BGRA2BGR)
                pre = ge3_encode(bgr, 24)
                width, height = w, h
            unlen = len(pre)
        else:
            raise ValueError("压缩类型必须为 1/2/3")
    
    # 压缩时传入进度回调
    comp = ge_pre_compress(pre, preset=preset, progress_cb=progress_cb)
    if out_pgd is None:
        base, _ = os.path.splitext(png_path)
        out_pgd = base + ".pgd"
    
    if current_template and os.path.exists(current_template):
        base_hdr = read_pgd_header(current_template)
        hdr = Pgd32Header(
            magic=b"GE", sizeof_header=0x20, orig_x=base_hdr.orig_x, orig_y=base_hdr.orig_y,
            width=width, height=height, orig_width=base_hdr.orig_width, orig_height=base_hdr.orig_height,
            compr_method=ctype, unknown=base_hdr.unknown,
        )
    else:
        hdr = Pgd32Header(
            magic=b"GE", sizeof_header=0x20, orig_x=0, orig_y=0,
            width=width, height=height, orig_width=width, orig_height=height,
            compr_method=ctype, unknown=0,
        )
    
    info = Pgd32Info(unlen, len(comp))
    p_out = _nt_longpath(out_pgd)
    with open(p_out, 'wb') as f:
        write_pgd_header(f, hdr)
        f.write(struct.pack('<II', info.uncomprlen, info.comprlen))
        f.write(comp)
    return out_pgd

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

def png2pgd_batch(in_png: str, ctype: int, out_pgd: Optional[str] = None,
                  template_pgd: Optional[str] = None, preset: str = 'normal',
                  fill_bgr: Tuple[int,int,int] = (255,255,255), recursive: bool = False,
                  file_progress_cb: Optional[Callable[[int, int], None]] = None,
                  batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                  quality_level: int = 2,
                  use_gpu: bool = False):
    """
    批量处理PNG到PGD，支持进度回调、优化级别和GPU加速
    
    使用BatchProgressManager统一管理进度，GUI模式下自动节流优化
    """
    input_files = find_files(in_png, ('.png',), recursive=recursive)
    if not input_files:
        raise ValueError(f"未找到PNG文件：{in_png}")
    is_batch = len(input_files) > 1 or os.path.isdir(in_png)
    if is_batch and not out_pgd:
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
            cli_desc="PNG→PGD",
            force_cli_progress=True  # GUI模式下也显示命令行进度
        )
    except ImportError:
        # 回退到原始实现
        progress_manager = None
        if batch_progress_cb:
            batch_progress_cb(0, len(input_files), 0.0)
    
    for i, png_path in enumerate(input_files):
        try:
            output_path = os.path.join(out_pgd, os.path.splitext(os.path.basename(png_path))[0] + ".pgd") if is_batch else (out_pgd or os.path.splitext(png_path)[0] + ".pgd")
            
            # 使用进度管理器的节流回调
            if progress_manager:
                progress_manager.start_file(png_path)
                file_cb = progress_manager.get_file_callback()
            else:
                file_cb = file_progress_cb
            
            png2pgd_single(png_path, ctype, output_path, template_pgd, preset, fill_bgr, 
                          progress_cb=file_cb, quality_level=quality_level, use_gpu=use_gpu)
            
            stats.total_input_size += os.path.getsize(png_path)
            stats.total_output_size += os.path.getsize(output_path)
            stats.file_count += 1
            
            if progress_manager:
                progress_manager.finish_file(success=True)
                progress_manager.log(f"OK 生成 PGD：{output_path}")
            else:
                log(f"OK 生成 PGD：{output_path}")
                if batch_progress_cb:
                    batch_progress_cb(i + 1, len(input_files), time.time() - stats.start_time)
                    
        except Exception as e:
            error_msg = f"ERROR 处理失败 {png_path}: {str(e)}"
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

def _parse_rgb_text(s: str) -> Tuple[int,int,int]:
    try:
        parts = s.strip().split(',')
        if len(parts) != 3:
            raise ValueError
        r, g, b = [int(p.strip()) for p in parts]
        for v in (r, g, b):
            if v < 0 or v > 255:
                raise ValueError
        return (b, g, r)
    except Exception:
        raise ValueError("填充色格式应为 R,G,B 且范围 0-255")

def main():
    parser = argparse.ArgumentParser(description="PNG → PGD 转换工具")
    parser.add_argument('input', help='输入 .png 文件或文件夹')
    parser.add_argument('-m', '--type', dest='ctype', type=int, required=True, choices=[1,2,3], help='压缩类型 (1/2/3)')
    parser.add_argument('output', nargs='?', default=None, help='输出 .pgd 文件或文件夹')
    parser.add_argument('-t', '--template', help='模板 .pgd 或文件夹（文件夹时自动匹配同名文件）')
    parser.add_argument('--preset', default='normal', choices=['fast','normal','max'], help='压缩预设')
    parser.add_argument('--fill-color', dest='fill_color', default='255,255,255', help='类型2透明混合填充色，格式 R,G,B （默认 255,255,255）')
    parser.add_argument('--recursive', action='store_true', default=False, help='递归子文件夹')
    args = parser.parse_args()
    try:
        fill_bgr = _parse_rgb_text(args.fill_color)
    except Exception as e:
        print(f"WARNING: --fill-color 解析失败：{e}，使用默认 255,255,255")
        fill_bgr = (255,255,255)
    try:
        png2pgd_batch(args.input, args.ctype, args.output, args.template, args.preset, fill_bgr, recursive=args.recursive)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()