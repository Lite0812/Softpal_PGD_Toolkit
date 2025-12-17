#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGD2PNG (Others 格式) - 将多种 PGD 变体转换为 PNG

功能介绍：
  支持解码以下 PGD 变体格式：
  - PGD/00_C：LZ 压缩 + TGA 格式
  - PGD/11_C：LZ 分平面 ARGB 格式
  - PGD/TGA：PGD 头 + 原生 TGA 数据
  - PGD3/PGD2：基于 GE 的增量图（XOR 叠加到基图）
  支持单文件和批量处理，支持进度回调

用法：
  单文件转换：
    python pgd2png_others.py <input.pgd> [output.png]
  
  批量转换：
    python pgd2png_others.py <input_folder> <output_folder> [--recursive]

命令行参数：
  input          输入 PGD 文件或文件夹路径
  output         输出 PNG 文件或文件夹路径（可选）
  --recursive    递归处理子文件夹（批量模式）

示例：
  # 单文件转换
  python pgd2png_others.py image.pgd
  python pgd2png_others.py image.pgd output.png
  
  # 批量转换（自动识别类型）
  python pgd2png_others.py input_folder output_folder
  python pgd2png_others.py input_folder output_folder --recursive

依赖：
  必需：numpy, pillow
  可选：pgd2png_ge (用于 PGD3 基图解码)

注意：
  - PGD3 格式需要同目录下有 pgd2png_ge.py 模块
  - 会自动检测 PGD 文件类型，无需手动指定
  - 支持与 pgd_type_detector.py 模块集成
"""
import io
import os
import sys
import time
import struct
import argparse
from typing import Optional, Tuple, List, Callable

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    print("需要安装依赖：pip install pillow numpy")
    raise

# --- 尝试导入 GE 解码器（用户已提供 pgd2png_ge.py） ---
try:
    import pgd2png_ge as ge
except ImportError:
    ge = None

def _u16(b, off=0): return struct.unpack_from('<H', b, off)[0]
def _u32(b, off=0): return struct.unpack_from('<I', b, off)[0]
def _i32(b, off=0): return struct.unpack_from('<i', b, off)[0]

def _nt_longpath(p: str) -> str:
    """Windows长路径兼容处理"""
    p = os.path.abspath(p)
    if os.name == 'nt':
        if not p.startswith('\\\\?\\'):
            p = '\\\\?\\' + p
    return p

# ----------------- 00/11 共用解压（look-behind LZ） -----------------
def _unpack_lookbehind(comp: bytes, out_len: int, look_behind: int,
                       progress_cb: Optional[Callable[[int, int], None]] = None) -> bytes:
    """
    Look-behind LZ解压，支持进度回调
    
    Args:
        comp: 压缩数据
        out_len: 解压后长度
        look_behind: 回看窗口大小
        progress_cb: 进度回调函数 (done, total)，范围0-100
    """
    out = bytearray(out_len)
    dst = 0
    idx = 0
    ctl = 2
    n = len(comp)
    
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
                raise ValueError("流结束：缺少控制字节")
            ctl = comp[idx] | 0x100
            idx += 1
        if ctl & 1:
            if idx + 3 > n:
                raise ValueError("流结束：复制段不足")
            src = comp[idx] | (comp[idx+1] << 8); idx += 2
            count = comp[idx]; idx += 1
            if dst > look_behind:
                src += dst - look_behind
            # 与 Binary.CopyOverlapped 等价（允许重叠）
            for i in range(count):
                out[dst+i] = out[src+i]
            dst += count
        else:
            if idx >= n:
                raise ValueError("流结束：literal 长度缺失")
            count = comp[idx]; idx += 1
            if idx + count > n:
                raise ValueError("流结束：literal 数据不足")
            out[dst:dst+count] = comp[idx:idx+count]
            idx += count
            dst += count
    
    # 确保最终进度为100%
    if progress_cb:
        progress_cb(100, 100)
    
    return bytes(out)

# ----------------- GE-LZ（PGD/GE 与 PGD3 共用）-----------------
def _unpack_ge_lz(comp: bytes, out_len: int,
                  progress_cb: Optional[Callable[[int, int], None]] = None) -> bytes:
    """
    GE-LZ解压，支持进度回调
    
    Args:
        comp: 压缩数据
        out_len: 解压后长度
        progress_cb: 进度回调函数 (done, total)，范围0-100
    """
    # 直接复用 pgd2png_ge 中的实现（若可用）
    if ge is not None and hasattr(ge, "_decompress_ge_lz_mem"):
        # TODO: 如果ge模块支持进度回调，传递progress_cb
        result = ge._decompress_ge_lz_mem(comp, out_len)  # type: ignore
        if progress_cb:
            progress_cb(100, 100)  # 确保完成
        return result
    
    # 轻量回退：拷贝自 ge 脚本逻辑，添加进度支持
    out = bytearray(out_len)
    dst = 0; idx = 0; n = len(comp); ctl = 2
    
    # 进度报告
    last_reported_pct = -1
    
    def report_progress():
        """报告解压进度（0-100%），每2%更新一次"""
        nonlocal last_reported_pct
        if progress_cb and out_len > 0:
            current_pct = int(dst * 100 / out_len)
            if current_pct >= last_reported_pct + 2 or (current_pct == 0 and last_reported_pct == -1):
                progress_cb(current_pct, 100)
                last_reported_pct = current_pct
    
    while dst < out_len:
        # 每次循环都检查并报告进度
        report_progress()
        
        ctl >>= 1
        if ctl == 1:
            if idx >= n: raise ValueError("GE LZ：缺少控制字节")
            ctl = comp[idx] | 0x100; idx += 1
        if ctl & 1:
            if idx + 2 > n: raise ValueError("GE LZ：复制项不足")
            lo = comp[idx]; hi = comp[idx+1]; idx += 2
            offset = (hi << 8) | lo
            count = (offset & 7)
            if (offset & 8) == 0:
                if idx >= n: raise ValueError("GE LZ：长复制项缺少附加字节")
                count = (count << 8) | comp[idx]; idx += 1
            count += 4
            offset >>= 4
            src_pos = dst - offset
            for i in range(count):
                out[dst+i] = out[src_pos+i]
            dst += count
        else:
            if idx >= n: raise ValueError("GE LZ：literal 长度缺失")
            count = comp[idx]; idx += 1
            if idx + count > n: raise ValueError("GE LZ：literal 数据不足")
            out[dst:dst+count] = comp[idx:idx+count]
            idx += count; dst += count
    
    # 确保最终进度为100%
    if progress_cb:
        progress_cb(100, 100)
    
    return bytes(out)

# ----------------- 11_C 平面合成 -----------------
def _planes_to_bgba_ar_order(planes: bytes, w: int, h: int) -> bytes:
    """
    FIX: 修复 11c 颜色通道顺序错误问题
    原实现使用的是 A,R,G,B 平面顺序，但 C# 实现期望的是 B,G,R,A 顺序
    按照 C# Pgd11Format.Read 方法的逻辑重新排列平面顺序
    """
    plane = w * h
    # C# 实现中的平面顺序：B, G, R, A
    b = planes[0*plane:1*plane]      # B 平面 (第一个平面)
    g = planes[1*plane:2*plane]      # G 平面 (第二个平面)  
    r = planes[2*plane:3*plane]      # R 平面 (第三个平面)
    a = planes[3*plane:4*plane]      # A 平面 (第四个平面)
    
    # 输出 BGRA 顺序
    out = bytearray(plane*4)
    dst = 0
    for i in range(plane):
        out[dst+0] = b[i]  # B
        out[dst+1] = g[i]  # G  
        out[dst+2] = r[i]  # R
        out[dst+3] = a[i]  # A
        dst += 4
    return bytes(out)

# ----------------- PGD/TGA 读取 -----------------
def _read_pgd_tga(fp) -> Image.Image:
    # FIX: 修复 TGA 格式读取问题
    # C# 代码 PgdTgaFormat.ReadMetaData 期望 TGA 数据从 0x18 (24字节) 开始
    # 原实现在错误位置读取 TGA 数据
    head = fp.read(0x18)
    if len(head) < 0x18:
        raise ValueError("PGD/TGA: 头过短")
    # 后续为原生 TGA
    tga_bytes = fp.read()
    im = Image.open(io.BytesIO(tga_bytes))
    im.load()
    return im

# ----------------- PGD3 读取（叠加 XOR） -----------------
def _read_pgd3(path: str, progress_cb: Optional[Callable[[int, int], None]] = None) -> Image.Image:
    """
    读取 PGD3 增量图，XOR 到基准 GE
    
    Args:
        path: PGD3文件路径
        progress_cb: 进度回调函数 (done, total)，范围0-100
        
    Returns:
        PIL图像对象
        
    Raises:
        ValueError: 当文件格式错误或找不到基准文件时
        RuntimeError: 当缺少pgd2png_ge模块时
    """
    def update_progress(step: int):
        """更新进度（0-100）"""
        if progress_cb:
            progress_cb(step, 100)
    
    update_progress(5)  # 5%: 开始读取
    
    if ge is None:
        raise RuntimeError("需要同目录 pgd2png_ge.py 以解码基准 GE 文件")
    
    update_progress(10)  # 10%: 模块检查完成
    
    with open(path, "rb") as f:
        update_progress(15)  # 15%: 开始读取头部
        hdr = f.read(0x30)
        if len(hdr) < 0x30:
            raise ValueError("PGD3: 头过短")
        sig = hdr[0:4]
        if sig not in (b'PGD3', b'PGD2'):
            raise ValueError("不是 PGD3/PGD2")
        off_x  = _u16(hdr, 4)
        off_y  = _u16(hdr, 6)
        width  = _u16(hdr, 8)
        height = _u16(hdr, 0x0A)
        bpp    = _u16(hdr, 0x0C)
        base_name = hdr[0x0E:0x0E+0x22].split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
        # 数据区
        unlen, clen = struct.unpack("<II", f.read(8))
        comp = f.read(clen)
        if len(comp) != clen:
            raise ValueError("PGD3: 数据长度不匹配")
    
    update_progress(25)  # 25%: 头部读取完成
    # 解压叠加块（GE-LZ + pal 预测）
    # 解压阶段 (30-45%)，嵌套进度回调
    def decompress_progress(done: int, total: int):
        """ 将解压进度0-100%映射到30-45% """
        overall_pct = 30 + int(done * 0.15)  # 30 + done * (45-30)/100
        if progress_cb:
            progress_cb(overall_pct, 100)
    
    update_progress(30)  # 30%: 开始解压叠加块
    unpacked = _unpack_ge_lz(comp, unlen, progress_cb=decompress_progress)
    update_progress(45)  # 45%: 解压完成
    update_progress(50)  # 50%: 开始后处理
    pixel_size = bpp // 8
    if ge is not None and hasattr(ge, "_postprocess_pal"):
        overlay = ge._postprocess_pal(unpacked, width, height, pixel_size)  # type: ignore
    else:
        # 内置 pal 解码
        stride = width * pixel_size
        overlay = bytearray(height * stride)
        ctl_pos = 0; src = height; dst = 0
        for _row in range(height):
            c = unpacked[ctl_pos]; ctl_pos += 1
            if c & 1:
                overlay[dst:dst+pixel_size] = unpacked[src:src+pixel_size]
                src += pixel_size; dst += pixel_size
                count = stride - pixel_size; prev = dst - pixel_size
                for _ in range(count):
                    overlay[dst] = (overlay[prev] - unpacked[src]) & 0xFF
                    dst += 1; prev += 1; src += 1
            elif c & 2:
                prev = dst - stride
                for _ in range(stride):
                    overlay[dst] = (overlay[prev] - unpacked[src]) & 0xFF
                    dst += 1; prev += 1; src += 1
            else:
                overlay[dst:dst+pixel_size] = unpacked[src:src+pixel_size]
                dst += pixel_size; src += pixel_size
                prev = dst - stride
                count = stride - pixel_size
                for _ in range(count):
                    avg = (overlay[prev] + overlay[dst - pixel_size]) // 2
                    overlay[dst] = (avg - unpacked[src]) & 0xFF
                    dst += 1; prev += 1; src += 1
        overlay = bytes(overlay)
    
    update_progress(60)  # 60%: 叠加块后处理完成
    
    # 解码基准 GE
    update_progress(65)  # 65%: 开始解码基准GE
    base_dir = os.path.dirname(path)
    base_path = os.path.join(base_dir, base_name)
    info = ge.load_pgd(base_path)  # {'width','height','method','unpacked'}
    update_progress(75)  # 75%: 基准GE解码完成
    bw, bh = info['width'], info['height']
    method = info['method']
    if method == 1:
        raw, mode = ge._postprocess_method1(info['unpacked'], bw, bh)  # type: ignore
        pil_mode = 'RGBA'; raw_mode = mode  # BGRA
    elif method == 2:
        raw, mode = ge._postprocess_method2(info['unpacked'], bw, bh)  # type: ignore
        pil_mode = 'RGB';  raw_mode = mode  # BGR
    elif method == 3:
        raw, mode, bw, bh = ge._postprocess_method3(info['unpacked'])  # type: ignore
        pil_mode = 'RGBA' if mode == 'BGRA' else 'RGB'
        raw_mode = mode
    else:
        raise ValueError(f"未知 GE 方法: {method}")
    
    update_progress(80)  # 80%: 开始XOR叠加
    # 叠加 XOR
    pixel_size = 4 if pil_mode == 'RGBA' else 3
    base = bytearray(raw)  # BGRA/BGR
    stride = bw * pixel_size
    src = 0
    # 叠加区起始
    dst = (off_y * bw + off_x) * pixel_size
    gap = (bw - width) * pixel_size
    for _y in range(height):
        for _x in range(width):
            for c in range(pixel_size):
                base[dst + c] ^= overlay[src + c]
            dst += pixel_size
            src += pixel_size
        dst += gap
    
    update_progress(95)  # 95%: XOR叠加完成
    # 输出 PNG
    im = Image.frombytes(pil_mode, (bw, bh), bytes(base), 'raw', raw_mode)
    update_progress(100)  # 100%: 完成
    return im

# ----------------- 类型判定 -----------------
def detect_kind(path: str) -> str:
    p = _nt_longpath(path)
    with open(p, 'rb') as f:
        head = f.read(0x2A)
    if len(head) >= 4 and head[:4] in (b'PGD3', b'PGD2'):
        return 'PGD3'
    if len(head) >= 0x20 and head[:2] == b'GE' and head[0x1C:0x20] == b'11_C':
        return 'PGD/11_C'
    if len(head) >= 0x24 and head[0x18:0x1C] == b'00_C':
        return 'PGD/00_C'
    # PGD/TGA 粗略检查：前 24 字节看作 PGD 头；随后 0x18+0x12 处的宽高应与 8/12 处一致
    if len(head) >= 0x2A:
        x = _i32(head, 0); y = _i32(head, 4)
        w = _u32(head, 8); h = _u32(head, 12)
        if abs(x) <= 0x2000 and abs(y) <= 0x2000 and w != 0 and h != 0 and w == _u16(head, 0x24) and h == _u16(head, 0x26):
            return 'PGD/TGA'
    return 'UNKNOWN'

# ----------------- 主转换逻辑 -----------------
def pgd_to_png(inp: str, out_png: Optional[str] = None,
               progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """
    单个PGD转换为PNG，支持进度回调
    
    参数：
        inp: 输入PGD文件路径
        out_png: 输出PNG文件路径
        progress_cb: 进度回调函数 (done, total)
    """
    # 使用 ConsoleProgressBar 如果没有 GUI 回调
    use_console_progress = progress_cb is None
    console_pbar = None
    
    if use_console_progress:
        try:
            prog_dir = os.path.dirname(os.path.abspath(__file__))
            if prog_dir not in sys.path:
                sys.path.insert(0, prog_dir)
            from progress_utils import ConsoleProgressBar
            # 增加到100步以获得更详细的进度
            console_pbar = ConsoleProgressBar(100, desc="解码PGD(Others)", width=40, show_speed=False)
        except ImportError:
            pass
    
    def update_progress(step: int):
        """更新进度（0-100）"""
        if progress_cb:
            progress_cb(step, 100)
        elif console_pbar:
            console_pbar.current = step
            console_pbar.update(0)
    
    update_progress(5)  # 5%: 开始检测
    kind = detect_kind(inp)
    if kind == 'UNKNOWN':
        raise ValueError("无法识别的 PGD 类型")
    
    update_progress(15)  # 15%: 类型检测完成
    if out_png is None:
        out_png = os.path.splitext(inp)[0] + ".png"
    p_inp = _nt_longpath(inp)
    p_out = _nt_longpath(out_png)
    
    update_progress(20)  # 20%: 读取文件
    with open(p_inp, "rb") as f:
        data = f.read()
    
    update_progress(30)  # 30%: 文件读取完成，开始解码
    if kind == 'PGD/00_C':
        # sizes 在 0x1C 处
        update_progress(35)  # 35%: 解析头部
        unlen = _u32(data, 0x1C)
        clen  = _u32(data, 0x20)
        comp  = data[0x24:0x24+clen]
        
        # 解压阶段 (45-70%)，嵌套进度回调
        def decompress_progress(done: int, total: int):
            """ 将解压进度0-100%映射到45-70% """
            overall_pct = 45 + int(done * 0.25)  # 45 + done * (70-45)/100
            update_progress(overall_pct)
        
        update_progress(45)  # 45%: 开始解压
        raw = _unpack_lookbehind(comp, unlen, 3000, progress_cb=decompress_progress)
        
        update_progress(70)  # 70%: 解压完成，创建图像
        im = Image.open(io.BytesIO(raw))
        im.load()
        
        update_progress(85)  # 85%: 图像加载完成
        im.save(p_out, "PNG")
        
    elif kind == 'PGD/11_C':
        update_progress(35)  # 35%: 解析头部
        w = _u32(data, 0x0C); h = _u32(data, 0x10)
        unlen = _u32(data, 0x20); clen = _u32(data, 0x24)
        comp  = data[0x28:0x28+clen]
        
        # 解压阶段 (45-65%)，嵌套进度回调
        def decompress_progress(done: int, total: int):
            """ 将解压进度0-100%映射到45-65% """
            overall_pct = 45 + int(done * 0.20)  # 45 + done * (65-45)/100
            update_progress(overall_pct)
        
        update_progress(45)  # 45%: 开始解压
        planes = _unpack_lookbehind(comp, unlen, 0xFFC, progress_cb=decompress_progress)
        
        update_progress(65)  # 65%: 解压完成，转换平面
        bgra = _planes_to_bgba_ar_order(planes, w, h)
        
        update_progress(80)  # 80%: 平面转换完成，创建图像
        im = Image.frombytes("RGBA", (w, h), bgra, "raw", "BGRA")
        im.save(p_out, "PNG")
        
    elif kind == 'PGD/TGA':
        # TGA格式优化：细分进度阶段
        update_progress(40)  # 40%: 开始读取TGA数据
        with open(p_inp, 'rb') as f:
            f.seek(0x18)
            tga_bytes = f.read()
        
        update_progress(55)  # 55%: TGA数据读取完成
        update_progress(65)  # 65%: 开始创建图像
        im = Image.open(io.BytesIO(tga_bytes)); im.load()
        
        update_progress(75)  # 75%: 图像解码完成
        update_progress(85)  # 85%: 开始保存
        im.save(p_out, "PNG")
        
    elif kind == 'PGD3':
        # PGD3解码（内部有详细进度：40-95%）
        def pgd3_progress(step: int, total: int):
            # 将PGD3内部的0-100%映射到整体的40-95%
            overall_step = 40 + int(step * 0.55)  # 40 + step * (95-40)/100
            update_progress(overall_step)
        
        update_progress(40)  # 40%: 开始PGD3解码
        im = _read_pgd3(inp, progress_cb=pgd3_progress)
        
        update_progress(95)  # 95%: PGD3解码完成
        im.save(p_out, "PNG")
    else:
        raise RuntimeError("不应到达")
    
    update_progress(95)  # 95%: 保存完成
    
    update_progress(100)  # 100%: 全部完成
    
    if console_pbar:
        console_pbar.close()
    
    return out_png

# ----------------- 批量 -----------------
def find_files(path: str, exts: Tuple[str, ...], recursive: bool) -> List[str]:
    """使用 os.scandir 提升查找速度"""
    if os.path.isfile(path):
        return [path] if path.lower().endswith(exts) else []
    files = []
    if not os.path.isdir(path):
        return files
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file() and entry.name.lower().endswith(exts):
                        files.append(entry.path)
                    elif recursive and entry.is_dir():
                        files.extend(find_files(entry.path, exts, True))
                except OSError:
                    continue
    except (PermissionError, OSError):
        pass
    return sorted(files)

def pgd2png_batch(in_path: str, out_path: Optional[str] = None, recursive: bool = False,
                  file_progress_cb: Optional[Callable[[int, int], None]] = None,
                  batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                  log_cb: Optional[Callable[[str], None]] = None) -> None:
    """
    批量处理 PGD → PNG（Others格式），支持进度回调和CLI进度条
    
    使用BatchProgressManager统一管理进度，GUI模式下自动节流优化
    
    参数：
        in_path: 输入文件或文件夹路径
        out_path: 输出文件夹路径（批量模式必需）
        recursive: 是否递归子目录
        file_progress_cb: 单文件进度回调 (done, total)
        batch_progress_cb: 批量进度回调 (processed, total, elapsed)
        log_cb: 日志回调函数
    """
    inputs = find_files(in_path, (".pgd",), recursive) if os.path.isdir(in_path) else [in_path]
    if not inputs:
        raise ValueError(f"未找到输入文件：{in_path}")
    
    batch = len(inputs) > 1 or os.path.isdir(in_path)
    if batch and not out_path:
        raise ValueError("批量模式需要输出目录")
    
    # 使用BatchProgressManager统一管理进度
    use_cli = batch_progress_cb is None and len(inputs) > 1
    
    try:
        prog_dir = os.path.dirname(os.path.abspath(__file__))
        if prog_dir not in sys.path:
            sys.path.insert(0, prog_dir)
        from progress_utils import BatchProgressManager
        
        progress_manager = BatchProgressManager(
            total_files=len(inputs),
            file_progress_cb=file_progress_cb,
            batch_progress_cb=batch_progress_cb,
            log_cb=log_cb,
            use_cli_progress=use_cli,
            cli_desc="PGD→PNG(Others)",
            force_cli_progress=True  # GUI模式下也显示命令行进度
        )
    except ImportError:
        # 回退到原始实现
        progress_manager = None
        if batch_progress_cb:
            batch_progress_cb(0, len(inputs), 0.0)
    
    # 处理文件
    start_time = time.time()
    for i, p in enumerate(inputs, 1):
        try:
            if batch:
                if out_path:
                    os.makedirs(out_path, exist_ok=True)
                    outp = os.path.join(out_path, os.path.splitext(os.path.basename(p))[0] + ".png")
                else:
                    outp = os.path.splitext(p)[0] + ".png"
            else:
                outp = out_path or (os.path.splitext(p)[0] + ".png")
            
            # 使用进度管理器的节流回调
            if progress_manager:
                progress_manager.start_file(p)
                file_cb = progress_manager.get_file_callback()
            else:
                file_cb = file_progress_cb
            
            # 调用单文件函数
            pgd_to_png(p, outp, progress_cb=file_cb)
            
            if progress_manager:
                progress_manager.finish_file(success=True)
                progress_manager.log(f"[+] {p} -> {outp}")
            else:
                if log_cb:
                    log_cb(f"[+] {p} -> {outp}")
                else:
                    print(f"[+] {p} -> {outp}")
                if batch_progress_cb:
                    batch_progress_cb(i, len(inputs), time.time() - start_time)
                
        except Exception as e:
            error_msg = f"[!] 失败 {p}: {e}"
            if progress_manager:
                # 异常处理时重置进度条状态
                if progress_manager.file_progress_cb:
                    progress_manager.file_progress_cb(100, 100)
                progress_manager.log(error_msg)
                progress_manager.finish_file(success=False)
            else:
                # 异常处理时重置进度回调状态
                if file_progress_cb:
                    file_progress_cb(100, 100)
                if log_cb:
                    log_cb(error_msg)
                else:
                    print(error_msg)
                if batch_progress_cb:
                    batch_progress_cb(i, len(inputs), time.time() - start_time)
            if not batch:
                raise
    
    # 关闭进度管理器
    if progress_manager:
        progress_manager.close()
    elif batch_progress_cb:
        batch_progress_cb(len(inputs), len(inputs), time.time() - start_time)

def main():
    ap = argparse.ArgumentParser(description="PGD(00/11/TGA/PGD3) → PNG (修复版)")
    ap.add_argument("input", help="输入 .pgd 文件或文件夹")
    ap.add_argument("output", nargs="?", default=None, help="输出 .png 文件或文件夹")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    args = ap.parse_args()

    try:
        # 使用新的批量函数，自动支持CLI进度条
        pgd2png_batch(args.input, args.output, recursive=args.recursive)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()