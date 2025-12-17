#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG2PGD (Others 格式) - 将 PNG 转换为多种 PGD 变体

功能介绍：
  支持将 PNG 图像编码为以下 PGD 变体格式：
  - PGD/00_C：TGA + 00_LZ 压缩
  - PGD/11_C：平面 BGRA + 11_LZ 压缩
  - PGD/TGA：PGD 头 + 原生 TGA 数据
  - PGD3：差分叠加格式（完整实现，支持压缩预设）
  支持单文件和批量处理，支持压缩预设和进度回调

用法：
  单文件转换：
    python png2pgd_others.py <input.png> -t <类型> [output.pgd] [options]
  
  批量转换：
    python png2pgd_others.py <input_folder> -t <类型> <output_folder> [options]

命令行参数：
  input              输入 PNG 文件或文件夹路径
  -t, --type <type>  目标 PGD 类型（必需）
                     可选值：00, 11, tga, pgd3
  output             输出 PGD 文件或文件夹路径（可选）
  --preset <level>   压缩预设（默认：max）
                     可选值：fast, normal, max
                     适用于：00, 11, pgd3 类型
  --offset <x,y>     像素偏移，格式：x,y（默认：0,0）
  --recursive        递归处理子文件夹（批量模式）
  --template <path>  PGD3 模板文件路径（PGD3 自动查找基准）
  --base-ge <path>   PGD3 基准 GE 文件路径（手动指定基准）

示例：
  # 单文件转换（PGD/11_C）
  python png2pgd_others.py input.png -t 11
  python png2pgd_others.py input.png -t 11 output.pgd
  
  # 单文件转换（PGD/00_C，最佳压缩）
  python png2pgd_others.py input.png -t 00 --preset max
  
  # 批量转换（PGD/TGA，递归）
  python png2pgd_others.py input_folder -t tga output_folder --recursive
  
  # PGD3 转换（使用模板自动查找基准）
  python png2pgd_others.py input.png -t pgd3 --template base.pgd3
  
  # PGD3 转换（手动指定基准 GE）
  python png2pgd_others.py input.png -t pgd3 --base-ge base.pgd

依赖：
  必需：numpy, pillow
  可选：png2pgd_ge, pgd2png_ge (用于 PGD3 压缩和解码)

注意：
  - PGD3 格式需要 png2pgd_ge.py 和 pgd2png_ge.py 模块
  - 00_C 和 11_C 支持压缩预设（fast/normal/max）
"""
import io
import os
import sys
import struct
import argparse
from typing import Optional, Tuple, List, Callable

try:
    import numpy as np
    from PIL import Image
except Exception:
    print("需要安装依赖：pip install pillow numpy")
    raise

# 复用 GE 压缩（png2pgd_ge）与 GE 解码（pgd2png_ge）
try:
    import png2pgd_ge as ge_enc
except ImportError:
    ge_enc = None

try:
    import pgd2png_ge as ge_dec
except ImportError:
    ge_dec = None



# ----------------- 辅助函数 -----------------
def _u16(b, off=0): return struct.unpack_from('<H', b, off)[0]
def _u32(b, off=0): return struct.unpack_from('<I', b, off)[0]

def _write_u16(x: int) -> bytes:
    return int(x).to_bytes(2, 'little', signed=False)

def _write_u32(x: int) -> bytes:
    return int(x).to_bytes(4, 'little', signed=False)

# ----------------- IO 辅助 -----------------
def _imread_rgba(path: str) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    return im

def _to_bgra_bytes(im: Image.Image) -> bytes:
    # Pillow 内部为 RGBA，转为 BGRA
    r, g, b, a = im.split()
    bgra = Image.merge("RGBA", (b, g, r, a))
    return bgra.tobytes()

def _to_bgr_bytes(im: Image.Image) -> bytes:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    r, g, b, a = im.split()
    bgr = Image.merge("RGB", (b, g, r))
    return bgr.tobytes()


# ----------------- 00/11 Look-Behind 压缩 -----------------
def _pack_lookbehind(raw: bytes, look_behind: int, preset: str = "normal",
                     progress_cb: Optional[Callable[[int, int], None]] = None) -> bytes:
    """
    生成与 PgdReader.Unpack 兼容的 LZ 流：
    - 控制字节：每 8 段，bit=1 为复制 [u16 offset][u8 count]，bit=0 为字面 [u8 len][len bytes]
    - 偏移 offset 为"窗口内偏移"，解码时若 dst>look_behind，会自动加上 (dst - look_behind)
    - 复制长度 count 范围 1..255（我们取 >=3 的匹配，其余走 literal）
    预设：fast/normal/max 调整匹配桶大小与惰性匹配。
    进度回调：支持细粒度进度更新（每处理5%数据更新一次）
    """
    if ge_enc is None or not hasattr(ge_enc, "FastMatcher"):
        # 退化到纯 literal（不会报错，但体积较大）
        return _pack_literal_blocks(raw)
    n = len(raw)
    if n == 0:
        return b"\x00"
    # 预设参数
    if preset == "fast":
        bucket, lazy = 32, 1
    elif preset == "max":
        bucket, lazy = 64, 2
    else:  # normal
        bucket, lazy = 48, 2
    # 匹配器（使用与 GE 压缩相同的哈希表实现）
    matcher = ge_enc.FastMatcher(window=look_behind, k=8, max_bucket=bucket)
    matcher.bind(raw)
    out = bytearray()
    pos = 0
    mv = memoryview(raw)
    
    # 进度报告（每处理2%数据就更新，确保至少15-20个更新点）
    last_reported_pct = -1  # 使用-1确保第一次能报告0%
    
    def emit_group(flag: int, payload: bytearray):
        out.append(flag & 0xFF)
        out.extend(payload)
    
    def report_progress():
        """报告压缩进度（内部0-100%），每2%更新一次"""
        nonlocal last_reported_pct
        if progress_cb and n > 0:
            current_pct = int(pos * 100 / n)
            # 每2%更新一次，避免重复更新
            if current_pct >= last_reported_pct + 2 or (current_pct == 0 and last_reported_pct == -1):
                progress_cb(current_pct, 100)
                last_reported_pct = current_pct
    
    # 主循环
    while pos < n:
        # 每次循环都检查并报告进度（函数内部会控制更新频率）
        report_progress()
        flag = 0
        payload = bytearray()
        blocks = 0
        while blocks < 8 and pos < n:
            # 查找匹配
            off, length = matcher.find(pos, min(255, n - pos))
            if length >= 3:
                # 惰性匹配：如果下一位置能得到更长匹配，则使用下一位置
                if lazy >= 1 and pos + 1 < n:
                    off2, len2 = matcher.find(pos + 1, min(255, n - (pos + 1)))
                    if len2 > length + 1:
                        # 先发一个 literal，再用更长的匹配
                        run = 1
                        payload.append(run)          # literal len
                        payload += mv[pos:pos+run]  # literal bytes
                        matcher.feed(pos)
                        pos += run
                        blocks += 1
                        continue
                # 选择当前匹配
                flag |= (1 << blocks)
                # 我们的 ge_enc.FastMatcher 返回的是距离（pos - cand），需要还原出匹配位置
                match_pos = pos - off
                # 存储窗口内偏移
                store_off = match_pos - max(0, pos - look_behind)
                if store_off < 0:
                    store_off = 0
                if store_off > 0xFFFF:
                    # 超界时退化为 literal（极少出现）
                    store_off = 0
                    length = 1
                    flag &= ~(1 << blocks)
                    payload.append(1)
                    payload += mv[pos:pos+1]
                    matcher.feed(pos)
                    pos += 1
                else:
                    payload += store_off.to_bytes(2, "little")
                    payload.append(length & 0xFF)
                    # 滑动窗口：把匹配区间内的起点都喂入哈希表
                    end = pos + length
                    feed_end = max(pos, end - matcher.k + 1)
                    while pos < feed_end:
                        matcher.feed(pos)
                        pos += 1
                    pos = end
            else:
                # 累积 literal 直到遇到可观匹配或达到255
                start = pos
                run = 1
                matcher.feed(pos)
                pos += 1
                while run < 255 and pos < n:
                    o2, l2 = matcher.find(pos, min(255, n - pos))
                    if l2 >= 3:
                        break
                    matcher.feed(pos)
                    pos += 1
                    run += 1
                payload.append(run)
                payload += mv[start:start+run]
            blocks += 1
        emit_group(flag, payload)
    
    # 确保最终进度为100%
    if progress_cb:
        progress_cb(100, 100)
    
    return bytes(out)


# ----------------- 00/11 简易打包（纯 literal） -----------------
def _pack_literal_blocks(raw: bytes) -> bytes:
    # 控制字节为 0x00，后跟最多 8 个 [len(<=255)][data] 段；重复直到耗尽
    out = bytearray()
    pos = 0
    n = len(raw)
    while pos < n:
        out.append(0x00)  # 8 段 literal
        for _ in range(8):
            if pos >= n:
                break
            run = min(255, n - pos)
            out.append(run)
            out.extend(raw[pos:pos+run])
            pos += run
    return bytes(out)


# ----------------- 写入 00_C -----------------
def _write_00c_from_png(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal',
                        progress_cb: Optional[Callable[[int, int], None]] = None):
    """
    将PNG编码为PGD/00_C格式，支持详细进度回调 (0-100%)
    
    Args:
        png_path: 输入PNG文件路径
        out_pgd: 输出PGD文件路径
        offset_xy: 偏移坐标 (x, y)
        preset: 压缩预设 ('fast', 'normal', 'max')
        progress_cb: 进度回调函数 (done, total)
    """
    def report(step: int):
        if progress_cb:
            progress_cb(step, 100)
    
    report(0)  # 开始
    
    # 阶段1: 读取PNG (0-20%)
    report(5)
    im = Image.open(png_path).convert("RGBA")
    w, h = im.size
    
    report(10)
    # 阶段2: 编码为TGA (20-40%)
    buf = io.BytesIO()
    im.save(buf, format="TGA")
    tga_bytes = buf.getvalue()
    
    report(40)  # 40%: 开始LZ压缩
    # 阶段3: LZ压缩 (40-85%)
    # 使用嵌套进度回调，将内部0-100%映射到整体40-85%
    def compress_progress(done: int, total: int):
        """压缩进度回调：将内部0-100%映射到40-85%"""
        if progress_cb:
            # 40 + (done / total) * (85 - 40)
            overall_pct = 40 + int(done * 0.45)
            progress_cb(overall_pct, 100)
    
    comp = _pack_lookbehind(tga_bytes, look_behind=3000, preset=preset, progress_cb=compress_progress)
    
    report(85)  # 85%: 压缩完成
    # 阶段4: 写入文件 (80-100%)
    unlen = len(tga_bytes); clen = len(comp)
    with open(out_pgd, "wb") as f:
        # 28字节头： x,y,w,h, reserved(2*4), '00_C'
        f.write(struct.pack('<iiII', int(offset_xy[0]), int(offset_xy[1]), w, h))
        f.write(struct.pack('<II', 0, 0))
        f.write(b'00_C')
        # sizes + comp 紧跟其后，数据起始为 0x1C
        f.write(struct.pack('<II', unlen, clen))
        f.write(comp)
    
    report(100)  # 完成


# ----------------- 写入 11_C -----------------
def _write_11c_from_png(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal',
                        progress_cb: Optional[Callable[[int, int], None]] = None):
    """
    将PNG编码为PGD/11_C格式，支持详细进度回调 (0-100%)
    修复 11c 颜色通道顺序错误问题：采用 B,G,R,A 平面顺序（与 C# 实现一致）
    
    Args:
        png_path: 输入PNG文件路径
        out_pgd: 输出PGD文件路径
        offset_xy: 偏移坐标 (x, y)
        preset: 压缩预设 ('fast', 'normal', 'max')
        progress_cb: 进度回调函数 (done, total)
    """
    def report(step: int):
        if progress_cb:
            progress_cb(step, 100)
    
    report(0)  # 开始
    
    # 阶段1: 读取PNG (0-15%)
    report(5)
    im = _imread_rgba(png_path)
    w, h = im.size
    
    report(10)
    bgra = _to_bgra_bytes(im)
    plane = w * h
    
    report(15)
    # 阶段2: 分离平面 (15-40%)
    # 正确顺序：B, G, R, A 四个平面
    planes = bytearray(plane * 4)
    src = 0
    for i in range(plane):
        b = bgra[src+0]; g = bgra[src+1]; r = bgra[src+2]; a = bgra[src+3]
        planes[0*plane + i] = b      # B 平面
        planes[1*plane + i] = g      # G 平面
        planes[2*plane + i] = r      # R 平面
        planes[3*plane + i] = a      # A 平面
        src += 4
        
        # 进度报告 (每25%)
        if i % (plane // 4) == 0:
            report(15 + int(25 * i / plane))
    
    report(40)  # 40%: 开始LZ压缩
    # 阶段3: LZ压缩 (40-85%)
    # 使用嵌套进度回调，将内部0-100%映射到整体40-85%
    def compress_progress(done: int, total: int):
        """压缩进度回调：将内部0-100%映射到40-85%"""
        if progress_cb:
            # 40 + (done / total) * (85 - 40)
            overall_pct = 40 + int(done * 0.45)
            progress_cb(overall_pct, 100)
    
    comp = _pack_lookbehind(bytes(planes), look_behind=0xFFC, preset=preset, progress_cb=compress_progress)
    
    report(85)  # 85%: 压缩完成
    # 阶段4: 写入文件 (85-100%)
    unlen = len(planes); clen = len(comp)
    with open(out_pgd, "wb") as f:
        # 32字节头
        f.write(b'GE' + bytes([0x1C, 0x00]))
        f.write(struct.pack('<iiII', int(offset_xy[0]), int(offset_xy[1]), w, h))
        f.write(struct.pack('<II', 0, 0))  # reserved
        f.write(b'11_C')
        f.write(struct.pack('<II', unlen, clen))
        f.write(comp)
    
    report(100)  # 完成


# ----------------- 写入 PGD/TGA -----------------
def _write_pgd_tga_from_png(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal',
                            progress_cb: Optional[Callable[[int, int], None]] = None):
    """
    将 PNG 写为 PGD/TGA 格式，支持详细进度回调 (0-100%)
    格式：24字节头 + TGA数据
    
    Args:
        png_path: 输入PNG文件路径
        out_pgd: 输出PGD文件路径
        offset_xy: 偏移坐标 (x, y)
        preset: 压缩预设（TGA格式不使用此参数）
        progress_cb: 进度回调函数 (done, total)
    """
    def report(step: int):
        if progress_cb:
            progress_cb(step, 100)
    
    report(0)  # 开始
    
    # 阶段1: 读取PNG (0-40%)
    report(10)
    im = Image.open(png_path).convert("RGBA")
    w, h = im.size
    
    report(40)
    # 阶段2: 编码为TGA (40-70%)
    buf = io.BytesIO()
    im.save(buf, format="TGA")
    tga_bytes = buf.getvalue()
    
    report(70)
    # 阶段3: 写入文件 (70-100%)
    with open(out_pgd, "wb") as f:
        # C# 期望 TGA 数据从 0x18 (24字节) 开始
        f.write(struct.pack('<iiII', int(offset_xy[0]), int(offset_xy[1]), w, h))
        f.write(struct.pack('<Q', 0))  # 8字节保留
        f.write(tga_bytes)
    
    report(100)  # 完成


# ----------------- PGD3 核心实现 -----------------
def read_pgd3_header(path: str) -> dict:
    """
    从 PGD3/PGD2 文件头读取元数据
    """
    with open(path, 'rb') as f:
        hdr = f.read(0x30)
        if len(hdr) < 0x30:
            raise ValueError("文件过短，非 PGD3/PGD2")
        magic = hdr[0:4]
        if magic not in (b'PGD3', b'PGD2'):
            raise ValueError(f"魔数不匹配：{magic!r}")

        offx   = _u16(hdr, 0x04)
        offy   = _u16(hdr, 0x06)
        width  = _u16(hdr, 0x08)
        height = _u16(hdr, 0x0A)
        bpp    = _u16(hdr, 0x0C)
        base_raw = hdr[0x0E:0x0E+0x22]
        base_name = base_raw.split(b'\x00', 1)[0].decode('shift_jis', errors='ignore').strip()

    return {
        'offx': offx, 'offy': offy,
        'width': width, 'height': height,
        'bpp': bpp, 'basename': base_name
    }

def _decode_ge_to_raw(path: str) -> Tuple[bytes, str, int, int]:
    """解码 GE 文件为原始字节"""
    info = ge_dec.load_pgd(path)  # type: ignore[union-attr]
    method  = info['method']
    width   = info['width']
    height  = info['height']
    unpack  = info['unpacked']

    if method == 1:
        raw, mode = ge_dec._postprocess_method1(unpack, width, height)  # type: ignore[union-attr]
        return raw, mode, width, height
    elif method == 2:
        raw, mode = ge_dec._postprocess_method2(unpack, width, height)  # type: ignore[union-attr]
        return raw, mode, width, height
    elif method == 3:
        raw, mode, w2, h2 = ge_dec._postprocess_method3(unpack)  # type: ignore[union-attr]
        return raw, mode, w2, h2
    else:
        raise ValueError(f"未知 GE 方法: {method}")

def _read_png_as_bgra(path: str) -> np.ndarray:
    """返回 numpy.uint8 的 BGRA 数组 (H,W,4)"""
    im = Image.open(path).convert('RGBA')
    arr = np.array(im, dtype=np.uint8)
    return arr[..., [2, 1, 0, 3]]  # RGBA -> BGRA

def _find_min_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """mask: (H,W) bool，返回 (x,y,w,h)"""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1

def _encode_row_mode1(row_bytes: bytes, pixel_size: int) -> bytes:
    """行内差分编码（c=1）"""
    out = bytearray()
    out += row_bytes[:pixel_size]
    for i in range(pixel_size, len(row_bytes)):
        out.append((row_bytes[i - pixel_size] - row_bytes[i]) & 0xFF)
    return bytes(out)

def png_to_pgd3(png_path: str,
                base_ge: Optional[str] = None,
                out_path: Optional[str] = None,
                bpp_mode: str = 'auto',
                fullframe: bool = False,
                preset: str = 'normal',
                progress_cb: Optional[Callable[[int, int], None]] = None) -> str:
    """
    将 PNG 与基准 GE 图比较，生成 PGD3（增量）文件。
    
    参数：
    - png_path: 输入 PNG 文件路径
    - base_ge: 基准 GE 文件路径（可选，None 时从模板 PGD （原来的 PGD3 文件）读取）
    - out_path: 输出 PGD3 文件路径（可选，默认同名 .pgd3）
    - bpp_mode: 'auto'自动选択24/32位，'24'强制24位，'32'强制32位
    - fullframe: 是否强制整幅画面作为增量区域
    - preset: 压缩预设（fast/normal/max），影响 GE-LZ 压缩效率
    - progress_cb: 进度回调函数 (done, total)，范围0-100
    """
    def update_progress(step: int):
        """更新进度（0-100）"""
        if progress_cb:
            progress_cb(step, 100)
    
    update_progress(5)  # 5%: 开始
    png_dir = os.path.dirname(os.path.abspath(png_path))
    stem = os.path.splitext(os.path.basename(png_path))[0]

    update_progress(10)  # 10%: 读取模板信息
    
    # 未提供 base_ge 时，从同名模版 .pgd 读取 BaseName
    if base_ge is None:
        cand1 = os.path.join(png_dir, stem + '.pgd3')
        cand2 = os.path.join(png_dir, stem + '.pgd')
        pgd3_hint = None
        if os.path.isfile(cand1):
            pgd3_hint = read_pgd3_header(cand1)
        elif os.path.isfile(cand2):
            try:
                pgd3_hint = read_pgd3_header(cand2)
            except Exception:
                pgd3_hint = None
        if not pgd3_hint or not pgd3_hint['basename']:
            raise FileNotFoundError("未提供 --base，且未能从同名模版 .pgd 读取到 BaseName")
        base_ge = os.path.join(png_dir, pgd3_hint['basename'])

    if not os.path.isfile(base_ge):
        raise FileNotFoundError(f"未找到基准 GE 文件：{base_ge}")

    update_progress(15)  # 15%: 开始读取PNG
    
    # 读取 PNG（BGRA）与基准 GE
    tgt = _read_png_as_bgra(png_path)
    
    update_progress(25)  # 25%: PNG读取完成，开始解码基准GE
    
    base_raw, base_mode, W, H = _decode_ge_to_raw(base_ge)
    if base_mode == 'BGRA':
        base = np.frombuffer(base_raw, dtype=np.uint8).reshape((H, W, 4))
    elif base_mode == 'BGR':
        tmp = np.frombuffer(base_raw, dtype=np.uint8).reshape((H, W, 3))
        base = np.dstack([tmp, np.full((H, W, 1), 255, dtype=np.uint8)])
    else:
        raise ValueError(f"不支持的 GE 模式：{base_mode}")

    if (tgt.shape[0], tgt.shape[1]) != (H, W):
        raise ValueError("PNG 与基准 GE 尺寸不一致")

    update_progress(35)  # 35%: 基准GE解码完成
    
    # 选择 bpp
    if bpp_mode == '24':
        pixel_size = 3
    elif bpp_mode == '32':
        pixel_size = 4
    else:
        alpha_diff = not np.array_equal(tgt[..., 3], base[..., 3])
        pixel_size = 4 if alpha_diff else 3

    update_progress(40)  # 40%: 开始计算差异矩形
    
    # 差异矩形
    cmp_ch = slice(0, pixel_size)
    diff_mask = np.any(tgt[..., cmp_ch] != base[..., cmp_ch], axis=2)
    if fullframe:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = _find_min_rect(diff_mask)
        if w == 0 or h == 0:
            x, y, w, h = 0, 0, 1, 1

    update_progress(50)  # 50%: 差异矩形计算完成，开始XOR叠加
    
    # 计算 overlay = base ^ target
    base_rect = base[y:y+h, x:x+w, :pixel_size].copy()
    tgt_rect  = tgt[y:y+h, x:x+w, :pixel_size].copy()
    overlay   = np.bitwise_xor(base_rect, tgt_rect)

    update_progress(60)  # 60%: XOR叠加完成
    
    # 行序列化 + 行内差分编码
    control = bytes([1] * h)
    body = bytearray()
    row_bytes = overlay.reshape(h, -1)
    for r in range(h):
        body += _encode_row_mode1(row_bytes[r].tobytes(), pixel_size)
    packed_body = control + bytes(body)

    update_progress(70)  # 70%: 行内差分编码完成
    
    # GE-LZ 压缩 - 尝试使用 preset 参数，如果不支持则使用默认参数
    try:
        comp = ge_enc.ge_pre_compress(packed_body, preset=preset)  # type: ignore[union-attr]
    except TypeError:
        # 兼容旧版本的 ge_pre_compress（不支持 preset 参数）
        comp = ge_enc.ge_pre_compress(packed_body)  # type: ignore[union-attr]
    
    update_progress(85)  # 85%: GE-LZ压缩完成
    
    unlen = len(packed_body)
    clen  = len(comp)

    update_progress(90)  # 90%: 开始写入文件
    
    # 写 PGD3
    if out_path is None:
        out_path = os.path.join(png_dir, stem + '.pgd3')
    basename = os.path.basename(base_ge).encode('shift_jis', errors='ignore')
    if len(basename) > 0x21:
        basename = basename[:0x21]
    base_field = basename + b'\x00' * (0x22 - len(basename))

    with open(out_path, 'wb') as f:
        f.write(b'PGD3')
        f.write(_write_u16(x))
        f.write(_write_u16(y))
        f.write(_write_u16(w))
        f.write(_write_u16(h))
        f.write(_write_u16(pixel_size * 8))
        f.write(base_field)
        f.write(_write_u32(unlen))
        f.write(_write_u32(clen))
        f.write(comp)

    update_progress(100)  # 100%: 完成
    
    return out_path


# ----------------- 写入 PGD3（包装函数） -----------------
def _write_pgd3_from_png(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal',
                         progress_cb: Optional[Callable[[int, int], None]] = None):
    """
    PGD3 包装函数：调用 png_to_pgd3 实现，支持详细进度回调 (0-100%)
    offset_xy 参数保留用于接口兼容，实际值从模板读取
    
    Args:
        png_path: 输入PNG文件路径
        out_pgd: 输出PGD3文件路径
        offset_xy: 偏移坐标 (实际从模板读取)
        preset: 压缩预设 ('fast', 'normal', 'max')
        progress_cb: 进度回调函数 (done, total)
    """
    def report(step: int):
        if progress_cb:
            progress_cb(step, 100)
    
    report(0)  # 开始
    
    # 将内部进度（0-100%）传递给外部
    def pgd3_progress(step: int, total: int):
        """PNG→PGD3内部进度直接传递"""
        if progress_cb:
            progress_cb(step, total)
    
    result = png_to_pgd3(
        png_path=png_path,
        base_ge=None,
        out_path=out_pgd,
        bpp_mode='auto',
        fullframe=False,
        preset=preset,
        progress_cb=pgd3_progress
    )
    
    report(100)  # 确保100%
    return result


# ----------------- 导出兼容函数 -----------------
def write_11c(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal'):
    return _write_11c_from_png(png_path, out_pgd, offset_xy, preset)

def write_00c(png_path: str, out_pgd: str, offset_xy=(0,0), preset: str = 'normal'):
    return _write_00c_from_png(png_path, out_pgd, offset_xy, preset)


# ----------------- 批量处理 -----------------
def png2pgd_batch(in_path: str, out_path: Optional[str] = None, pgd_type: str = "11",
                  recursive: bool = False,
                  file_progress_cb: Optional[Callable[[int, int], None]] = None,
                  batch_progress_cb: Optional[Callable[[int, int, float], None]] = None,
                  log_cb: Optional[Callable[[str], None]] = None,
                  preset: str = "normal") -> None:
    """
    批量处理 PNG → PGD(Others格式)，支持进度回调和CLI进度条
    
    使用BatchProgressManager统一管理进度，GUI模式下自动节流优化
    
    参数：
        in_path: 输入文件或文件夹路径
        out_path: 输出文件夹路径（批量模式必需）
        pgd_type: 目标PGD类型 ('00', '11', 'tga', 'pgd3')
        recursive: 是否递归子目录
        file_progress_cb: 单文件进度回调 (done, total)
        batch_progress_cb: 批量进度回调 (processed, total, elapsed)
        log_cb: 日志回调函数
        preset: 压缩预设 ('fast', 'normal', 'max')
    """
    import time
    
    # 查找文件
    inputs = find_files(in_path, (".png",), recursive) if os.path.isdir(in_path) else [in_path]
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
            cli_desc=f"PNG→PGD({pgd_type.upper()})",
            force_cli_progress=True  # GUI模式下也显示命令行进度
        )
    except ImportError:
        # 回退到原始实现
        progress_manager = None
        if batch_progress_cb:
            batch_progress_cb(0, len(inputs), 0.0)
    
    # 选择编码函数
    encode_func_map = {
        "00": _write_00c_from_png,
        "11": _write_11c_from_png,
        "tga": _write_pgd_tga_from_png,
        "pgd3": _write_pgd3_from_png
    }
    
    if pgd_type not in encode_func_map:
        raise ValueError(f"不支持的PGD类型: {pgd_type}")
    
    encode_func = encode_func_map[pgd_type]
    
    # 处理文件
    success_count = 0
    start_time = time.time()
    
    for i, p in enumerate(inputs, 1):
        try:
            if batch:
                if out_path:
                    os.makedirs(out_path, exist_ok=True)
                    outp = os.path.join(out_path, os.path.splitext(os.path.basename(p))[0] + ".pgd")
                else:
                    outp = os.path.splitext(p)[0] + ".pgd"
            else:
                outp = out_path or (os.path.splitext(p)[0] + ".pgd")
            
            # 使用进度管理器的节流回调
            if progress_manager:
                progress_manager.start_file(p)
                file_cb = progress_manager.get_file_callback()
            else:
                file_cb = file_progress_cb
            
            # 调用单文件编码函数
            encode_func(p, outp, offset_xy=(0,0), preset=preset, progress_cb=file_cb)
            
            success_count += 1
            
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
                progress_manager.log(error_msg)
                progress_manager.finish_file(success=False)
            else:
                if log_cb:
                    log_cb(error_msg)
                else:
                    print(error_msg)
            if not batch:
                raise
    
    # 关闭进度管理器
    if progress_manager:
        progress_manager.close()
    elif batch_progress_cb:
        batch_progress_cb(len(inputs), len(inputs), time.time() - start_time)
    
    # 输出统计
    summary = f"\n完成: {success_count}/{len(inputs)} 个文件"
    if progress_manager:
        progress_manager.log(summary)
    elif log_cb:
        log_cb(summary)
    else:
        print(summary)


# ----------------- CLI -----------------
def find_files(path: str, exts: Tuple[str, ...], recursive: bool) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(exts) else []
    files = []
    if recursive:
        for root, _, names in os.walk(path):
            for n in names:
                if n.lower().endswith(exts):
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(path):
            p = os.path.join(path, n)
            if os.path.isfile(p) and n.lower().endswith(exts):
                files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="PNG → PGD(00/11/TGA/PGD3)")
    ap.add_argument("input", help="输入 .png 文件或文件夹")
    ap.add_argument("-t", "--type", dest="kind", required=True, choices=["00","11","tga","pgd3"], help="目标类型")
    ap.add_argument("output", nargs="?", default=None, help="输出 .pgd 文件或目录")
    ap.add_argument("--recursive", action="store_true", help="递归子目录（输入为目录时）")
    ap.add_argument("--preset", default="normal", choices=["fast","normal","max"], help="压缩预设（00/11/PGD3 有效）")
    args = ap.parse_args()

    try:
        # 使用统一的批量处理函数，自动支持CLI进度条
        png2pgd_batch(
            in_path=args.input,
            out_path=args.output,
            pgd_type=args.kind,
            recursive=args.recursive,
            preset=args.preset
        )
    except Exception as e:
        print(f"\n[!] 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()