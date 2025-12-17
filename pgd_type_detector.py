#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGD 类型检测模块 - 统一的 PGD 格式识别接口

功能介绍：
  提供统一的 PGD 格式检测接口，确保整个项目使用一致的检测逻辑
  支持的格式：
  - GE：标准 GE 格式（压缩方法 1/2/3）
  - PGD/00_C：LZ 压缩的 TGA 格式
  - PGD/11_C：分平面 ARGB 格式
  - PGD/TGA：原生 TGA 格式
  - PGD3/PGD2：增量/XOR 叠加格式

用法：
  作为模块导入：
    from pgd_type_detector import detect_pgd_type
    pgd_type = detect_pgd_type('sample.pgd')
  
  命令行检测：
    python pgd_type_detector.py <pgd_file>

API 说明：
  detect_pgd_type(filepath: str) -> str
    检测 PGD 文件类型
    返回值：'GE', 'PGD/00_C', 'PGD/11_C', 'PGD/TGA', 'PGD3', 'UNKNOWN'
  
  is_ge_format(pgd_type: str) -> bool
    判断是否为 GE 格式
  
  is_others_format(pgd_type: str) -> bool
    判断是否为 Others 系列格式
  
  get_format_description(pgd_type: str) -> str
    获取格式的描述信息

命令行参数：
  pgd_file       要检测的 PGD 文件路径

示例：
  # 命令行检测
  python pgd_type_detector.py sample.pgd
  
  # Python 代码中使用
  from pgd_type_detector import detect_pgd_type, get_format_description
  pgd_type = detect_pgd_type('image.pgd')
  print(f"类型: {pgd_type}")
  print(f"描述: {get_format_description(pgd_type)}")

检测逻辑：
  1. 检测 PGD3/PGD2（魔数 b'PGD3', b'PGD2' 在偏移 0）
  2. 检测 GE 系列（魔数 b'GE' 在偏移 0）
     - PGD/11_C：标记 b'11_C' 在偏移 0x1C
     - PGD/00_C：标记 b'00_C' 在偏移 0x18
     - 标准 GE：压缩方法在 (1,2,3)
  3. 检测 PGD/TGA（无魔数，通过结构验证）

依赖：
  必需：无（仅需 Python 标准库）
"""

import struct
import os
from typing import Optional


def detect_pgd_type(filepath: str) -> str:
    """
    基于文件头魔数和结构特征准确识别 PGD 类型
    
    检测逻辑：
    1. 首先检测 PGD3/PGD2 (魔数 b'PGD3', b'PGD2' 在偏移 0)
    2. 检测 GE 系列 (魔数 b'GE' 在偏移 0)
       - PGD/11_C: 标记 b'11_C' 在偏移 0x1C, 头大小=0x1C
       - PGD/00_C: 标记 b'00_C' 在偏移 0x18, 头大小=0x20
       - 标准 GE: 压缩方法在(1,2,3), 头大小=0x20
    3. 检测 PGD/TGA (无魔数，通过结构验证)
       - 前16字节: x,y,w,h 坐标和尺寸
       - TGA数据从 0x18 开始，宽高匹配
    
    参数:
        filepath: PGD 文件路径
    
    返回:
        'GE'       - 标准 GE 格式 (压缩方法 1/2/3)
        'PGD/00_C' - LZ压缩的TGA格式
        'PGD/11_C' - 分平面ARGB格式  
        'PGD/TGA'  - 原生TGA格式
        'PGD3'     - 增量/XOR叠加格式
        'UNKNOWN'  - 无法识别的格式
    
    示例:
        >>> detect_pgd_type('sample.pgd')
        'GE'
        >>> detect_pgd_type('overlay.pgd3')
        'PGD3'
    """
    try:
        with open(filepath, 'rb') as f:
            # 读取前 0x30 字节用于检测
            header = f.read(0x30)
            if len(header) < 4:
                return 'UNKNOWN'

            # 1. 首先检测 PGD3/PGD2 (魔数在偏移 0)
            if header[:4] in (b'PGD3', b'PGD2'):
                # 验证头长度和基本结构
                if len(header) >= 0x30:
                    # 读取宽高作为额外验证
                    w = struct.unpack_from('<H', header, 8)[0]
                    h = struct.unpack_from('<H', header, 0x0A)[0]
                    if 0 < w <= 16384 and 0 < h <= 16384:  # 合理尺寸范围
                        return 'PGD3'
                return 'UNKNOWN'

            # 2. 检测 GE 系列格式 (魔数在偏移 0)
            if header[:2] == b'GE':
                # 读取头大小
                if len(header) < 4:
                    return 'UNKNOWN'
                hdr_size = struct.unpack_from('<H', header, 2)[0]
                
                # 检测 PGD/11_C (标记在偏移 0x1C)
                if len(header) >= 0x20 and hdr_size == 0x1C:
                    marker = header[0x1C:0x20]
                    if marker == b'11_C':
                        # 验证宽高
                        w = struct.unpack_from('<I', header, 0x0C)[0]
                        h = struct.unpack_from('<I', header, 0x10)[0]
                        if 0 < w <= 16384 and 0 < h <= 16384:
                            return 'PGD/11_C'
                
                # 检测 PGD/00_C (标记在偏移 0x18)
                if len(header) >= 0x24 and hdr_size == 0x20:  # 注意：00_C 的头大小通常是 0x20
                    marker = header[0x18:0x1C]
                    if marker == b'00_C':
                        # 验证宽高
                        w = struct.unpack_from('<I', header, 8)[0]
                        h = struct.unpack_from('<I', header, 0xC)[0]
                        if 0 < w <= 16384 and 0 < h <= 16384:
                            return 'PGD/00_C'
                
                # 检测标准 GE 格式
                if len(header) >= 0x20 and hdr_size == 0x20:
                    # 验证压缩方法
                    method = struct.unpack_from('<H', header, 0x1C)[0]
                    if method in (1, 2, 3):
                        # 验证宽高
                        w = struct.unpack_from('<I', header, 0xC)[0]
                        h = struct.unpack_from('<I', header, 0x10)[0]
                        if 0 < w <= 16384 and 0 < h <= 16384:
                            return 'GE'
                
                return 'UNKNOWN'

            # 3. 检测 PGD/TGA 格式 (无魔数，通过结构验证)
            if len(header) >= 0x2A:
                # 读取前16字节：x, y, w, h
                x, y = struct.unpack_from('<ii', header, 0)
                w, h = struct.unpack_from('<II', header, 8)
                
                # 验证坐标和尺寸合理性
                if abs(x) <= 0x2000 and abs(y) <= 0x2000 and 0 < w <= 16384 and 0 < h <= 16384:
                    # TGA 格式在 0x18 开始有原生 TGA 数据
                    # TGA 文件头第12-13字节是宽度，14-15字节是高度（小端序）
                    if len(header) >= 0x2A:
                        tga_w = struct.unpack_from('<H', header, 0x24)[0]
                        tga_h = struct.unpack_from('<H', header, 0x26)[0]
                        if w == tga_w and h == tga_h:
                            return 'PGD/TGA'
            
            return 'UNKNOWN'
            
    except Exception:
        return 'UNKNOWN'


def is_ge_format(pgd_type: str) -> bool:
    """判断是否为 GE 格式"""
    return pgd_type == 'GE'


def is_others_format(pgd_type: str) -> bool:
    """判断是否为 Others 系列格式"""
    return pgd_type in ('PGD/00_C', 'PGD/11_C', 'PGD/TGA', 'PGD3')


def get_format_description(pgd_type: str) -> str:
    """获取格式的描述信息"""
    descriptions = {
        'GE': '标准 GE 格式 (支持压缩类型 1/2/3)',
        'PGD/00_C': 'LZ 压缩的 TGA 格式',
        'PGD/11_C': '分平面 ARGB 格式',
        'PGD/TGA': '原生 TGA 格式',
        'PGD3': '增量/XOR 叠加格式',
        'UNKNOWN': '未知格式'
    }
    return descriptions.get(pgd_type, '未知格式')


# 兼容性别名（与旧代码保持兼容）
detect_kind = detect_pgd_type


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pgd_type_detector.py <pgd_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在: {filepath}")
        sys.exit(1)
    
    pgd_type = detect_pgd_type(filepath)
    desc = get_format_description(pgd_type)
    
    print(f"文件: {filepath}")
    print(f"类型: {pgd_type}")
    print(f"描述: {desc}")
