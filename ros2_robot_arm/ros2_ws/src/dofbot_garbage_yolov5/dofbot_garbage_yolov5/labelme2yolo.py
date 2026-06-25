#!/usr/bin/env python3
# coding: utf-8
"""
LabelMe JSON → YOLO 格式转换工具
使用方法：
  python3 labelme2yolo.py <json文件或目录>
"""

import json
import os
import sys
import base64
import numpy as np


def poly_to_rect(points):
    """多边形点集转矩形 (x_center, y_center, w, h) 归一化"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # 转为中心点 + 宽高，归一化到 0-1
    x_center = ((x1 + x2) / 2) / 640
    y_center = ((y1 + y2) / 2) / 480
    w = (x2 - x1) / 640
    h = (y2 - y1) / 480

    return x_center, y_center, w, h


def convert_one(json_path, class_map):
    """转换单个 json 文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取图像尺寸
    h, w = data['imageHeight'], data['imageWidth']

    # 生成对应的 txt 文件名
    txt_path = json_path.replace('.json', '.txt')

    lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            print(f"  警告：未知标签 '{label}'，跳过")
            continue

        cls_id = class_map[label]

        # 转换坐标
        if len(shape['points']) >= 3:
            # 多边形 → 外接矩形
            xc, yc, bw, bh = poly_to_rect(shape['points'])
        else:
            # 点或线 → 跳过
            print(f"  警告：'{label}' 点数太少，跳过")
            continue

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # 写入 txt
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(lines)


def main():
    if len(sys.argv) < 2:
        print("用法：python3 labelme2yolo.py <目录或json文件>")
        return

    target = sys.argv[1]

    # 类别映射（必须和 power_fitting.yaml 一致）
    class_map = {
        'sleeve': 0,
        'terminal_lug': 1,
    }

    # 查找所有 json 文件
    json_files = []
    if os.path.isfile(target) and target.endswith('.json'):
        json_files = [target]
    elif os.path.isdir(target):
        json_files = [f for f in os.listdir(target) if f.endswith('.json')]
        json_files.sort()
        json_files = [os.path.join(target, f) for f in json_files]

    if not json_files:
        print("没有找到 .json 文件")
        return

    print(f"找到 {len(json_files)} 个标注文件")
    total_shapes = 0

    for jf in json_files:
        n = convert_one(jf, class_map)
        total_shapes += n
        print(f"  ✓ {os.path.basename(jf)} → {n} 个标注")

    print(f"\n✅ 转换完成！共 {total_shapes} 个标注框")
    print(f"标签文件保存在同一目录下（.txt 格式）")


if __name__ == "__main__":
    main()
