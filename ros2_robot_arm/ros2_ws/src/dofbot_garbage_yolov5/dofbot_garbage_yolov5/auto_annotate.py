#!/usr/bin/env python3
# coding: utf-8
"""git remote add origin https://github.com/TinoCODE04/-Atlas-200DK--2026-14-.git
自动标注工具 - 电力金具 (纯 PIL + NumPy，不需要 OpenCV)
基于颜色分割 + 轮廓检测，自动框出金具并分类

使用方法：
  python3 auto_annotate.py data/train/images data/train/labels
  python3 auto_annotate.py data/val/images data/val/labels
"""

import os
import sys
import numpy as np
from PIL import Image


def bgr_to_hsv(img_bgr):
    """BGR → HSV (纯numpy实现)"""
    img_rgb = img_bgr[..., ::-1]  # BGR → RGB
    r, g, b = img_rgb[..., 0] / 255.0, img_rgb[..., 1] / 255.0, img_rgb[..., 2] / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # H
    h = np.zeros_like(max_c)
    mask = delta != 0
    idx = np.where(mask)
    d = delta[mask]
    mm = max_c[mask]

    h_r = np.mod((g[mask] - b[mask]) / d, 6) * 60
    h_g = ((b[mask] - r[mask]) / d + 2) * 60
    h_b = ((r[mask] - g[mask]) / d + 4) * 60

    h[idx] = np.where(mm == r[mask], h_r,
             np.where(mm == g[mask], h_g, h_b))

    # S
    s = np.zeros_like(max_c)
    s[mask] = np.where(max_c[mask] == 0, 0, delta[mask] / max_c[mask])

    # V
    v = max_c

    h = h.astype(np.float32)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)
    h = ((h / 2)).astype(np.uint8)  # OpenCV: H ∈ [0,180]

    return np.stack([h, s, v], axis=-1)


def morph_open(mask, ksize=3):
    """形态学开操作"""
    from scipy import ndimage
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = ndimage.binary_erosion(mask, kernel)
    return ndimage.binary_dilation(eroded, kernel).astype(np.uint8)


def morph_close(mask, ksize=5):
    """形态学闭操作"""
    from scipy import ndimage
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = ndimage.binary_dilation(mask, kernel)
    return ndimage.binary_erosion(dilated, kernel).astype(np.uint8)


def find_contours_uint8(mask):
    """从二值图中提取轮廓 (简化版，用label + 边界检测)"""
    from scipy import ndimage
    labeled, num = ndimage.label(mask)
    contours = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)  # (y, x) pairs
        if len(coords) < 10:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # 构建简化的轮廓点（矩形四角 + 边缘采样点）
        yy, xx = coords[:, 0], coords[:, 1]
        n = len(coords)
        step = max(1, n // 50)
        pts = coords[::step][:, ::-1]  # → (x, y)
        if len(pts) < 3:
            pts = np.array([[x_min, y_min], [x_max, y_min], 
                           [x_max, y_max], [x_min, y_max]])
        contours.append(pts)
    return contours


def auto_annotate(image_path, output_dir):
    """自动标注单张图片"""
    img_bgr = np.array(Image.open(image_path).convert('RGB'))[:, :, ::-1]
    if img_bgr is None or img_bgr.size == 0:
        print(f"  无法读取: {image_path}")
        return 0

    h, w = img_bgr.shape[:2]

    # HSV颜色分割
    hsv = bgr_to_hsv(img_bgr)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 铜/金色范围
    mask1 = ((h_ch >= 0) & (h_ch <= 30) & (s_ch >= 30) & (v_ch >= 30)).astype(np.uint8)
    mask2 = ((h_ch >= 150) & (h_ch <= 180) & (s_ch >= 30) & (v_ch >= 30)).astype(np.uint8)
    mask3 = ((s_ch >= 15) & (v_ch >= 25)).astype(np.uint8)  # 非灰度区域

    mask = np.clip(mask1 + mask2 + mask3, 0, 1)

    # 形态学操作
    mask = morph_close(mask, ksize=5)
    mask = morph_open(mask, ksize=3)

    # 查找连通区域
    from scipy import ndimage
    labeled, num = ndimage.label(mask)

    # 筛选
    results = []
    min_area = (w * h) * 0.008
    max_area = (w * h) * 0.85

    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)  # (y, x)
        area = len(coords)
        if area < min_area or area > max_area:
            continue

        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        x1, y1 = x_coords.min(), y_coords.min()
        x2, y2 = x_coords.max(), y_coords.max()
        bw, bh = x2 - x1, y2 - y1

        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

        if aspect_ratio > 2.5:
            cls_id = 0  # sleeve（长条形）
        elif area > (w * h) * 0.12:
            cls_id = 1  # terminal_lug（大块状）
        else:
            cls_id = 0

        xc = ((x1 + bw / 2) / w)
        yc = ((y1 + bh / 2) / h)
        nw = bw / w
        nh = bh / h

        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        results.append((cls_id, xc, yc, nw, nh))

    # 写文件
    basename = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(output_dir, basename + ".txt")

    with open(txt_path, 'w') as f:
        for cls_id, xc, yc, nw, nh in results:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    return len(results)


def main():
    if len(sys.argv) < 3:
        print("用法: python3 auto_annotate.py <图片目录> <标签输出目录>")
        print("例:   python3 auto_annotate.py data/train/images data/train/labels")
        return

    img_dir = sys.argv[1]
    label_dir = sys.argv[2]

    os.makedirs(label_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

    if not images:
        print(f"在 {img_dir} 中没有找到图片")
        return

    print("=" * 55)
    print(f"  自动标注工具")
    print(f"  图片目录: {img_dir}")
    print(f"  标签目录: {label_dir}")
    print(f"  图片数量: {len(images)}")
    print("=" * 55)
    print(f"  分类规则：")
    print(f"    sleeve (0)       → 长条形  aspect_ratio > 2.5")
    print(f"    terminal_lug (1) → 块状/大块")
    print(f"  注意：纯PIL实现，无OpenCV依赖")
    print("=" * 55)

    success = 0
    empty = 0
    errors = 0

    for i, img_name in enumerate(images):
        img_path = os.path.join(img_dir, img_name)
        try:
            n = auto_annotate(img_path, label_dir)
            if n > 0:
                success += 1
                if (i + 1) % 20 == 0:
                    print(f"  进度: {i + 1}/{len(images)} ...")
            else:
                empty += 1
                print(f"  ⚠️  未检测到物体: {img_name}")
        except Exception as e:
            errors += 1
            print(f"  ❌ 错误: {img_name} - {e}")

    print(f"\n{'=' * 55}")
    print(f"  ✅ 完成！")
    print(f"  成功标注: {success} 张")
    print(f"  未检测到: {empty} 张")
    print(f"  出错:     {errors} 张")
    print(f"  标签目录: {label_dir}")
    print(f"{'=' * 55}")
    print(f"\n建议操作：")
    print(f"  1. 抽查几张标注结果，确认框的位置和类别")
    print(f"  2. 如果不对，调整颜色参数后重新运行")


if __name__ == "__main__":
    main()
