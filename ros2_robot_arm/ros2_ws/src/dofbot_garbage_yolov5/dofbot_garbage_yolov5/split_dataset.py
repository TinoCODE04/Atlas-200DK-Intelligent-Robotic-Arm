#!/usr/bin/env python3
# coding: utf-8
"""
数据集整理工具
功能：将 data_collect/ 中的照片按比例分入 train/ 和 val/
用法：python3 split_dataset.py
"""

import os
import random
import shutil

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_COLLECT = os.path.join(BASE_DIR, "data_collect")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")

# 类别列表
CLASSES = ["sleeve", "terminal_lug"]

# 验证集比例
VAL_RATIO = 0.2


def split():
    for cls in CLASSES:
        src_dir = os.path.join(DATA_COLLECT, cls)
        if not os.path.exists(src_dir):
            print(f"  跳过: {cls} (目录不存在)")
            continue

        images = sorted(os.listdir(src_dir))
        random.shuffle(images)

        n_total = len(images)
        n_val = max(1, int(n_total * VAL_RATIO))
        n_train = n_total - n_val

        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        # 复制图片
        for img_list, split_dir in [(train_imgs, TRAIN_DIR), (val_imgs, VAL_DIR)]:
            dst_img_dir = os.path.join(split_dir, "images")
            dst_label_dir = os.path.join(split_dir, "labels")
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_label_dir, exist_ok=True)

            for img in img_list:
                src = os.path.join(src_dir, img)
                dst = os.path.join(dst_img_dir, img)
                shutil.copy2(src, dst)

        print(f"  {cls}: 总计 {n_total} 张 | 训练 {n_train} | 验证 {n_val}")

    print("\n✅ 数据集整理完成！")
    print(f"\n请确认数据目录结构：")
    print(f"  data/train/images/  ← 训练图片")
    print(f"  data/train/labels/  ← 训练标签（标注后才有）")
    print(f"  data/val/images/    ← 验证图片")
    print(f"  data/val/labels/    ← 验证标签（标注后才有）")
    print(f"\n下一步：用 LabelImg 标注图片")


if __name__ == "__main__":
    print("=" * 50)
    print("  数据集整理工具")
    print("=" * 50)
    split()
