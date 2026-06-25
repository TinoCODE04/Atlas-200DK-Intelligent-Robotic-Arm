#!/usr/bin/env python3
# coding: utf-8
"""
电力金具数据集采集工具
使用方法：
  python3 collect_data.py <类别名称>
  例如：python3 collect_data.py sleeve
  按空格键拍照，按q退出
"""

import cv2
import os
import sys
import time
from datetime import datetime

# 类别名称
CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "unknown"

# 保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data_collect", CLASS_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 55)
print(f"  类别: {CLASS_NAME}")
print(f"  保存路径: {SAVE_DIR}")
print(f"  按 [空格] 拍照  |  按 [q] 退出")
print("=" * 55)
print(f"  目标: 每个类别至少拍 80 张")
print(f"  提示: 变换角度、位置，部分叠放、遮挡也要拍")
print("=" * 55)

count = 0
last_save_time = 0
SAVE_INTERVAL = 0.3  # 防止连按太快重复保存

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头失败")
        break

    # 显示预览
    display = frame.copy()
    cv2.putText(display, f"{CLASS_NAME} | Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "SPACE=拍照  Q=退出", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 显示十字准线（帮助对齐）
    h, w = display.shape[:2]
    cv2.line(display, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (255, 0, 0), 1)
    cv2.line(display, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (255, 0, 0), 1)

    cv2.imshow("Data Collection - " + CLASS_NAME, display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print(f"\n采集结束，共 {count} 张")
        print(f"文件保存在: {SAVE_DIR}")
        break

    elif key == ord(' '):
        now = time.time()
        if now - last_save_time < SAVE_INTERVAL:
            continue
        last_save_time = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{CLASS_NAME}_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        count += 1
        print(f"  [{count}] {filename}")

cap.release()
cv2.destroyAllWindows()
