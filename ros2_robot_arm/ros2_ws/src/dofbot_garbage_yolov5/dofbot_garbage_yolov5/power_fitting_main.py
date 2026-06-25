#!/usr/bin/env python3
# coding: utf-8
"""
电力金具识别 - 主程序
使用 ONNX Runtime 推理，替代原 ais_bench NPU 推理
"""

import os
import sys
from time import sleep

# 确保能导入同包的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2 as cv
import Arm_Lib

from utils.power_fitting_identify import PowerFittingIdentify


def open_camera(preferred=0):
    for index in [preferred, 1, 2, 3]:
        capture = cv.VideoCapture(index)
        if capture.isOpened():
            print("Open camera index:", index)
            return capture
        capture.release()
    raise RuntimeError("Cannot open USB camera. Check /dev/video* or close other camera programs.")


def main(args=None):
    # 创建识别实例
    detector = PowerFittingIdentify()
    # 创建机械臂实例
    arm = Arm_Lib.Arm_Device()
    # 初始位置参数（标定：joint1=84, joint2=130）
    xy = [84, 130]

    joints_0 = [xy[0], xy[1], 0, 0, 90, 30]

    # ========== 重置机械臂位置 ==========
    print("Start Reset Robot Arm Position, Please Wait..")
    arm.Arm_Buzzer_On(1)
    sleep(0.3)
    # 直接回到初始位置（参考垃圾分类代码的 reset 方式）
    joints_0 = [xy[0], xy[1], 0, 0, 90, 30]
    arm.Arm_serial_servo_write6_array(joints_0, 2000)
    sleep(3)
    print("Finish Robot Arm Position Reset!")

    # 打开摄像头
    capture = open_camera(0)

    # 放大显示窗口
    cv.namedWindow("Power Fitting Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("Power Fitting Detection", 1280, 960)

    # 稳定检测参数
    warm_up_count = 0
    last_num = 0
    last_count = 0
    WARMUP_BUFFER = 3

    print("\n开始运行！按 [q] 退出")
    print("=" * 55)

    while capture.isOpened():
        ret, img = capture.read()
        if not ret:
            print("读取图像失败")
            continue

        # 统一图像大小
        img = cv.resize(img, (640, 480))

        # 识别
        result, msg = detector.run(img)

        # 显示
        cv.imshow("Power Fitting Detection", result)

        # 按q退出
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("退出程序")
            break

        # 稳定检测逻辑：连续检测到才执行抓取（参考原代码）
        print("Model warm up at stage:", warm_up_count)
        if warm_up_count != 0 and last_num == warm_up_count:
            last_count += 1
            if last_count > 5:
                warm_up_count = 0
                last_count = 0
        last_num = warm_up_count

        if len(msg) != 0:
            print(f"  [检测中] 识别到: {list(msg.keys())}, 计数: {warm_up_count}/{WARMUP_BUFFER}")
            warm_up_count += 1
            if warm_up_count > WARMUP_BUFFER:
                print(f"\n>>> 稳定检测到物体，开始抓取！")
                detector.grab(msg)
                warm_up_count = 0
                last_count = 0
                print(">>> 抓取完成，继续检测...\n")
        else:
            # 没检测到物体时减少计数
            if warm_up_count > 0:
                warm_up_count = max(0, warm_up_count - 1)
                print(f"  [丢失目标] 计数减少至: {warm_up_count}")

    capture.release()
    cv.destroyAllWindows()

    # 退出前复位
    arm.Arm_serial_servo_write6_array(joints_0, 1000)
    sleep(1)
    print("程序退出，机械臂已复位")


if __name__ == "__main__":
    main()
