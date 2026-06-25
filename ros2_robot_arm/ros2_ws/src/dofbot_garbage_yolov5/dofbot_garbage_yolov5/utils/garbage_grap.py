#!/usr/bin/env python
# coding: utf-8

from time import sleep

import Arm_Lib


class GarbageGrapMove:
    def __init__(self):
        # 设置移动状态
        self.move_status = True
        # 创建机械臂实例
        self.arm = Arm_Lib.Arm_Device()
        # 夹爪加紧角度
        self.grap_joint = 130
        # 初始校准位置
        self.xy = [90, 130]

    def move(self, joints, joints_down, xy=None):
        """
        移动过程 - 改进版：垂直下降抓取，避免斜着抓
        :param joints: 移动到物体位置的各关节角度（IK反解结果）
        :param joints_down: 放置位置各关节角度
        :param xy: 初始校准位置 [joint1, joint2]
        """
        # 0. 先回到初始校准位置
        if xy is not None:
            joints_0 = [xy[0], xy[1], 0, 0, 90, 30]
            self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
            sleep(1.5)

        # 1. 在物体正上方（保持水平位置，抬高 z）
        joints_up = [joints[0], joints[1], max(joints[2] + 35, 55), joints[3], 265, 30]
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        sleep(1.2)

        # 2. 张开夹爪
        self.arm.Arm_serial_servo_write(6, 0, 500)
        sleep(0.5)

        # 3. 垂直下降到物体位置
        self.arm.Arm_serial_servo_write6_array(joints, 800)
        sleep(0.8)

        # 4. 闭合夹爪（抓取）
        self.arm.Arm_serial_servo_write(6, self.grap_joint, 800)
        sleep(1.0)

        # 5. 垂直架起
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        sleep(1.2)

        # 6. 旋转到放置位置
        self.arm.Arm_serial_servo_write(1, joints_down[0], 800)
        sleep(0.8)

        # 7. 下降到放置位置
        self.arm.Arm_serial_servo_write6_array(joints_down, 1000)
        sleep(1)

        # 8. 释放物体
        self.arm.Arm_serial_servo_write(6, 30, 500)
        sleep(0.5)

        # 9. 抬起
        joints_up2 = [joints_down[0], 80, 50, 50, 265, 30]
        self.arm.Arm_serial_servo_write6_array(joints_up2, 1000)
        sleep(1)

    def arm_run(self, name, joints):
        """
        机械臂移动函数
        :param name:识别的垃圾名称
        :param joints: 反解求得的各关节角度
        """
        # 有害垃圾--红色
        if (
            name == "Syringe"
            or name == "Used_batteries"
            or name == "Expired_cosmetics"
            or name == "Expired_tablets"
            and self.move_status
        ):
            # 此处设置,需执行完本次操作,才能向下运行
            self.move_status = False
            # 获得目标关节角
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            # 移动到垃圾桶位置放下对应姿态
            joints_down = [45, 50, 20, 60, 265, self.grap_joint]
            # 移动（传入初始位置用于抓前复位）
            self.move(joints, joints_down, self.xy)
            # 移动完毕
            self.move_status = True
        # 可回收垃圾--蓝色
        if (
            name == "Zip_top_can"
            or name == "Newspaper"
            or name == "Old_school_bag"
            or name == "Book"
            and self.move_status
        ):
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [27, 75, 0, 50, 265, self.grap_joint]
            self.move(joints, joints_down, self.xy)
            self.move_status = True
        # 厨余垃圾--绿色
        if (
            name == "Fish_bone"
            or name == "Watermelon_rind"
            or name == "Apple_core"
            or name == "Egg_shell"
            and self.move_status
        ):
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [147, 75, 0, 50, 265, self.grap_joint]
            self.move(joints, joints_down, self.xy)
            self.move_status = True

        # 其他垃圾--灰色
        if (
            name == "Yellow"
            or name == "Cigarette_butts"
            or name == "Toilet_paper"
            or name == "Peach_pit"
            or name == "Disposable_chopsticks"
            and self.move_status
        ):
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [133, 50, 20, 60, 265, self.grap_joint]
            self.move(joints, joints_down, self.xy)
            self.move_status = True
