#!/usr/bin/env python3
# coding: utf-8
"""
电力金具识别模块 - 使用 ONNX Runtime（不需要 OM 模型）
识别两类金具：sleeve（接线管）、terminal_lug（接线鼻子）
"""

import sys
import os
from time import sleep
import rclpy
import cv2 as cv
import numpy as np
import onnxruntime as ort
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pathlib import Path
from dofbot_info.srv import Kinemarics
import Arm_Lib


class PowerFittingIdentify:
    """电力金具识别类 - ONNX Runtime 版本"""

    def __init__(self):
        self.cfg = {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "input_shape": [640, 640],
        }

        # 创建ROS节点
        rclpy.init(args=sys.argv)
        self.node = rclpy.create_node("dofbot_power_fitting")
        self.node_pub = rclpy.create_node("dofbot_pf_img_node")

        # 路径计算：直接使用包目录下的 model/ 和 config/
        FILE = Path(__file__).resolve()
        lib_root = os.path.dirname(FILE.parents[0])
        model_folder = os.path.join(lib_root, "model")
        cfg_folder = os.path.join(lib_root, "config")
        offset_cfg_path = os.path.join(cfg_folder, "offset.txt")

        # ONNX 模型路径
        self.onnx_model_path = os.path.join(model_folder, "best.onnx")
        label_path = os.path.join(model_folder, "power_fitting_names.txt")

        # 加载 ONNX Runtime 模型
        print(f"加载 ONNX 模型: {self.onnx_model_path}")
        self.session = ort.InferenceSession(
            self.onnx_model_path, providers=['CPUExecutionProvider']
        )

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"输入节点: {self.input_name}, 输出节点: {self.output_name}")

        # 加载类别标签
        self.labels_dict = {}
        with open(label_path) as f:
            for i, label in enumerate(f.readlines()):
                self.labels_dict[i] = label.strip()
        print(f"类别标签: {self.labels_dict}")

        # 初始化 - 参考标定界面初始位置 joint1=84, joint2=133
        self.frame = None
        self.arm = Arm_Lib.Arm_Device()
        self.xy = [84, 130]  # 机械臂初始识别位置（标定：joint1=84, joint2微调为130使摄像头俯视更好）
        self.warmup = 0
        self.WARMUP_MAX = 3

        # 读取偏移量
        self.offset = 0.0
        self.x_offset = 0.0
        if os.path.exists(offset_cfg_path):
            with open(offset_cfg_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 1:
                    self.offset = float(lines[0].strip())
                if len(lines) >= 2:
                    self.x_offset = float(lines[1].strip())
        print(f"偏移量: y={self.offset}, x={self.x_offset}")

        # ROS服务
        self.client = self.node.create_client(Kinemarics, "trial_service")

        # 图像发布
        self.image_pub = self.node_pub.create_publisher(Image, "cam_data", 10)
        self.bridge = CvBridge()

        # 移动状态锁
        self.move_status = True

        print("电力金具识别模块(ONNX Runtime)初始化完成!")

    def run(self, frame):
        """
        识别并返回结果
        :param frame: BGR图像 (640x480)
        :return: (绘制结果的图像, 识别结果dict {name: (x, y)})
        """
        self.warmup += 1

        # 预热阶段
        if self.warmup < self.WARMUP_MAX:
            cv.putText(frame, "Model Loading...", (200, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, {}

        # 推理
        try:
            pred, names, drawed = self._infer(frame)
        except Exception as e:
            print(f"推理出错: {e}")
            return frame, {}

        if len(pred) > 0:
            cv.putText(drawed, "Detected!", (30, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 发布图像
        data = self.bridge.cv2_to_imgmsg(drawed, encoding="bgr8")
        self.image_pub.publish(data)

        return drawed, self._parse_result(pred, names, drawed)

    def _infer(self, img_bgr):
        """ONNX Runtime 推理"""
        # Letterbox 预处理
        img, ratio, pad = letterbox(img_bgr, new_shape=(640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis, ...]

        # 推理
        output = self.session.run([self.output_name], {self.input_name: img})[0]
        print(f"  [调试] ONNX输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")

        # 安全处理ONNX输出形状（可能为 (1,N,7) 或 (N,7)）
        pred = np.squeeze(np.array(output))
        if pred.ndim == 1:
            pred = pred[np.newaxis, :]
        print(f"  [调试] squeeze后形状: {pred.shape}")
        if len(pred) == 0:
            return np.empty((0, 6)), self.labels_dict, img_bgr

        # 取最高置信度类别
        conf_scores = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_confs = np.max(cls_scores, axis=1)
        final_conf = conf_scores * cls_confs

        # 阈值过滤
        mask = final_conf >= self.cfg["conf_thres"]
        pred = pred[mask]
        cls_ids = cls_ids[mask]
        final_conf = final_conf[mask]

        if len(pred) == 0:
            return np.empty((0, 6)), self.labels_dict, img_bgr

        # 简单NMS：按中心点距离去重（修复索引 bug）
        kept_indices = []
        boxes = pred[:, :4].copy()
        scores = final_conf.copy()

        order = np.argsort(-scores)
        while order.size > 0:
            i = order[0]
            kept_indices.append(i)
            if len(kept_indices) >= 10:
                break
            cx1 = (boxes[i, 0] + boxes[i, 2]) / 2
            cy1 = (boxes[i, 1] + boxes[i, 3]) / 2
            cx2 = (boxes[order[1:], 0] + boxes[order[1:], 2]) / 2
            cy2 = (boxes[order[1:], 1] + boxes[order[1:], 3]) / 2
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            remain = dist > 50
            order = order[1:][remain]

        kept = pred[kept_indices]
        kept = np.column_stack([kept[:, :4], final_conf[kept_indices], cls_ids[kept_indices]])

        # 还原坐标到原图
        scale_coords((640, 640), kept[:, :4], img_bgr.shape, ratio_pad=(ratio, pad))

        # 画框
        drawed = draw_prediction(kept, img_bgr, self.labels_dict)
        print(f"  [调试] NMS后检测数: {len(kept)}")

        return kept, self.labels_dict, drawed

    def _parse_result(self, pred, names, frame):
        """解析检测结果，计算抓取坐标"""
        msg = {}

        if len(pred) == 0:
            return msg

        for det in reversed(pred):
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            conf = det[4]
            cls = int(det[5])

            name = names.get(cls, f"class_{cls}")

            # 计算中心点（像素坐标）
            point_x = int((x1 + x2) / 2)
            point_y = int((y1 + y2) / 2)

            # 画中心点
            cv.circle(frame, (point_x, point_y), 5, (0, 0, 255), -1)

            # 坐标映射：像素 → 机械臂工作空间
            a = round((point_x - 320) / 2500, 5)
            b = round((480 - point_y) / 3000 * 0.8 + 0.15, 5)
            # 加偏移让机械臂往后退（和原垃圾分类代码一致：加上偏移）
            a = a + self.x_offset
            b = b + self.offset

            msg[name] = (a, b)
            print(f"  检测到: {name}, 位置: ({a}, {b}), 置信度: {conf:.2f}")

        return msg

    # ============ 机械臂抓取 ============

    def grab(self, msg):
        """执行抓取 - 参考原垃圾分类代码"""
        if not msg or not self.move_status:
            return

        self.move_status = False
        self.arm.Arm_Buzzer_On(1)
        sleep(0.3)

        # 按y坐标排序（先抓前面的）
        new_msg = sorted(list(msg.items()), key=lambda x: x[1][1])
        print("new msg is", new_msg)

        for elm in new_msg:
            try:
                name = elm[0]
                raw_pos = msg[name]
                joints = self.server_joint(raw_pos)
                print(f"  [坐标] {name}: 像素坐标→工作空间({raw_pos[0]:.4f}, {raw_pos[1]:.4f}) → 关节角度{[round(j,1) for j in joints]}")
                self._do_grab(name, joints)
            except Exception as e:
                print(f"抓取失败 {name}: {e}")

        # 回到初始位置（复位）
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1)
        print("机械臂已复位到初始位置")
        self.move_status = True

    def server_joint(self, posxy):
        """调用ROS反解服务（坐标已在 _parse_result 中应用偏移）"""
        self.client.wait_for_service(timeout_sec=1.0)
        req = Kinemarics.Request()
        req.tar_x = posxy[0]
        req.tar_y = posxy[1]
        req.kin_name = "ik"

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        res = future.result()

        if res:
            joints = [res.joint1, res.joint2, res.joint3, res.joint4, res.joint5]
            if joints[2] < 0:
                joints[1] += joints[2] / 2
                joints[3] += joints[2] * 3 / 4
                joints[2] = 0
            return joints
        raise RuntimeError("IK求解失败")

    def _do_grab(self, name, joints):
        """执行单次抓取 - 参考原代码 move 函数"""
        grap_joint = 150  # 夹爪闭合角度（最大力度）
        # 安全抬起位置（抬高，joint2=60更垂直）
        joints_uu = [90, 60, 50, 50, 265, 30]
        # 目标位置（张开夹爪）
        joints_grab = [joints[0], joints[1], joints[2], joints[3], 265, 30]
        # 放置位置（sleeve放左边45°，terminal_lug放右边135°）
        if name == "sleeve":
            joints_down = [45, 60, 20, 60, 265, grap_joint]
        else:
            joints_down = [135, 60, 20, 60, 265, grap_joint]

        print(f"  抓取 {name}: 物体位置={[round(j,1) for j in joints]}, 放置位置={joints_down}")

        # 0. 先回到初始校准位置（确保摄像头正确定位）
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1.5)

        # 1. 移动到安全抬起位置（抬高，避免碰撞）
        self.arm.Arm_serial_servo_write6_array(joints_uu, 1500)
        sleep(1.5)
        # 2. 张开夹爪
        self.arm.Arm_serial_servo_write(6, 0, 500)
        sleep(0.5)
        # 3. 下降到物体位置
        self.arm.Arm_serial_servo_write6_array(joints_grab, 1000)
        sleep(1)
        # 4. 闭合夹爪（抓取）- 给足时间完全闭合
        self.arm.Arm_serial_servo_write(6, grap_joint, 800)
        sleep(1.0)
        # 5. 架起
        self.arm.Arm_serial_servo_write6_array(joints_uu, 1500)
        sleep(1.5)
        # 6. 移动到放置位置上方
        self.arm.Arm_serial_servo_write(1, joints_down[0], 1000)
        sleep(1)
        # 7. 下降到放置位置
        self.arm.Arm_serial_servo_write6_array(joints_down, 1500)
        sleep(1.5)
        # 8. 释放物体
        self.arm.Arm_serial_servo_write(6, 30, 500)
        sleep(0.5)
        # 9. 抬起
        self.arm.Arm_serial_servo_write6_array(joints_uu, 1500)
        sleep(1.5)

        print(f"  ✓ {name} 抓取完成！")


# ============ 工具函数 ============

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
    return coords


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    import torchvision
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xyxy2xywh(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        c = x[:, 5:] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]
    return output


def draw_bbox(bbox, img0, color, wt, names):
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4]) < 0.05:
            continue
        img0 = cv.rectangle(
            img0,
            (int(bbox[idx][0]), int(bbox[idx][1])),
            (int(bbox[idx][2]), int(bbox[idx][3])),
            color, wt,
        )
        img0 = cv.putText(
            img0,
            f"{names[int(class_id)]} {bbox[idx][4]:.2f}",
            (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    return img0


def draw_prediction(pred, img_bgr, labels):
    img_dw = draw_bbox(pred, img_bgr, (0, 255, 0), 2, labels)
    return img_dw
