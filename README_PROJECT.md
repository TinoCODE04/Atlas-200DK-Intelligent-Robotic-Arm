# DOFBOT 垃圾分类 & 电力金具智能识别系统

基于 ROS2 Humble + YOLOv5 + 华为 Atlas 200I DK A2 的智能垃圾分类与电力金具识别抓取系统。机械臂通过摄像头实时识别目标物体，自动规划路径并执行抓取。

---

## 功能特性

- **垃圾分类识别**：识别 4 类垃圾（有害/可回收/厨余/其他），共 16 种物品
- **电力金具识别**：识别 sleeve（铜管）和 terminal_lug（线鼻子）两类电力金具
- **自动抓取分类**：识别到物体后自动执行 9 步抓取流程，放置到对应位置
- **双模式推理**：
  - 原始模式：CANN NPU 推理（Ascend 310B4 OM 模型）
  - ONNX 模式：ONNX Runtime CPU 推理（通用部署）

## 系统架构

```
┌─────────────────────────────────────────────────┐
│                    ROS2 Humble                   │
├─────────────┬───────────────────────────────────┤
│ 运动学服务器 │     dofbot_moveit                 │
│ (IK 反解)   │     kinematic_server.launch.py   │
├─────────────┴───────────────────────────────────┤
│           主程序 (main.py / power_fitting_main.py)  │
│  - 摄像头采集   - 稳定检测   - 抓取编排          │
├────────────────────┬────────────────────────────┤
│  识别模块           │  抓取模块                   │
│  garbage_identify  │  garbage_grap.py            │
│  power_fitting_    │     ↓                       │
│  identify.py       │  9步运动流程                 │
│  (YOLOv5推理)      │  - 复位 → 抬起 → 下降 →     │
│                    │    抓取 → 架起 → 旋转 →      │
│  NPU/ONNX推理      │    释放 → 抬起               │
├────────────────────┴────────────────────────────┤
│              Arm_Lib 机械臂驱动                    │
│          (串口通信 / Servo 控制)                   │
└─────────────────────────────────────────────────┘
```

## 硬件平台

| 组件 | 型号/规格 |
|------|----------|
| 开发板 | 华为 Atlas 200I DK A2 |
| 芯片 | Ascend 310B4 |
| CANN 版本 | 23.0.rc3 |
| 机械臂 | DOFBOT 6-DOF |
| 摄像头 | USB 摄像头（640×480） |
| 夹爪 | Servo 控制（张开 0° / 闭合 130°-150°） |

## 软件环境

### 开发板（Linux）
- OS: Ubuntu 22.04 / 华为 Mind Studio 镜像
- ROS2: Humble Hawksbill
- Python: 3.8+
- 依赖: `Arm_Lib`, `rclpy`, `cv_bridge`, `opencv-python`, `numpy`, `onnxruntime`

### 训练环境（Windows）
- OS: Windows 11
- GPU: NVIDIA RTX 4060 Laptop (8GB)
- Python: 3.10+
- 依赖: `torch`, `torchvision`, `opencv-python`

---

## 项目结构

```
dofbot_garbage_yolov5/
├── main.py                        # 垃圾分类主程序入口
├── power_fitting_main.py          # 电力金具主程序入口
├── collect_data.py                # 数据采集（拍照保存）
├── auto_annotate.py               # 自动标注（基于预训练模型）
├── split_dataset.py               # 数据集划分（train/val/test）
├── train_simple.py                # YOLOv5 训练脚本
├── offset.txt                     # 坐标偏移量配置
│
├── config/
│   ├── XYT_config.txt             # 机械臂初始位置校准参数
│   ├── dp.bin                     # 透视变换矩阵
│   ├── offset.txt                 # x/y 坐标偏移
│   └── data_collect_idx.txt       # 数据采集索引
│
├── data/
│   ├── power_fitting.yaml         # 电力金具数据集配置
│   ├── train/                     # 训练集
│   │   ├── images/
│   │   └── labels/
│   └── val/                       # 验证集
│       ├── images/
│       └── labels/
│
├── data_collect/
│   ├── sleeve/                    # 铜管原始照片
│   └── terminal_lug/              # 线鼻子原始照片
│
├── model/
│   ├── coco_names.txt             # COCO 类别名
│   ├── power_fitting_names.txt    # 电力金具类别名
│   ├── best.pt                    # YOLOv5 最佳权重
│   ├── best.onnx                  # 导出 ONNX 模型
│   └── best.onnx.data             # ONNX 外部权重数据
│
├── utils/
│   ├── garbage_identify.py        # 垃圾分类识别模块
│   ├── garbage_grap.py            # 垃圾分类抓取模块
│   ├── power_fitting_identify.py  # 电力金具识别模块（ONNX）
│   ├── dofbot_config.py           # 相机标定/透视变换
│   ├── det_utils.py               # 检测工具函数
│   └── npu_utils.py               # CANN NPU 推理工具
│
├── test/                          # 单元测试
├── tools/                         # 辅助工具（相机校准 Notebook）
└── dofbot_garbage_yolov5/         # ROS2 包构建输出
    ├── setup.py
    ├── package.xml
    └── install/                   # colcon 安装目录
```

---

## 快速开始

### 前置条件

1. 启动运动学服务器（提供 IK 反解服务）
2. 确保机械臂和摄像头已连接

### 终端 1 — 启动运动学服务器

```bash
source /opt/ros/humble/setup.bash
source /home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/install/local_setup.bash
ros2 launch dofbot_moveit kinematic_server.launch.py
```

### 终端 2 — 垃圾分类

```bash
su - HwHiAiUser
cd /home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/src/dofbot_garbage_yolov5/dofbot_garbage_yolov5
source /opt/ros/humble/setup.bash
source /home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/install/local_setup.bash
python3 main.py
```

### 终端 2 — 电力金具识别

```bash
su - HwHiAiUser
cd /home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/src/dofbot_garbage_yolov5/dofbot_garbage_yolov5
source /opt/ros/humble/setup.bash
source /home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/install/local_setup.bash
python3 power_fitting_main.py
```

---

## 工作流程

### 识别流程

1. 摄像头采集帧（640×480）
2. Letterbox 预处理 → 送入 YOLOv5 推理
3. 后处理（置信度过滤 + NMS）
4. 坐标映射（像素 → 机械臂工作空间）
5. 叠加偏移量修正（offset.txt）

### 抓取流程（9 步）

```
复位校准 → 物体上方 → 张开夹爪 → 垂直下降 → 闭合抓取
    → 垂直架起 → 旋转到放置位 → 下降释放 → 抬起复位
```

- 每次抓取前先回到校准位置，确保摄像头正确定位
- 垂直下降方式抓取，避免斜着抓导致滑落
- 全部抓取完成后自动复位到初始位置

### 稳定检测

连续 3 帧检测到相同目标才触发抓取，避免误触发。丢失目标时计数器递减。

---

## 配置参数

### offset.txt — 坐标偏移

```
<y偏移>    # 第1行：y轴偏移，控制抓取前后位置
<x偏移>    # 第2行：x轴偏移，控制抓取左右位置
```

| 方向 | 调整方法 |
|------|---------|
| 抓取太靠前 | 减小第1行 |
| 抓取太靠后 | 增大第1行 |
| 抓取太靠左 | 减小第2行 |
| 抓取太靠右 | 增大第2行 |

### 初始位置

| 参数 | 垃圾分类 | 电力金具 |
|------|---------|---------|
| 关节角度 | `[90, 130, 0, 0, 90, 30]` | `[84, 130, 0, 0, 90, 30]` |

从 `config/XYT_config.txt` 读取。

---

## 模型训练

### 数据采集

```bash
# 拍照采集
python3 collect_data.py sleeve          # 采集铜管
python3 collect_data.py terminal_lug    # 采集线鼻子

# 自动标注
python3 auto_annotate.py data_collect/sleeve data/train/images_sleeve data/train/labels_sleeve

# 数据集划分（8:1:1）
python3 split_dataset.py
```

### 训练（Windows + GPU）

```bash
cd C:\Users\Y9000P\yolov5

python train.py --data ../project/data/power_fitting.yaml `
    --weights yolov5s.pt `
    --epochs 100 `
    --batch-size 8 `
    --img 640 `
    --project runs/train `
    --name power_fitting_100ep `
    --cache --workers 0
```

### 导出 ONNX

```bash
cd C:\Users\Y9000P\yolov5

python export.py --weights runs/train/power_fitting_100ep/weights/best.pt `
    --include onnx --simplify --opset 11
```

---

## 类别说明

### 垃圾分类（4 类 16 种）

| 类别 | 颜色 | 物品 |
|------|------|------|
| 有害垃圾 | 🔴 | Syringe, Used_batteries, Expired_cosmetics, Expired_tablets |
| 可回收 | 🔵 | Zip_top_can, Newspaper, Old_school_bag, Book |
| 厨余垃圾 | 🟢 | Fish_bone, Watermelon_rind, Apple_core, Egg_shell |
| 其他垃圾 | ⚫ | Cigarette_butts, Toilet_paper, Peach_pit, Disposable_chopsticks |

### 电力金具（2 类）

| ID | 名称 | 说明 |
|----|------|------|
| 0 | sleeve | 铜管/套管 |
| 1 | terminal_lug | 线鼻子/接线端子 |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `No module named 'rclpy'` | 先 `source /opt/ros/humble/setup.bash` |
| 机械臂初始位置不对 | 修改 `config/XYT_config.txt` 或代码中的 `xy` 参数 |
| 抓取位置偏前/偏后 | 调整 `offset.txt` 第 1 行 |
| 抓取位置偏左/偏右 | 调整 `offset.txt` 第 2 行 |
| 斜着抓不住 | 已修复为垂直下降抓取 |
| 推理速度慢 | 当前用 CPU ONNX Runtime，预计 3-5 秒/帧 |
| ONNX 输出报错 | 检查 `output_name` 是否为 `output0` |

---

## 技术栈

- **ROS2 Humble** — 机器人操作系统，运动学服务通信
- **YOLOv5** — 目标检测模型（训练 + 推理）
- **ONNX Runtime** — 模型推理引擎
- **CANN** / **Ascend NPU** — 华为昇腾芯片推理（原始方案）
- **MoveIt** — 机械臂运动规划框架
- **OpenCV** — 图像处理 + 透视变换
- **Arm_Lib** — DOFBOT 机械臂驱动库

---

## 作者

电力智能系统综合实践项目
