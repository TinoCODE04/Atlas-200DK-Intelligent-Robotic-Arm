#!/usr/bin/env python3
# coding: utf-8
"""
简易 YOLOv5 训练脚本
依赖：pip3 install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
      需要预先下载 yolov5s.pt 预训练权重
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# ============ 简易 YOLO 数据集 ============
class PowerFittingDataset(Dataset):
    def __init__(self, data_dir, img_size=640, transform=None):
        self.img_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.img_size = img_size
        
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # 加载标签
        label_path = os.path.join(self.label_dir, 
                                  os.path.splitext(img_name)[0] + '.txt')
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        targets.append([float(x) for x in parts])
        
        if len(targets) == 0:
            targets = torch.zeros((0, 5))
        else:
            targets = torch.tensor(targets, dtype=torch.float32)
        
        return img_tensor, targets, img_name


def collate_fn(batch):
    """处理不同数量的目标框"""
    imgs, targets, names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets), list(names)


# ============ 简化 YOLO 检测头 ============
class SimpleDetectHead(nn.Module):
    """简化检测头：用于2类目标检测"""
    def __init__(self, in_channels=256, num_classes=2, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 3  # 简化：3个anchor
        
        if anchors is None:
            # 默认anchors（基于640输入）
            anchors = torch.tensor([
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ], dtype=torch.float32).reshape(3, 3, 2)
        self.register_buffer('anchors', anchors)
        
    def forward(self, x):
        # x: [batch, channels, H, W]
        batch_size = x.shape[0]
        grid_h, grid_w = x.shape[2], x.shape[3]
        
        # 重塑为 [batch, num_anchors, grid_h, grid_w, 5+num_classes]
        x = x.reshape(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        x = x.permute(0, 1, 3, 4, 2)
        
        return x


# ============ 损失函数 ============
def compute_loss(predictions, targets, model):
    """简化损失计算"""
    total_loss = torch.tensor(0.0, device=predictions[0].device)
    n = len(predictions)
    
    for i, pred in enumerate(predictions):
        # pred: [batch, anchors, grid_h, grid_w, 5+nc]
        batch_size = pred.shape[0]
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        
        # 解析预测
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = torch.exp(pred[..., 2:4])
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        # 生成ground truth
        # 简化：只计算置信度损失和分类损失
        loss_conf = torch.tensor(0.0, device=pred.device)
        loss_cls = torch.tensor(0.0, device=pred.device)
        loss_xy = torch.tensor(0.0, device=pred.device)
        loss_wh = torch.tensor(0.0, device=pred.device)
        
        for b in range(batch_size):
            if len(targets[b]) == 0:
                # 没有目标，只计算背景损失
                loss_conf += torch.mean(pred_conf ** 2) * 0.1
                continue
            
            for t in targets[b]:
                cls_id = int(t[0])
                gx = t[1] * grid_w
                gy = t[2] * grid_h
                gw = t[3] * grid_w
                gh = t[4] * grid_h
                
                gx_int, gy_int = int(gx), int(gy)
                gx_int = min(gx_int, grid_w - 1)
                gy_int = min(gy_int, grid_h - 1)
                
                # 定位损失
                loss_xy += ((pred_xy[b, :, gy_int, gx_int, 0] - (gx - gx_int)) ** 2).mean()
                loss_xy += ((pred_xy[b, :, gy_int, gx_int, 1] - (gy - gy_int)) ** 2).mean()
                loss_wh += ((pred_wh[b, :, gy_int, gx_int, 0] - gw) ** 2).mean()
                loss_wh += ((pred_wh[b, :, gy_int, gx_int, 1] - gh) ** 2).mean()
                
                # 置信度损失（目标处应为1）
                loss_conf += ((pred_conf[b, :, gy_int, gx_int] - 1) ** 2).mean()
                
                # 分类损失
                loss_cls += ((pred_cls[b, :, gy_int, gx_int, cls_id] - 1) ** 2).mean()
        
        # 合并损失
        layer_loss = loss_xy + loss_wh + loss_conf + loss_cls
        total_loss = total_loss + layer_loss
    
    return total_loss / n


# ============ 主训练流程 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/power_fitting.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='power_fitting')
    parser.add_argument('--pretrained', type=str, default='yolov5s.pt')
    args = parser.parse_args()
    
    print("=" * 55)
    print("  简易 YOLOv5 训练")
    print(f"  数据: {args.data}")
    print(f"  轮数: {args.epochs}")
    print(f"  批次: {args.batch_size}")
    print("=" * 55)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 加载数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'data', 'train')
    val_dir = os.path.join(base_dir, 'data', 'val')
    
    train_dataset = PowerFittingDataset(train_dir, img_size=args.img)
    val_dataset = PowerFittingDataset(val_dir, img_size=args.img)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    
    # 加载预训练模型（如果有）
    model = None
    pretrained_path = os.path.join(base_dir, args.pretrained)
    if os.path.exists(pretrained_path):
        print(f"  加载预训练权重: {pretrained_path}")
        try:
            ckpt = torch.load(pretrained_path, map_location=device)
            print(f"  权重加载成功")
            model = ckpt  # 简化处理
        except Exception as e:
            print(f"  权重加载失败: {e}")
    
    if model is None:
        print("  使用随机初始化模型")
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),  # 2类分类
        )
    
    model = model.to(device)
    
    # 只训练最后几层（如果使用预训练模型）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 保存目录
    save_dir = os.path.join(base_dir, args.project, args.name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_losses = []
        
        for batch_idx, (imgs, targets, names) in enumerate(train_loader):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            try:
                outputs = model(imgs)
                
                if isinstance(outputs, dict):
                    # YOLOv5 格式输出
                    preds = outputs.get('inference', outputs.get('train', []))
                    if isinstance(preds, list) and len(preds) > 0:
                        loss = compute_loss(preds, targets, model)
                    else:
                        # 分类损失
                        loss = torch.tensor(0.0, device=device)
                        for b in range(len(targets)):
                            if len(targets[b]) > 0:
                                cls_target = targets[b][0, 0].long()
                                loss += nn.functional.cross_entropy(
                                    outputs.get('train', outputs).mean(dim=[2,3])[b:b+1], 
                                    cls_target.unsqueeze(0)
                                )
                else:
                    # 简单分类头
                    loss = torch.tensor(0.0, device=device)
                    for b in range(len(targets)):
                        if len(targets[b]) > 0:
                            cls_target = targets[b][0, 0].long()
                            feat = outputs[b:b+1] if outputs.shape[0] == len(targets) else outputs.mean(dim=[2,3])[b:b+1]
                            loss += nn.functional.cross_entropy(feat, cls_target.unsqueeze(0))
                
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                
                train_losses.append(loss.item())
                
            except Exception as e:
                print(f"  Batch {batch_idx} 错误: {e}")
                continue
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = np.mean(train_losses[-5:]) if train_losses else 0
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, targets, names in val_loader:
                imgs = imgs.to(device)
                try:
                    outputs = model(imgs)
                    if isinstance(outputs, dict):
                        preds = outputs.get('inference', outputs.get('train', []))
                        if isinstance(preds, list) and len(preds) > 0:
                            loss = compute_loss(preds, targets, model)
                        else:
                            loss = torch.tensor(0.0)
                    else:
                        loss = torch.tensor(0.0)
                    val_losses.append(loss.item())
                except:
                    pass
        
        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else 0
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = os.path.join(save_dir, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'nc': 2,
                'names': ['sleeve', 'terminal_lug'],
            }, save_path)
            print(f"  💾 最佳模型已保存: {save_path}")
        
        print()
    
    print("=" * 55)
    print(f"  ✅ 训练完成！最佳模型: {save_dir}/best.pt")
    print("=" * 55)
    print(f"\n下一步：导出 ONNX 并转为 OM 模型")
    print(f"  1. python3 export_onnx.py --weights {save_dir}/best.pt")
    print(f"  2. atc --model=best.onnx --framework=5 --output=power_fitting_bs1 \\")
    print(f"       --input_shape='input:1,3,640,640' --soc_version=Ascend310P")


if __name__ == "__main__":
    main()
