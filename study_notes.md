<h1 style="text-align:center; font-family:Times New Roman; color:blue;">
  YOLOv8<span style="font-family:SimSun;">学习笔记</span

</h1>
<h2 style="text-align:center; font-family:Times New Roman; font-size:20pt;">
  郗高佑

## 一、YOLO原理理解

### 1.1 YOLO核心思想
- **You Only Look Once**: 将目标检测视为单次回归问题
- **Anchor-Free设计**: YOLOv8取消了Anchor Boxes
- **多尺度预测**: 使用FPN结构检测不同大小的物体

### 1.2 YOLOv8架构特点
- **Backbone**: CSPDarknet53
- **Neck**: PAN-FPN
- **Head**: 解耦头（分类和回归分离）

### 1.3 损失函数
- 分类损失：BCE Loss
- 回归损失：CIoU Loss
- 目标损失：DFL Loss

