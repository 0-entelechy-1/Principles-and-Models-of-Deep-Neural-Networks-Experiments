# 深度神经网络原理与模型 — 实验合集

本仓库包含《深度神经网络原理与模型》课程的四个实验项目，分别对应回归、分类、目标检测与图像分割任务。每个实验均提供完整代码、输出结果（图片/日志）以及 Markdown 版实验报告。

## 仓库结构

| 实验 | 主题 | 报告链接 |
|------|------|----------|
| **Experiment-1** | 多层感知机（MLP）曲线拟合与梯度分析 | [实验总结一.md](Experiment-1/docs/实验总结一.md) |
| **Experiment-2** | 经典 CNN 图像分类（LeNet / VGG / ResNet / 注意力与轻量化网络） | [实验总结二.md](Experiment-2/docs/实验总结二.md) |
| **Experiment-3** | 目标检测：Faster R-CNN vs YOLO | [实验总结三.md](Experiment-3/docs/实验总结三.md) |
| **Experiment-4** | 图像分割：FCN vs DeepLabV3 | [实验总结四.md](Experiment-4/docs/实验总结四.md) |

## 实验简介

- **Experiment-1**：利用 PyTorch 构建多层感知机，在 `sin(x) + 0.5·cos(2x)` 函数上完成曲线拟合；对比不同优化器、网络结构、学习率，并可视化梯度消失与梯度爆炸现象。
- **Experiment-2**：在 CIFAR-10 数据集上实现 MLP、VGG-style、ResNet-18 等分类模型，验证 SE 注意力、深度可分离卷积、MobileNetV2 / ShuffleNetV2 等轻量化设计的效果。
- **Experiment-3**：使用 MS COCO 2017 真实图像，对比二阶段检测器 Faster R-CNN 与一阶段检测器 YOLO 的检测精度、推理速度与参数量。
- **Experiment-4**：使用真实场景图像，对比 FCN-ResNet50 与 DeepLabV3-ResNet50 的语义分割效果，分析 ASPP、空洞卷积与多尺度特征对分割边界的影响。

## 运行环境

- Python 3.9
- PyTorch 2.7.0 / torchvision 0.22.0
- numpy、matplotlib、Pillow、opencv（部分实验）
- ultralytics（Experiment-3 YOLO）

## 使用说明

1. 进入对应实验目录：`cd Experiment-X`
2. 运行主程序：`python experimentX.py`
3. 输出图片与日志保存在 `Experiment-X/outputs/` 目录下
4. 实验报告位于 `Experiment-X/docs/实验总结X.md`

## 备注

- 本仓库仅上传四个实验目录；`Experiment-2/data/` 未上传（运行时会自动下载 CIFAR-10）。
- `docs/` 目录下仅保留 Markdown 版报告，原始 `.docx` / `.doc` 模板未纳入版本控制。
- 图片文件使用 Git LFS 托管，若本地查看请确保已安装并启用 [Git LFS](https://git-lfs.com/)。
