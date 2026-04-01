# STMF: Spatio-Temporal Multi-modal Fusion for Hand Pose Estimation
## 技术设计与 Bug 修复记录文档 (v1.0)

本文件详细记录了 STMF 模型在研发过程中遇到的核心问题及其解决方案，并对最终实现的架构设计进行了系统性总结。

---

## 1. 核心 Bug 修复日志

### 1.1 时间轴“4 帧错位”问题 (The 4-Frame Shift Bug)
*   **问题现象**: 模型在 FreiHAND 验证集上的 MPJPE 高达 21mm+，样本数比 GT 少 4 个。
*   **根本原因**: 原始 `_build_sequences` 使用滑动窗口 `range(0, n_samples - 5 + 1)`。当序列长度为 5 时，第一个窗口是 `[0, 1, 2, 3, 4]`，对应的 Target 是第 4 帧。这导致了：
    1. 视频的前 4 帧永远无法作为 Target 帧被预测。
    2. 评估脚本比对时产生索引错位（预测帧 4 对齐真值帧 0），产生了约 4 帧的时间距离误差。
*   **修复方案**: 引入 **序列首帧填充 (Padding)** 机制。
    *   对于任何索引 `t`，强制构造长度为 5 的序列。若 `t < 4`，则重复使用该视频序列的第一帧进行填充（e.g., `t=0` 时窗口为 `[0,0,0,0,0]`）。
*   **结果**: 预测结果样本数恢复至 3960 帧，与真值严格 1对1 对齐。

### 1.2 关节索引顺序不匹配 (Joint Index Mismatch)
*   **问题现象**: 模型预测结果在视觉上正确，但数值评测异常。
*   **根本原因**: HaMeR 内部将 MANO 关节重排为了 OpenPose 风格（`mano_to_openpose`），导致大拇指和食指等关节索引与 FreiHAND/HO3D 官方定义的 [Wrist, Thumb, Index, Middle, Ring, Pinky] 顺序不符。
*   **修复方案**: 
    1. 统一数据集命名为 `HO3D-VAL` 和 `FREIHAND-VAL`，激活 `Evaluator` 内部针对 `HO3D-VAL` 的自动重排逻辑。
    2. 验证并确保在导出 JSON 前，关节坐标已还原为官方 MANO 顺序。

### 1.3 训练“教师作弊”与曝光偏差 (Exposure Bias)
*   **问题现象**: 模型在训练集中表现极好，但在推理时若不给定完美真值，误差显著增加。
*   **根本原因**: 训练时输入的是完美的上一帧姿态真值 ($GT_{t-1}$)。在实际推理中，上一帧输入的是带噪的模型预测值。
*   **修复方案**: 在 `TemporalImageDataset` 训练阶段为输入的 `pose_seq` 增加 **Gaussian Noise Augmentation**（约 0.02 rad）。强制模型学会在历史姿态不完美的情况下，依靠图像特征和物理传感器进行纠偏。

---

## 2. 最终架构设计 (Final STMF Design)

### 2.1 时序数据加载层 (Temporal DataLoader)
*   **Sliding Window**: 支持窗口大小 `T`（默认 5）和步长 `stride`。
*   **Multi-modal Return**: 每个 Batch 返回：
    *   `img`: `[B, T, 3, H, W]`（视觉序列）
    *   `sensor_seq`: `[B, T, 5]`（五指物理传感器序列）
    *   `pose_seq`: `[B, T, 48]`（历史关节参数序列，训练时带噪）
*   **Person Boundary Check**: 自动识别 `person_id`，防止窗口跨越不同的视频序列或拍摄对象。

### 2.2 多模态 Token 化 (Modality Tokenization)
*   **Visual Token**: 通过冻结的 ViT Backbone 提取 Patch 特征，注入 **Temporal Positional Encoding**。
*   **Physical Token**: 将 `(B, T, 5)` 的物理传感器序列通过 MLP 并行映射，保留时间轴序列作为独立的 Query Tokens。
*   **Kinematic Token**: 将 `(B, T-1, 48)` 的历史姿态序列通过 MLP 映射，同样转化为 Token。
*   **设计理念**: 所有的历史信息（过去几帧发生了什么）都被建模为 Transformer 能够理解的 Token 序列，而不仅仅是一个单一的上一帧快照。

### 2.3 交叉模态融合头 (Cross-Modal Fusion Head)
*   **Cross-Attention**: 使用多模态 Tokens（传感器+历史姿态）作为 Query，去主动查询（Interrogate）视觉特征存储库（Memory Bank）。
*   **Residual Refinement**: STMF 不直接预测绝对姿态，而是基于 Base ViT 对当前帧的初始预测预测出一个 **$\Delta R$ (Delta Rotation)**。
*   **物理约束**: 通过高权重的传感器 Loss（`FK_SENSOR`）和时序平滑 Loss（`SMOOTHNESS`），确保预测出的手部动作符合物理规律。

---

## 3. 运行指南

### 3.1 训练
```bash
python scripts/train_stmf.py \
    checkpoint=./checkpoints/hamer_vit_l.ckpt \
    batch_size=8 \
    epochs=100
```

### 3.2 评估 (Stateful Mode)
在推理时，模型会维护一个内部状态，`pose_seq` 将由上一时刻的 `pred_pose` 实时填充。

---
**维护者**: Antigravity AI
**最后更新**: 2026-03-16
