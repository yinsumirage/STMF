# 后续开发架构提案：HaMeR + 传感器残差精化网络

## 1. 为什么选择这条路线

此前两天我们实现了从头训练 ResNet50 预测 MANO 参数的基线模型，但因为缺乏相机外参预测（Global Translation）而遇到了 ~18mm PA-MPJPE 的瓶颈。这是目前大模型与我们的自定义层实现上的重要区别。

结合落地场景（VR 或数据手套接收器会提供手腕的全局位置和 6DoF），我们**不需要**让单目网络去费尽心思算全局平移。

最佳性价比和学术贡献的选择是：**Sensor-Guided Post-Refinement for Vision-based Hand Estimation**。

## 2. 核心系统架构方案

我们将采用最新的视觉手部大模型（如 `HaMeR` 或 `WILOR`）作为基础先验模块，加上我们自己的轻量级传感器修正网络。

```text
       [RGB 图像]
           │
           ▼
[预训练大模型 (HaMeR / WILOR)] ──────────┐
           │                             │
           ▼                             ▼
 (粗略但全局对齐的 MANO Pose & Shape)   (图像特征，可选)
           │                             │
           └──────────────────┐          │    [数据手套 5 指弯曲度]
                              │          │          │
                              ▼          ▼          ▼
                       [ 后处理残差精化网络 (Residual Refiner) ] 
                       (这是我们要主要开发并在 FreiHAND 或者自己的数据上 Train 的)
                                         │
                                         ▼
                            (精化后高精度的 MANO Pose & Shape)
                                         │
                                         ▼
                             [VR/落地渲染端，结合手环全局追踪]
```

## 3. 具体实现步骤

### A. 准备预置条件
1. 下载 HaMeR / WILOR 的预训练模型。
2. 对于 FreiHAND 里我们要用来做训练/残差精化的图，先批量用 HaMeR 跑一遍推理，得到它的初始预测 `hamer_pose`, `hamer_shape`，直接落盘存成 JSON。
3. 这样在我们训练 Refiner 时，可以**完全剥离**笨重的图片推理过程，光靠处理 JSON 参数表和一维传感器数值跑极速训练。

### B. 输入与输出总结

**【输入】**
- `hamer_pose`: (48,) 基础网络给出的有瑕疵但合理的位姿预测。
- `hamer_shape`: (10,) 基础网络给出的手部形状。
- `sensor_dist`: (5,) 五个手指的归一化距离。
- `features` (可选): 拿 HaMeR 最后一层前的 256/512 维特征也可以输进来（如果有必要端到端）。

**【输出】**
- `refined_pose`: (48,) 
- `refined_shape`: (10,)

**【网络结构建议】**
一个小型的 MLP 或者 Transformer Encoder。
```python
Residual(x) = hamer_pose + MLP( Concat(hamer_pose, sensor_dist) )
```
如果只针对弯曲度相关的手指关节位姿做补偿，收敛会极其快，同时在实验（也就是发文章）时的消融实验很好写：直接比对 `Refined PA-MPJPE` vs `HaMeR baseline PA-MPJPE`。

## 4. 后续新系统（新对话）推荐步骤

开启新对话时，可以告诉 AI 助手：
> “我已经有了一个名为 `training_finger_distances.json` 的手套数据集，并且有了包含相机原图坐标的 `training_xyz.json` 和 `training_mano.json`。我现在的方向是写一个 **Residual Refiner**。请你帮我写一个基于 PyTorch 的微调小型网络（输入预估的 MANO + 我们归一化的传感器 5个维度的值），输出新的 MANO。”
