# STMF Refactor And Final Design

这份文档关注“为什么这样设计”和“模型/数据逻辑具体改了什么”。
如果你只是想知道当前怎么运行，请先看 `01_current_status.md`。
如果你想知道某个功能对应仓库里的哪个文件，请看 `03_code_structure.md`。

## 背景

本轮重构的核心目标，是把 STMF 从“当前帧视觉 + 历史视觉 buffer + 历史 pose + 历史 sensor 的大一锅融合”，收敛成一个更适合实时场景的小型纠偏器：

- 当前图像负责主预测
- 历史 pose 负责时序稳定
- 历史 sensor 负责遮挡条件下的物理纠偏

重构动机有三点：

1. 历史视觉 buffer 会显著增加实现复杂度和时序边界问题。
2. 在现有数据规模下，历史视觉不一定比历史 pose/sensor 更有收益。
3. 实时系统更需要“稳定的小修正器”，而不是“重型时序重建器”。

## 当前最终设计

### 1. 主干结构

- Backbone 仍然使用冻结的 HaMeR backbone + original mano head。
- STMF 不再处理整段图像窗口，只处理目标时刻 `t` 的当前图像特征。
- STMF 的输入改为：
  - 当前图像的 patch tokens
  - 历史 sensor 序列 `sensor_seq`
  - 历史 pose 序列 `pose_seq[:, :-1]`
- STMF 的输出改为：
  - pose residual
  - camera residual
- `betas` 不再由 STMF head 直接回归。

### 2. Betas 策略

`betas` 目前采用保守策略：

- 主体仍来自当前帧 HaMeR 的 base beta
- 如果存在上一时刻 beta，则做指数平滑

公式：

`beta_t = m * beta_{t-1} + (1 - m) * beta_base_t`

其中默认 `m = 0.9`。

这样做的原因：

- shape 是慢变量，不应该像 pose 一样频繁变化
- 直接让小型 STMF 头去回归 beta，容易把单帧噪声放大成时序抖动
- 但完全不处理 beta 连续性，也会保留 HaMeR 单帧估计带来的轻微跳动

### 3. Stateful Eval

评测时默认启用 autoregressive / stateful 模式：

- 上一帧预测的 `pred_pose` 会缓存
- 下一帧推理时，把缓存写回 `pose_seq`
- 上一帧 `betas` 也会按 sequence 级别缓存，用于 beta 平滑

这一步是必须的。否则 eval 阶段会退化成：

- 当前图像 + 当前 sensor + 全零历史 pose

那样无法真实反映时序模型的线上行为。

如需回退到旧行为，可使用：

```bash
python scripts/eval_stmf.py --stateless ...
```

## 这轮修复内容

### 1. 数据集与边界

- `ImageDataset` 现在优先读取 NPZ 中的 `personid`
- `TemporalImageDataset` 会补出：
  - `sequence_key`
  - `temporal_indices`
  - `sensor_valid_mask`
  - `pose_valid_mask`
  - `prev_betas`
  - `has_prev_betas`
- 如果旧 NPZ 的 `personid` 不可信，但 `imgname` 中包含序列信息，会自动从路径恢复 sequence id
- 左侧 padding 仍然会保留固定长度窗口，但现在会显式标记哪些历史槽位是真实历史、哪些只是占位

### 2. STMF 结构收敛

- 删除了历史视觉 buffer 作为 memory bank 的依赖
- memory 改为当前帧视觉 tokens
- query 保留历史 sensor / pose tokens
- query token 增加了：
  - temporal positional encoding
  - modality embedding

### 3. FK_SENSOR Loss 修复

原版实现的问题是：

- 预测端使用绝对 3D 距离
- 标签端使用 `[0, 1]` 归一化 sensor

两边量纲不一致。

现在的实现改为：

- 从预测 joints 恢复到官方顺序
- 根据骨长估计 `lmax`
- 用固定 `fist_ratio` 估计 `lmin`
- 将预测距离归一化到 `[0, 1]`
- 再与 sensor 标签做 MSE

这至少保证了监督目标和标签空间一致。

### 4. Smoothness Loss 接通

原来 `SMOOTHNESS` 是设计存在、实现未接通。

现在模型前向会显式输出：

- `pred_pose`
- `pred_poses_seq`

因此 smoothness loss 已经可以真正参与训练。

### 5. History Valid Mask

本轮新增了标准的 history valid mask 机制：

- `sensor_valid_mask`: 长度为 `T`
- `pose_valid_mask`: 长度为 `T-1`

语义如下：

- `1` 表示该时序槽位对应真实历史
- `0` 表示该槽位只是为了补齐固定窗口长度而左侧 padding 出来的占位

例如 `T=5` 时：

- 第 0 帧: `sensor_valid_mask = [0, 0, 0, 0, 1]`
- 第 1 帧: `sensor_valid_mask = [0, 0, 0, 1, 1]`
- 第 1 帧: `pose_valid_mask = [0, 0, 0, 1]`

模型侧的用法：

- 仍保留数值 padding，保证 tensor 形状固定
- Transformer decoder 使用 `tgt_key_padding_mask`
- query token 聚合改为 masked mean，而不是无条件平均
- smoothness loss 只对完整有效的时间三元组生效
- `prev_betas` 只有在上一时刻真实存在时才启用

这样做的意义是：

- 不再把“缺历史”误当成“真实的 0 sensor / 0 pose”
- 单帧冷启动不会被伪历史强行污染
- 序列边界行为更清晰，也更方便后续继续迭代

## 当前已知限制

### 1. Autoregressive Drift

即使 stateful eval 已接通，长期闭环仍然会面临漂移风险：

- 上一帧预测误差会传到下一帧
- 长序列下可能出现缓慢累计偏差

这部分目前暂未彻底解决，后续可考虑：

- scheduled sampling
- confidence-gated history injection
- history reset / re-anchor 机制

### 2. 训练与推理分布仍不完全一致

虽然训练阶段已有 pose noise augmentation，但这并不等同于真实 autoregressive 误差分布。

### 3. 当前实现仍会加载整段图像窗口

目前模型只使用当前帧图像做视觉主干，但数据集仍返回整段 `img` 窗口。
这样做的好处是兼容现有接口，代价是数据加载仍略显冗余。

后续如果继续追求实时性，可以再把 dataset 侧也收紧到：

- 当前图像
- 历史 pose/sensor

## 当前推荐理解

现阶段最合适的 STMF，不应该被理解成一个“重新生成整只手的时序大模型”，而应该理解成：

- HaMeR 是主预测器
- Sensor 是物理纠偏器
- Prev pose 是时序稳定器
- STMF 是一个轻量 residual refiner

这个理解和当前数据规模、实时需求、工程复杂度，是更匹配的。
