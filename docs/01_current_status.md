## STMF Project Progress

这份文档用于记录当前仓库在原始 HaMeR 基础上的改动、现阶段结论、以及推荐使用方式。

推荐阅读顺序：

1. 先读本文件，了解当前状态和推荐命令
2. 再读 `02_stmf_design.md`，了解设计和修复背景
3. 最后读 `03_code_structure.md`，按代码结构定位具体文件

历史草稿和已经过时的文档已移到 `docs/archive/`。

### 1. 当前项目目标

当前项目不是单纯复现原版 HaMeR，而是在 HaMeR 上增加一个 `STMF` 时序纠偏模块，利用：

- 当前帧图像
- 历史 pose
- 历史 sensor

来提升手部重建的稳定性和遮挡场景表现。

同时，为了在 HO3D 上继续训练和评测，也补了一套本地数据预处理和评测流程。

### 2. 已完成的主要改动

#### 2.1 STMF 模型和评测链路

已经完成：

- 增加 `scripts/train_stmf.py`
  - 用于训练 STMF 版本模型
- 增加 `scripts/eval_stmf.py`
  - 用于评测 STMF
  - 支持 `--base_hamer`
  - 可以用相同数据链路直接评测原始 HaMeR baseline
- `hamer/models/stmf.py`
  - STMF 结构收敛为“当前图像 + 历史 pose/sensor”
  - 不再依赖整段历史图像 buffer
- `eval_stmf.py`
  - 已支持 stateful / autoregressive 推理
  - 会把上一帧预测 pose 回填给下一帧

#### 2.2 Dataset 和时序逻辑

已经完成：

- `hamer/datasets/temporal_dataset.py`
  - 增加时序窗口
  - 增加 sequence 边界处理
  - 增加 `temporal_indices`
  - 增加 `sensor_valid_mask`
  - 增加 `pose_valid_mask`
- `hamer/datasets/image_dataset.py`
  - 读取 NPZ 中的 `personid`
  - 图片扩展名 `.png/.jpg/.jpeg` 自动兼容
  - 缺失图片在评测阶段自动跳过，不再整次崩溃

#### 2.3 训练链路

已经完成：

- `hamer/datasets/stmf_datamodule.py`
  - 训练/验证 dataloader 初始化更稳
  - 修复了 `train_dataset is not initialized`
  - 修复了 `val_dataloader() -> None`
- `scripts/train_stmf.py`
  - 默认关闭无意义的 validation/test/lr logging
  - checkpoint 改成按 step 保存并保留 `last.ckpt`

#### 2.4 HO3D 数据预处理

已经完成：

- `tools/data_prep/ho3d_process.py`
  - 修正了 `scale` 保存协议
  - `scale` 现在按 HaMeR 预期保存为像素宽高
  - 修复了 evaluation bbox 曾经被错误 `/200` 两次的问题
  - 支持两种 bbox 来源：
    - `gt`
    - `vitpose`
  - `vitpose` 模式复用 HaMeR demo 流程：
    - body detector
    - ViTPose
    - 从右手关键点生成 hand bbox
  - `vitpose` 检测失败时会自动回退到 GT bbox，避免样本数下降

### 3. 目前已经确认的结论

#### 3.1 当前评测链路已经基本对齐

`scripts/eval.py` 和 `scripts/eval_stmf.py --base_hamer` 在同一数据输入下，结果已经对齐。

这说明：

- `eval_stmf.py` 的 `base_hamer` 分支本身没有额外引入误差

#### 3.2 之前 HO3D 分数异常低，主要不是模型本体问题

已经确认至少存在过两个问题：

- HO3D bbox / scale 协议不一致
- 本地数据集存在缺图

这些问题都会直接影响 HaMeR baseline 分数。

#### 3.3 现在最重要的是统一 bbox protocol

原版 HaMeR 训练并不是在线跑 detector，而是：

- 先离线生成 bbox / annotation
- 再把样本打包成 tar 数据

所以当前项目里，`tools/data_prep/ho3d_process.py` 的作用就是：

- 充当我们自己的离线预处理器

当前建议是：

- 不要混入和 HaMeR 无关的新 detector 作为主协议
- 先优先尝试 HaMeR 自己 demo 所用的 `detector + ViTPose` 流程

### 4. 当前推荐使用方式

#### 4.1 导出 HO3D NPZ

GT bbox:

```bash
conda run -n STMF python tools/data_prep/ho3d_process.py \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --split both \
  --bbox_source gt
```

ViTPose bbox:

```bash
conda run -n STMF python tools/data_prep/ho3d_process.py \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --split both \
  --bbox_source vitpose \
  --body_detector regnety
```

#### 4.2 评测原始 HaMeR baseline

```bash
conda run -n STMF python scripts/eval_stmf.py \
  --base_hamer \
  --checkpoint /path/to/hamer.ckpt \
  --dataset HO3D-VAL \
  --batch_size 64 \
  --window_size 5 \
  --results_folder results_hamer_v3
```

#### 4.3 训练 STMF

```bash
conda run -n STMF python scripts/train_stmf.py \
  checkpoint=/path/to/hamer.ckpt \
  batch_size=64 \
  devices=2 \
  epochs=20 \
  window_size=5
```

#### 4.4 评测 STMF

```bash
conda run -n STMF python scripts/eval_stmf.py \
  --checkpoint /path/to/stmf_last.ckpt \
  --dataset HO3D-VAL \
  --batch_size 64 \
  --window_size 5 \
  --results_folder results_stmf
```

### 5. 当前还在做的工作

当前还没有完全收敛的点：

- HO3D 上最优 bbox protocol 还没有最终定稿
- 本地评分脚本和官方评分链路是否完全一致，还需要继续确认
- STMF 在 HO3D-v3 上的实际增益还不稳定
- 还没有形成一份统一的“数据导出 -> 训练 -> 评测”完整 README

### 6. 下一步最推荐做什么

优先级建议：

1. 先用 `vitpose` 版本导出 HO3D train/eval NPZ
2. 先测 `base_hamer` baseline
3. 如果 baseline 变好，再训练 STMF
4. 最后再比较：
   - GT bbox
   - ViTPose bbox

不建议当前马上切到 WiLoR 重新开新坑，因为会同时引入太多不确定性。
