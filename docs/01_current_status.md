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
  - HO3D 的 `hand_keypoints_2d / hand_keypoints_3d` 现在按 split 区分处理：
    - `train`: 导出时从官方 MANO 顺序转换到 HaMeR/OpenPose 顺序
    - `evaluation`: 默认保留 HO3D 官方顺序
  - 原因：
    - 训练监督必须和模型输出顺序一致
    - benchmark 评测文件最好保持官方语义，避免混淆
    - `pose_utils.py` 里的 HO3D reorder 只发生在评测导出，不会自动修正训练数据

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

#### 3.4 HO3D 训练监督里还有一个很关键的顺序问题

除了 bbox 之外，HO3D 训练数据还有一个容易反复踩坑的点：

- HO3D 原始关节顺序和 HaMeR / `render_openpose()` 假设的顺序不是同一套
- 如果直接把 HO3D 官方顺序写进训练 NPZ / webdataset tar
  - TensorBoard 里的 GT skeleton 会连线混乱
  - 更重要的是，训练 loss 也会拿错位的 joints 做监督

当前处理方式：

  - `tools/data_prep/ho3d_process.py`
  - `train` split: 先把 HO3D 官方顺序转换到模型内部顺序
  - `evaluation` split: 默认保持官方顺序
  - `sensor` 特征计算时，无论 train/eval，都会先把 HO3D joints 转到模型/OpenPose 顺序
    再交给 `MANOHandProcessor`，因为它内部假设的是 thumb/index/middle/ring/pinky 的模型顺序
- `hamer/utils/pose_utils.py`
  - 评测导出预测结果前，把模型内部顺序转回 HO3D 官方顺序

这两个方向不能混淆：

- 训练监督：必须转成模型顺序
- 评测导出：必须转回官方顺序

#### 3.5 HO3D 本地打包数据目前已经确认的自洽性

最近新增了一个检查脚本：

- `tools/data_prep/check_packed_mano_consistency.py`

它会读取打包后的 `npz`，并检查：

- `hand_pose`
- `betas`
- `hand_keypoints_3d`

这三者在 MANO 几何上是否自洽。

当前在本地 `ho3d_train.npz` 上已经确认：

- `hand_pose / betas / hand_keypoints_3d` 是高度自洽的
- `as-is` 顺序和 MANO 重建结果几乎完全一致
- 再额外做一次 `official -> openpose` 重排反而会变差

这说明至少对当前这份本地 train NPZ 而言：

- `MANO` 参数监督本身没有和 `keypoints_3d` 打架
- 当前保存的 `hand_keypoints_3d` 已经是模型内部顺序

注意：

- 这个结论是针对当前本地导出的那份 `ho3d_train.npz`
- 如果后面重新导出远程训练集，仍然建议重新跑一次这个检查，确认不同机器上的数据协议没有漂移

#### 3.6 plain HaMeR 在 HO3D-v3 上的当前结论

截至目前，最稳定的 baseline 仍然是：

- 直接使用原始 HaMeR checkpoint
- 在修正后的 self-eval 链路上评测

大致结果：

- `PA-MPJPE ≈ 13.61`
- `PA-MPVPE ≈ 13.04`

而 plain HaMeR 的多轮 HO3D finetune 目前都还没有稳定超过这个 baseline。

已经观察到的现象：

- `pose_only` 版本不会像最早那样立刻塌缩，但仍然容易把评测结果训差
- 调整后的任务导向 loss 比原始 loss 配方更好，但目前仍未超过 base checkpoint
- 训练过程中经常出现：
  - `keypoints_3d loss` 下降
  - `keypoints_2d loss` 升高
  - 可视化里的整体手方向、投影位置、或手形变差

这不是单纯的“loss 写错了”，而更像是当前训练目标和项目真实需求还没有完全对齐。

#### 3.7 为什么 `global_orient loss` 会下降，但视觉上手反而更怪

这个现象已经在多轮实验里出现过，目前更合理的解释是：

- `global_orient loss` 是参数空间里的 `rotmat` 误差，不是“图像里看起来方向是否正确”的直接指标
- `keypoints_3d loss` 是 root-relative 的
  - 它更关心相对骨架结构
  - 不强约束整手在图像里的朝向和投影
- `pose_only` 训练里，实际可训练的 `decpose` 同时控制：
  - `global_orient`
  - `hand_pose`

这意味着：

- 即使只训练很少一部分参数，也足以把整手根方向和手指姿态一起带偏
- 模型可能找到一种“3D 相对骨架更像 GT，但 2D 投影更差、视觉更怪”的局部解

所以：

- 不能只看 `train/loss` 或 `loss_global_orient` 是否下降
- 还必须同时看：
  - `loss_keypoints_2d`
  - 可视化里的 2D overlay
  - 最终 `PA-MPJPE / PA-MPVPE`

### 3.8 当前对 STMF 线的意义

plain HaMeR 这条线目前还没有给出稳定正收益，但最近已经确认：

- HO3D 的 sensor 计算之前确实需要先做 joints 顺序转换
- 这一点已经修到了 `tools/data_prep/ho3d_process.py`

所以当前更值得继续验证的是：

- 重新导出带正确 sensor 的 HO3D NPZ
- 在这份数据上重新训练/评测 STMF

相比继续盲目放大 plain HaMeR 的 finetune 强度，这条线现在更可能给出可解释的结果

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

导出后建议先检查 GT 是否正常：

```bash
conda run -n STMF python tools/data_prep/inspect_packed_gt.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --img_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --out_dir /home/mirage/STMF/_DATA/HO-3D_v3/inspect_train \
  --dataset_type ho3d \
  --packed_order openpose \
  --order both \
  --num_samples 12
```

检查时要这样理解：

- `openpose/model order` 那一列才是拿来判断训练监督是否正常的主视图
- `official MANO order` 那一列现在只画点和索引编号，用来对照原始官方索引，不用于判断骨架连线是否“像手”
- 如果检查旧版未修复的 HO3D NPZ，需要改成：
  - `--packed_order official`

如果训练表现持续异常，建议再做一次 MANO 监督自洽检查：

```bash
conda run -n STMF python tools/data_prep/check_packed_mano_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --num_samples 4096
```

这一步的目的不是检查 loss 代码，而是确认打包出来的：

- `hand_pose`
- `betas`
- `hand_keypoints_3d`

是否本来就在同一套 MANO 几何语义下自洽。如果这里误差很大，就说明训练时不同监督项本身在“互相打架”。

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
- HO3D plain HaMeR finetune 仍在继续排查，当前已确认“全量微调容易把 base 训坏”，保守 head-only / pose-only 方案还在验证

### 6. 下一步最推荐做什么

优先级建议：

1. 先用 `vitpose` 版本导出 HO3D train/eval NPZ
2. 先测 `base_hamer` baseline
3. 如果 baseline 变好，再训练 STMF
4. 最后再比较：
   - GT bbox
   - ViTPose bbox

不建议当前马上切到 WiLoR 重新开新坑，因为会同时引入太多不确定性。
