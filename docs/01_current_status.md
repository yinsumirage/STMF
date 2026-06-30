## STMF Project Progress

这份文档用于记录当前仓库在原始 HaMeR 基础上的改动、现阶段结论、以及推荐使用方式。

当前主线已经从 **HO3D-v3 plain HaMeR finetune 提分** 收束到
**sensor-guided temporal MANO refinement**：
HO3D camera / patchK / overfit 线保留为协议分析和 negative result，
新主线重点验证低维拉线 sensor 是否能在遮挡、视觉歧义和检测跳变时稳定 MANO 时序轨迹。

推荐阅读顺序：

1. 先读本文件，了解当前状态和推荐命令
2. 再读 `07_sensor_guided_temporal_refinement.md`，了解新的 sensor-guided 主线
3. 再读 `08_sensor_guided_temporal_research_plan.md`，了解长期研究愿景和 benchmark 路线
4. 如果是准备给老师汇报 STMF v1，再读 `06_stmf_v1_report_protocol.md`
5. 最后读 `03_code_structure.md`，按代码结构定位具体文件
6. 如果正在排查 HO3D hard case 的 camera / patch 投影问题，补读 `04_ho3d_camera_intrinsics_note.md`
7. 如果要把当前结论带出去做论文/方法调研，再读 `05_hamer_hand_recovery_research_brief.md`
8. 如需了解早期 STMF 设计背景，再读 `02_stmf_design.md`
9. 如果要远程跑实验、同步远程仓库或处理本地/远程路径差异，读 `09_remote_workflow.md`

历史草稿和已经过时的文档已移到 `docs/archive/`。

当前如果是继续排查 plain HaMeR 在 HO3D 上的 hard case，
建议额外阅读：

- `docs/04_ho3d_camera_intrinsics_note.md`
- `docs/05_hamer_hand_recovery_research_brief.md`

它专门总结了：

- 为什么当前更像是 patch 相机近似问题
- 为什么这件事在大规模无相机监督数据上又很难直接解决
- 当前更稳的验证路线是什么

### 1. 当前项目目标

当前项目不是单纯复现原版 HaMeR。新的主线是
sensor-guided temporal MANO refinement：在 HaMeR/WiLoR 这类单帧 RGB 基座之外，
利用低维拉线 sensor 和历史 pose 约束时序 MANO 输出。

已有的 `STMF` v1 时序纠偏模块利用：

- 当前帧图像
- 历史 pose
- 历史 sensor

来提升手部重建的稳定性和遮挡场景表现。

同时，为了在 HO3D 上继续训练和评测，也补了一套本地数据预处理和评测流程。

### 2. 已完成的主要改动

#### 2.1 SensorTemporalRefiner v2 缓存训练协议

已经新增一条更收窄的 v2 训练/评测入口，用来回答“sensor 是否能在时序和视觉歧义下稳定 finger articulation”：

- `hamer/models/components/sensor_temporal_refiner.py`
  - `SensorTemporalRefiner` 默认只预测 `hand_pose` residual。
  - `global_orient` 和 `camera` 默认不改，只保留 ablation 开关。
- `scripts/cache_base_hamer_predictions.py`
  - 离线缓存 base HaMeR 每帧预测。
  - 输出必须和 packed GT NPZ 完全等长、同顺序。
  - 默认不跳过缺图，避免训练 cache 和 GT NPZ 错位。
  - 默认 `--split train`，不会套 HO3D official evaluation whitelist；远程 smoke 曾暴露硬编码 evaluation split 会把 train 子集过滤成 0 帧。
  - 如果后续要生成官方 evaluation 子集预测，再显式传 `--split evaluation`。
- `hamer/datasets/sensor_refiner_dataset.py`
  - 从 packed GT NPZ + base cache 构造时序窗口。
  - 每个 sample 是一个目标帧，但内部自带历史 pose window 和 sensor window，因此训练 batch 可以 shuffle。
  - sequence 开头用第一帧左 padding，并用 `pose_valid_mask / sensor_valid_mask` 标记无效历史。
- `scripts/train_sensor_refiner.py`
  - 训练 image-free refiner，不在线跑 RGB backbone。
  - 支持 `history_source={base,gt,mixed}`。
  - 支持 `sensor_mode={sensor,zero}`，分别对应 sensor-guided refiner 和 temporal pose-only refiner。
- `scripts/eval_sensor_refiner.py`
  - 支持 stateless eval 和 `--stateful` autoregressive eval。
  - 正式 temporal 结论优先看 `--stateful`，因为它会把上一帧 refined pose 回填给下一帧窗口。
  - 支持 cached stress：`--blackout_len`、`--base_pose_noise_std`、`--sensor_dropout`。
- `scripts/eval_sensor_refiner_metrics.py`
  - 读取 refiner NPZ 输出，用 MANO layer 计算 base/refined 的 3D/temporal 指标。
  - 当前输出 `PA-MPJPE / PA-MPVPE / MPJVE / MPJAE / PredJitter / Stress_PA-MPJPE`。

当前这个 v2 入口是最小协议，不等于完整 benchmark。后续还需要接入：

- pseudo-sensor FK consistency loss
- 更完整的 bbox jitter / frame dropout 扰动
- clean / blackout / sensor dropout 的统一汇总表

2026-06-03 远程 smoke 结果：

- 在 `dual4090` 上用 HO3D train 前 128 帧跑通了 `cache -> train_sensor_refiner 3 step -> eval_sensor_refiner --stateful`。
- 输出 `refined_pose (128, 48)`、`delta_hand_pose (128, 45)`，`stateful=True`。
- 这只证明真实远程链路可运行，不作为正式指标；详细记录见 `docs/09_remote_workflow.md`。

2026-07-01 HO3D-v3 full cached v2 结果：

- 全量 HaMeR cache 已生成并复用：
  - train: `/data/hand_data/HO-3D_v3/ho3d_train_hamer_base_cache.npz`
  - evaluation: `/data/hand_data/HO-3D_v3/ho3d_evaluation_hamer_base_cache.npz`
- 默认 20 epoch、`history_source=base` 已能超过 base HaMeR：
  - base clean: `PA-MPJPE=18.920`, `PA-MPVPE=17.434`, `MPJVE=2.948`, `PredJitter=4.378`
  - sensor clean: `PA-MPJPE=18.121`, `PA-MPVPE=16.737`, `MPJVE=2.935`, `PredJitter=4.351`
  - zero clean: `PA-MPJPE=18.308`, `PA-MPVPE=16.911`, `MPJVE=2.940`, `PredJitter=4.363`
- 100 epoch 会过拟合 primary PA 指标，不作为推荐配置。
- `SMOOTHNESS_WEIGHT=0.01` 明显拉低 primary PA/MPVPE，不作为推荐配置。
- 新增训练扰动 `TRAIN_BASE_POSE_NOISE_STD` 后，当前 sensor-guided 最有用的配置是：
  - `RUN_DATE=20260701_sensdrop02`
  - `EPOCHS=20`
  - `HISTORY_SOURCE=mixed`
  - `MIXED_GT_PROB=0.25`
  - `TRAIN_BASE_POSE_NOISE_STD=0.07`
  - `TRAIN_SENSOR_DROPOUT=0.2`
  - `LR=5e-5`
- 该配置下 sensor-guided refiner 在同配置内明显优于 pose-only：
  - sensor clean: `PA-MPJPE=17.614`, `PA-MPVPE=16.316`, `MPJVE=2.873`, `PredJitter=4.244`
  - zero clean: `PA-MPJPE=17.903`, `PA-MPVPE=16.573`, `MPJVE=2.875`, `PredJitter=4.249`
- `LR=5e-5` 比默认 `1e-4` 明显更好；`2.5e-5` 不够，`7.5e-5` 稍差，`5e-5 + 40 epoch` 会过拟合。
- `TRAIN_SENSOR_DROPOUT=0.2` 比 `0.05/0.1/0.15/0.25/0.3` 略好；额外 `TRAIN_SENSOR_NOISE_STD=0.02/0.05` 没有继续提升。
- 用 `TRAIN_SENSOR_DROPOUT=0.2` 复核 `SEED=2024/2025/2026`：
  - sensor clean `PA-MPJPE=17.732 / 17.889 / 17.815`
  - 对应 zero clean `PA-MPJPE=17.882 / 18.096 / 18.060`
  - 结论：sensor 同 seed 稳定优于 zero，但 `SEED=12345` 的 `17.614` 是当前最好单次结果，汇报时应同时说明 seed 方差。
  - 加入 sensor dropout 后，`SEED=12345` 的最好单次结果进一步到 `17.614`。
- `blackout_strategy=hold/zero` 的 stress-frame PA 不稳定，暂时只作为诊断，不作为主结论。当前更可靠的结论是：base-pose noise augmentation + lower LR + sensor dropout 能明显改善 cached refiner，sensor 在 `sensdrop02` 同配置下有 clean + temporal 收益。
- `HaMeR + EMA` baseline 已补齐，入口是 `scripts/export_base_ema_predictions.py`：
  - sweep `alpha=0.05/0.1/0.2/0.3/0.4/0.6/0.8`
  - 最好 PA 是 `alpha=0.05`: `PA-MPJPE=18.717`, `PA-MPVPE=17.226`, `MPJVE=2.840`, `PredJitter=4.206`
  - 最低 jitter 是 `alpha=0.4`: `PA-MPJPE=18.897`, `PA-MPVPE=17.411`, `MPJVE=2.847`, `PredJitter=4.169`
  - 结论：EMA 能降低 jitter，但 primary 3D 指标明显弱于当前 sensor-guided refiner；当前 sensor 不是只靠过度平滑赢 PA。

#### 2.2 STMF 模型和评测链路

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

#### 2.3 Dataset 和时序逻辑

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

#### 2.4 训练链路

已经完成：

- `hamer/datasets/stmf_datamodule.py`
  - 训练/验证 dataloader 初始化更稳
  - 修复了 `train_dataset is not initialized`
  - 修复了 `val_dataloader() -> None`
- `scripts/train_stmf.py`
  - 默认关闭无意义的 validation/test/lr logging
  - checkpoint 改成按 step 保存并保留 `last.ckpt`

#### 2.5 HO3D 数据预处理

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

### 3.9 新增 overfit 诊断入口

为了回答“loss 到底有没有在把单样本往正确方向推”，当前新增了一个独立工具：

- `scripts/train_overfit.py`

它和主训练入口的区别是：

- 不走 WebDataset tar 随机采样
- 直接读取打包好的 NPZ
- 只取固定的 `1 / 8` 个样本做过拟合
- 默认关闭训练增强，保持 crop 和监督可重复

它的主要用途不是追求分数，而是排查：

- 单样本是否能被当前 loss 快速压住
- `keypoints_2d / keypoints_3d / hand_pose / betas` 的 raw loss 和 weighted contribution 分别是什么
- base checkpoint 到 finetune 过程中，同一批固定样本的 overlay 是变好还是变坏

当前这个工具还额外支持一个临时诊断开关：

- `--ho3d_coord_change_before_projection`

以及一组只改 GT 侧监督的临时诊断开关：

- `--gt_coord_recipe none`
- `--gt_coord_recipe flip_gt_keypoints_3d`
- `--gt_coord_recipe flip_gt_keypoints_3d_global_orient`
- `--gt_coord_recipe flip_gt_keypoints_3d_mano`

它只会在 overfit 脚本内部，把预测 `3D joints` 先做一次 HO3D/OpenGL 风格的
`diag(1, -1, -1)` 坐标变换，再投影成 `pred_keypoints_2d`。

`--gt_coord_recipe` 则是另一类实验：

- `flip_gt_keypoints_3d`
  - 只在算 loss 前，把 `batch['keypoints_3d']` 的 `xyz` 乘一次 `diag(1, -1, -1)`
  - 用来回答：如果只改 `3D GT` 所在参考系，`2d_3d` 是否还会把手拉翻
- `flip_gt_keypoints_3d_global_orient`
  - 在算 loss 前，同时把：
    - `batch['keypoints_3d']`
    - `batch['mano_params']['global_orient']`
    映射到同一套变换后的 GT 参考系
  - 但不改 `hand_pose / betas`
  - 其中 `global_orient` 当前按“左乘外部坐标变换”处理，而不是局部 pose 使用的共轭变换
  - 用来回答：当前剩下的 rigid misalignment，是否主要是 `global_orient` 没对齐，而不是局部手指 pose 没对齐
- `flip_gt_keypoints_3d_mano`
  - 在算 loss 前，同时把：
    - `batch['keypoints_3d']`
    - `batch['mano_params']['global_orient']`
    - `batch['mano_params']['hand_pose']`
    映射到同一套变换后的 GT 参考系
  - 其中：
    - `global_orient` 用左乘外部坐标变换
    - `hand_pose` 仍使用局部坐标系的共轭变换
  - 用来回答：如果把当前可见的 `3D + MANO` GT supervision package 一起切到另一套坐标语义，单样本 overfit 是否会恢复正常

注意：

- 当前 plain HaMeR 的训练 batch 里没有显式单独带出 `handTrans`
- 但 `keypoints_3d loss` 本来就是 root-relative
- 所以这个诊断里不需要再额外引入 `handTrans loss`
- 因此 `flip_gt_keypoints_3d_mano` 的语义可以理解成：
  - “翻当前 loss 真正会用到的那部分 `3D + MANO` GT”

注意：

- 这不是正式训练修复
- 只是为了快速验证 plain HaMeR 在 HO3D 上的“2D 越训越翻”
  是否就是投影坐标系少了一步转换导致的

如果后面继续出现“总 loss 在降，但图越来越歪”，建议优先先跑：

- `num_samples=1`
- `num_samples=8`

先确认这个最小诊断实验能不能自洽，再决定要不要继续做全量 HO3D 短跑。

当前 `train_overfit.py` 还新增了几个常用诊断配方：

- `--loss_recipe full`
- `--loss_recipe 2d_only`
- `--loss_recipe 3d_only`
- `--loss_recipe mano_only`
- `--loss_recipe 2d_3d`
- `--loss_recipe 2d_mano`

以及一个常用结构开关：

- `--unfreeze_camera_head`

目的是把 plain HaMeR 在 HO3D 上最常见的 ablation 固化下来，避免每次都手写一长串 `--override`。

### 3.10 HO3D 当前新增的投影一致性诊断

最近在 `1 sample overfit` 上观察到了一个更具体的现象：

- `keypoints_3d loss` 可以降到接近 0
- `global_orient loss` 也可以降到接近 0
- 但 `keypoints_2d loss` 会持续升高
- 可视化里整只手会逐渐翻到和图像里相反的方向

这个现象更像是：

- `HO3D 3D / MANO 参数监督`
- 和
- `HO3D 2D 投影监督`

没有完全使用同一套坐标语义。

为此新增了一个专门的对照工具：

- `tools/data_prep/inspect_ho3d_projection_consistency.py`

它会对同一帧同时比较：

- packed `GT 2D`
- packed `GT 3D -> proj`
- meta `handJoints3D -> proj`
- meta `handJoints3D(reordered) -> proj`
- `GT MANO local -> proj`
- `GT MANO + handTrans -> proj`

并且各自再分成两种投影：

- `with coord_change`
- `without coord_change`

当前这个工具还会额外把下面这些值写进 `summary.json`：

- `handTrans`
- `camMat`
- per-joint reprojection error
- packed 3D / meta joints / MANO joints 之间的 3D 均值差
- 如果开启 `--fit_patch_pred_cam_t`
  - 还会额外输出在 HaMeR patch 坐标里，直接用 GT `MANO local joints`
    拟合最优 `pred_cam_t` 后的 2D 误差

因此它现在不仅能回答“是不是要做 coord_change”，也能更细地回答：

- packed `hand_keypoints_3d` 和 meta `handJoints3D` 是否一致
- meta `handJoints3D` 如果先转到 model/OpenPose 顺序，再投影后是否就能和 packed 2D 对齐
- `MANO` 重建后如果不加 `handTrans`，会差多少
- `MANO + handTrans` 和 meta joints / packed 2D 之间，到底还剩多少像素差
- 当前 HaMeR 的 `pred_cam_t` 这套相机参数化，本身能不能把 GT 局部手 joints 投到足够贴近的 patch 2D

当前怀疑的重点是：

- `ho3d_process.py` 在生成 2D GT 时使用了 `diag(1, -1, -1)` 的坐标变换
- 但 plain HaMeR 训练前向里的投影链路并没有显式经过这一步

所以后续如果还要继续追 plain HaMeR 在 HO3D 上“越训越翻”的问题，
建议优先先用这个工具把：

#### 3.8.1 当前 inspect 结果的新结论

在 remote hard case `11903 / 23807` 上，已经确认：

- `meta handJoints3D` 只要先 reorder 到 model/OpenPose 顺序
  - 就能和 packed `hand_keypoints_3d` 完全对齐
  - `meta_joints_reordered_with_coord_change_px = 0`
  - `packed3d_vs_meta3d_reordered_mm = 0`
- `MANO + handTrans + coord_change`
  - 投影到 packed `GT 2D` 后也只剩约 `0.64 ~ 0.73 px`

这说明目前更像是：

- HO3D 的 packed `GT 2D / GT 3D / MANO + handTrans`
  本身是自洽的
- 剩下 hardest case 的问题更可能发生在
  plain HaMeR 当前训练目标如何去逼近这条 GT 生成链

此外，新增的 `--fit_patch_pred_cam_t` 诊断在本地 `idx=0` smoke test 上给出了一个很重要的信号：

- 直接用 `coord_change(MANO local joints)` 在 HaMeR 的 patch 坐标里拟合最优 `pred_cam_t`
  - 可以把 patch 2D 平均误差压到约 `1 px`
  - `patch_fit_mano_local_coord_l1 ≈ 0.10`
  - `patch_fit_mano_local_coord_px ≈ 1.02 px`

这个结果当前更支持：

- HaMeR 的 `pred_cam_t` 参数化本身并不是明显“不够表达”
- 问题更像是：
  - 训练时没有稳定学到这组相机/姿态的联合解
  - 而不是单纯“camera 结构缺了一个 handTrans 参数”

在此基础上，又进一步加入了“真实 patch intrinsics”对照：

- 直接从 `camMat + crop affine`
  构造每个样本自己的 patch 相机
- 并同时比较：
  - `exact crop(*)`
  - `patchK(*)`
  - `fit cam_t(*)`

本地 `idx=0` 当前结果是：

- `patch_exact_packed_px = 0.0000`
- `patch_exact_mano_px = 0.2354`
- `patch_patchK_packed_px = 0.0000`
- `patch_patchK_mano_px = 0.2354`
- `patch_fit_mano_local_coord_px = 1.0240`

这个结果目前更支持：

- protocol-correct 的真实 patch intrinsics 可以几乎无损复现 GT patch supervision
- 当前 HaMeR 的 fixed patch camera surrogate 则仍然会留下残差

在当时的诊断阶段，最值得继续验证的是 hard case 上：

- `真实 patch intrinsics`
- 是否也能把那几个固定 5px 左右的残差明显压下去

当前 inspect 工具还进一步支持了一个“固定 GT 手，只优化 camera family”的实验：

- 固定 `coord_change(MANO local joints)`
- 只优化：
  - `cam_t`
  - 以及 `fx, fy, cx, cy`

新增输出指标：

- `patch_fit_mano_local_coord_camk_l1`
- `patch_fit_mano_local_coord_camk_px`

本地 `idx=0` 当前结果是：

- `patch_fit_mano_local_coord_px ≈ 1.02 px`
- `patch_fit_mano_local_coord_camk_px ≈ 0.86 px`

说明：

- 在固定 GT 手的条件下
- 给 camera family 更多自由度
- 确实还能进一步压低 patch 2D 残差

所以这条实验现在很适合直接拿 hard case 去判断：

- 仅靠 camera family
- 到底能不能把 residual 继续拉回来

进一步在远程 hard case 上已经确认：

- `patch_exact_*` / `patch_patchK_*`
  - 仍然几乎完美
- 但：
  - `patch_fit_mano_local_coord_px`
  - `patch_fit_mano_local_coord_camk_px`
  都还停在约 `5px`

典型结果：

- `11903`
  - `patch_fit_mano_local_coord_px ≈ 5.64`
  - `patch_fit_mano_local_coord_camk_px ≈ 5.57`
- `23807`
  - `patch_fit_mano_local_coord_px ≈ 5.30`
  - `patch_fit_mano_local_coord_camk_px ≈ 5.09`

这说明：

- 问题不只是 fixed `fx/fy/cx/cy` 太死
- 单纯把 intrinsics family 放宽到 `cam_t + fx + fy + cx + cy`
  仍然不足以等价 HO3D 的 protocol-correct patch 相机链

当时进一步要验证的是：

- 不是继续加更多 intrinsics 参数
- 而是检查当前 camera translation family 是否本身就不兼容
- 尤其是：
  - `tz > 0` 这个 HaMeR 风格的正深度约束
  - 会不会正是 hard case 上剩余 `~5px` 残差的来源

因此 `inspect_ho3d_projection_consistency.py` 现在又新增了 signed-`tz` 诊断：

- `patch_fit_mano_local_coord_free_tz_*`
- `patch_fit_mano_local_coord_camk_free_tz_*`

这条新诊断的目的很明确：

- 固定 GT 手不动
- 只让 camera family 继续拟合
- 但放开 `tz` 正负号限制
- 看剩余 hard-case 残差能不能直接从 `~5px` 打到接近 `patchK`

随后又继续补了一个很关键的参数对照：

- 直接把 protocol-correct 的 `patch_cam_mat`
  拆成：
  - `fx`
  - `fy`
  - `cx`
  - `cy`
- 再把 `cam+K fit` 学到的：
  - `focal_length`
  - `camera_center`
  并排打印出来

本地 `idx=0` 的结果已经暴露出一个非常强的信号：

- `patchK_params`
  - `fx ≈ 171.8`
  - `fy ≈ 171.6`
  - `cx ≈ 125.8`
  - `cy ≈ 122.9`
- 但 `cam+K fit` 学到的却是：
  - `fx ≈ 12699`
  - `fy ≈ 13552`
  - `cx ≈ -0.061`
  - `cy ≈ -0.030`

这说明：

- 现在 inspect 里的 `patchK`
- 和 `perspective_projection` 拟合出来的 `cam+K`

虽然都叫“focal / center”，
但它们实际上不在同一套数值语义里。

因此当前更强的判断已经不是：

- “相机自由度还不够”

而是：

- `patchK` 这条 protocol-correct patch 相机链
- 和 HaMeR 当前 `perspective_projection` 这套 patch 相机参数化
  从定义层面就还没有完全对齐

基于这个结论，`scripts/train_overfit.py` 现在又新增了一个隔离诊断模式：

- `--projection_mode ho3d_patchK`

这个模式的作用是：

- 不改主训练入口
- 只在 overfit 诊断里
- 用每个样本自己的：
  - `camMat`
  - `crop affine`
  显式构造 `patchK`
- 再用这条 protocol-correct patch 投影链来计算 `pred_keypoints_2d`

后来这里又补了一个关键修正：

- 不能只换 projection backend
- 还要把 `pred_cam_t` 的深度语义一起切到 `patchK`

因此当前 `projection_mode=ho3d_patchK` 下：

- `tz` 不再沿用 fixed `5000` focal 的默认语义
- 而是改成使用每个样本自己的 patch focal

这是必要的，因为如果只把投影公式换成 `patchK`，
但 `pred_cam_t` 还沿用 old HaMeR 的 fixed-focal 语义，
step 0 的 keypoint 会直接 collapse 成一小团，诊断本身就失真。

同时为了避免 mesh 渲染再次用错相机语义把判断带偏，
这个模式下的 TensorBoard 预测图会改成 skeleton-only 可视化。

这条线后续已经继续做完了。

当前结论不是：

- `patchK` 再多试几轮也许就能直接落主训练

而是：

- `patchK` 这条 protocol-correct patch 相机链本身没有问题
- 但它和 plain HaMeR 当前 camera head / `perspective_projection` 的参数语义不兼容
- 因此不能把 `projection_mode=ho3d_patchK` 当成当前 plain HaMeR 主训练的可用修复

也就是说，这条线的诊断价值已经足够，但工程上暂时应收束：

- 不继续把主训练往 `patchK` 相机协议上硬改
- 不再把 residual camera / intrinsics overfit 当作当前 plain HaMeR 提分的主方向
- HO3D plain finetune 的短期主线回到更保守的 `gtcoord + 训练配方` 调整

### 3.10.5 当前主训练线的最小 HO3D 修复

在反复的 overfit 和 inspect 诊断后，当前最稳妥的主训练修复不是：

- 直接改 `ho3d_process.py` 重打数据
- 也不是立刻把主训练改成 protocol-correct `patchK` 投影

而是先做一个更保守的主线修复：

- 保持 HO3D packed NPZ 不变
- 在主训练 `HAMER.compute_loss()` 里增加可配置的 GT supervision reinterpretation

当前已经接入：

- `hamer/models/hamer.py`
  - 支持 `gt_coord_recipe`
- 旧诊断 experiment 已归档：
  - `docs/archive/ho3d_diagnostic_configs/hamer_ho3d_pose_only_finetune_gtcoord.yaml`

这个 archived experiment 只做一件事：

- `gt_coord_recipe: flip_gt_keypoints_3d_global_orient`

也就是：

- `keypoints_3d`
- `global_orient`

在算 loss 前切到 empirically 更合理的参考系，
但：

- `keypoints_2d`
- `hand_pose`
- 数据打包结果本身

都先保持不变。

之所以先走这条路，是因为目前最强证据支持的是：

- plain HaMeR 在 HO3D 上最明显的 catastrophic flip
  与 `keypoints_3d + global_orient` 的 supervision 参考系有关

但当前还没有足够证据支持：

- 直接永久改写 HO3D NPZ
- 或直接重写主训练的 patch 相机协议

因此现在推荐先用这个更小的主训练修复去跑分，
确认 HO3D 指标和可视化是否真的改善，
再决定要不要把协议修改固化到数据打包阶段。

到目前为止的主线结论是：

- `gt_coord_recipe=flip_gt_keypoints_3d_global_orient`
  仍然是当前最保守、最适合继续做 plain HaMeR HO3D finetune 的修复
- 而 camera / patchK 这条线当前已经完成阶段性诊断，
  结论是“协议不等价”，不是“再加几个参数就能直接修好”

在这之后，又继续补了一条更干净的二分实验：

- 不再只对 `coord_change(MANO local)` 做 camera-family 拟合
- 同时对 `coord_change(MANO + handTrans)` 也做同样的拟合

新增输出项：

- `patch_fit_mano_with_trans_coord_*`
- `patch_fit_mano_with_trans_coord_camk_*`
- `patch_fit_mano_with_trans_coord_free_tz_*`
- `patch_fit_mano_with_trans_coord_camk_free_tz_*`

这条诊断的用途是直接回答：

- 如果 3D 手本身已经带上 GT `handTrans`
- 只剩 patch 相机这一步
- 那么当前 HaMeR 风格 `perspective_projection` 这一族相机
  到底能不能逼近 protocol-correct 的 `patchK`

如果 hard case 上这组 `MANO+trans` 拟合能接近 `patchK`，
就说明问题主要出在：

- `MANO local`
- 用一个 surrogate camera translation 去替代 GT `handTrans`

如果这组也仍然卡在 `~5px`，
那就更说明问题在：

- `perspective_projection / focal / camera_center`
- 这套 patch 相机语义和 protocol-correct `patchK`
  之间仍然没有真正对齐

在此基础上，已经继续沿这条路线做了一个最小 residual camera 诊断：

- 只加在 `scripts/train_overfit.py`
- 不改主训练入口
- 新增一个 tiny patch-camera head
  - 输入：frozen backbone 的 pooled feature
  - 输出：`delta_f, delta_cx, delta_cy`
- 只修正 patch 投影，不改 MANO 参数语义

当前入口参数是：

- `--camera_residual_mode focal_center`
- `--camera_residual_mode focal_xy_center`

本地 smoke test 已确认：

- 新增模块 `camera_residual_head` 已接入模型和优化器
- 在 conservative freeze 下仅新增约 `328K` 参数
- 总 trainable params 约 `430K`

这条线当前的目的不是直接替换 plain HaMeR 主相机，而是先回答：

- 一个很小的 patch 相机 residual
- 能不能把 HO3D hard case 上 fixed patch camera 留下的那几个像素残差拉回来

当前两个 residual mode 的区别是：

- `focal_center`
  - 学 `delta_f, delta_cx, delta_cy`
- `focal_xy_center`
  - 学 `delta_fx, delta_fy, delta_cx, delta_cy`

其中 4 参数版本更接近 inspect 工具里显式构造的 `patchK`，
但它仍然不表示“真正学会了相机旋转”。

另外还确认过一个容易误导判断的点：

- 之前 overfit 可视化里，mesh 渲染链没有完整使用动态的
  - `focal_length`
  - `camera_center`
- 结果会出现：
  - mesh overlay 看起来偏了很多
  - 但 `pred_keypoints_2d / loss_keypoints_2d` 并没有对应变坏

现在这个可视化链已经改成向后兼容的方式：

- 默认仍保持原行为
- 只有显式传入 `focal_length / camera_center` 时才会启用新的渲染参数

因此后续再看 residual camera 实验时：

- mesh overlay
- 2D skeleton
- 2D loss

三者会更接近同一套相机语义，不再像之前那样互相打架。

- packed 2D
- packed 3D
- GT MANO

三者的相机语义彻底对齐，再决定要不要改 loss 或训练配置。

#### 3.10.1 当前单样本 overfit 已经确认的结论

最近已经实际跑过下面几组 `1 sample overfit`：

- `2d_only`
- `3d_only`
- `mano_only`
- `pose_only + camera head unfrozen`
- 以及额外的 `ho3d_coord_change_before_projection`

目前已经确认：

- `2d_only`
  - 在图像里最终可以非常贴近 GT
  - 说明裁图、2D GT、2D 投影链路本身至少能形成一个合理低损解
- `3d_only`
  - 会把手训到朝上翻转
  - 而且整只手会向外伸出更多
  - 说明 `3D` 监督单独就足以把模型拉向一个和图像语义冲突的解
- `mano_only`
  - 也会把手翻到朝上
  - 但形态更像是整体转了约 `180°`
  - 比 `3d_only` 更像参数空间里的稳定 flipped 解
- `pose_only + camera head unfrozen`
  - 没有根本解决翻转问题
  - 只是让根部位置发生了明显偏移
  - 说明 camera head 更像是在补投影位置，而不是解决朝向语义冲突
- `ho3d_coord_change_before_projection`
  - 在 `train_overfit.py` 里只改预测 3D -> 2D 的投影坐标变换
  - 实际 overfit 结果和原先基本一致，没有明显好转
  - 所以“仅仅是少了一步投影前 `coord_change`”这个解释已经不足以说明当前翻转现象

因此当前更强的判断是：

- plain HaMeR 在 HO3D 上的冲突，不只是一个简单的投影符号问题
- 更像是：
  - `2D` 监督代表的是图像语义
  - `3D / MANO` 监督代表的是另一套更稳定的参数/结构语义
  - 两者在当前 HO3D finetune 目标下，会把模型拉向不同解

所以现在更值得继续拆的是：

- `2d_3d`
- `2d_mano`

先判断到底是：

- `3D` 监督更主导翻转
- 还是 `MANO(global_orient / hand_pose / betas)` 更主导翻转

#### 3.10.2 当前 `2d_3d / 2d_mano` 进一步结论

在上面的基础上，又继续跑了：

- `2d_3d`
- `2d_mano`

目前观察到：

- `2d_3d`
  - 最终仍然会翻转
  - 视觉效果和之前 `3d_only` / 原始 pose-only 翻坏的感觉很接近
- `2d_mano`
  - 最终不翻
  - 至少在当前单样本 overfit 上，没有出现 `3D` 那种把手整体拉到朝上的现象

这说明当前更强的可疑点已经收缩到：

- 不是 `MANO(global_orient / hand_pose / betas)` 这一组监督在主导翻转
- 更像是 `KEYPOINTS_3D` 这组监督本身，在当前 plain HaMeR + HO3D 设定下把模型拉向了和图像语义冲突的解

但这里还不能直接下结论说：

- “HO3D 的 3D GT 数值一定错了”

因为还有至少两种可能：

1. `3D loss` 的目标虽然数值自洽，但它的权重和量纲明显压过了 `2D loss`
   - 当前默认继承的权重里：
     - `KEYPOINTS_3D = 0.05`
     - `KEYPOINTS_2D = 0.01`
   - 再加上 raw loss 本身量级不同，`3D` 很容易在总目标里占主导
2. `3D loss` 采用的是 root-relative 监督
   - 它强约束骨架结构
   - 但不直接保证图像里“看起来方向对”
   - 在当前冻结 camera head 的设定下，模型更容易通过改变 pose 来满足 3D，而牺牲 2D 外观

因此当前更稳妥的表述是：

- `3D supervision` 很可能是主导 plain HaMeR 在 HO3D 上翻转退化的关键项
- 但这还不足以证明 HO3D 的 3D GT “本身错了”
- 更可能是：
  - `3D GT` 的语义
  - `2D GT` 的图像语义
  - 以及当前 HaMeR 训练目标的权重/约束方式
  三者组合起来后，出现了一个对图像更差、但对 `3D loss` 更优的局部解

所以当时进一步安排的验证是：

- `flip_gt_keypoints_3d`
- `flip_gt_keypoints_3d_mano`

- 重新导出带正确 sensor 的 HO3D NPZ
- 在这份数据上重新训练/评测 STMF

新增这两个 GT-side 实验的原因是：

- 之前只试过改预测侧的 `pred 3D -> pred 2D` 投影
- 但用户提出另一个非常合理的怀疑：
  - 如果 HO3D 当前打包进训练的数据里，
  - `2D GT` 来自 `coord_change + projection`
  - 而 `3D / MANO GT` 仍保留在原始 3D 语义
  - 那么真正该试的，不只是“改预测投影”
  - 还要试“把 GT 侧 supervision 统一切到另一套参考系再算 loss”

截至目前，这两个 GT-side recipe 只是新增到 `train_overfit.py`，尚未记录稳定结论；
后续如果实验结果明确，需要继续把观察补回本节。

#### 3.10.3 当前 `flip_gt_keypoints_3d(_global_orient)` 的进一步结论

在远程 HO3D train NPZ 上，继续对 `spread` 选出的 8 个样本做了 overfit。

当前远程 `spread` 样本为：

- `0` -> `train/ABF10/rgb/0000.jpg`
- `11903` -> `train/BB14/rgb/0188.jpg`
- `23807` -> `train/GSF13/rgb/0701.jpg`
- `35710` -> `train/MDF12/rgb/0555.jpg`
- `47614` -> `train/SB14/rgb/0327.jpg`
- `59517` -> `train/SMu42/rgb/1208.jpg`
- `71421` -> `train/SiBF10/rgb/1531.jpg`
- `83324` -> `train/SiS1/rgb/0897.jpg`

目前已经观察到：

- `full + flip_gt_keypoints_3d_mano`
  - 视觉上比原始配方稳定很多，不再出现之前那种“越训越翻”
  - 但 `loss_global_orient / loss_hand_pose` 仍然不够理想
  - 说明把整套 `MANO` 都一起翻，并不是最精确的最终语义
- `full + flip_gt_keypoints_3d`
  - 8-sample 下整体 2D/3D 外观已经基本合理
  - `keypoints_3d` 可以降得很低
  - `hand_pose` 也能降到较低水平
  - 但有少数样本的手腕根部位置从 step 0 到后续 step 都几乎不动
  - 典型例子是：
    - `11903` -> `train/BB14/rgb/0188.jpg`
    - `23807` -> `train/GSF13/rgb/0701.jpg`
- `full + flip_gt_keypoints_3d + unfreeze_camera_head`
  - 允许 camera head 一起训练后，预测会出现前后移动和缩放变化
  - 但当前最顽固的问题仍不是平移/尺度，而更像是整手 rigid orientation 没有完全对齐
- `full + flip_gt_keypoints_3d_global_orient`
  - 把 `keypoints_3d + global_orient` 一起翻到同一参考系后
  - 单样本总 loss 会立刻降到非常低
  - `loss_hand_pose` 也会降到很小
  - `loss_global_orient` 则会从原先接近 `64` 明显降到接近 `8`
  - 但会长期停在 `8` 左右，后续几乎不再下降

这个 `8` 很可疑，因为它不像普通的“没完全收敛”，更像是：

- 当前 `global_orient` 监督和预测之间，仍然残留一个很稳定的刚体旋转差
- 而这个差不能靠 camera 平移/缩放解决

当前更合理的判断是：

- `KEYPOINTS_3D` 的参考系问题已经基本抓住
- `hand_pose` 很可能本来就接近正确，不应该随外部参考系整体翻转
- 但 `global_orient` 这项监督仍然没有完全对齐
- 现在 plain HaMeR 在 HO3D 上剩余最核心的问题，不再是“整体翻转”，而是“整手 rigid orientation 的最后一段对齐”

另外还需要注意：

- `freeze_shape_head=True` 时，`decshape` 虽然冻结，但如果上游 transformer 被放开，`betas` 预测仍然可以跟着变
- 因此最近观察到的 `loss_betas` 下降，不代表 shape head 本身已经单独解冻
- 它更像是 frozen linear head 接收到了变化后的 transformer feature

#### 3.10.4 当前 hard case 上 `2D / 3D / hand_pose` 冲突的进一步结论

针对远程 hard case，又继续做了更细的单样本 ablation，重点看：

- `23807` -> `train/GSF13/rgb/0701.jpg`
- `11903` -> `train/BB14/rgb/0188.jpg`

并统一使用：

- `gt_coord_recipe=flip_gt_keypoints_3d_global_orient`
- `unfreeze_camera_head=True`

当前观察到：

- `2D only`
  - hardest sample 可以被拉到非常贴近 GT
  - 例如 `23807` 的 `loss_keypoints_2d` 可以降到约 `0.06`
  - 这说明模型能力本身并不是问题，camera 也不是完全无效
- `2D + 3D` 在 `23807`
  - 仍然可以把 2D 拉得比较低
  - `loss_keypoints_2d` 可降到约 `0.075 ~ 0.08`
  - 但同时：
    - `loss_keypoints_3d` 会反弹到约 `0.33`
    - `loss_global_orient` 反弹到约 `0.06`
    - `loss_hand_pose` 反弹到约 `6`
  - 说明在这个样本上，`2D` 和 `3D` 还存在明显竞争，但还能找到一个折中解
- `2D + hand_pose` 在 `23807`
  - 收敛更快
  - `loss_keypoints_2d` 可进一步降到约 `0.04`
  - 但同时：
    - `loss_keypoints_3d` 会涨到约 `0.5`
    - `loss_global_orient` 维持在约 `0.14`
  - 说明在这个样本上，`hand_pose` 并不会像 `3D` 那样严重阻碍 2D 贴合
- `2D + 3D` 在 `11903`
  - 无法把 hand base 完全拉回
  - 手指 2D 大致能贴上
  - 但手腕根部位置仍然明显不对
  - 最终大致：
    - `loss_keypoints_2d ≈ 0.2`
    - `loss_keypoints_3d ≈ 0.2`
    - `loss_global_orient ≈ 0.01`
    - `loss_hand_pose ≈ 15`
  - 说明这个样本上的主要冲突项更明显
- `2D + hand_pose` 在 `11903`
  - 反而能把 base 尽量拉近
  - 最终 2D 可降到约 `0.06`
  - 但同时：
    - `loss_keypoints_3d` 会涨到约 `1.3`
    - `loss_global_orient` 会涨到约 `0.4`
    - `loss_hand_pose` 从约 `0.2` 升到 `0.5 ~ 0.6`

此外，还进一步尝试了在 full recipe 下持续增大 `KEYPOINTS_2D` 权重：

- `0.01`
- `0.02`
- `0.05`
- `0.1`

但当前观察是：

- 即使把 `KEYPOINTS_2D` 提到 `0.1`
- 在单样本 full 训练里，`2D` 贡献已经占总 loss 的大头
- `loss_keypoints_2d` 仍然不能像 `2D only` 那样降到极低
- 同时继续打开：
  - `camera head`
  - `mano transformer`
  - `shape head`
  也没有根本解决这个问题

这个结果目前更支持下面这个判断：

- 现在已经不能简单解释成“`2D weight` 太小”
- 因为即使 `2D` 权重显著增大，模型仍然会被其它监督拉回去
- 更像是：
  - `2D only` 能找到一个图像上非常好的解
  - 但一旦加回 `3D / hand_pose / 其它 MANO` 监督
  - 模型会被推向另一个参数空间更舒服、但图像上更差的折中解

因此当前更强的判断是：

- 当前 hard case 的剩余问题，核心不是“容量不够”或“camera 不会学”
- 更像是 supervision 之间仍存在真实冲突
- 并且从现有单样本结果看：
  - `KEYPOINTS_3D` 很可能仍是最强的冲突源
  - `hand_pose` 在部分样本上也会和 2D 产生竞争
  - 但它不像 `3D` 那样在所有样本上都稳定地主导错误解

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
  window_size=5 \
  history_mode=pose_sensor \
  sensor_mode=pseudo
```

Pose-only baseline:

```bash
conda run -n STMF python scripts/train_stmf.py \
  checkpoint=/path/to/hamer.ckpt \
  batch_size=32 \
  devices=1 \
  epochs=20 \
  window_size=5 \
  history_mode=pose \
  sensor_mode=off
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

#### 4.5 评测 HaMeR + EMA baseline

```bash
conda run -n STMF python scripts/eval_stmf.py \
  --base_hamer \
  --ema_decay 0.8 \
  --checkpoint /path/to/hamer.ckpt \
  --dataset HO3D-VAL \
  --batch_size 64 \
  --window_size 5 \
  --results_folder results_hamer_ema
```

#### 4.6 导出 InterHand2.6M 30fps STMF NPZ

```bash
conda run -n STMF python tools/data_prep/interhand_process.py \
  --base_dir /path/to/InterHand2.6M_30fps \
  --split train,val \
  --hand_filter single \
  --output_dir /path/to/InterHand2.6M_30fps
```

如果要把 interacting 样本拆成单手 track：

```bash
conda run -n STMF python tools/data_prep/interhand_process.py \
  --base_dir /path/to/InterHand2.6M_30fps \
  --split train \
  --hand_filter split
```

#### 4.7 输出时序稳定性 / blackout stress test

```bash
conda run -n STMF python scripts/eval_temporal_metrics.py \
  --checkpoint /path/to/stmf_last.ckpt \
  --dataset HO3D-VAL \
  --window_size 5 \
  --results_folder results_temporal \
  --blackout_lengths 1,3
```

如果要同时导出 clean / blackout 视频：

```bash
conda run -n STMF python scripts/eval_temporal_metrics.py \
  --checkpoint /path/to/stmf_last.ckpt \
  --dataset HO3D-VAL \
  --window_size 5 \
  --results_folder results_temporal \
  --blackout_lengths 1,3 \
  --save_video_dir results_temporal/videos
```

#### 4.8 输出 HInt `ALL / VIS / OCC` 和 phase profile

```bash
conda run -n STMF python scripts/eval_hint_phase.py \
  --base_hamer \
  --checkpoint /path/to/hamer.ckpt \
  --dataset EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC \
  --results_folder results_hint_phase
```

STMF 版本：

```bash
conda run -n STMF python scripts/eval_hint_phase.py \
  --checkpoint /path/to/stmf_last.ckpt \
  --dataset EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC \
  --window_size 5 \
  --results_folder results_hint_phase
```

注意：

- `hamer/configs/datasets_eval.yaml` 现在已经改成当前仓库可直接运行的 repo-relative 路径
- HInt 的 `EGO4D / EPICK / NEWDAYS` split 默认都从本地 `_DATA/HInt_annotation_partial/...` 读图
- 如果后面换机器，只需要改这一份 eval config，不要再把旧的绝对路径写死进脚本
- `TemporalImageDataset` / temporal eval 里的 HO3D 序列解析现在同时兼容：
  - `train/ABF10/rgb/0000.jpg`
  - `SM1/rgb/0000.png`
  这两种路径形式，避免把整个 validation set 错归到同一个 `seq_0`
- 当前这台机器上的 `TEST_ego4d_img` 目录只有标注 `json`，没有原图
  - 所以 `EGO4D-TEST-*` 在本地会被 missing-image filter 清空
  - `eval_hint_phase.py` 现在会自动跳过这类空 split
  - 真正跑 HInt 表格时，先确认该 split 的图片资源是否齐全，或者先跑 `EPICK / NEWDAYS`

### 5. 当前还在做的工作

当前还没有完全收敛的点：

- HO3D 上最优 bbox protocol 还没有最终定稿
- 本地评分脚本和官方评分链路是否完全一致，还需要继续确认
- STMF 在 HO3D-v3 上的实际增益还不稳定
- InterHand2.6M 30fps 的导出脚本已经接入，但真实大规模预训练结果还没跑完
- `eval_temporal_metrics.py` / `eval_hint_phase.py` 已经补上 v1 所需指标和汇报入口，但表格里的正式结果还需要实际跑数
- HInt phase 评测链路已经修到可用的本地路径协议，但还需要补一轮真实跑数 smoke test
- 还没有形成一份统一的“数据导出 -> 训练 -> 评测”完整 README
- HO3D plain HaMeR finetune 仍在继续排查，当前已确认“全量微调容易把 base 训坏”，保守 head-only / pose-only 方案还在验证

### 6. 下一步最推荐做什么

优先级建议：

1. 先导出 `InterHand2.6M 30fps -> STMF NPZ`
2. 先训 `STMF pose-only`
3. 再训 `STMF pose+sensor`
4. 统一输出四组基线：
   - `HaMeR`
   - `HaMeR + EMA`
   - `STMF pose-only`
   - `STMF pose+sensor`
5. 用 `eval_temporal_metrics.py` 跑：
   - `PA-MPJPE / PA-MPVPE`
   - `MPJVE / MPJAE / PredJitter`
   - `Blackout-1 / Blackout-3`
6. 用 `eval_hint_phase.py` 跑：
   - `ALL / VIS / OCC`
   - `Occlusion Phase Profile`

不建议当前马上切到 WiLoR 重新开新坑，因为会同时引入太多不确定性。
