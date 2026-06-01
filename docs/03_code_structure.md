# Code Structure Guide

这份文档用于快速说明当前仓库里和 STMF 项目直接相关的代码结构。  
目标是让后续新开对话时，可以先读这份文档，再直接定位要改的文件。

## 1. 推荐阅读顺序

建议按下面顺序看：

1. `docs/01_current_status.md`
2. `docs/07_sensor_guided_temporal_refinement.md`
3. `docs/08_sensor_guided_temporal_research_plan.md`
4. `docs/06_stmf_v1_report_protocol.md`
5. `docs/04_ho3d_camera_intrinsics_note.md`
6. `docs/05_hamer_hand_recovery_research_brief.md`
7. `docs/02_stmf_design.md`
8. 本文件

如果只是为了继续开发，不建议先读 `docs/archive/` 里的旧文档，因为里面大多是早期草稿或已经过时的设计。

其中：

- `docs/07_sensor_guided_temporal_refinement.md`
  - 当前新的主线文档
  - 重点是 sensor-guided temporal MANO refinement 的问题定义、baseline、数据接口和 v2 refiner 边界
- `docs/08_sensor_guided_temporal_research_plan.md`
  - 当前长期研究愿景
  - 重点是外部 benchmark 顺序、pseudo-sensor 合法性、真实手套数据定位和对老师的汇报口径
- `docs/04_ho3d_camera_intrinsics_note.md`
  - 专门记录 plain HaMeR 在 HO3D hard case 上暴露出来的 patch 相机近似问题
  - 以及为什么“自由预测内参”虽然可能有帮助，但在无真实相机监督的大规模数据上又很容易语义塌陷
- `docs/05_hamer_hand_recovery_research_brief.md`
  - 更适合带出去做 deep research 的摘要版
  - 重点是当前强证据、开放问题、以及跨数据集/相机协议研究方向

## 2. 顶层结构

当前项目里最重要的目录是：

- `scripts/`
  - 训练、评测、demo 入口
- `hamer/models/`
  - 模型定义，包括原始 HaMeR 和新增的 STMF
- `hamer/datasets/`
  - 数据加载逻辑，包括单帧和时序数据集
- `tools/data_prep/`
  - 数据预处理和 NPZ 导出脚本
- `docs/`
  - 当前项目文档
- `test/`
  - 针对 STMF 新加的一些测试

## 3. 运行入口文件

### 3.1 原始 HaMeR 训练入口

文件：

- `scripts/train.py`

作用：

- 使用原始 HaMeR 的 WebDataset tar 训练链路
- 支持 conservative finetune
- 支持从原始 `hamer.ckpt` 初始化
- 支持 exact resume / weight-only fallback

这个文件现在是 plain HaMeR 的主入口，和 STMF 不是同一条配置/训练模式。

当前特别容易需要改的点：

- `checkpoint` / `ckpt_path` 恢复逻辑
- freeze 策略
- Lightning resume 兼容
- checkpoint 输出目录
- top-level Hydra 参数如何映射到 `TRAIN.* / GENERAL.*`

### 3.2 STMF 训练入口

文件：

- `scripts/train_stmf.py`

作用：

- 加载 HaMeR checkpoint
- 注入 STMF 配置
- 创建 STMF datamodule
- 训练 `STMF_HAMER`

当你需要改这些内容时，优先看这个文件：

- batch size
- epochs
- checkpoint 保存策略
- validation/test 默认行为
- STMF 训练超参数注入
- `history_mode={pose,pose_sensor}`
- `sensor_mode={off,pseudo}`
- pseudo sensor augmentation 参数

注意：

- `train_stmf.py` 当前并不强依赖 `experiment/*.yaml`
- 它会先读取 base HaMeR config，再在脚本里直接注入 STMF 训练超参数
- 所以 plain HaMeR 的 experiment yaml 和 STMF 的顶层 Hydra 配置，不要混为一谈

### 3.3 plain HaMeR overfit 诊断入口

文件：

- `scripts/train_overfit.py`

作用：

- 针对 plain HaMeR 做 `1 / 8 sample` 过拟合诊断
- 直接读取 NPZ，而不是训练 tar
- 固定同一批样本反复训练，帮助理解当前 loss 是否真的在往 GT 靠

这个文件特别适合回答这些问题：

- 单样本到底能不能被当前 loss 拟合下来
- `keypoints_2d` 和 `keypoints_3d` 谁在主导退化
- 总 loss 下降时，fixed batch 的 overlay 是变好还是变坏

当前这个诊断入口还支持一个只在本脚本内部生效的试验开关：

- `--ho3d_coord_change_before_projection`

也支持一个只在 overfit 入口内部生效的 residual patch-camera 诊断头：

- `--camera_residual_mode focal_center`

它会在原有 `pred_cam_t` 之外，再从 pooled image feature 预测一个很小的：

- `delta_f`
- `delta_cx`
- `delta_cy`

或者在 4 参数版本下预测：

- `delta_fx`
- `delta_fy`
- `delta_cx`
- `delta_cy`

只用于验证：

- 当前 HO3D hard case 上的固定 patch 相机残差
- 是否能通过一个轻量 patch intrinsics residual 被拉回来

以及一组只改 GT 侧监督的临时配方：

- `--gt_coord_recipe flip_gt_keypoints_3d`
- `--gt_coord_recipe flip_gt_keypoints_3d_global_orient`
- `--gt_coord_recipe flip_gt_keypoints_3d_mano`

也支持常用 loss ablation 配方：

- `--loss_recipe 2d_only`
- `--loss_recipe 3d_only`
- `--loss_recipe mano_only`
- `--loss_recipe 2d_3d`
- `--loss_recipe 2d_mano`

以及一个结构开关：

- `--unfreeze_camera_head`

用途是验证：

- 如果只改预测 3D -> 2D 的投影坐标变换
- 不改 3D / MANO loss

那么 HO3D 单样本 overfit 里的“手越训越翻”是否会缓解。

现在这个入口还可以进一步验证：

- 如果只在算 loss 前翻 `gt_keypoints_3d`
  - `2d_3d` 是否仍然会把手拉翻
- 如果只在算 loss 前翻 `gt_keypoints_3d + global_orient`
  - 当前剩余的整手刚体朝向问题，是否主要来自 `global_orient` GT 参考系没有对齐
- 如果在算 loss 前把 `gt_keypoints_3d + MANO(global_orient/hand_pose)` 一起翻到同一参考系
  - 当前可见的 `3D + MANO` supervision package 是否会重新和图像语义对齐

当前实现细节上需要注意：

- `global_orient` 的 GT 变换按“左乘外部坐标变换”处理
- `hand_pose` 的 GT 变换按局部坐标系的共轭变换处理

这两者不是同一种变换语义，不应该混用。

注意：

- 当前 plain HaMeR 训练 batch 里没有显式 `handTrans`
- 但 `keypoints_3d loss` 本来就是 root-relative
- 所以这个 GT-side 诊断主要覆盖的是当前 loss 真正读取的那部分 3D supervision，而不是额外引入一个新的 translation loss

另外，这个入口现在也专门用于回答：

- `2D` 单独是否能拟合图像
- `3D` 单独是否会把手翻转
- `MANO` 单独是否会把手翻转
- `2D + 3D` 是否仍然翻转
- `2D + MANO` 是否还能保持不翻
- 放开 camera head 后，问题是缓解还是只是换成位置漂移

如果是做“理解 loss / 看固定样本 / 排查过拟合失败”，优先看这个文件，不要直接上全量 `train.py`。

### 3.4 STMF / baseline 评测入口

文件：

- `scripts/eval_stmf.py`

作用：

- 评测 STMF 模型
- 也支持 `--base_hamer`，直接在相同数据链路下评测原始 HaMeR

这个文件里最重要的逻辑是：

- `prepare_base_hamer_batch`
  - 把 STMF 时序 batch 转成单帧 HaMeR 输入
- `inject_stateful_history`
  - 在 STMF 推理时，把上一帧预测写回历史 pose / betas
- `ema_decay`
  - 对 `pred_pose / pred_betas / pred_cam` 做 sequence-wise EMA

如果你要改这些内容，进这个文件：

- stateful / stateless 推理
- baseline 对照实验
- EMA baseline
- 评测输出 JSON 格式

### 3.5 时序指标 / blackout 评测入口

文件：

- `scripts/eval_temporal_metrics.py`

作用：

- 在 HO3D / InterHand 这类时序数据上统一输出：
  - `PA-MPJPE`
  - `PA-MPVPE`
  - `MPJVE`
  - `MPJAE`
  - `PredJitter`
- 也支持：
  - `Blackout-1`
  - `Blackout-3`
  的 stress test
- 可选导出 clean / blackout 视频

如果你后面要改：

- 时序稳定性指标
- 抖动指标
- 失败帧恢复
- blackout 视频导出

优先看这个文件。

### 3.6 SensorTemporalRefiner v2 缓存训练入口

相关文件：
- `scripts/cache_base_hamer_predictions.py`
- `scripts/train_sensor_refiner.py`
- `scripts/eval_sensor_refiner.py`
- `hamer/datasets/sensor_refiner_dataset.py`
- `hamer/models/components/sensor_temporal_refiner.py`

定位：
- 这是当前新主线的 v2 最小训练协议。
- 它不在线训练 HaMeR backbone，而是先把 HaMeR 逐帧预测缓存为 frame-aligned NPZ。
- 然后用缓存的 `base_pose / base_cam`、历史 pose window 和 sensor window 训练 `SensorTemporalRefiner`。

关键约束：
- base cache 必须和 packed GT NPZ 完全等长、同顺序。
- `scripts/cache_base_hamer_predictions.py` 默认不会跳过缺图；如果缺图，应先修数据或重新打包 NPZ。
- `SensorRefinerDataset` 会检查 cache 里的 `imgname` 是否和 GT NPZ 对齐。
- 训练 sample 可以 shuffle，因为每个 sample 内部已经带了自己的局部历史窗口。
- sequence 开头使用第一帧左 padding，并用 `pose_valid_mask / sensor_valid_mask` 标记 padding 无效。

训练入口：
- `scripts/train_sensor_refiner.py`
  - 默认只训练 `hand_pose` residual。
  - `global_orient` 和 `camera` 默认不改。
  - `--history_source base` 更接近真实推理分布。
  - `--history_source gt` 适合 sanity check。
  - `--history_source mixed` 用于减少 teacher-forcing 和推理分布的差距。

评测入口：
- `scripts/eval_sensor_refiner.py`
  - 默认 stateless：使用 cache/dataset 里构造好的历史窗口。
  - `--stateful`：按 sequence 顺序逐帧回填上一帧 refined pose，才是正式 temporal 评测应优先看的模式。

当前这个入口还不是完整 benchmark：
- 已经能验证 cache/window/stateful 协议是否跑通。
- 后续还需要接 MANO FK loss、pseudo-sensor consistency、blackout/bbox jitter 扰动和统一 temporal metrics。

### 3.7 HInt phase 评测入口

文件：

- `scripts/eval_hint_phase.py`

作用：

- 输出 HInt 的：
  - `ALL / VIS / OCC`
  - `mode_kpl2`
  - `PCK@0.05 / 0.10 / 0.15`
- 并按文件名阶段标签统计：
  - `pre_45`
  - `pre_30`
  - `pre_15`
  - `pre_frame`
  - `contact_frame`
  - `pnr_frame`
  - `post_frame`

这个文件是当前 “Occlusion Phase Profile” 的标准入口。

补充：

- 这个入口默认读取 `hamer/configs/datasets_eval.yaml`
- 现在这份 eval config 已经改成 repo-relative 路径
- 本地默认从 `_DATA/HInt_annotation_partial/...` 读取 HInt 图像，不再依赖旧机器上的绝对路径
- 如果某个 split 被 missing-image filter 清到 0，脚本会自动跳过而不是直接报错

### 3.8 原始 HaMeR 评测入口

文件：

- `scripts/eval.py`

作用：

- 原始 HaMeR 的官方评测入口

现在主要用于做对照，确认 `eval_stmf.py --base_hamer` 是否和原版一致。

### 3.9 Demo 入口

文件：

- `demo.py`

作用：

- 任意图片推理
- 用 detectron2 + ViTPose 找 hand bbox

这个文件非常重要，因为我们现在已经把它的 bbox 流程迁移到了 HO3D 离线预处理里。

## 4. 模型相关文件

### 4.1 STMF 主模型

文件：

- `hamer/models/stmf.py`

作用：

- 定义 `STMF_HAMER`
- 包含 tokenizer、fusion head、loss、beta 平滑等逻辑

这里是后续最容易继续改的核心文件。常见改动点：

- STMF 输入结构
- pose / cam residual
- history mask 使用方式
- FK sensor loss
- smoothness loss

### 4.1.1 Sensor-guided v2 refiner

文件：

- `hamer/models/components/sensor_temporal_refiner.py`

作用：

- 定义 `SensorTemporalRefiner`
- 作为新的 sensor-guided temporal MANO refinement 主线骨架
- 输入 base HaMeR pose、历史 pose window、sensor window，以及可选当前帧 image feature
- 默认只输出 `delta_hand_pose (B, 45)` 和 `refined_pose (B, 48)`
- residual head 零初始化，step 0 等价于 base model

设计边界：

- 默认不改 `global_orient`
- 默认不改 `camera`
- `predict_global_orient=True` / `predict_cam=True` 只作为 ablation
- 用来验证“低维拉线物理先验主要约束 finger articulation”，不要和 HO3D camera / patchK 诊断线混在一起

### 4.2 原始 HaMeR 加载

文件：

- `hamer/models/__init__.py`

作用：

- `load_hamer`
- `load_stmf`

如果 checkpoint 加载行为不对，或者 base / STMF 权重读取不一致，要从这里查。

### 4.3 plain HaMeR 主训练的 GT supervision 修正

文件：

- `hamer/models/hamer.py`

当前这里除了原始 HaMeR forward / loss 之外，又新增了一个很小的 HO3D 主线修复入口：

- `gt_coord_recipe`

当前支持的 recipe 与 overfit 诊断线保持一致：

- `none`
- `flip_gt_keypoints_3d`
- `flip_gt_keypoints_3d_global_orient`
- `flip_gt_keypoints_3d_mano`

主训练当前推荐只先用：

- `flip_gt_keypoints_3d_global_orient`

作用：

- 不改 packed NPZ
- 不改主训练投影链
- 只在 loss 计算前，按 recipe 重解释 GT supervision

如果后面 plain HaMeR 在 HO3D 上：

- loss 趋势变合理
- 可视化不再 catastrophic flip
- benchmark 分数也提升

那就说明这条“先修 supervision 语义，再考虑改数据协议”的路线是成立的。

## 5. Dataset 与数据流

### 5.1 单帧数据集

文件：

- `hamer/datasets/image_dataset.py`

作用：

- 读取 NPZ
- 根据 `center/scale` 裁图
- 组装单帧样本

当前额外补过的逻辑：

- 读取 `personid`
- 自动兼容 `.png/.jpg/.jpeg`
- 评测时缺图样本自动跳过

如果后面 bbox 裁图不对、图片路径不对、样本数不对，先查这个文件。

### 5.2 时序数据集

文件：

- `hamer/datasets/temporal_dataset.py`

作用：

- 在 `ImageDataset` 之上构建滑动窗口
- 输出：
  - `img`
  - `sensor_seq`
  - `pose_seq`
  - `sensor_valid_mask`
  - `pose_valid_mask`
  - `temporal_indices`
  - `sequence_key`

这里负责：

- sequence 边界
- 左侧 padding
- HO3D evaluation whitelist 过滤
- 历史数据的有效性标记
- 从文件名恢复 `sequence_key / frame_order`
  - 现在兼容 HO3D train 的 `train/SEQ/rgb/frame.jpg`
  - 也兼容 HO3D eval 的 `SEQ/rgb/frame.png`

如果后面时序样本有错位、跨序列混窗、stateful 历史回填不对，先查这个文件。

### 5.3 STMF DataModule

文件：

- `hamer/datasets/stmf_datamodule.py`

作用：

- 为 `train_stmf.py` 提供 train / val dataloader
- 直接使用 NPZ，而不是原始 HaMeR 的 tar WebDataset

这里现在负责：

- 路径解析
- 训练集和验证集 lazy build
- 避免 `setup(stage)` 不稳定导致 dataloader 为 `None`
- repo-relative `_DATA/...` 路径兼容

如果后面训练启动不了、dataset 没初始化、路径在不同机器上不一致，先查这个文件。

## 6. HO3D 预处理

### 6.1 HO3D 导出脚本

文件：

- `tools/data_prep/ho3d_process.py`

作用：

- 导出 `ho3d_train.npz`
- 导出 `ho3d_evaluation.npz`
- 计算 `sensor`
- 生成 `center/scale`

当前支持两种 bbox 来源：

- `gt`
- `vitpose`

另外，这个文件现在还负责一个很重要的训练监督修复：

- 把 HO3D 官方 MANO 顺序的 `hand_keypoints_2d / hand_keypoints_3d`
- 转成 HaMeR / OpenPose 风格的模型内部顺序

但这里现在是按 split 区分的：

- `train` split:
  - 转成模型内部顺序
- `evaluation` split:
  - 默认保留官方顺序

原因：

- `hamer/utils/pose_utils.py` 里的 HO3D reorder 只在评测导出时生效
- 训练 dataloader 不会自动帮你修 joints 顺序
- 如果导出时不转换，训练 loss 会直接用错位的关节监督模型
- 如果把 evaluation 也默默改成模型顺序，反而容易和 benchmark 评测语义混在一起

当前的 `vitpose` 模式复用了 `scripts/demo.py` 的流程：

- body detector
- ViTPose
- 从右手关键点生成 hand bbox

如果你后面要改：

- bbox protocol
- 是否使用 detector
- detector 成功/失败回退逻辑
- HO3D joints 顺序导出逻辑

就进这个文件。

### 6.2 InterHand2.6M 30fps 导出脚本

文件：

- `tools/data_prep/interhand_process.py`

作用：

- 把 InterHand2.6M 官方 JSON 导出成 STMF 可直接读取的 NPZ
- 输出：
  - `imgname`
  - `center/scale`
  - `hand_keypoints_2d`
  - `hand_keypoints_3d`
  - `hand_pose/betas`
  - `personid`
  - `sensor`
- 默认保留单手样本
- 也支持把 interacting 帧拆成 per-hand sample

如果后面要做 Stage A 的时序预训练，先跑这个文件。

### 6.3 打包结果可视化检查

文件：

- `tools/data_prep/inspect_packed_gt.py`

作用：

- 读取已经导出的 `npz`
- 把原图、bbox、GT keypoints 可视化出来
- 用来快速检查当前打包协议是不是和模型监督顺序一致

这里有两个关键概念：

- `packed_order`
  - 当前 `npz` 里 keypoints 的实际保存顺序
- `order`
  - 输出哪些可视化视图

当前推荐用法：

- 新版 train NPZ：
  - `--packed_order openpose`
- 旧版未修复 train NPZ：
  - `--packed_order official`

判断训练监督是否正常时，主要看：

- `openpose/model order` 这一列

### 6.4 MANO 监督一致性检查

文件：

- `tools/data_prep/check_packed_mano_consistency.py`

作用：

- 检查打包后的：
  - `hand_pose`
  - `betas`
  - `hand_keypoints_3d`
- 是否在同一套 MANO 几何语义下自洽

这个工具不是评测脚本，而是一个“训练监督是否互相打架”的排查工具。

如果后面 plain HaMeR 再次出现：

- loss 在降
- 可视化和评测却越来越差

建议优先用这个脚本再确认一遍当前导出的 train NPZ。

### 6.5 HO3D 投影一致性检查

文件：

- `tools/data_prep/inspect_ho3d_projection_consistency.py`

作用：

- 对同一帧同时读取：
  - packed `hand_keypoints_2d`
  - packed `hand_keypoints_3d`
  - 原始 meta 里的 `handPose / handBeta / handTrans / handJoints3D / camMat`
- 然后比较：
  - packed 3D 投影到 2D
  - meta `handJoints3D` 投影到 2D
  - meta `handJoints3D` 转到 model 顺序后再投影到 2D
  - GT MANO local joints 投影到 2D
  - GT MANO 重建后投影到 2D
- 并分别画出：
  - `with coord_change`
  - `without coord_change`

同时还会把：

- `handTrans`
- `camMat`
- per-joint reprojection error
- packed 3D / meta 3D / MANO 3D 的均值差

写入 `summary.json`

如果开启：

- `--fit_patch_pred_cam_t`

它还会额外：

- 用与 plain HaMeR 训练一致的 deterministic patch sample
- 直接拿 GT `MANO local joints`
- 在 HaMeR 当前的 patch projection 公式下拟合一个最优 `pred_cam_t`
- 分别比较：
  - `MANO local`
  - `coord_change(MANO local)`
- 输出 patch 2D 的：
  - `L1 sum`
  - 平均像素误差
  - 拟合出来的 `cam_t`
  - 对应的 patch 可视化

这样它不仅能看“原始 HO3D GT 是否自洽”，也能看：

- 当前 HaMeR 的 `pred_cam_t` 参数化
- 在 hard case 上到底有没有能力表达 GT 那条投影链

当前它还进一步支持：

- 从 `camMat + crop affine`
  显式构造每个样本自己的 `patch intrinsics`

并输出：

- `exact crop(packed3D / MANO+trans)`
- `patchK(packed3D / MANO+trans)`
- `fit cam_t(MANO local / coord_change(MANO local))`
- `fit free-z cam_t(coord_change(MANO local))`
- `fit free-z cam+K(coord_change(MANO local))`
- `fit cam_t / cam+K on coord_change(MANO+trans)`
- `fit free-z cam_t / cam+K on coord_change(MANO+trans)`

这样可以直接回答：

- hard case 的固定 patch 残差
- 到底是 protocol-correct 的 patch 相机就已经能解释
- 还是必须依赖 learnable camera surrogate 才能拟合
- 当前 HaMeR 风格 `tz > 0` 相机参数化
  是否正是 hard case 上 residual camera 仍然卡在 `~5px` 的原因
- 当前剩余误差
  到底主要发生在：
  - `MANO local -> surrogate translation`
  - 还是 patch 相机语义本身

### 6.4 Overfit patchK 诊断

文件：

- `scripts/train_overfit.py`

当前又额外支持：

- `--projection_mode ho3d_patchK`

作用：

- 只在 overfit 诊断里启用 HO3D-specific 的 protocol-correct patch projection
- 从样本对应的 `meta/*.pkl` 读取 `camMat`
- 用当前样本的 bbox crop 仿射生成 `patchK`
- 用这条 patch 投影链替代默认的 `perspective_projection`
- 同时把 `pred_cam_t` 的深度语义切到 patch focal，
  避免出现“新投影 + 旧 fixed-focal camera 语义”导致的 step-0 collapse

注意：

- 这个模式目前是诊断用，不是主训练协议
- 为了避免 mesh 渲染继续误导判断，这个模式下的 TensorBoard 可视化会切成 skeleton-only

这条线当前已经完成阶段性结论：

- `patchK` / protocol-correct patch projection 在 HO3D 上是成立的
- 但它和 plain HaMeR 当前 camera head / `perspective_projection` 的参数语义不兼容
- 因此 `projection_mode ho3d_patchK` 目前只保留为诊断工具，
  不建议直接推广到主训练配置

### 6.5 Mesh 渲染和 keypoint 可视化的相机一致性

文件：

- `hamer/utils/mesh_renderer.py`
- `hamer/models/hamer.py`

当前已经补了一个向后兼容的小修复：

- 如果显式传入：
  - `focal_length`
  - `camera_center`
- mesh 渲染会真正使用这些动态相机参数
- 如果不传，则仍保持原来的默认中心主点和默认 focal 行为

这个修复的目的主要是避免 overfit 诊断里出现：

- mesh overlay 看起来偏了很多
- 但 `pred_keypoints_2d / loss_keypoints_2d` 没有同步变坏

也就是让：

- mesh 可视化
- keypoint 可视化
- 2D loss

尽量落到同一套 patch 相机语义下。

这个工具主要用来回答：

- 当前 `keypoints_2d` 和 `keypoints_3d / MANO` 到底是不是在同一套相机坐标下
- HO3D 的 `diag(1, -1, -1)` 坐标变换是不是正是导致单样本 overfit 时手越训越翻的关键原因
- 当前 hard case 上，到底是 packed 3D、meta joints、MANO local、还是 `handTrans` 这一步在引入剩余差异
- 当前 `meta handJoints3D` 的残差到底是不是单纯来自 joint order 没重排
- 当前 patch-space 里剩下的误差，到底是 camera 参数化表达不了，还是训练没有学到那个解

## 7. 配置与日志目录

当前最常用的 Hydra 配置文件：

- `hamer/configs_hydra/train.yaml`
- `hamer/configs_hydra/hydra/default.yaml`

当前目录结构约定：

- `task_name`
  - 控制顶层日志桶，例如 `logs/hamer/...`
- `exp_name`
  - 控制实验名，例如 `hamer_ho3d_pose_only_finetune`
- 时间戳
  - 控制具体某次 run

所以当前 run 目录长这样：

- `logs/<task_name>/<exp_name>/<timestamp>/`

注意：

- plain HaMeR experiment yaml 里应该显式写 `exp_name`
- STMF 当前主要走 `train_stmf.py` 内部注入配置，不依赖 plain HaMeR 那套 experiment yaml 体系

### 6.2 打包结果检查工具

文件：

- `tools/data_prep/inspect_packed_gt.py`

作用：

- 读取已经打包好的 NPZ
- 把原图、bbox、GT keypoints 渲染出来
- 用来快速检查：
  - bbox 是否合理
  - 关节顺序是否已经转换到模型顺序

注意：

- `openpose/model order` 视图：
  - 用 OpenPose 风格连线
  - 这是判断训练监督是否正常的主视图
- `official MANO order` 视图：
  - 只画点和索引编号
  - 不能拿它判断 OpenPose 风格骨架是否“像手”
- `packed_order` 需要和当前 NPZ 实际保存的顺序一致：
  - 旧 HO3D NPZ 通常是 `official`
  - 修复后的 train NPZ 应该是 `openpose`

### 6.3 FreiHAND 导出脚本

文件：

- `tools/data_prep/freihand_npz_exporter.py`

作用：

- 导出 FreiHAND 的 NPZ

现在主要用于对照和补全 STMF 训练所需字段。

## 7. 评测与指标

### 7.1 预测缓存与导出

文件：

- `scripts/eval_stmf.py`
- `scripts/eval.py`

作用：

- 把预测导出成官方 JSON 格式

### 7.2 Evaluator

文件：

- `hamer/utils/pose_utils.py`

作用：

- 在内存中累积 3D joints / vertices
- HO3D 关节顺序重排

如果后面怀疑：

- HO3D joints 顺序不对
- evaluator 记录逻辑不对

就查这里。

### 7.3 本地评分脚本

文件：

- `results/freihand_score_cal.py`

作用：

- 对导出的 JSON 做本地打分

注意：

- 这个脚本适合快速本地比较
- 但它不一定和官方服务器 100% 等价
- 中间缺样本时，要特别注意预测和 GT 的对齐方式

## 8. 当前建议的开发路线

如果你后面继续开发，推荐按这个顺序：

1. `tools/data_prep/ho3d_process.py`
   - 先把 bbox protocol 定稳
2. `scripts/eval_stmf.py --base_hamer`
   - 先把 baseline 跑稳
3. `scripts/train_stmf.py`
   - 再训练 STMF
4. `hamer/models/stmf.py`
   - 最后才继续改模型结构

这样做的好处是：

- 先把数据和评测协议固定
- 再讨论模型增益
- 避免把“模型问题”和“bbox / 评分问题”混在一起

## 9. 目前建议优先读哪些文件

如果后面要继续改代码，最值得先读的是：

- `scripts/train_stmf.py`
- `scripts/eval_stmf.py`
- `hamer/models/stmf.py`
- `hamer/datasets/temporal_dataset.py`
- `hamer/datasets/stmf_datamodule.py`
- `tools/data_prep/ho3d_process.py`

这 6 个文件基本已经覆盖了当前 STMF 项目的主链路。
