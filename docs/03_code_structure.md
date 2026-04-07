# Code Structure Guide

这份文档用于快速说明当前仓库里和 STMF 项目直接相关的代码结构。  
目标是让后续新开对话时，可以先读这份文档，再直接定位要改的文件。

## 1. 推荐阅读顺序

建议按下面顺序看：

1. `docs/01_current_status.md`
2. `docs/02_stmf_design.md`
3. 本文件

如果只是为了继续开发，不建议先读 `docs/archive/` 里的旧文档，因为里面大多是早期草稿或已经过时的设计。

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

### 3.1 训练入口

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

### 3.2 评测入口

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

如果你要改这些内容，进这个文件：

- stateful / stateless 推理
- baseline 对照实验
- 评测输出 JSON 格式

### 3.3 原始 HaMeR 评测入口

文件：

- `scripts/eval.py`

作用：

- 原始 HaMeR 的官方评测入口

现在主要用于做对照，确认 `eval_stmf.py --base_hamer` 是否和原版一致。

### 3.4 Demo 入口

文件：

- `scripts/demo.py`

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

### 4.2 原始 HaMeR 加载

文件：

- `hamer/models/__init__.py`

作用：

- `load_hamer`
- `load_stmf`

如果 checkpoint 加载行为不对，或者 base / STMF 权重读取不一致，要从这里查。

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
