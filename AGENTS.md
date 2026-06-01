# STMF Project Agent Guide

这份文件是给本仓库里的本地 agent / 协作者看的项目约定。目标是让新开对话时能快速进入当前主线，少重复排查已经有结论的问题。

## 1. 当前项目主线

当前短期主线已经从 **plain HaMeR on HO3D-v3 finetune 提分** 转为：

- **sensor-guided temporal MANO refinement**
- 利用低维拉线 sensor / pseudo-sensor、历史 pose 和 valid mask，在遮挡、视觉歧义、bbox 抖动、失败帧之后稳定 MANO 时序轨迹

不要把当前目标简单理解成继续调 plain HaMeR 的学习率、冻结策略或 HO3D 单帧分数。HO3D camera / patchK / overfit 线目前主要是协议分析和 negative result；除非用户明确要求继续排查，否则不要把它当成默认主训练方向。

默认对照口径：

- `HaMeR`
- `HaMeR + EMA`
- `Temporal pose-only refiner`
- `Sensor-guided refiner`

成功标准不要只看 clean `PA-MPJPE / PA-MPVPE`，还要看 temporal stability、blackout / jitter / occlusion 下的恢复能力，以及是否至少打过 `HaMeR + EMA`。

## 2. 推荐阅读顺序

开始做任何较大改动前，按这个顺序读文档：

1. `docs/01_current_status.md`
2. `docs/07_sensor_guided_temporal_refinement.md`
3. `docs/08_sensor_guided_temporal_research_plan.md`
4. `docs/03_code_structure.md`
5. `docs/06_stmf_v1_report_protocol.md`

按任务补读：

- 排查 HO3D camera / patch / projection hard case：读 `docs/04_ho3d_camera_intrinsics_note.md`
- 需要带出去做方法调研或论文口径：读 `docs/05_hamer_hand_recovery_research_brief.md`
- 理解早期 STMF v1 设计：读 `docs/02_stmf_design.md`
- 旧草稿和已过时方案在 `docs/archive/`，默认不要先读 archive。

## 3. 当前关键入口

当前新主线优先看：

- `scripts/cache_base_hamer_predictions.py`
  - 离线缓存 base HaMeR 每帧预测
  - 输出必须和 packed GT NPZ 完全等长、同顺序
- `scripts/train_sensor_refiner.py`
  - 训练 image-free `SensorTemporalRefiner`
  - 默认只修正 `hand_pose` residual
  - `history_source={base,gt,mixed}` 分别对应真实推理分布、sanity check、减少 exposure bias
- `scripts/eval_sensor_refiner.py`
  - 默认 stateless eval
  - 正式 temporal 结论优先使用 `--stateful`
- `hamer/datasets/sensor_refiner_dataset.py`
  - packed GT NPZ + base cache 构造时序窗口
  - 检查 cache 和 GT 的 `imgname` 对齐
- `hamer/models/components/sensor_temporal_refiner.py`
  - v2 refiner 骨架
  - residual head 零初始化，step 0 应等价于 base model

STMF v1 baseline 保留：

- `scripts/train_stmf.py`
- `scripts/eval_stmf.py`
- `hamer/models/stmf.py`
- `hamer/datasets/temporal_dataset.py`
- `hamer/datasets/stmf_datamodule.py`

plain HaMeR / HO3D 诊断入口：

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/train_overfit.py`
- `tools/data_prep/ho3d_process.py`
- `tools/data_prep/inspect_packed_gt.py`
- `tools/data_prep/check_packed_mano_consistency.py`
- `tools/data_prep/inspect_ho3d_projection_consistency.py`

## 4. 文档维护约定

只要改了下面内容，必须同步更新 `docs/`：

- 训练入口行为
- 评测入口行为
- 数据预处理协议
- 新增工具脚本
- 新增 experiment 配置
- 任何会影响结论解释的 bug 修复
- 远程实验结果、关键负结果、协议结论

默认更新规则：

- 当前状态、最新结论、推荐命令：更新 `docs/01_current_status.md`
- 文件入口、职责、代码结构变化：更新 `docs/03_code_structure.md`
- sensor-guided v2 主线变化：更新 `docs/07_sensor_guided_temporal_refinement.md`
- 长期研究路线或汇报口径变化：更新 `docs/08_sensor_guided_temporal_research_plan.md`
- STMF v1 汇报指标变化：更新 `docs/06_stmf_v1_report_protocol.md`
- 较大的模型设计变化：视情况更新 `docs/02_stmf_design.md`

不要只改代码不记结论。

## 5. 顶部使用说明约定

如果修改了入口脚本的行为，顺手更新文件头注释和典型命令。重点包括：

- `scripts/cache_base_hamer_predictions.py`
- `scripts/train_sensor_refiner.py`
- `scripts/eval_sensor_refiner.py`
- `scripts/eval_temporal_metrics.py`
- `scripts/train_stmf.py`
- `scripts/eval_stmf.py`
- `scripts/train.py`
- `scripts/eval.py`
- `scripts/train_overfit.py`
- `tools/data_prep/ho3d_process.py`
- `tools/data_prep/inspect_packed_gt.py`
- `tools/data_prep/check_packed_mano_consistency.py`
- `tools/data_prep/inspect_ho3d_projection_consistency.py`

目标是让后续打开文件时，先看头部就知道当前推荐怎么用。

## 6. HO3D 协议特别约定

HO3D 很容易反复踩协议问题。凡是涉及 HO3D 数据、训练监督、sensor 或评测导出，必须显式确认并记录：

- `train` split 的 joints 保存顺序
- `evaluation` split 的 joints 保存顺序
- sensor 是按哪套顺序计算的
- 评测导出时是否需要转回 HO3D 官方顺序
- packed `2D / 3D / MANO / handTrans / camMat` 是否仍然自洽

当前默认结论：

- `train` split: 训练监督应使用 HaMeR / OpenPose 风格的模型内部顺序
- `evaluation` split: 默认保留 HO3D 官方顺序
- sensor: 无论 train/eval，进入 `MANOHandProcessor` 前应转到模型/OpenPose 顺序
- 评测导出: 预测结果需要在 `hamer/utils/pose_utils.py` 中转回 HO3D 官方顺序

如果这些语义发生变化，必须同时更新：

- `docs/01_current_status.md`
- `docs/03_code_structure.md`

## 7. 训练和评测解释约定

不要只记录：

- `train/loss` 下降了

必须同时记录：

- 关键分项 loss 的趋势
- TensorBoard / overlay 可视化有没有明显异常
- 最终 `PA-MPJPE / PA-MPVPE` 有没有真的变好
- `MPJVE / MPJAE / PredJitter` 是否改善
- blackout / bbox jitter / frame dropout 下是否更稳

特别注意：

- plain HaMeR on HO3D 曾多次出现 loss 下降但 overlay、2D loss 或 benchmark 变差
- `global_orient loss` 下降不等于图像里整手朝向正确
- clean PA-MPJPE 小幅变化不足以证明 sensor 有用；sensor 主价值应体现在遮挡、视觉缺失、检测跳变和 temporal recovery

## 8. 排查优先级

训练结果异常时，优先按这个顺序排查：

1. 数据协议是否变了
2. joints 顺序是否一致
3. packed GT 的 MANO、keypoints、handTrans、camMat 是否自洽
4. sensor 是否按正确顺序和正确归一化方式计算
5. cache 是否与 GT NPZ 完全等长、同顺序
6. stateful eval 是否真的按 sequence 顺序回填上一帧 refined pose
7. 再讨论学习率、冻结策略、loss 权重

不要一上来只调优化器超参。

## 9. 新增实验配置约定

新增 `hamer/configs_hydra/experiment/*.yaml` 时：

- 显式写 `exp_name`
- 只覆盖真正不同的配置
- 不要把无关默认值重复抄一遍
- 在 `docs/01_current_status.md` 里说明这个实验配置是为了解决什么问题

注意：

- plain HaMeR 训练主要走 experiment yaml
- `train_stmf.py` 和 sensor refiner 入口很多配置是脚本参数 / 脚本注入型配置，不一定需要新建 experiment yaml

## 10. 协作沟通约定

当用户还在讨论方案、排查方向、或者只是在确认一个想法是否成立时：

- 默认先交流结论、风险、验证方案
- 不要直接开始改训练 / 评测 / 模型代码
- 只有在用户明确要求“去实现 / 去写脚本 / 去改文件”后，才进入代码修改

如果确实需要写代码做诊断，优先选择：

- 新增独立脚本
- 新增诊断工具
- 不干扰现有主入口行为的最小改动

目标是：

- 先把问题定义清楚
- 再动代码
- 避免把讨论中的假设提前固化成实现

## 11. 远程机器和数据路径

远程双卡 4090 机器：

- SSH alias: `dual4090`
- 实际连接信息：`user@220.196.173.235:39029`
- 远程项目目录：`/home/user/code/STMF`
- 远程 conda 环境：`STMF`
- 远程 conda 初始化：`. /home/user/miniconda3/etc/profile.d/conda.sh`
- 远程数据目录：`/data/hand_data`
- HO3D 数据目录：`/data/hand_data/HO-3D_v3`
- GPU：双 NVIDIA GeForce RTX 4090

远程执行项目命令时默认使用：

```bash
ssh dual4090 '. /home/user/miniconda3/etc/profile.d/conda.sh && conda activate STMF && cd /home/user/code/STMF && <command>'
```

远程长任务默认放进 tmux，不要直接在普通 SSH 前台跑训练：

```bash
ssh dual4090 'tmux new -d -s stmf_v2'
ssh dual4090 'tmux send-keys -t stmf_v2 "cd /home/user/code/STMF && . /home/user/miniconda3/etc/profile.d/conda.sh && conda activate STMF" C-m'
ssh dual4090 'tmux capture-pane -t stmf_v2 -p -S -200'
```

训练命令建议使用 `tee` 保存完整日志，例如 `2>&1 | tee logs_remote/<run_name>.log`。启动训练后先用 `tmux capture-pane` 检查最近输出，确认环境、CUDA、数据路径、loss 和 dataloader 都正常；不要只看命令是否成功发出。

准备在远程跑实验前，先在远程项目目录检查并拉取代码：

```bash
ssh dual4090 'cd /home/user/code/STMF && git status --short --branch && git pull --ff-only'
```

如果远程存在未提交改动、未跟踪文件覆盖风险、分叉或冲突，不要自动 stash、reset、clean 或覆盖；先把 `git status` 和 `git pull` 报错反馈给用户确认。

## 12. Git 和认证约定

本地 WSL 仓库路径：

- `/home/mirage/STMF`

推荐本地 GitHub remote 使用 SSH：

```bash
git remote set-url origin git@github.com:yinsumirage/STMF.git
```

本地提交后，如果用户要求同步到 GitHub，需要执行：

```bash
git push
```

如果 push 遇到冲突、认证失败、non-fast-forward 或远程保护规则，停止并反馈给用户，不要强推。

禁止事项：

- 不要把 GitHub token / PAT 写进仓库、文档、脚本、命令历史或 `AGENTS.md`
- 不要为了通过 push 把 token 放进 remote URL
- 不要自动执行 `git reset --hard`、`git clean -fd`、强制 push，除非用户明确要求

提交信息建议带类别：

- `data protocol`
- `training config`
- `eval fix`
- `docs`
- `tooling`
- `remote workflow`

## 13. 工作区安全

当前工作区可能有用户或其它 agent 的未提交改动。修改前先看：

```bash
git status --short --branch
```

只改和当前任务直接相关的文件。遇到自己没改的变更：

- 相关：读懂后协同处理
- 不相关：不要碰
- 会阻塞任务：先反馈给用户

不要为了“整理干净”回滚别人的改动。
