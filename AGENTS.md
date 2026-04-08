# STMF Project Agent Guide

这份文件是给本仓库里的本地 agent / 协作者看的项目约定。

目标不是限制修改，而是减少“代码改了、结论没记下来、后面又重复排查”的情况。

## 1. 文档维护约定

只要改了下面这些内容，必须同步更新 `docs/`：

- 训练入口行为
- 评测入口行为
- 数据预处理协议
- 新增工具脚本
- 新增 experiment 配置
- 任何会影响结论解释的 bug 修复

默认更新规则：

- 当前状态和最新结论：
  - 更新 `docs/01_current_status.md`
- 文件入口、职责、代码结构变化：
  - 更新 `docs/03_code_structure.md`
- 如果是较大的设计变化、训练思路变化、模型结构变化：
  - 视情况补充 `docs/02_stmf_design.md`

不要只改代码不记结论。

## 2. 顶部使用说明约定

如果修改了下面这些入口脚本的行为，顺手更新文件头注释和典型命令：

- `scripts/train.py`
- `scripts/train_stmf.py`
- `scripts/eval.py`
- `scripts/eval_stmf.py`
- `tools/data_prep/ho3d_process.py`
- `tools/data_prep/inspect_packed_gt.py`
- `tools/data_prep/check_packed_mano_consistency.py`

目标是让后续打开文件时，先看头部就知道当前推荐怎么用。

## 3. HO3D 相关特别约定

HO3D 很容易反复踩协议问题，修改时必须显式记录：

- `train` split 的 joints 顺序
- `evaluation` split 的 joints 顺序
- sensor 是按哪套顺序计算的
- 评测导出时是否还需要回到官方顺序

如果这些语义发生变化，必须在：

- `docs/01_current_status.md`
- `docs/03_code_structure.md`

里同时写清楚。

## 4. 新增实验配置约定

新增 `hamer/configs_hydra/experiment/*.yaml` 时：

- 显式写 `exp_name`
- 在文件里只覆盖真正不同的配置
- 不要把无关默认值重复抄一遍
- 在 `docs/01_current_status.md` 里说明这个实验配置是为了解决什么问题

如果只是 STMF 训练入口里的局部调参，不一定必须新建 experiment yaml；先看该入口是不是脚本注入型配置。

## 5. 训练结果解释约定

不要只记录：

- `train/loss` 下降了

还要同时记录：

- 关键分项 loss 的趋势
- TensorBoard 可视化有没有明显异常
- 最终评测指标有没有真的变好

特别是 HO3D plain HaMeR finetune，必须避免“loss 变小 = 模型变好”的想当然判断。

## 6. 优先排查顺序

当训练结果异常时，优先按这个顺序排查：

1. 数据协议是否变了
2. joints 顺序是否一致
3. MANO 参数和 keypoints 是否自洽
4. sensor 是否按正确顺序计算
5. 再讨论学习率、冻结策略、loss 权重

不要一上来只调优化器超参。

## 7. 提交信息建议

提交时尽量在 message 里说明属于哪一类：

- data protocol
- training config
- eval fix
- docs
- tooling

这样后面回看历史更快。
