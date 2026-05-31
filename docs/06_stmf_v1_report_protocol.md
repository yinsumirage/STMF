# STMF v1 report protocol

这份文档只关注一件事：

- **给老师汇报时，STMF v1 应该固定展示哪些结果。**

当前项目已经不再把短期目标定义成：

- “plain HaMeR 在 HO3D 上继续稳定涨多少单帧分数”

而是定义成：

- **在尽量不伤害 HO3D 3D 精度的前提下，提升时序稳定性、遮挡鲁棒性和失败帧恢复能力。**

## 1. 固定比较的 4 个 baseline

后续所有正式表格，默认都比较下面 4 条线：

1. `HaMeR`
2. `HaMeR + EMA`
3. `STMF pose-only`
4. `STMF pose+sensor`

说明：

- `HaMeR + EMA`
  - 是最便宜也最重要的时序 baseline
  - 如果 STMF 连它都打不过，结论必须如实写清楚
- `STMF pose-only`
  - 是 v1 最关键的安全线
  - 即使 pseudo-sensor 没带来额外收益，它也应该能单独作为主结果交付
- `STMF pose+sensor`
  - 当前默认先用 pseudo-sensor
  - 如果后面拿到小规模真实 sensor，再作为 phase-2 做额外适配

## 2. 主结果表

主表固定放在 HO3D 上，至少包含：

- `PA-MPJPE`
- `PA-MPVPE`
- `MPJVE`
- `MPJAE`
- `PredJitter`

解释：

- `PA-MPJPE / PA-MPVPE`
  - 负责回答“3D 精度有没有明显退化”
- `MPJVE / MPJAE`
  - 负责回答“时序速度 / 加速度是否更接近 GT”
- `PredJitter`
  - 负责回答“预测轨迹本身是不是更平稳”

当前建议的成功标准：

- 相比 `HaMeR`，`PA-MPJPE / PA-MPVPE` 退化不超过 `0.5 mm`
- `MPJVE / MPJAE / PredJitter` 至少有一个稳定改善 `>= 10%`

## 3. Stress test

失败帧恢复固定汇报两组：

- `Blackout-1`
- `Blackout-3`

固定指标：

- `Blackout1_PeakError`
- `Blackout1_RecoveryFrames@10%`
- `Blackout3_PeakError`
- `Blackout3_RecoveryFrames@10%`

解释：

- `PeakError`
  - blackout 区间内的平均误差峰值
- `RecoveryFrames@10%`
  - corruption 结束后，需要多少帧才回到 clean baseline 的 `1.1x` 以内

## 4. HInt 遮挡表

HInt 部分固定输出两层结果。

### 4.1 标准分层

每次至少给出：

- `ALL`
- `VIS`
- `OCC`

每层固定指标：

- `mode_kpl2`
- `PCK@0.05`
- `PCK@0.10`
- `PCK@0.15`

### 4.2 Occlusion Phase Profile

如果数据集文件名带阶段标签，再额外输出：

- `pre_45`
- `pre_30`
- `pre_15`
- `pre_frame`
- `contact_frame`
- `pnr_frame`
- `post_frame`

当前固定展示：

- `PCK@0.10`
- 必要时补 `mode_kpl2`

注意：

- 这条结果当前的定位是：
  - **interaction phase aggregate robustness curve**
- 不宣称是严格逐事件 recovery benchmark

## 5. 当前推荐汇报顺序

1. 先给出 `HaMeR` 在 HO3D 上的 base 精度
2. 再给出 `HaMeR + EMA`
3. 再给出 `STMF pose-only`
4. 最后给出 `STMF pose+sensor`
5. 然后补：
   - blackout stress test
   - HInt `ALL / VIS / OCC`
   - HInt phase profile
6. 最后用一页可视化视频收尾：
   - clean
   - blackout-1
   - blackout-3

## 6. 当前默认解释口径

如果结果是：

- `pose-only` 已经优于 `HaMeR + EMA`
- 但 `pose+sensor` 还没有稳定超过 `pose-only`

当前默认解释口径是：

- v1 已经证明：
  - 时序 residual refiner 这条线是成立的
- pseudo-sensor 分支的机制已经接通
- 但要拿到稳定额外收益，还需要：
  - 更大规模时序预训练
  - 更贴近真实分布的 sensor 数据

如果结果是：

- `pose+sensor` 明显优于 `pose-only`

则当前主卖点可以升级成：

- **sensor-guided temporal refinement**

否则不要强讲 sensor，先把主结果定位成：

- **temporal pose refinement**
