# Sensor-Guided Temporal MANO Research Plan

这份文档是当前项目的长期研究愿景。

短期执行细节放在：

- `docs/07_sensor_guided_temporal_refinement.md`

本文件用于回答更长期的问题：

- 这个课题最终想证明什么
- 拉线 sensor 的价值应该如何表述
- 公开 benchmark 和真实手套数据分别承担什么角色
- 后续如果扩展到 egocentric / hand-object 场景，应该按什么顺序推进

## Summary

主线建议改成：**sensor-guided temporal MANO refinement under visual ambiguity**。

不要再把核心问题定义成“HaMeR / WiLoR 在 HO3D-v3 单帧指标能不能继续降”。更合理的论文 / 汇报口径是：

- 单帧 RGB 模型在遮挡、运动模糊、hand-object interaction、检测跳变时天然不稳定。
- 拉线数据虽然低维，但它直接约束手指弯曲状态。
- 因此它最适合做 **物理先验 / temporal stabilizer**，而不是替代 RGB 做完整 3D hand recovery。

外部数据集方向上，优先看：

- **HOT3D**：egocentric、多视角、MANO、手 / 物体 / 相机 GT，很适合作为未来强 benchmark。
- **HOI4D**：大规模 egocentric RGB-D，有 3D hand pose / object / action / camera 信息，适合做 ego hand-object 场景。
- **HO3D / DexYCB / InterHand**：继续作为受控 benchmark，不是最终应用场景。
- **HInt / Ego4D 类数据**：适合做遮挡、可见性和 qualitative，不作为 3D MANO 主监督。

参考来源：

- [HOT3D](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d)
- [HOT3D paper](https://arxiv.org/abs/2406.09598)
- [HOI4D](https://hoi4d.github.io/)
- [AnyHand](https://chen-si-cs.github.io/projects/AnyHand/)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [HaMeR](https://github.com/geopavlakos/hamer)

## Key Changes

- **STMF 设计要收窄**：拉线 sensor 不应该强行预测 `global_orient / camera`，它主要约束 finger articulation。新设计里 sensor 分支默认只影响 `hand_pose` 的 finger residual，global orient 和 camera 主要由 RGB / base model 决定。
- **benchmark 用 pseudo-sensor 合法化**：公开数据集没有真实手套，所以先从 GT joints / MANO 生成 5D pseudo-sensor，再加噪声、dropout、延迟、标定偏差，模拟真实拉线信号。这样可以在 HO3D / HOT3D / HOI4D 上定量证明“额外低维物理信号在遮挡时有价值”。
- **真实手套数据用于校准和 demo**：真实 sensor 不直接拿来刷 benchmark，而是用于验证 pseudo-sensor 分布是否合理，并做最终实时系统展示。
- **评测重点换掉**：主指标不是单帧 PA-MPJPE 提分，而是遮挡 / 失败帧下的 temporal stability 和 recovery：
  - `PA-MPJPE / PA-MPVPE`：保证不明显退化
  - `MPJVE / MPJAE / PredJitter`：证明更稳
  - `Blackout-1 / Blackout-3`：证明视觉缺失时 sensor 有用
  - occlusion subset：证明遮挡帧更贴手、不乱跳

## Implementation Direction

### 1. 先做 STMF-v2 最小改版

- 输入：当前 RGB、上一段 MANO pose、5D sensor sequence、valid mask。
- 输出：只预测 `hand_pose` residual，默认不改 `global_orient / cam`。
- 保留一个 ablation：允许预测 full pose / cam，用来证明“收窄到 finger residual 更合理”。

### 2. 训练协议

- Base model：先用 HaMeR，后续把 WiLoR / AnyHand checkpoint 作为更强 RGB baseline。
- Sensor target：从 GT joints 生成 5D normalized fingertip-wrist distance。
- 训练扰动：image dropout、bbox jitter、blackout、sensor noise、sensor dropout、temporal delay。
- Loss：原 MANO / 3D / 2D loss + sensor FK consistency + temporal smoothness。
- 不再继续主攻 HO3D patch camera 修复；那条线作为已完成的负结果 / 协议分析。

### 3. Benchmark 顺序

1. **HO3D-v3**
   - 用现有仓库最快跑通 controlled benchmark。
   - 主要验证 pseudo-sensor、blackout、bbox jitter、temporal metrics 的完整链路。
2. **InterHand 或 DexYCB**
   - 验证非 HO3D 场景泛化。
   - 用来避免所有结论只绑定 HO3D-v3 协议。
3. **HOT3D 优先，HOI4D 备选**
   - 做 egocentric hand-object benchmark。
   - 更接近最终应用和论文叙事。
4. **HInt / Ego4D**
   - 只做遮挡 / 可视化 / 2D robustness。
   - 不作为主 3D 分数来源。

### 4. 对老师的汇报口径

- “前期发现 plain HaMeR 在 HO3D-v3 上直接 finetune 不稳定，问题不是简单超参，而是 2D / 3D / MANO / camera protocol 在 hard case 上有冲突。”
- “所以我把目标从单帧 finetune 提分，调整为 sensor-guided temporal refinement。”
- “下一步用公开 benchmark 的 pseudo-sensor 做定量验证，再用真实拉线数据做实时系统展示。”

## Test Plan

- **Ablation 1**：HaMeR vs HaMeR + EMA vs STMF pose-only vs STMF sensor-guided。
- **Ablation 2**：sensor 只改 finger residual vs sensor 改 full pose / cam。
- **Ablation 3**：clean RGB vs blackout-1 / blackout-3 vs bbox jitter vs occlusion subset。
- **Ablation 4**：oracle pseudo-sensor vs noisy pseudo-sensor vs dropout pseudo-sensor。

成功标准：

- clean `PA-MPJPE / PA-MPVPE` 不退化超过约 `0.5-1.0 mm`
- blackout / occlusion 下 `PredJitter` 或 recovery 指标明显优于 HaMeR + EMA
- finger articulation 在遮挡帧的错误明显减少

## Assumptions

- 拉线数据主要反映五指弯曲，不可靠地提供手腕全局旋转、深度或相机。
- benchmark 上没有真实手套，因此 pseudo-sensor 是必要桥梁，不是造假；它回答的是“如果有这种低维物理信号，模型能不能用好”。
- 当前仓库继续保留；先在 STMF 线上收敛研究问题，不急着新开仓库。
