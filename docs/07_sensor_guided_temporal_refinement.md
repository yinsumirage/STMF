# Sensor-Guided Temporal MANO Refinement

这份文档记录当前新的主线问题定义。

当前项目不再把短期目标定义成：

- 继续把 plain HaMeR 在 HO3D-v3 上 finetune 到稳定超过 base checkpoint

而是定义成：

- **利用低维拉线 sensor 作为物理先验，在遮挡、视觉歧义、检测跳变时稳定 MANO 时序轨迹。**

## 1. 为什么换主线

HO3D-v3 排查已经给出一个阶段性结论：

- HO3D packed `2D / 3D / MANO` 链路整体是自洽的
- plain HaMeR 在 hard case 上存在真实的 supervision conflict
- 这个冲突不是简单的 bbox、joint order、单个 loss 权重或相机残差可以直接修好
- `patchK` / camera residual / GT coord recipe 这条线的价值主要是协议分析和负结果，不适合作为当前主训练提分主线

因此，HO3D-v3 仍然保留为 benchmark 和 sanity check，
但不再作为项目最终目标。

## 2. 拉线 sensor 的合理角色

当前 5D 拉线信号最可靠的信息是：

- 五指整体弯曲 / 伸展状态
- 手指 articulation 的时序连续性
- RGB 被遮挡或检测不稳定时的低维物理约束

它不应该默认负责：

- 手腕全局朝向 `global_orient`
- 相机平移 / 深度
- bbox 或 hand detector 纠错

所以新主线里的默认设计是：

- RGB / base model 负责全局手、相机、图像对齐
- temporal pose history 负责连续性
- sensor 负责约束 finger articulation

## 3. Baseline 固定口径

后续正式表格默认比较四条线：

1. `HaMeR`
2. `HaMeR + EMA`
3. `Temporal pose-only refiner`
4. `Sensor-guided refiner`

其中：

- `HaMeR + EMA` 是最便宜、必须打过的时序 baseline
- `Temporal pose-only refiner` 用来判断时序模型本身有没有用
- `Sensor-guided refiner` 用来判断拉线物理先验是否提供额外收益

AnyHand / WiLoR 暂时只作为未来更强 RGB backbone 候选，
不要在第一阶段引入训练迁移，否则会混淆：

- RGB backbone 更强
- sensor 真的有用

## 4. STMF-v1 和 v2 的分工

### STMF-v1

位置：

- `hamer/models/stmf.py`
- `scripts/train_stmf.py`
- `scripts/eval_stmf.py`

定位：

- 作为当前已有 baseline 保留
- 输入当前图像、历史 pose、历史 sensor
- 输出 pose / camera residual

注意：

- v1 可以继续跑对照
- 但它不一定是新主线最终结构
- 不建议直接在 v1 上继续堆 camera / HO3D-specific 修复

### SensorTemporalRefiner v2

位置：

- `hamer/models/components/sensor_temporal_refiner.py`

默认接口：

- `base_pose`: `(B, 48)`
- `pose_window`: `(B, T, 48)`
- `sensor_window`: `(B, T, 5)`
- optional `image_feature`

默认输出：

- `delta_hand_pose`: `(B, 45)`
- `refined_pose`: `(B, 48)`

默认行为：

- 只修正 `hand_pose`
- 不改 `global_orient`
- 不改 `camera`
- residual head 零初始化，因此 step 0 等价于 base model

可选 ablation：

- `predict_global_orient=True`
- `predict_cam=True`

这些只用于验证“sensor 是否不应该控制全局手腕/相机”，
不要作为默认主线。

## 5. 数据接口

统一 sensor 格式：

- shape: `(T, 5)`
- value range: `[0, 1]`
- 语义：thumb / index / middle / ring / pinky 的 normalized pull value

公开 benchmark 没有真实拉线数据，因此第一阶段使用 pseudo-sensor：

1. 从 GT joints / MANO 生成五指 wrist-to-tip normalized distance
2. 加入 noise / dropout / temporal delay / calibration bias
3. 用这些扰动模拟真实拉线数据

真实手套数据后续只需要转成同一套 `(T, 5)` 接口，
即可复用 benchmark 模型和 demo pipeline。

## 6. Benchmark 设计

第一阶段先用 HO3D-v3 跑通 protocol。

核心 stress tests：

- clean
- blackout-1
- blackout-3
- bbox jitter
- frame dropout

核心指标：

- `PA-MPJPE / PA-MPVPE`
  - 证明 3D 精度没有明显退化
- `MPJVE / MPJAE`
  - 证明时序速度 / 加速度更合理
- `PredJitter`
  - 证明输出更稳
- blackout recovery
  - 证明视觉缺失或失败帧后恢复更快

成功标准不应只看 clean PA-MPJPE 是否下降，
而应重点看遮挡 / blackout / jitter 场景下 sensor-guided refiner
是否稳定优于 `HaMeR + EMA` 和 `pose-only refiner`。

## 7. 下一步

近期优先级：

1. 保留 STMF-v1 作为 baseline
2. 用 `SensorTemporalRefiner` 做 v2 最小训练入口
3. 先在 HO3D-v3 pseudo-sensor 上跑 clean / blackout / bbox jitter
4. 如果 v2 明显优于 pose-only，再考虑接真实拉线数据 demo
5. 如果 v2 在 HaMeR backbone 上成立，再考虑 WiLoR / AnyHand checkpoint

暂不建议：

- 新开仓库重做所有训练链路
- 继续把 HO3D patch camera 修复当主线
- 第一阶段就迁移到 WiLoR / AnyHand 训练
