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

当前 `HaMeR + EMA` 已通过 `scripts/export_base_ema_predictions.py` 实现为 cached baseline：
只对 `hand_pose` 做 sequence-local EMA，并复用 `scripts/eval_sensor_refiner_metrics.py` 计算同一套指标。

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

## 6. v2 训练和评测协议

当前 v2 不再把“时序训练”理解成 dataloader 必须按视频顺序逐帧喂模型。
新的最小协议是两阶段：

1. 先离线缓存 base HaMeR 的逐帧预测。
   - 入口：`scripts/cache_base_hamer_predictions.py`
   - 输入：packed NPZ、图片目录、HaMeR checkpoint
   - 输出：与 packed NPZ 完全对齐的 `base_pose / base_cam / sequence_key / frame_order`
   - 约束：训练 cache 必须覆盖 NPZ 里的每一帧，不能默认跳过缺图，否则 temporal window 会错位。
   - 训练 split 默认使用 `--split train`，避免 HO3D official evaluation whitelist 把 train frames 过滤掉。
   - 只有给官方 evaluation scorer 准备预测时，才显式使用 `--split evaluation`。

2. 再训练 image-free 的 `SensorTemporalRefiner`。
   - 入口：`scripts/train_sensor_refiner.py`
   - dataset：`hamer/datasets/sensor_refiner_dataset.py`
   - 每个训练 sample 仍然是“目标帧”，因此 batch 可以 shuffle。
   - 但 sample 内部自带 `(window_size, 48)` pose history 和 `(window_size, 5)` sensor history。
   - sequence 开头用第一帧左 padding，同时用 `pose_valid_mask / sensor_valid_mask` 标出 padding 无效。

这样做的好处是：

- 训练阶段不需要每一步都在线跑 HaMeR，远程实验成本更低。
- 可以先验证“sensor 是否能修正 finger articulation”，不被 RGB backbone 训练细节干扰。
- 可以同时支持三种 history source：
  - `base`：历史 pose 来自 HaMeR cache，更接近推理时分布
  - `gt`：teacher forcing，适合 sanity check
  - `mixed`：在 GT history 和 base history 之间随机混合，减少 exposure bias

评测阶段要和训练区分开：

- 默认可做 stateless eval：直接使用 cache 里的历史 pose window。
- 正式 temporal 结论应使用 stateful eval：按 sequence 顺序逐帧跑，把上一帧 refined pose 回填给下一帧窗口。
- 入口：`scripts/eval_sensor_refiner.py`
- stateful eval 才能回答“前一帧预测会不会影响后一帧、是否真的减少跳变”。

当前 v2 最小 loss 是：

- `hand_pose` residual MSE：默认主 loss
- optional `global_orient` MSE：只用于 ablation
- optional smoothness：只约束最后两帧 history 和当前 refined pose 的加速度
- optional `--base_pose_noise_std`：训练阶段只扰动当前 `base_pose[:, 3:]`，用于让 refiner 见过 RGB/base hand pose 抖动；默认 `0.0`，不影响旧实验。
- optional `--base_pose_hold_dropout`：训练阶段随机用上一帧窗口 pose 替换当前 base pose，模拟当前帧 RGB/base 失败；当前 sweep 未超过 `sensdrop02`，暂不作为默认。
- optional `--sensor_dropout / --sensor_noise_std`：训练阶段只扰动有效 sensor timestep；当前 HO3D-v3 cached sweep 里 `sensor_dropout=0.2` 有收益，sensor Gaussian noise 暂未带来额外提升。

当前固定两个 v2 refiner baseline：

- `--sensor_mode sensor`：sensor-guided refiner，使用 `(T, 5)` sensor window。
- `--sensor_mode zero`：temporal pose-only refiner，同结构但 sensor 输入全零。

这样能在同一 base cache、同一 history window、同一 stateful eval 协议下隔离“sensor 是否提供额外收益”。

当前 cached stress-test 入口：

- `scripts/eval_sensor_refiner.py --blackout_len 1/3 --blackout_strategy hold`
  - 模拟连续视觉失败帧：当前帧 base pose 使用上一帧 clean base pose hold 住。
  - sensor window 仍然来自当前序列，因此可以检查 sensor 是否帮助 finger articulation recovery。
- `--base_pose_noise_std`
  - 模拟 base RGB pose 抖动。
- `--sensor_dropout`
  - 模拟拉线信号缺失，用于 sensor robustness ablation。

当前 cached metrics 入口：

- `scripts/eval_sensor_refiner_metrics.py`
  - 读取 `eval_sensor_refiner.py` 输出的 NPZ。
  - 用 HaMeR 的 MANO layer 从 `base_pose / refined_pose / GT pose` 生成 joints / vertices。
  - 输出 base/refined 的 `PA-MPJPE / PA-MPVPE / MPJVE / MPJAE / PredJitter / Stress_PA-MPJPE`。

后续如果要把目标从 smoke protocol 推进到正式 benchmark，应继续补：

- pseudo-sensor FK consistency loss
- clean 与 stress-test 的固定评测表格

## 7. Benchmark 设计

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

## 8. 下一步

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
