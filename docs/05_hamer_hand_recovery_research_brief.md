# HaMeR x HO3D Hand Recovery Research Brief

这份文档是给后续做 deep research / 查论文 / 对比其它手部数据集协议时用的摘要。

目标不是替代实验记录，而是把目前已经拿到的强证据、当前最像真的机制、以及最值得查的外部问题整理成一个更适合研究的入口。

---

## 1. 当前已经确认的强结论

### 1.1 HO3D 原始 GT 链路本身是自洽的

当前已经通过 `tools/data_prep/inspect_ho3d_projection_consistency.py` 验证：

- packed `GT 2D`
- packed `GT 3D`
- reordered `meta handJoints3D`
- `GT MANO + handTrans`

在 `coord_change = diag(1, -1, -1)` 的语义下是彼此一致的。

对 hard case：

- `11903` -> `train/BB14/rgb/0188.jpg`
- `23807` -> `train/GSF13/rgb/0701.jpg`

已经得到：

- `patch_exact_packed_px ≈ 0`
- `patch_patchK_packed_px ≈ 0`
- `patch_exact_mano_px ≈ 0.29 / 0.38 px`
- `patch_patchK_mano_px ≈ 0.29 / 0.38 px`

这说明：

- HO3D 的 `MANO + handTrans + camMat + crop`
- 本身可以几乎完美重投影到 patch GT 2D

因此当前问题不是：

- `global_orient GT` 单独坏了
- `MANO GT` 整体坏了
- `handTrans` 本身坏了

### 1.2 plain HaMeR 在 HO3D hard case 上确实存在 supervision conflict

overfit 诊断已经确认：

- `2D only`
  - 可以把 hard case 拉得很贴
- `2D + 3D`
  - 会在部分 hard case 上把结果从图像最优解拉走
- `2D + hand_pose`
  - 在某些样本上比 `2D + 3D` 更容易贴近

这说明：

- 模型不是完全学不会
- 也不只是 capacity 不够
- 当前 full supervision 确实会在 hard case 上互相竞争

### 1.3 HaMeR 当前 patch camera surrogate 与 HO3D protocol-correct patch camera 不完全等价

inspect 工具的 patch-space 拟合结果已经很明确：

- protocol-correct `patchK`
  - 可以接近 `0.3 px`
- 但当前 HaMeR 风格 camera family 拟合：
  - `coord_change(MANO local)`
  - `coord_change(MANO + handTrans)`
  - `cam_t`
  - `cam_t + fx + fy + cx + cy`
  - `free tz`
  都仍然卡在 `~5 px`

hard case 典型结果：

- `11903`
  - `patch_fit_mano_local_coord_camk_px ≈ 5.57`
  - `patch_fit_mano_with_trans_coord_camk_px ≈ 5.56`
- `23807`
  - `patch_fit_mano_local_coord_camk_px ≈ 5.09`
  - `patch_fit_mano_with_trans_coord_camk_px ≈ 5.10`

这说明：

- 问题不只是 `handTrans` 没带进去
- 也不只是 `tz > 0`
- 更像是 HaMeR 当前 `perspective_projection + pred_cam_t` 这套 patch 相机参数化
  与 HO3D 的 `patchK` 从定义层面就还没有对齐

---

## 2. 当前最像真的机制

### 2.1 HO3D 原始链

HO3D 对一帧手的几何表达可以概括成：

1. `MANO(global_orient, hand_pose, betas)` 生成局部手
2. `+ handTrans` 把手平移到相机前的 3D 位置
3. `coord_change = diag(1, -1, -1)` 把 HO3D/OpenGL 坐标转到投影使用的相机坐标
4. 用样本自己的 `camMat`
5. 再经过 crop / resize，形成 patch-space 的 2D 监督

### 2.2 plain HaMeR 当前链

plain HaMeR 当前的 patch-space 链可以概括成：

1. 预测 `global_orient / hand_pose / betas`
2. 通过 MANO 得到 `pred_keypoints_3d`
3. 预测简化 camera 参数 `pred_cam`
4. 通过固定规则生成 `pred_cam_t`
5. 用 `perspective_projection(...)` 和 fixed-focal/centered patch 相机做投影

### 2.3 两条链不等价的核心

表面上二者都有：

- 旋转
- 平移
- 手部形状与姿态

但实际差别在于：

- HO3D 用的是：
  - 真实 `camMat`
  - 真实 `handTrans`
  - 显式 `crop_affine`
- HaMeR 用的是：
  - `pred_cam_t`
  - fixed patch camera
  - centered/normalized patch projection surrogate

最关键的新证据是：

- `patchK_params`
  的真实量级
- 和 `cam+K fit` 学出来的 `focal_length / camera_center`
  完全不在一套数值语义里

例如本地 `idx=0`：

- `patchK`
  - `fx ≈ 171`
  - `fy ≈ 171`
  - `cx ≈ 126`
  - `cy ≈ 123`
- `cam+K fit`
  - `fx ≈ 1.2e4`
  - `fy ≈ 1.35e4`
  - `cx ≈ 0`
  - `cy ≈ 0`

这说明不是“少几个参数”那么简单，而是：

- 两套相机定义本身不在同一坐标和归一化语义里

---

## 3. 当前不再支持的解释

下面这些说法，现在都没有足够证据支持，甚至已经被实验弱化了：

### 3.1 “global_orient GT 本身坏了”

不支持。

因为：

- `MANO + handTrans + coord_change + camMat`
  已经能接近完美重投影到 GT 2D

如果 `global_orient GT` 本身坏了，这一步不可能这么准。

### 3.2 “只要给 HaMeR 多预测几个内参参数就自然会好”

不支持。

因为：

- 加 `fx/fy/cx/cy`
- 甚至放开 `tz`

都没有把 hard case 从 `~5 px` 拉到 `~0.3 px`

所以当前问题不是单纯“自由度数量”。

### 3.3 “只要把 `handTrans` 等价成 `pred_cam_t` 就行”

不支持。

因为：

- `coord_change(MANO local)`
- 和
- `coord_change(MANO + handTrans)`

在当前 HaMeR-style patch camera family 下拟合结果几乎一样，都卡在 `~5 px`。

这说明当前问题不只是缺失 GT translation，而是整个 camera/projection 语义不对齐。

---

## 4. 用户当前最关心的研究问题

下面这些问题，适合拿去做 deep research。

### 4.1 别的手部数据集也像 HO3D 这样吗？

当前推断：

- **不一定。**

更可能存在几类不同协议：

1. **真实相机协议**
   - 有真实 `camMat`
   - 有真实 3D hand translation
   - 有 parametric MANO 标注
   - HO3D 属于这一类的典型代表

2. **弱相机/近似相机协议**
   - 只有 crop 后 patch 监督
   - 没有真实内参
   - 训练时通常用 fixed focal / centered patch camera
   - 很多互联网弱标注数据更接近这一类

3. **伪标签/拟合协议**
   - MANO 参数、2D/3D 点来自外部拟合器或 pseudo-label
   - 相机量可能只是“有效变量”，未必有真实物理意义

所以非常值得研究的是：

- 哪些手部数据集提供真实内参
- 哪些提供真实 translation
- 哪些只提供 MANO + 2D/3D 点
- 各家训练时是否真的使用真实 `K`
- 还是把数据全部统一到 centered patch camera surrogate

### 4.2 如果给 HaMeR 加 `fx/fy/cx/cy`，理论自由度是不是就够了？

当前判断：

- **从“自由度计数”上看，接近够。**
- 但从“参数语义是否一致”上看，**还不够**。

因为当前问题不是只差四个数，而是：

- HaMeR 当前 camera head 的输出语义
- 和 HO3D 的 protocol-correct `patchK`
  根本不在一套定义里

所以即使理论上自由度匹配，
如果参数化和坐标语义不统一，模型也不一定能学到那个解。

### 4.3 这是不是说明 HaMeR 预训练时就把 camera head 限制住了？

当前判断：

- **很可能是。**

更准确地说：

- 预训练阶段的 camera head 学到的是：
  - centered patch camera
  - fixed focal surrogate
  - 与大量互联网/弱监督数据相容的一套“有效相机变量”

因此它能表达的，不一定是：

- HO3D 这种带真实 `camMat + handTrans + crop` 的 protocol-correct patch 相机

所以 HO3D hard case 出现：

- base / wrist 总有固定残差
- overfit 也掉不干净

是很合理的。

### 4.4 WiLoR 这种直接预测内参的方法是不是更对？

当前判断：

- 方向上值得参考
- 但语义上要很谨慎

因为如果没有真实 `K` 监督，模型预测出来的“内参”很可能只是：

- 一个为了最小化 loss 的 effective camera variable

而不是真实物理相机。

这会带来两个现实问题：

1. 训练时效果可能更好
2. 但可解释性差
3. 推理时如果换成真实内参，反而可能更差

所以更值得研究的是：

- WiLoR 一类方法的 camera parameters 到底是不是物理相机
- 还是只是有效潜变量
- 它们在不同数据协议之间的迁移性如何

---

## 5. 当前最值得查的外部 research 方向

### 5.1 手部数据集相机协议综述

想查的问题：

- HO3D / FreiHAND / InterHand / DexYCB / H2O / OakInk / Ego4D hand / ARCTIC hand 等
  是否提供真实内参、外参、translation、MANO
- 训练时各方法到底有没有显式使用这些量
- 还是最终都统一回 fixed patch camera surrogate

### 5.2 Hand mesh recovery 的相机头设计

想查的问题：

- 哪些论文使用 weak-perspective / centered patch camera
- 哪些论文显式预测 `fx/fy/cx/cy`
- 哪些论文显式用真实 `K`
- 哪些方法在 mixed datasets 上做过 camera-protocol unification

### 5.3 MANO + camera 的耦合方式

想查的问题：

- 当前主流方法里：
  - `global_orient`
  - `handTrans`
  - `camera translation`
  - `intrinsics`
  是怎样分工的
- 哪些方法直接回归 hand/world translation
- 哪些方法只回归 patch camera surrogate

### 5.4 patch-space projection 的标准做法

想查的问题：

- 是否有通用做法把真实 `camMat + crop affine`
  统一折算成每样本 patch intrinsics
- 是否有方法在训练时同时支持：
  - 有真实内参的数据集
  - 没有真实内参的互联网数据集

---

## 6. 当前项目里最稳的工程判断

从目前证据看，最稳的工程判断是：

1. HO3D 数据本身没有明显坏掉
2. plain HaMeR 在 HO3D 上的 hard case 问题是真实存在的
3. 它不是单一 `global_orient` bug
4. 也不是单纯多加几个相机参数就自动能好
5. 更像是：
   - current HaMeR camera surrogate
   - 与 HO3D protocol-correct camera chain
   的系统性不对齐

因此如果后续要继续做工程扩展，路线应该更谨慎：

- 先做协议研究
- 再决定是：
  - 保守地留在 HaMeR surrogate 相机里调 recipe
  - 还是真正设计一个新的 camera/projection head

---

## 7. 当前最适合带去 deep research 的一句话摘要

当前最像真的问题不是：

- “HO3D 的 MANO / orient 标注错了”

而是：

- **HaMeR 当前基于 centered patch camera 的相机头，与 HO3D 基于 `camMat + handTrans + crop` 的真实相机协议不等价；这种不等价在 hard case 上表现为无法消掉的 patch-space 残差与 supervision conflict。**

