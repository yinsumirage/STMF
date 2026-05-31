# HO3D Camera Intrinsics Note

这份文档专门总结 plain HaMeR 在 HO3D hard case 上暴露出来的“相机内参与 patch 投影近似”问题。

目标不是直接给出最终修复，而是把下面几件事说清楚：

- 当前到底在怀疑什么
- 为什么这件事在 HO3D 上特别容易暴露
- 为什么它在互联网大规模数据上又很难直接监督
- 如果要继续验证，什么路线最稳

## 1. 当前问题的来源

我们现在已经确认了两件事：

1. HO3D 原始 GT 链路本身是自洽的
2. plain HaMeR 在少数 hard case 上，full supervision 下的 patch-space 拟合仍然存在固定残差

已经确认自洽的 HO3D 原始链路是：

1. `MANO(global_orient, hand_pose, betas)` 生成局部手
2. `+ handTrans` 把手平移到相机前的 3D 位置
3. `coord_change = diag(1, -1, -1)` 进行 HO3D/OpenGL 到投影坐标的转换
4. 用样本自己的 `camMat` 投影到原图 2D
5. 再经过 crop / resize 变成训练 patch 上的监督

当前 inspect 工具已经验证：

- packed `GT 2D`
- packed `GT 3D`
- reordered `meta handJoints3D`
- `GT MANO + handTrans`

在 `coord_change` 语义下基本一致。

这说明目前不是：

- `global_orient GT` 单独坏了
- `hand_pose GT` 单独坏了
- `MANO + handTrans` 无法重投影到 GT 2D

补充一个很具体的矩阵定义：

- `camMat`
  - 是 HO3D 原始相机内参矩阵，形状是 `3 x 3`
  - 典型形式：

```text
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

- `crop_affine`
  - 是从原图像素坐标映射到 patch 像素坐标的仿射变换
  - OpenCV / 当前实现里先返回的是 `2 x 3`
  - 在齐次坐标下补成 `3 x 3` 后再与 `camMat` 相乘：

```text
patchK = crop_affine_h @ camMat
```

在当前 deterministic overfit 设定里：

- `train=False`
- `rot=0`
- `do_flip=False`

所以这个 `crop_affine` 主要表达的是：

- 缩放
- 平移

而不是一个“真实相机旋转矩阵”。

## 2. HaMeR 当前真正做的事

plain HaMeR 不是用 HO3D 的真实相机链路做训练，而是用一个简化后的 patch 相机。

当前 HaMeR 的主链路是：

1. 网络回归 `global_orient / hand_pose / betas`
2. 通过 MANO 得到 `pred_keypoints_3d`
3. 网络再回归一个简化 camera 参数 `pred_cam`
4. 把 `pred_cam` 转成 `pred_cam_t = [tx, ty, tz]`
5. 用固定 focal、固定主点的 patch 相机，把 `pred_keypoints_3d + pred_cam_t` 投到 patch 2D

也就是说，当前 HaMeR 并没有直接使用：

- 每张图自己的真实 `camMat`
- 每张图真实的 `handTrans`

它使用的是一个 patch-space surrogate：

- `pred_cam_t`
- fixed focal
- zero principal point

这条链在大多数样本上够用，但在 HO3D hard case 上，可能已经不再严格等价于：

- `camMat + handTrans + crop`

## 3. `handTrans` 和 `pred_cam_t` 的差别

这两个量形式上都像 3 维向量，但语义并不相同。

### 3.1 `handTrans`

`handTrans` 是 HO3D 原始 meta 里的 GT 3D 平移向量。

它的作用是：

- 把 MANO 在局部坐标系里生成的手
- 平移到真实相机空间里的位置

它是数据集给出的真实几何量。

### 3.2 `pred_cam_t`

`pred_cam_t` 是 HaMeR 在 patch-space 里学出来的相机平移替代量。

它的作用是：

- 帮助当前 3D 手在 patch 上投到正确的位置和尺度

它不是 HO3D `handTrans` 的直接预测值，也没有被 GT `handTrans` 单独监督。

因此：

- `handTrans` 是真实原图相机空间里的量
- `pred_cam_t` 是 patch 相机里的 surrogate

二者不能简单视为“同一个东西，只是名字不一样”。

## 4. 为什么“自由度看起来一样”，但系统还是不等价

看起来两边都有：

- 一个 3D 平移
- 一个整手旋转
- 一套 MANO 形状和局部 pose

但真正的差别在于：

### 4.1 HO3D 有真实内参 `camMat`

HO3D 的原始投影使用每张图自己的 `camMat`。

### 4.2 HaMeR 用固定 patch 相机

HaMeR 默认使用：

- fixed focal
- zero principal point
- patch 坐标系

因此当前训练并不是在“精确复刻 HO3D 的原始投影”，而是在用一个近似相机去解释 patch 上的监督。

### 4.3 训练监督已经是 crop 后的 patch 坐标

训练里的 `gt_keypoints_2d` 不是原图像素，而是：

1. 原图 keypoints
2. 经过 bbox crop / resize 仿射
3. 再归一化到 patch 空间

所以如果想让 HO3D 的原始 `camMat + handTrans` 与 HaMeR 的 patch 相机严格等价，
其实需要显式构造“每个样本自己的 patch intrinsics”。

当前 HaMeR 没做这一步，而是用统一的简化 patch camera 近似。

## 5. 为什么这件事在 hard case 上会暴露

目前 hard case 的症状是：

- `2D only` 可以拉得很贴
- 但 full supervision 下会出现固定 base / wrist 残差
- inspect 里用 `coord_change(MANO local)` 拟合最优 `pred_cam_t`，也会得到和训练后很像的“带固定间隙”的解

这说明：

- 不是简单的“网络没优化到”
- 而是很可能当前简化 patch 相机下的最优解，本来就离 GT patch 2D 有固定残差

也就是说，问题更像是：

- `HO3D 原始相机链`
- 和
- `HaMeR patch surrogate 相机链`

在这些样本上不再精确等价。

## 6. 为什么这件事很难直接靠“预测内参”解决

理论上，如果给模型更多相机自由度，例如：

- 预测 `fx, fy`
- 预测 `cx, cy`
- 或者额外预测 patch-space 的 residual intrinsics

那么 hard case 可能会更容易拟合。

但这里有一个实际困难：

### 6.1 大量互联网数据没有真实内参

在大规模网络数据、弱标注数据、无标注数据上，通常只有：

- 图像
- 2D keypoints
- 有时有 3D keypoints
- 有时有 MANO pseudo-label

但没有：

- 真实相机内参
- 真实 `handTrans`

这意味着如果直接让模型自由预测 `camMat`，又没有强监督，它很容易学成一个“有效内参”而不是真实内参。

### 6.2 预测内参很容易语义塌陷

一旦没有真实 `K` 的监督，模型预测的“内参”可能会和下面这些量耦合：

- crop 位置
- 目标尺度
- depth surrogate
- pose prior
- backbone 特征

最后得到的是：

- 一个有利于最小化 loss 的相机替代变量
- 而不是可解释的真实物理内参

这会带来两个问题：

1. 训练时可能有效，但难解释
2. 推理时如果换成真实内参，效果反而可能更差

这和 WiLoR 一类方法里常见的现象是相符的：

- 网络内部自己预测的相机参数效果更好
- 但它未必意味着这些参数具有真实几何语义

## 7. 当前更稳的验证路线

目前不建议直接第一步就把 HaMeR 改成“自由预测完整内参”。

更稳的路线是分三层：

### 7.1 先做 protocol-correct 版本

先不学习内参，而是尝试：

- 从 HO3D 的真实 `camMat`
- 加上当前样本的 crop / resize 仿射
- 显式推导每个样本自己的 patch intrinsics

然后用这个“真实 patch intrinsics”替代当前 fixed patch camera。

这个版本最有解释力，可以直接回答：

- 当前 hard case 是否真的是相机近似误差导致的

当前已经在 `tools/data_prep/inspect_ho3d_projection_consistency.py`
里加了一个不改训练主线的诊断版本：

- `--fit_patch_pred_cam_t`

它现在会同时输出：

- `exact crop(packed3D)`
- `exact crop(MANO+trans)`
- `patchK(packed3D)`
- `patchK(MANO+trans)`
- `fit cam_t on MANO local`
- `fit cam_t on coord_change(MANO local)`

其中：

- `exact crop(*)`
  - 指先在原图上用真实 `camMat` 投影，再用真实 crop 仿射映射到 patch
- `patchK(*)`
  - 指把 `camMat + crop affine` 显式合成为每个样本自己的 patch intrinsics

本地 `idx=0` smoke test 当前结果是：

- `patch_exact_packed_px = 0.0000`
- `patch_exact_mano_px = 0.2354`
- `patch_patchK_packed_px = 0.0000`
- `patch_patchK_mano_px = 0.2354`
- `patch_fit_mano_local_coord_px = 1.0240`

这组结果当前说明：

- protocol-correct 的真实 patch intrinsics 可以几乎无损复现 GT patch supervision
- 而 HaMeR 当前的 `pred_cam_t + fixed patch camera` 拟合，即使在同一样本上，也仍然会留下可见残差

所以当前路线是有根据的，不是拍脑袋猜相机问题。

### 7.2 再做小范围 learnable correction

如果 7.1 证明问题确实在 patch 相机近似，
再考虑加一个小的 residual 相机修正模块，例如：

- `delta_f`
- `delta_cx`
- `delta_cy`

或者其他少量 offset / residual 参数。

这个阶段的目标不是学完整相机，而是：

- 在真实 patch intrinsics 的基础上做小修正

这样比直接自由预测整套内参更稳，也更不容易塌语义。

当前仓库里已经按这个思路接了一个最小验证版本：

- 位置：`scripts/train_overfit.py`
- 参数：`--camera_residual_mode focal_center`
- 参数：`--camera_residual_mode focal_xy_center`

这个 tiny residual head 当前只预测：

- `delta_f`
- `delta_cx`
- `delta_cy`

或者在 4 参数版本里预测：

- `delta_fx`
- `delta_fy`
- `delta_cx`
- `delta_cy`

并且只作用在 overfit 诊断脚本内部，不影响主训练入口。

它当前的用途非常明确：

- 不是为了立刻替换 plain HaMeR 的相机实现
- 而是为了先回答：
  - 在 hard case 上，一个非常小的 patch intrinsics residual
- 是否就足以把 fixed patch camera 留下的那几个像素残差拉回来

但在继续拿远程 hard case 做 patch-space 拟合后，又得到了一个更具体的结论：

- `patch_exact_*` / `patch_patchK_*`
  - 仍然几乎完美
- 但只要换回当前 HaMeR 风格的 camera family 去拟合
  - 即使把自由度放宽到：
    - `cam_t`
    - `fx`
    - `fy`
    - `cx`
    - `cy`
  - hard case 也仍然会残留约 `5px` 的 patch 误差

这说明：

- 问题不只是 fixed intrinsics 太粗糙
- 当前用于 patch-space 拟合的 camera translation family 本身也很可疑

目前最怀疑的是：

- `tz > 0` 这个 HaMeR 风格的正深度参数化

因为在 inspect 诊断里，我们拟合的是：

- `coord_change(MANO local joints)`

而 HO3D 的 protocol-correct 链是：

- `MANO local`
- `+ handTrans`
- 再 `coord_change`

如果想在这套坐标语义下，用一个“等效 translation”去逼近真实链路，
那么其 `z` 分量不一定天然应该被限制为正。

因此当前 inspect 工具又新增了两组 signed-`tz` 诊断：

- `patch_fit_mano_local_coord_free_tz_*`
- `patch_fit_mano_local_coord_camk_free_tz_*`

这条诊断的目标不是改训练，而是先回答一个更干净的问题：

- 如果只放开 `tz` 的符号约束
- 剩余 hard-case 的那 `~5px` patch 残差
- 会不会就直接掉到接近 `patchK` 的 `~0.3px`

当前又增加了一步更直接的 sanity check：

- 把 protocol-correct 的 `patch_cam_mat`
  直接拆成真实的：
  - `fx`
  - `fy`
  - `cx`
  - `cy`
- 再和 `cam+K fit` 学出来的对应量并排打印

这个对照目前已经暴露出一个更根本的问题：

- `patchK` 的真实参数
- 和 `perspective_projection` 拟合出来的 `focal_length / camera_center`

并不在同一套数值语义里。

本地 `idx=0` 的典型现象是：

- `patchK`
  - `fx/fy` 约 `171`
  - `cx/cy` 约 `126/123`
- 但 `cam+K fit`
  - `fx/fy` 却跑到 `1.2e4 ~ 1.35e4`
  - `cx/cy` 则是接近 `0` 的小数

这说明问题已经不仅仅是：

- `fx/fy/cx/cy` 自由度不够
- `tz` 是否允许为负

而更像是：

- `patchK`
- 和 HaMeR 当前 `perspective_projection`

在坐标定义 / 数值归一化层面本身就不等价。

为了进一步区分“translation surrogate”问题和“patch 相机语义”问题，
inspect 工具还新增了一组更直接的对照：

- `coord_change(MANO local)`
- `coord_change(MANO + handTrans)`

两者都分别用下面几类 camera family 去拟合 patch GT：

- `cam_t`
- `cam_t + fx + fy + cx + cy`
- 以及 signed-`tz` 版本

这组对照的意义是：

- 如果 `MANO + handTrans` 一旦给进来，camera family 就能接近 `patchK`
  - 说明主问题在“用 camera translation surrogate 代替 GT handTrans”这一步
- 如果即使 `MANO + handTrans` 给进来，仍然离 `patchK` 很远
  - 就说明问题更像在：
    - `perspective_projection`
    - `focal / camera_center`
    - patch-space 投影语义本身

因此现在这条线已经不再只是“相机参数够不够多”的问题，
而是在检查：

- HaMeR 当前 patch 相机族
  是否从定义上就和 protocol-correct 的 `patchK`
  不完全等价

### 7.3 最后才考虑完全 learnable intrinsics

只有在前两步都证明确实还不够时，
才值得尝试让网络直接预测更自由的 `K_cam`。

否则容易把：

- 相机误差
- crop 误差
- pose 误差
- depth surrogate

全部混在一个黑盒变量里。

## 8. 对当前项目的实际意义

对当前 HO3D plain HaMeR finetune 而言，
这条相机问题带来的实际意义是：

1. 不能再简单把 hard case 解释成 `global_orient GT` 坏了
2. 也不能只靠继续调 `2D / 3D / MANO` loss 权重来期待问题自然消失
3. 需要认真验证：
   - 当前 fixed patch camera 是否本身就是 hard case 的主误差来源
4. 如果这件事被验证为真，
   - 那么后续再讨论 camera residual head / patch intrinsics 修正才有意义

但结合后续已经完成的整套 inspect / overfit 诊断，现在这条线还有一个更具体的工程结论：

- `patchK` 这条 protocol-correct patch 相机链在 HO3D 上是成立的
- 但它和 plain HaMeR 当前 camera head / `perspective_projection` 的参数语义并不兼容
- 因此不能把“构造出真实 `patchK`”直接等价成“主训练里可直接替换成 `patchK` 投影”
- 也不能再把 `patchK` 线当成当前 plain HaMeR 提分的短期可落地方案

目前已经完成并确认的点是：

- `patch_exact_*` / `patch_patchK_*` 几乎完美
- `cam_t`
- `cam_t + fx + fy + cx + cy`
- `free-tz`
- `MANO local`
- `MANO + handTrans`

这些组合无论怎么拟合，hard case 都仍然稳定卡在约 `5px` 的 patch 残差。

这说明当前剩余问题不是：

- 单纯少了 `handTrans`
- 单纯 `tz > 0`
- 单纯多给几个 intrinsics 参数就能解决

而是：

- HaMeR 当前 patch 相机 surrogate
- 和 HO3D protocol-correct `patchK`

从定义层面就不在同一套语义里。

## 9. 当前建议

当前更推荐把这件事当成一个“相机协议验证问题”，而不是立刻当成“网络结构不够强”的问题。

这一步到目前为止已经基本完成。当前文档不再把这条线定义成“继续验证中的问题”，而是把它作为一个已完成的诊断结论：

1. `inspect_ho3d_projection_consistency.py` 已经足够说明 HO3D 原始 GT 链路是自洽的
2. 也已经足够说明：
   - `patchK` 是 protocol-correct 的
   - 但 plain HaMeR 当前 camera family 不能直接对齐到这条链
3. 因此当前不建议继续把精力投入：
   - patch-space 相机族的小修小补
   - residual intrinsics overfit
   - 直接把 `projection_mode=ho3d_patchK` 推到主训练

当前更实际的工程建议是：

1. 承认 plain HaMeR 当前 camera surrogate 路线，先不重写主训练相机协议
2. 主训练优先采用已经证明有帮助的最小修复：
   - `gt_coord_recipe=flip_gt_keypoints_3d_global_orient`
3. 后续 HO3D plain finetune 的主要工作重点应转回：
   - 冻结策略
   - batch size
   - augmentation 强度
   - checkpoint 选择
   - 以及最终评测分数

如果后面要继续研究 camera 问题，更合适的定位应该是：

- 新模型分支 / 新 camera head 设计
- 而不是继续假设当前 HaMeR camera 头只差几个参数就能补齐 HO3D 协议
