# Remote Workflow Notes

这份文档专门记录远程双 4090 机器的同步、数据路径和实验运行约定。

## 1. 远程机器

- SSH alias: `dual4090`
- 连接信息：`user@220.196.173.235:39029`
- 项目目录：`/home/user/code/STMF`
- conda 环境：`STMF`
- conda 初始化：`. /home/user/miniconda3/etc/profile.d/conda.sh`
- 数据根目录：`/data/hand_data`
- HO3D-v3：`/data/hand_data/HO-3D_v3`
- GPU：双 NVIDIA GeForce RTX 4090

常用远程命令模板：

```bash
ssh dual4090 '. /home/user/miniconda3/etc/profile.d/conda.sh && conda activate STMF && cd /home/user/code/STMF && <command>'
```

## 2. 本地和远程路径差异

不要假设本地 WSL 和远程机器的数据路径一致。

当前远程实验默认使用：

- `/data/hand_data/HO-3D_v3/ho3d_train.npz`
- `/data/hand_data/HO-3D_v3/ho3d_evaluation.npz`
- `/data/hand_data/HO-3D_v3/datasets_tar_ho3d_v3.yaml`
- `/data/hand_data/FreiHAND_pub_v2`

如果某些 config / 命令只改了数据路径、checkpoint 路径、日志路径或远程机器绝对路径，不要直接理解成模型或协议差异。

反过来，如果改动影响了：

- joints 顺序
- sensor 计算方式
- HO3D bbox / scale 协议
- eval export 顺序
- model forward / loss
- temporal window / stateful eval

就必须按真实代码行为处理，并同步更新 `docs/01_current_status.md` 和 `docs/03_code_structure.md`。

## 3. 远程同步规则

远程跑实验前先检查：

```bash
ssh dual4090 'cd /home/user/code/STMF && git status --short --branch && git pull --ff-only'
```

如果远程有未提交改动或 untracked 文件，不要自动 `reset` / `clean` / 覆盖。

优先使用安全备份分支：

```bash
ssh dual4090 'cd /home/user/code/STMF && git switch -c backup/remote-wip-YYYYMMDD && git add -A && git commit -m backup_remote_wip_YYYYMMDD'
ssh dual4090 'cd /home/user/code/STMF && git switch main && git pull --ff-only'
```

说明：

- 备份分支只保存在远程本机，默认不推送到 GitHub。
- 如果只是少量明确有价值的代码，应先在本地 main 上做干净 cherry-pick / patch，而不是直接把远程工作区整体合并。
- 结果 JSON、debug 图片、日志 symlink 通常不要直接进 main；需要时把关键数字和结论写进 docs。

## 4. 2026-06-03 远程同步记录

远程 `main` 曾停在 `17b3716`，本地工作区包含早期 HO3D finetune / projection 诊断改动和结果产物。

已处理：

- 创建远程本地备份分支：`backup/remote-wip-20260603`
- 备份提交：`d540147 backup_remote_wip_2026_06_03`
- 远程 `main` 已 fast-forward 到 `origin/main` 的 `56be755`

当时检查结论：

- 多数核心 HO3D 修复已被 `origin/main` 吸收。
- 远程 `scripts/train_stmf.py` 比 main 旧，不应覆盖 main。
- 远程三个 HO3D 诊断 yaml 已在 `docs/archive/ho3d_diagnostic_configs/` 归档。
- 远程 `docs/04_hamer_stmf_training_guide.md` 是早期训练背景文档，不应替换当前 `docs/04_ho3d_camera_intrinsics_note.md`。
- 远程结果文件支持当前结论：plain HaMeR HO3D finetune 和早期 STMF v1 没有稳定超过 base HaMeR。

关键远程结果：

| Run | PA-MPJPE mean | PA-MPVPE mean |
| --- | ---: | ---: |
| HaMeR baseline | 13.496 mm | 12.932 mm |
| HaMeR 3w finetune | 13.499 mm | 12.937 mm |
| STMF 20 epoch | 18.176 mm | 17.767 mm |
| STMF 5k step | 18.661 mm | 18.219 mm |

如果后续需要找回旧远程文件：

```bash
ssh dual4090 'cd /home/user/code/STMF && git show --stat backup/remote-wip-20260603'
ssh dual4090 'cd /home/user/code/STMF && git diff main..backup/remote-wip-20260603 -- <path>'
```

## 5. tmux 运行长任务

远程长任务默认放进 tmux：

```bash
ssh dual4090 'tmux new -d -s stmf_v2'
ssh dual4090 'tmux send-keys -t stmf_v2 "cd /home/user/code/STMF && . /home/user/miniconda3/etc/profile.d/conda.sh && conda activate STMF" C-m'
ssh dual4090 'tmux capture-pane -t stmf_v2 -p -S -200'
```

训练命令建议保存完整日志：

```bash
python <entrypoint> <args> 2>&1 | tee logs_remote/<run_name>.log
```

启动后必须检查最近输出，确认：

- conda 环境正确
- CUDA / GPU 可见
- 数据路径存在
- dataloader 没卡住
- loss 不是 NaN
- checkpoint / log 目录符合预期

## 6. 2026-06-03 STMF-v2 remote smoke

目的：

- 验证远程 `dual4090` 上新的 cached sensor refiner 链路是否能跑通真实 HO3D 图片和 HaMeR checkpoint。
- 只做小规模 protocol smoke，不作为正式指标。

环境：

- 远程 HEAD：`497a431`
- GPU：使用 `CUDA_VISIBLE_DEVICES=1`
- checkpoint：`/home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- 输入子集：`/data/hand_data/HO-3D_v3/stmf_v2_debug/ho3d_train_128.npz`
- base cache：`/data/hand_data/HO-3D_v3/stmf_v2_debug/ho3d_train_128_hamer_base_cache.npz`
- train log/checkpoint：`/home/user/code/STMF/logs_remote/stmf_v2_smoke_20260603`
- eval output：`/home/user/code/STMF/results/sensor_refiner/debug/stmf_v2_smoke_20260603_stateful.npz`
- success summary：`/home/user/code/STMF/logs_remote/stmf_v2_smoke_20260603_success.txt`
- note：`logs_remote/stmf_v2_smoke_20260603.log` 保留了第一次引号污染导致失败的日志，不作为成功 smoke 的依据。

结果：

- HO3D train 前 128 帧子集创建成功：`train/ABF10/rgb/0000.jpg` 到 `train/ABF10/rgb/0127.jpg`。
- HaMeR base cache 成功写出，8 个 batch，shape 与 128 帧子集对齐。
- `train_sensor_refiner.py` 跑通 3 step，`loss_hand_pose` 约从 `0.00303` 降到 `0.00187`。
- `eval_sensor_refiner.py --stateful` 跑通 128 帧，输出：
  - `refined_pose`: `(128, 48)`
  - `delta_hand_pose`: `(128, 45)`
  - `stateful`: `True`
  - `delta_abs_mean`: `0.030069`

暴露并修复的问题：

- `scripts/cache_base_hamer_predictions.py` 曾硬编码 `ImageDataset(train=False)`。
- 对 HO3D train 子集运行时会错误套用 official evaluation whitelist，导致 `keeping 0/128`。
- 现在 cache 脚本新增 `--split {train,evaluation}`，默认 `train`；只有 official evaluation 子集才显式使用 `--split evaluation`。

结论：

- 远程 CUDA、真实图片读取、HaMeR forward、frame-aligned cache、shuffleable v2 training、stateful eval 这条最小链路已经打通。
- 下一步可以把规模扩大到单个 HO3D sequence 或几千帧，并接入 temporal metrics / blackout stress，而不是继续做代码级 smoke。
