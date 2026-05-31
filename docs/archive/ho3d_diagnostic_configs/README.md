# HO3D Diagnostic Experiment Configs

These configs are archived because the current mainline is
sensor-guided temporal MANO refinement rather than plain HaMeR HO3D-v3
finetuning.

They are kept for reproducibility of the HO3D protocol and overfit
diagnostics:

- `hamer_ho3d_pose_cam_shape_gtcoord.yaml`
- `hamer_ho3d_pose_only_finetune_gtcoord.yaml`
- `hamer_ho3d_pose_only_worient.yaml`

If one of these needs to be rerun, copy it back to
`hamer/configs_hydra/experiment/` explicitly and record the reason in
`docs/01_current_status.md`.
