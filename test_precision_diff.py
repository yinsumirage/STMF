"""
Quick script to verify numeric difference between FP32 and TF32/AMP outputs.

TF32 vs 纯FP32:
  Mean vertex error: 0.0587 mm (毫米)
  Max vertex error:  0.0893 mm
AMP(FP16) vs 纯FP32:
  Mean vertex error: 0.2317 mm (毫米)
  Max vertex error:  0.5140 mm
平均误差只有 0.2 毫米，最大不超过 0.5 毫米
"""
import torch
from hamer.models import load_hamer, DEFAULT_CHECKPOINT

def test_precision_diff():
    device = 'cuda'
    model, _ = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    torch.manual_seed(42)
    batch = {'img': torch.randn(1, 3, 256, 256, device=device)}

    print("Comparing FP32 vs TF32 vs AMP (FP16)")
    print("-" * 50)

    # 1. True FP32 Baseline
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        out_fp32 = model.forward_step(batch, train=False)
        verts_fp32 = out_fp32['pred_vertices']

    # 2. TF32
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    with torch.inference_mode():
        out_tf32 = model.forward_step(batch, train=False)
        verts_tf32 = out_tf32['pred_vertices']

    # 3. AMP (FP16)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        out_amp = model.forward_step(batch, train=False)
        verts_amp = out_amp['pred_vertices']

    # Compare (L2 distance in meters)
    diff_tf32 = torch.norm(verts_fp32 - verts_tf32, dim=-1).mean().item() * 1000  # convert to mm
    max_tf32 = torch.norm(verts_fp32 - verts_tf32, dim=-1).max().item() * 1000

    diff_amp = torch.norm(verts_fp32 - verts_amp, dim=-1).mean().item() * 1000
    max_amp = torch.norm(verts_fp32 - verts_amp, dim=-1).max().item() * 1000

    print(f"TF32 difference from FP32:")
    print(f"  Mean vertex error: {diff_tf32:.4f} mm")
    print(f"  Max vertex error:  {max_tf32:.4f} mm\n")

    print(f"AMP (FP16) difference from FP32:")
    print(f"  Mean vertex error: {diff_amp:.4f} mm")
    print(f"  Max vertex error:  {max_amp:.4f} mm")

if __name__ == '__main__':
    test_precision_diff()
