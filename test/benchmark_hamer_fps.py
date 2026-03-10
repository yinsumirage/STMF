"""
Benchmark HaMeR inference FPS.
Tests: FP32 baseline, TF32, AMP (autocast fp16), AMP + torch.compile.
Usage: python benchmark_hamer_fps.py

====================================================================
Device: cuda
GPU:    NVIDIA GeForce RTX 3060 Laptop GPU
SMs:    30  |  VRAM: 6.4 GB
Summary:
  fp32              173.4 ms    5.8 FPS  (1.00x vs baseline)
  tf32               80.0 ms   12.5 FPS  (2.17x vs baseline)
  amp                82.1 ms   12.2 FPS  (2.11x vs baseline)
  amp_compile        77.1 ms   13.0 FPS  (2.25x vs baseline)
====================================================================
"""

import time
import torch
from hamer.models import load_hamer, DEFAULT_CHECKPOINT

WARMUP = 5
RUNS   = 30
BATCH  = 1

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench(model, batch, label, use_amp=False):
    model.eval()
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else torch.inference_mode()
    def run_one():
        if use_amp:
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                model.forward_step(batch, train=False)
        else:
            with torch.inference_mode():
                model.forward_step(batch, train=False)

    for _ in range(WARMUP):
        run_one(); sync()
    sync()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        run_one(); sync()
    elapsed = (time.perf_counter() - t0) / RUNS * 1000
    fps = 1000 / elapsed
    print(f"  [{label:<32}]  {elapsed:6.1f} ms  →  {fps:.1f} FPS")
    return elapsed

def load_fresh(device):
    m, _ = load_hamer(DEFAULT_CHECKPOINT)
    return m.to(device).eval()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"SMs:    {props.multi_processor_count}  |  "
              f"VRAM: {props.total_memory/1e9:.1f} GB\n")

    batch = {'img': torch.randn(BATCH, 3, 256, 256, device=device)}
    results = {}

    print("=" * 68)
    print("Benchmarking:")
    print("=" * 68)

    # ── 1. FP32 baseline (TF32 off, PyTorch default) ─────────────
    torch.set_float32_matmul_precision('highest')   # explicit default
    torch.backends.cudnn.allow_tf32 = False
    model = load_fresh(device)
    results['fp32'] = bench(model, batch, "FP32 (TF32 disabled)")
    del model; torch.cuda.empty_cache()

    # ── 2. TF32 (Ampere Tensor Cores for fp32 matmul) ────────────
    # Just flipping this flag — no model change, no precision change in
    # the exponent range, only the mantissa is rounded to 10 bits.
    torch.set_float32_matmul_precision('high')   # enables TF32
    torch.backends.cudnn.allow_tf32 = True
    model = load_fresh(device)
    t = bench(model, batch, "TF32  (1 flag, same model weights)")
    results['tf32'] = t
    print(f"    → speedup vs FP32: {results['fp32']/t:.2f}×")
    del model; torch.cuda.empty_cache()

    # ── 3. AMP autocast FP16 ─────────────────────────────────────
    # torch.autocast handles dtype boundaries automatically;
    # matmuls / convolutions run fp16, reductions stay fp32.
    # MANO's .float() cast inside forward_step is preserved.
    torch.set_float32_matmul_precision('high')
    model = load_fresh(device)
    try:
        # quick sanity check
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model.forward_step(batch, train=False)
        ok = not any(torch.isnan(v).any()
                     for v in [out['pred_cam'], out['pred_keypoints_3d']])
    except Exception as e:
        ok = False; print(f"  AMP sanity check failed: {e}")

    if ok:
        t = bench(model, batch, "AMP autocast fp16 + TF32", use_amp=True)
        results['amp'] = t
        print(f"    → speedup vs FP32: {results['fp32']/t:.2f}×")
    else:
        print(f"  [AMP                            ]  SKIPPED")
    del model; torch.cuda.empty_cache()

    # ── 4. AMP + torch.compile ────────────────────────────────────
    print(f"\n  torch.compile warmup (expect 60-120s for JIT compilation)...")
    torch.set_float32_matmul_precision('high')
    model = load_fresh(device)
    try:
        model.backbone = torch.compile(model.backbone, mode='default')
        # trigger compilation
        for i in range(3):
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                model.forward_step(batch, train=False)
            sync()
            print(f"    warmup {i+1}/3 done")
        t = bench(model, batch, "AMP + torch.compile backbone", use_amp=True)
        results['amp_compile'] = t
        print(f"    → speedup vs FP32: {results['fp32']/t:.2f}×")
    except Exception as e:
        print(f"  [AMP + compile                  ]  SKIPPED: {e}")
    del model; torch.cuda.empty_cache()

    print("=" * 68)
    print("Summary:")
    for k, v in results.items():
        fps = 1000/v
        rel = results['fp32']/v
        print(f"  {k:<16} {v:6.1f} ms  {fps:5.1f} FPS  ({rel:.2f}x vs baseline)")
    print("=" * 68)
