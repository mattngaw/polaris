"""Standalone gfx803 GEMM correctness check.

Run *inside* the polaris container:

    python env/check-matmul.py                # check fp32 (default)
    python env/check-matmul.py --dtype bf16   # check a specific dtype

Exit 0  → matmul is correct in the chosen dtype (you can train in it).
Exit 1  → matmul is broken in the chosen dtype.

Background: the project's validated stack is ROCm 5.7 + PyTorch 2.3.0a0
(image robertrosenbusch/rocm6_gfx803_comfyui:5.7), where fp32 GEMM is
correct. Every ROCm 6.x image tested (6.1.2, 6.4.3) produced
structured-wrong fp32 GEMM on gfx803 for transformer-shape matmuls while
bf16 / fp16 / fp64 stayed correct — which is why the stack is pinned to
5.7. This check is the guard against drifting back onto a broken image.
See KNOWN_GOTCHAS.md HEADLINE for the full story.
"""
import argparse
import sys
import torch
import torch.nn.functional as F


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp64": torch.float64}
# atol per dtype: order-of-magnitude precision noise floor
ATOL = {"fp32": 1e-3, "bf16": 1.0, "fp16": 1.0, "fp64": 1e-9}


def check(label, max_diff, atol):
    ok = max_diff < atol
    print(f"  [{'OK' if ok else 'BAD'}] {label}: max_diff={max_diff:.6f} (atol={atol})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=list(DTYPES.keys()), default="fp32")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA/HIP device visible — cannot check matmul.")
        return 2

    dtype = DTYPES[args.dtype]
    atol = ATOL[args.dtype]
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device[0]={torch.cuda.get_device_name(0)}")
    print(f"checking dtype={args.dtype}\n")

    all_ok = True

    # 1. F.linear at a typical transformer shape
    torch.manual_seed(42)
    x = torch.randn(32, 128, dtype=dtype)
    W = torch.randn(65, 128, dtype=dtype)
    y_cpu = F.linear(x, W).float()
    y_gpu = F.linear(x.cuda(), W.cuda()).cpu().float()
    all_ok &= check("F.linear(x:(32,128), W:(65,128))",
                    (y_cpu - y_gpu).abs().max().item(), atol)

    # 2. nn.Linear forward
    torch.manual_seed(42)
    x = torch.randn(32, 128, dtype=dtype)
    head = torch.nn.Linear(128, 65, bias=False).to(dtype)
    y_cpu = head(x).float()
    y_gpu = head.cuda()(x.cuda()).cpu().float()
    all_ok &= check("nn.Linear(128, 65)(x:(32,128))",
                    (y_cpu - y_gpu).abs().max().item(), atol)

    # 3. Transposed-view matmul (attention's Q @ K^T pattern)
    torch.manual_seed(42)
    a = torch.randn(32, 128, dtype=dtype)
    W = torch.randn(65, 128, dtype=dtype)
    y_cpu = (a @ W.T).float()
    y_gpu = (a.cuda() @ W.cuda().T).cpu().float()
    all_ok &= check("a @ W.T (non-contig B)",
                    (y_cpu - y_gpu).abs().max().item(), atol)

    # 4. ones() — exact result known by hand
    x = torch.ones(32, 128, dtype=dtype)
    W = torch.ones(65, 128, dtype=dtype)
    y_gpu = F.linear(x.cuda(), W.cuda()).cpu().float()
    # All entries should be 128.0 exactly.
    deviation = (y_gpu - 128.0).abs().max().item()
    all_ok &= check("ones @ ones.T (exact answer = 128.0)",
                    deviation, atol)

    print()
    if all_ok:
        print(f"ALL OK — gfx803 GEMM is correct in {args.dtype} on this image; "
              f"you can train in this dtype.")
        return 0
    print(f"FAILED in {args.dtype}. See KNOWN_GOTCHAS.md HEADLINE.")
    if args.dtype == "fp32":
        print("Suggestion: try `python env/check-matmul.py --dtype bf16`. "
              "bf16 is the documented workaround on this stack.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
