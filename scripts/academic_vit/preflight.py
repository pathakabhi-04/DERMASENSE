"""Preconditions 2 and 3 of docs/academic_vit_spec.md, runnable on CPU.

1. Our CV-4 ViT trunk equals torchvision's own vit_b_16(IMAGENET1K_V1) on
   10 random tensors (max abs diff of the CLS feature < 1e-4). This is the
   spec's fallback for precondition 2, since ImageNet val is not on disk.
2. forward_with_attention() -- the hand-run blocks used for rollout --
   reproduces the model's own encoder output, at 224 and at 512 (< 1e-5).
3. Rollout rows are distributions (sum to 1).
4. Precondition 3: one CV-3 batch at 512 through AcademicModel(S, vit_b_16)
   runs forward and backward; time (and CUDA peak memory, on a GPU) recorded.

    PYTHONPATH=. python3 scripts/academic_vit/preflight.py [--out preflight.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.academic.model import AcademicModel
from src.academic.vit import ViTClassifier, forward_with_attention, rollout, tokens


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"device": str(dev)}

    ours = ViTClassifier().vit.eval().to(dev)
    ref = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(dev)
    x = torch.randn(10, 3, 224, 224, device=dev)
    d1 = (tokens(ours, x)[:, 0] - tokens(ref, x)[:, 0]).abs().max().item()
    res["trunk_vs_torchvision_maxdiff"] = d1
    assert d1 < 1e-4, d1

    t_ref = tokens(ref, x)
    t_att, attns = forward_with_attention(ref, x)
    d2 = (t_ref - t_att).abs().max().item()
    res["manual_blocks_vs_forward_224_maxdiff"] = d2
    assert d2 < 1e-5, d2
    r = rollout(attns)
    res["rollout_row_sum_maxdev"] = (r.sum(-1) - 1).abs().max().item()
    assert res["rollout_row_sum_maxdev"] < 1e-4

    seg = AcademicModel("S", backbone="vit_b_16", image_size=512, seed=42).to(dev).eval()
    x512 = torch.randn(2, 3, 512, 512, device=dev)
    d3 = (tokens(seg.encoder.vit, x512) - forward_with_attention(seg.encoder.vit, x512)[0]).abs().max().item()
    res["manual_blocks_vs_forward_512_maxdiff"] = d3
    assert d3 < 1e-5, d3
    feats = seg.encoder(x512)
    res["pyramid_shapes"] = [list(f.shape) for f in feats]
    assert [tuple(f.shape[1:]) for f in feats] == [(64, 256, 256), (256, 128, 128), (512, 64, 64), (1024, 32, 32), (2048, 16, 16)]

    with torch.enable_grad():
        seg.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        xb = torch.randn(8 if dev.type == "cuda" else 2, 3, 512, 512, device=dev)
        t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = seg(xb)["seg"]
        out.float().mean().backward()
        res["seg_512_fwd_bwd_batch"] = xb.shape[0]
        res["seg_512_fwd_bwd_seconds"] = round(time.time() - t0, 2)
        if dev.type == "cuda":
            res["seg_512_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["seg_param_millions"] = round(sum(p.numel() for p in seg.parameters()) / 1e6, 1)
    res["all_passed"] = True
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
