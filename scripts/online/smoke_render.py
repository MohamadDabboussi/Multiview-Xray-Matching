"""Smoke test: render one acquisition from a cached cube (CPU or GPU), print timings, save a preview.

    python scripts/online/smoke_render.py --cube <cache>/knee_256.npy --out <dir>/smoke_knee.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.online_drr import OnlineDRRRenderer, preprocess_views  # noqa: E402


class _Bank:
    def __init__(self, cube):
        self.cube = cube

    def get(self, i, device):
        return self.cube.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    cube = torch.from_numpy(np.load(a.cube).astype(np.float32)).permute(2, 1, 0).contiguous().half()
    r = OnlineDRRRenderer(_Bank(cube), dim=a.dim, n_patches=16, delx=2.0 * 256 / a.dim, n_points=300).to(device)
    angles = torch.tensor([[0.0, 180.0], [90.0, 180.0]])
    sid, sad = torch.tensor([1050.0, 1050.0]), torch.tensor([1000.0, 1000.0])
    t0 = time.time()
    views, corr, fps, w = r.render_sample(cube.to(device), angles, sid, sad, translation=torch.tensor([0.0, 0, 0]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    pre = preprocess_views(views)
    print(f"device={device} views {tuple(views.shape)} corr {tuple(corr.shape)} time {dt:.2f}s")
    print(f"views range [{views.min():.3f}, {views.max():.3f}]  corr max {corr.max():.3f} "
          f"frac>0.3 {(corr > 0.3).float().mean():.4f}  nonzero rows {(corr[0].amax(1) > 0.3).sum().item()}/256")
    print(f"weights w: min {w.min():.3f} max {w.max():.3f} mean {w.mean():.3f}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
    axs[0].imshow(views[0, 0].cpu(), cmap="gray"); axs[0].set_title("view 1 (raw)")
    axs[1].imshow(views[1, 0].cpu(), cmap="gray"); axs[1].set_title("view 2 (raw)")
    axs[2].imshow(pre[0, 0].cpu(), cmap="gray"); axs[2].set_title("view 1 (preprocessed)")
    axs[3].imshow(pre[1, 0].cpu(), cmap="gray"); axs[3].set_title("view 2 (preprocessed)")
    axs[4].imshow(corr[0].cpu(), cmap="magma", vmin=0, vmax=1); axs[4].set_title("correspondence 256x256")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plt.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
