"""Throughput of the on-the-fly pipeline vs. reading the offline .h5 dataset.

    python scripts/online/benchmark.py --cache_dir <cache> [--h5 data.h5] [--batch 16] [--iters 20]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.online_drr import OnlineDRRParams, OnlineDRRRenderer, VolumeBank, preprocess_views  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--h5", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n_points", type=int, default=300)
    ap.add_argument("--render_chunk", type=int, default=4)
    ap.add_argument("--max_volumes", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubes = sorted(glob.glob(os.path.join(a.cache_dir, "*_256.npy")))[: a.max_volumes]
    bank = VolumeBank(cubes, cache_dir=a.cache_dir, dim=256, verbose=False)
    renderer = OnlineDRRRenderer(bank, n_points=a.n_points, render_chunk=a.render_chunk).to(device)
    ds = OnlineDRRParams(len(bank), a.batch * (a.iters + 2), pairs_per_volume=4, seed=1)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    times = []
    for i, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        views, corr = renderer(batch)
        views = preprocess_views(views, jitter_p=0.7)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t0)
    t = float(np.mean(times))
    print(f"[online] device={device} batch={a.batch}: {t:.3f} s/batch -> {a.batch / t:.1f} pairs/s "
          f"(n_points={a.n_points}, chunk={a.render_chunk}); peak mem "
          f"{torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.2f} GiB")
    if a.h5:
        import torchvision.transforms as transforms

        from src.data.dataloader import DRRDataset
        from src.utils.utils import NamedLambda, postprocess_view

        tv = transforms.Compose([NamedLambda(lambda img: postprocess_view(img), name="pp"), transforms.Resize((256, 256))])
        hds = DRRDataset(a.h5, transform_view=tv, data_type="train")
        for nw in (0, 8):
            hdl = DataLoader(hds, batch_size=a.batch, shuffle=True, num_workers=nw)
            times = []
            t0 = time.time()
            for i, (c, v) in enumerate(hdl):
                if i >= a.iters:
                    break
                times.append(time.time() - t0)
                t0 = time.time()
            t = float(np.mean(times[2:]))
            print(f"[offline h5] workers={nw}: {t:.3f} s/batch -> {a.batch / t:.1f} pairs/s")


if __name__ == "__main__":
    main()
