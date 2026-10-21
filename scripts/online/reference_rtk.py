"""Stage A of the parity check: render *reference* DRRs + correspondence GT with the ORIGINAL
offline pipeline (ITK-RTK Joseph projector + per-voxel projections) for a cached HU cube.

Run in the original environment (itk-rtk).  Produces <out>.npz with, per setting:
views (V, 256, 256), corr (V(V-1)/2, 256, 256) and the acquisition parameters.

    python scripts/online/reference_rtk.py --cube <cache>/knee_256.npy --out <dir>/ref_knee.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import apply_translation  # noqa: E402
from src.utils.utils_drr_fast import apply_transform  # noqa: E402

# A small, deliberately diverse set of acquisitions (CSV convention: [a0, a1], a1 ~ 180).
DEFAULT_SETTINGS = [
    {"angles": [[0.0, 180.0], [90.0, 180.0]], "sid": [1050, 1050], "sad": [1000, 1000], "tx": 0},
    {"angles": [[0.0, 180.0], [0.0, 180.0]], "sid": [1050, 1050], "sad": [1000, 1000], "tx": 0},
    {"angles": [[30.0, 170.0], [120.0, 195.0]], "sid": [1000, 1100], "sad": [800, 1200], "tx": 0},
    {"angles": [[0.0, 180.0], [90.0, 180.0]], "sid": [1050, 1050], "sad": [1000, 1000], "tx": 40},
    {"angles": [[200.0, 160.0], [275.0, 210.0]], "sid": [1080, 1020], "sad": [900, 1100], "tx": -25},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True, help="HU cube .npy (z, y, x) from prepare_volumes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default=None, help="JSON list of settings (default: 5 built-in)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--scale_factor", type=int, default=16)
    a = ap.parse_args()
    settings = json.load(open(a.settings)) if a.settings else DEFAULT_SETTINGS
    cube = np.load(a.cube).astype(np.float64)  # (z, y, x) HU, as produced by preprocess_volume
    axis = [1, 1, -1]
    out_views, out_corr, meta = [], [], []
    for i, s in enumerate(settings):
        t0 = time.time()
        vol = apply_translation(cube, (0, 0, s["tx"]), fill_value=-1000)
        vol = np.clip(vol, -1000, 3000) + 1000
        corr, views = apply_transform(vol, s["angles"], axis, a.dim, s["sid"], s["sad"], a.scale_factor, "average")
        views = torch.stack([v.squeeze() for v in views]).numpy()
        out_views.append(views.astype(np.float32))
        out_corr.append(corr.numpy().astype(np.float32))
        meta.append(dict(s, seconds=round(time.time() - t0, 1)))
        print(f"setting {i}: views {views.shape} corr {tuple(corr.shape)}  {time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(a.out, views=np.stack(out_views), corr=np.stack(out_corr), meta=json.dumps(meta), cube=a.cube)
    print("saved", a.out)


if __name__ == "__main__":
    main()
