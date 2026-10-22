"""Rank cached CT cubes for page visuals: checkpoint AP on a few azimuth / roll pairs + AP & lateral previews.

    python scripts/online/candidate_sheet.py --cache_dir <cache> --ckpt <ckpt> --out <dir> [--n 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gifs import Scene, _ap, _pck1, _sample_variation, font, gray_rgb, load_model, predict, to_display  # noqa: E402

N = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--size", type=int, default=180)
    ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(a.cache_dir, "*_256.npy")))
    seen, names = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        names.append(os.path.basename(p)[:-8])
    if a.names:
        names = [n for n in names if n in set(a.names)]
    model = load_model(a.ckpt, device)
    scene = Scene(a.cache_dir, device)
    rng = np.random.default_rng(0)
    rows, tiles = [], []
    for name in names:
        res = {"name": name}
        for kind in ("azimuth", "roll2"):
            aps, pcks = [], []
            for k in range(a.n):
                v = _sample_variation(kind, rng)
                views, corr, _, _ = scene.pair(name, v["cams"], sid=v["sid"], sad=v["sad"], tx=v["tx"])
                C = corr[0].cpu().numpy()
                v1, v2 = to_display(views[0, 0], "legacy"), to_display(views[1, 0], "legacy")
                P = predict(model, v1, v2, device)
                aps.append(_ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32)))
                pcks.append(_pck1(P, C))
            res[f"ap_{kind}"] = float(np.mean(aps))
            res[f"pck_{kind}"] = float(np.nanmean(pcks))
        views, corr, _, _ = scene.pair(name, [(0, 0, 0), (90, 0, 0)])
        res["frac_pos"] = float((corr[0] > 0.3).float().mean().item())
        v1, v2 = gray_rgb(to_display(views[0, 0], "legacy")), gray_rgb(to_display(views[1, 0], "legacy"))
        t = Image.new("RGB", (2 * a.size + 6, a.size + 36), (250, 250, 252))
        t.paste(Image.fromarray(v1).resize((a.size, a.size)), (0, 36))
        t.paste(Image.fromarray(v2).resize((a.size, a.size)), (a.size + 6, 36))
        d = ImageDraw.Draw(t)
        d.text((4, 2), name[:34], fill=(17, 24, 39), font=font(13))
        d.text((4, 18), f"AP az {res['ap_azimuth']:.2f} · roll {res['ap_roll2']:.2f} · PCK {res['pck_azimuth']:.2f}", fill=(14, 124, 138), font=font(12))
        tiles.append(t)
        rows.append(res)
        print(f"{name:48s} AP az {res['ap_azimuth']:.2f}  roll {res['ap_roll2']:.2f}  PCK {res['pck_azimuth']:.2f}", flush=True)
        scene.cubes.clear()
    json.dump(rows, open(os.path.join(a.out, "candidates.json"), "w"), indent=1)
    cols = 4
    W, H = tiles[0].size
    import math

    r = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (W + 10) + 10, r * (H + 10) + 10), (250, 250, 252))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % cols) * (W + 10), 10 + (i // cols) * (H + 10)))
    sheet.save(os.path.join(a.out, "candidates.png"))
    print("->", os.path.join(a.out, "candidates.png"))


if __name__ == "__main__":
    main()
