"""Contact sheet of all cached CT cubes: AP + lateral DRR per volume (to choose volumes for figures).

    python scripts/online/contact_sheet.py --cache_dir <cache> --out <png> [--device cpu]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gifs import Scene, font, gray_rgb, to_display  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    names = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(a.cache_dir, "*_256.npy")))
    scene = Scene(a.cache_dir, torch.device(a.device))
    tiles = []
    for name in names:
        views, corr, _, _ = scene.pair(name, [(0, 0), (90, 0)])
        v1, v2 = gray_rgb(to_display(views[0, 0])), gray_rgb(to_display(views[1, 0]))
        t = Image.new("RGB", (2 * a.size + 6, a.size + 22), (250, 250, 252))
        t.paste(Image.fromarray(v1).resize((a.size, a.size)), (0, 22))
        t.paste(Image.fromarray(v2).resize((a.size, a.size)), (a.size + 6, 22))
        ImageDraw.Draw(t).text((4, 3), f"{name}  (frac>0.3: {(corr[0] > 0.3).float().mean().item() * 100:.2f}%)", fill=(17, 24, 39), font=font(13))
        tiles.append(t)
        print("ok", name, flush=True)
        scene.cubes.clear()
    rows = math.ceil(len(tiles) / a.cols)
    W, H = tiles[0].size
    sheet = Image.new("RGB", (a.cols * (W + 10) + 10, rows * (H + 10) + 10), (250, 250, 252))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % a.cols) * (W + 10), 10 + (i // a.cols) * (H + 10)))
    sheet.save(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
