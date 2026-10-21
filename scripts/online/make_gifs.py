"""Project-page visuals rendered with the on-the-fly pipeline (DiffDRR views + analytic correspondence).

Outputs (in --out_dir):
  orbit_<vol>.{mp4,gif}        view 1 fixed (AP) with a query patch, view 2 orbiting 360 deg with the GT
                               many-to-many correspondence of the query overlaid, and the full matrix.
  build_<vol>.{mp4,gif}        construction of the correspondence matrix as max over depth slabs of voxels.
  prediction_<vol>.{mp4,gif}   (needs --ckpt) GT vs. predicted correspondence of query patches on unseen CTs.
  prediction_grid.png          static version of the above (several queries per volume).
  teaser_pairs.png             DRR pairs of several anatomies with sampled GT matches drawn as lines.
  pairs_with_matrices.png      pairs next to their 256x256 correspondence matrices.

    python scripts/online/make_gifs.py --cache_dir <cache> --out_dir <dir> --volumes knee ankle hip shoulder \
        [--ckpt <model.ckpt>] [--pose_mapping <validation report.json>] [--frames 72] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.online_drr import OnlineDRRRenderer  # noqa: E402
from src.utils.utils import postprocess_view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_preprocess import postprocess_view as postprocess_view_legacy  # noqa: E402

PREPROCESS = {"public": postprocess_view, "legacy": postprocess_view_legacy}

CORAL = (255, 107, 74)
TEAL = (46, 196, 182)
GOLD = (255, 196, 61)
INK = (17, 24, 39)
PAPER = (250, 250, 252)
N = 16


# ----------------------------------------------------------------------------------------------
class _Bank:
    def __init__(self, cube):
        self.cube = cube

    def get(self, i, device):
        return self.cube.to(device)


def font(size):
    try:
        import matplotlib.font_manager as fm

        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def to_display(view_raw: torch.Tensor, kind: str = "public") -> np.ndarray:
    """Raw DRR (H, W) -> [0, 1] float.  'public' = percentile stretch + CLAHE (src/utils/utils.py),
    'legacy' = histogram equalisation used to train the released checkpoints (legacy_preprocess.py)."""
    out = PREPROCESS[kind](view_raw.detach().cpu().float()).squeeze().numpy().astype(np.float32)
    return np.clip(out, 0, 1)


def gray_rgb(img01: np.ndarray) -> np.ndarray:
    g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def heat_overlay(rgb: np.ndarray, heat: np.ndarray, color, alpha_max=0.85, gamma=0.7) -> np.ndarray:
    """heat: (N, N) in [0, 1] (patch grid) or (H, W). Blocky upsampling keeps the patch structure visible."""
    H, W = rgb.shape[:2]
    if heat.shape != (H, W):
        heat = np.kron(heat, np.ones((H // heat.shape[0], W // heat.shape[1])))
    a = np.clip(heat, 0, 1) ** gamma * alpha_max
    out = rgb.astype(np.float32) * (1 - a[..., None]) + np.array(color, np.float32) * a[..., None]
    return out.clip(0, 255).astype(np.uint8)


def save_thumbnail(rgb: np.ndarray, path, size: int = 256, gamma: float = 0.8):
    """Square thumbnail cropped to the anatomy (content bounding box) and slightly brightened."""
    g = rgb.mean(-1)
    ys, xs = np.where(g > 0.15 * g.max())
    if len(ys) == 0:
        Image.fromarray(rgb).save(path)
        return
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.12 + 4
    H, W = g.shape
    half = min(half, H / 2, W / 2)
    box = (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, W)), int(min(cy + half, H)))
    im = Image.fromarray(rgb).crop(box).resize((size, size), Image.LANCZOS)
    arr = (np.asarray(im).astype(np.float32) / 255) ** gamma
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def draw_grid(rgb: np.ndarray, ps: int, alpha: float = 0.18) -> np.ndarray:
    """Faint patch grid (every ps pixels) so that the patch extent is visible."""
    out = rgb.astype(np.float32)
    out[::ps, :, :] = out[::ps, :, :] * (1 - alpha) + 255 * alpha
    out[:, ::ps, :] = out[:, ::ps, :] * (1 - alpha) + 255 * alpha
    return out.clip(0, 255).astype(np.uint8)


def draw_patch(rgb: np.ndarray, r: int, c: int, color, ps: int, width=3) -> np.ndarray:
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    d.rectangle([c * ps, r * ps, (c + 1) * ps - 1, (r + 1) * ps - 1], outline=color, width=width)
    return np.array(im)


def matrix_rgb(C: np.ndarray, size: int, row=None, cmap="magma", gamma=0.6) -> np.ndarray:
    """Matrix as an image; a display gamma < 1 lifts the soft (tissue) entries without changing the data."""
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    rgb = (cm(np.clip(C, 0, 1) ** gamma)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
    if row is not None:
        d = ImageDraw.Draw(im)
        y = int((row + 0.5) * size / C.shape[0])
        d.line([0, y, size, y], fill=CORAL, width=2)
    return np.array(im)


def panel(rgb: np.ndarray, size: int, caption: str = "", sub: str = "", bg=PAPER, fg=INK):
    im = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    cap_h = 46 if caption else 0
    out = Image.new("RGB", (size, size + cap_h), bg)
    out.paste(im, (0, cap_h))
    if caption:
        d = ImageDraw.Draw(out)
        d.text((6, 4), caption, fill=fg, font=font(17))
        if sub:
            d.text((6, 25), sub, fill=(110, 110, 125), font=font(13))
    return out


def compose(panels, gap=14, bg=PAPER, margin=14):
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1) + 2 * margin
    out = Image.new("RGB", (w, h + 2 * margin), bg)
    x = margin
    for p in panels:
        out.paste(p, (x, margin))
        x += p.width + gap
    return out


def save_anim(frames, stem: str, fps: int, gif_width: int = 760, gif_colors: int = 128, hold_last=0):
    import imageio

    frames = [np.array(f) for f in frames] + [np.array(frames[-1])] * hold_last
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2  # even dims for yuv420p
    try:
        with imageio.get_writer(stem + ".mp4", fps=fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as wr:
            for f in frames:
                wr.append_data(f[:h2, :w2])
        print("  ->", stem + ".mp4")
    except Exception as e:  # noqa: BLE001
        print("  mp4 skipped:", e)
    scale = gif_width / w
    small = [Image.fromarray(f).resize((int(w * scale), int(h * scale)), Image.LANCZOS) for f in frames]
    small = [im.quantize(colors=gif_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for im in small]
    small[0].save(stem + ".gif", save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print("  ->", stem + ".gif", f"({os.path.getsize(stem + '.gif') / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------------------------
class Scene:
    """Renderer with the RTK mapping (images come out in the orientation of the offline dataset)."""

    def __init__(self, cache_dir, device, mapping=None, n_points=300):
        self.cache_dir, self.device = Path(cache_dir), device
        self.cubes = {}
        self.r = OnlineDRRRenderer(_Bank(None), dim=256, n_patches=16, delx=2.0, n_points=n_points,
                                   render_chunk=2, mapping=mapping).to(device)

    def cube(self, name):
        if name not in self.cubes:
            arr = np.load(self.cache_dir / f"{name}_256.npy").astype(np.float32)
            self.cubes[name] = torch.from_numpy(arr).permute(2, 1, 0).contiguous().half().to(self.device)
        return self.cubes[name]

    def pair(self, name, cams, sid=(1050, 1050), sad=(1000, 1000), tx=0.0):
        """cams: list of (theta, phi[, roll]) in degrees. Returns raw views (V,1,H,W), corr, footprints, w."""
        theta = torch.tensor([c[0] for c in cams], dtype=torch.float32)
        phi = torch.tensor([c[1] for c in cams], dtype=torch.float32)
        roll = torch.tensor([c[2] if len(c) > 2 else 0.0 for c in cams], dtype=torch.float32)
        return self.r.render_tpr(self.cube(name), theta, phi, roll, torch.tensor(sid, dtype=torch.float32),
                                 torch.tensor(sad, dtype=torch.float32), translation_x=tx, apply_flips=True)


def pick_queries(view01: np.ndarray, C: np.ndarray, k: int, min_mass=0.5, margin=1, min_peak=0.5):
    """Query patches on bright (bony) structures whose GT row carries mass, spread over the image.
    ``view01`` should be the *raw* line-integral view (CLAHE turns air into grey noise)."""
    ps = view01.shape[0] // N
    v = (view01 - view01.min()) / (view01.max() - view01.min() + 1e-8)
    bright = v.reshape(N, ps, N, ps).mean((1, 3)).reshape(-1)
    mass = C.sum(1)
    cand = np.where((mass > min_mass) & (C.max(1) > min_peak) & (bright > 0.25 * bright.max()))[0]
    cand = cand[(cand // N >= margin) & (cand // N < N - margin) & (cand % N >= margin) & (cand % N < N - margin)]
    if len(cand) == 0:
        return []
    order = cand[np.argsort(-bright[cand])]
    chosen = []
    for q in order:
        r, c = divmod(int(q), N)
        if all(abs(r - rr) + abs(c - cc) >= 3 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) >= k:
            break
    chosen.sort(key=lambda rc: (rc[0], rc[1]))
    return chosen


# ----------------------------------------------------------------------------------------------
def make_orbit(scene: Scene, name: str, out_dir: Path, frames: int, size: int, fps: int):
    print(f"[orbit] {name}")
    ps = 256 // N
    v_ref, C_ref, _, _ = scene.pair(name, [(0, 0), (90, 0)])
    view1 = to_display(v_ref[0, 0])
    queries = pick_queries(v_ref[0, 0].cpu().numpy(), C_ref[0].cpu().numpy(), 3)
    if not queries:
        queries = [(8, 8)]
    rgb1 = gray_rgb(view1)
    out_frames = []
    for i in range(frames):
        theta = 360.0 * i / frames
        q = queries[min(len(queries) - 1, int(len(queries) * i / frames))]
        qi = q[0] * N + q[1]
        views, corr, _, _ = scene.pair(name, [(0, 0), (theta, 0)])
        C = corr[0].cpu().numpy()
        v2 = gray_rgb(to_display(views[1, 0]))
        p1 = draw_patch(rgb1.copy(), q[0], q[1], CORAL, ps)
        p2 = heat_overlay(v2, C[qi].reshape(N, N), CORAL)
        pm = matrix_rgb(C, size, row=qi)
        fr = compose([
            panel(p1, size, "View 1  (fixed, AP)", "query patch in red"),
            panel(p2, size, f"View 2  (θ = {theta:5.1f}°)", "ground-truth matches of the query"),
            panel(pm, size, "Correspondence matrix C", "256 × 256 patches · row of the query highlighted"),
        ])
        out_frames.append(fr)
    save_anim(out_frames, str(out_dir / f"orbit_{name}"), fps)


def make_build(scene: Scene, name: str, out_dir: Path, size: int, fps: int, grid: bool = True):
    print(f"[build] {name}")
    ps = 256 // N
    views, corr, fps_, w = scene.pair(name, [(0, 0), (90, 0)])
    C_full = corr[0].cpu().numpy()
    v1, v2 = gray_rgb(to_display(views[0, 0])), gray_rgb(to_display(views[1, 0]))
    if grid:
        v1, v2 = draw_grid(v1, ps), draw_grid(v2, ps)
    Nv = fps_.shape[1]
    idx = torch.arange(Nv, device=fps_.device)
    slab = (idx // N) % N  # coarse Y index = depth along the AP viewing direction
    g1 = float((fps_[0] * w[:, None]).amax().item()) + 1e-6  # global normalisation: air slabs stay dark
    g2 = float((fps_[1] * w[:, None]).amax().item()) + 1e-6
    frames = []
    for s in range(N):
        sel = slab <= s
        cur = slab == s
        Cs = scene.r.correspondence(fps_[0][sel], fps_[1][sel], w[sel]).cpu().numpy()
        f1 = (fps_[0][cur] * w[cur, None]).amax(0).cpu().numpy().reshape(N, N) / g1
        f2 = (fps_[1][cur] * w[cur, None]).amax(0).cpu().numpy().reshape(N, N) / g2
        fr = compose([
            panel(heat_overlay(v1, f1, TEAL, 0.6, gamma=0.5), size, "View 1 (AP)", f"voxel slab {s + 1}/{N} along the viewing depth"),
            panel(heat_overlay(v2, f2, TEAL, 0.6, gamma=0.5), size, "View 2 (lateral)", "the same slab seen from the side"),
            panel(matrix_rgb(Cs, size), size, "C = max over voxels so far", f"{int(sel.sum())}/{Nv} voxels accumulated"),
        ])
        frames += [fr] * 2
    frames += [compose([panel(v1, size, "View 1 (AP)"), panel(v2, size, "View 2 (lateral)"),
                        panel(matrix_rgb(C_full, size), size, "Final correspondence matrix")])] * 8
    save_anim(frames, str(out_dir / f"build_{name}"), fps)


def load_model(ckpt: str, device):
    """Load a repo checkpoint on any node: no weight download, hparams filtered to the public signature."""
    import inspect

    import torchvision.models as tvm

    from src.model.model import CorrespondenceModel

    orig = tvm.resnet50

    def _resnet50(*a, **k):
        k["weights"] = None
        return orig(*a, **k)

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = dict(state.get("hyper_parameters", {}))
    sig = inspect.signature(CorrespondenceModel.__init__).parameters
    kwargs = {k: v for k, v in hp.items() if k in sig}
    dropped = sorted(set(hp) - set(kwargs))
    tvm.resnet50 = _resnet50
    try:
        model = CorrespondenceModel(**kwargs)
    finally:
        tvm.resnet50 = orig
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    print(f"[ckpt] {os.path.basename(ckpt)}: epoch {state.get('epoch')}, dropped hparams {dropped}, "
          f"missing {len(missing)} / unexpected {len(unexpected)} keys")
    if missing:
        print("   missing:", missing[:8])
    if unexpected:
        print("   unexpected:", unexpected[:8])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, view1_01: np.ndarray, view2_01: np.ndarray, device) -> np.ndarray:
    v1 = torch.from_numpy(view1_01)[None, None].float().to(device)
    v2 = torch.from_numpy(view2_01)[None, None].float().to(device)
    out = model(v1.repeat(1, 3, 1, 1), v2.repeat(1, 3, 1, 1))["correspondence_matrix"]
    return out[0].float().cpu().numpy()


def _ap(scores, labels):
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    return float((tp / (np.arange(len(lab)) + 1) * lab).sum() / max(lab.sum(), 1))


def _pck1(P, C, thr=0.3):
    """Fraction of query patches (rows with a GT match) whose predicted argmax lies within 1 patch of a GT match."""
    rows = np.where(C.max(1) > thr)[0]
    if len(rows) == 0:
        return float("nan")
    ok = 0
    for r in rows:
        j = int(P[r].argmax())
        tgt = np.where(C[r] > thr)[0]
        dr, dc = np.abs(tgt // N - j // N), np.abs(tgt % N - j % N)
        ok += int((np.maximum(dr, dc) <= 1).any())
    return ok / len(rows)


def make_prediction(scene: Scene, model, names, out_dir: Path, size: int, n_queries: int, device, preprocess="both"):
    print("[prediction]", names)
    ps = 256 // N
    cams = [(0, 0), (80, 0)]
    kinds = ["public", "legacy"] if preprocess == "both" else [preprocess]
    anim, grid_rows, stats = [], [], {}
    for name in names:
        views, corr, _, _ = scene.pair(name, cams, sid=(1050, 1080), sad=(1000, 950))
        C = corr[0].cpu().numpy()
        preds, st = {}, {}
        for kind in kinds:
            v1, v2 = to_display(views[0, 0], kind), to_display(views[1, 0], kind)
            P = predict(model, v1, v2, device)
            preds[kind] = (v1, v2, P)
            st[kind] = {"ap_0.2": _ap(P.reshape(-1), (C > 0.2).reshape(-1).astype(np.float32)),
                        "ap_0.3": _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32)),
                        "pearson": float(np.corrcoef(P.reshape(-1), C.reshape(-1))[0, 1]), "pck1_0.3": _pck1(P, C)}
            print(f"  {name:9s} {kind:7s}: AP@0.2 {st[kind]['ap_0.2']:.3f}  AP@0.3 {st[kind]['ap_0.3']:.3f}  "
                  f"r {st[kind]['pearson']:.3f}  PCK@1 {st[kind]['pck1_0.3']:.3f}")
        stats[name] = st
        best_kind = max(kinds, key=lambda k: st[k]["ap_0.3"])
        v1, v2, P = preds[best_kind]
        rgb1, rgb2 = gray_rgb(v1), gray_rgb(v2)
        qs = pick_queries(views[0, 0].cpu().numpy(), C, n_queries)
        for (r, c) in qs:
            qi = r * N + c
            fr = compose([
                panel(draw_patch(rgb1.copy(), r, c, CORAL, ps), size, f"View 1 · {name}", "query patch"),
                panel(heat_overlay(rgb2, C[qi].reshape(N, N), CORAL), size, "View 2 · ground truth", "analytic correspondence"),
                panel(heat_overlay(rgb2, P[qi].reshape(N, N), TEAL), size, "View 2 · prediction", "pre-trained model, unseen CT"),
            ])
            anim += [fr] * 7
        for (r, c) in qs[:3]:
            qi = r * N + c
            grid_rows.append(compose([
                panel(draw_patch(rgb1.copy(), r, c, CORAL, ps), size),
                panel(heat_overlay(rgb2, C[qi].reshape(N, N), CORAL), size),
                panel(heat_overlay(rgb2, P[qi].reshape(N, N), TEAL), size),
            ], margin=4))
    if anim:
        save_anim(anim, str(out_dir / ("prediction_" + "_".join(names))), fps=10)
    if grid_rows:
        W = max(r.width for r in grid_rows)
        H = sum(r.height for r in grid_rows) + 50
        g = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(g)
        for k, cap in enumerate(["View 1 · query", "View 2 · ground truth", "View 2 · prediction"]):
            d.text((14 + k * (size + 14), 14), cap, fill=INK, font=font(18))
        y = 50
        for r in grid_rows:
            g.paste(r, (0, y))
            y += r.height
        g.save(out_dir / "prediction_grid.png")
        print("  ->", out_dir / "prediction_grid.png")
    json.dump(stats, open(out_dir / "prediction_stats.json", "w"), indent=1)


def make_teaser(scene: Scene, names, out_dir: Path, size: int, n_lines=10, seed=0):
    print("[teaser]", names)
    rng = np.random.default_rng(seed)
    ps = 256 // N
    tiles, tiles_m = [], []
    palette = [CORAL, TEAL, GOLD, (168, 85, 247), (59, 130, 246), (16, 185, 129), (244, 63, 94), (245, 158, 11)]
    for name in names:
        th = float(rng.uniform(60, 120))
        views, corr, _, _ = scene.pair(name, [(0, 0), (th, float(rng.uniform(-10, 10)))])
        C = corr[0].cpu().numpy()
        v1, v2 = gray_rgb(to_display(views[0, 0])), gray_rgb(to_display(views[1, 0]))
        qs = pick_queries(views[0, 0].cpu().numpy(), C, n_lines)
        canvas = Image.new("RGB", (2 * size + 24, size), PAPER)
        im1 = Image.fromarray(v1).resize((size, size), Image.LANCZOS)
        im2 = Image.fromarray(v2).resize((size, size), Image.LANCZOS)
        canvas.paste(im1, (0, 0)); canvas.paste(im2, (size + 24, 0))
        d = ImageDraw.Draw(canvas)
        sc = size / 256
        for k, (r, c) in enumerate(qs):
            col = palette[k % len(palette)]
            qi = r * N + c
            row = C[qi]
            tgt = np.where(row > 0.3)[0]
            if len(tgt) == 0:
                continue
            x1, y1 = (c + 0.5) * ps * sc, (r + 0.5) * ps * sc
            for t in tgt[np.argsort(-row[tgt])][:4]:
                tr, tc = divmod(int(t), N)
                x2, y2 = size + 24 + (tc + 0.5) * ps * sc, (tr + 0.5) * ps * sc
                d.line([x1, y1, x2, y2], fill=col, width=2)
                d.rectangle([x2 - 4, y2 - 4, x2 + 4, y2 + 4], fill=col)
            d.rectangle([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=col, outline=INK)
        tiles.append(canvas)
        if name == names[0]:  # thumbnails used inside the project page's architecture figure
            save_thumbnail(v1, out_dir / f"thumb_{name}_v1.png")
            save_thumbnail(v2, out_dir / f"thumb_{name}_v2.png")
            Image.fromarray(matrix_rgb(C, 256)).save(out_dir / f"thumb_{name}_corr.png")
        # pair + matrix tile
        m = Image.fromarray(matrix_rgb(C, size))
        t2 = Image.new("RGB", (3 * size + 48, size), PAPER)
        t2.paste(im1, (0, 0)); t2.paste(im2, (size + 24, 0)); t2.paste(m, (2 * size + 48, 0))
        tiles_m.append(t2)
    cols = 2
    rows = math.ceil(len(tiles) / cols)
    W = cols * tiles[0].width + (cols + 1) * 20
    H = rows * (tiles[0].height + 20) + 20
    g = Image.new("RGB", (W, H), PAPER)
    for i, t in enumerate(tiles):
        g.paste(t, (20 + (i % cols) * (t.width + 20), 20 + (i // cols) * (t.height + 20)))
    g.save(out_dir / "teaser_pairs.png")
    print("  ->", out_dir / "teaser_pairs.png")
    H2 = len(tiles_m) * (size + 20) + 20
    g2 = Image.new("RGB", (tiles_m[0].width + 40, H2), PAPER)
    for i, t in enumerate(tiles_m):
        g2.paste(t, (20, 20 + i * (size + 20)))
    g2.save(out_dir / "pairs_with_matrices.png")
    print("  ->", out_dir / "pairs_with_matrices.png")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--volumes", nargs="+", default=["knee", "ankle", "hip", "shoulder"])
    ap.add_argument("--orbit_volumes", nargs="*", default=None)
    ap.add_argument("--build_volume", default=None, help="volume for the construction animation (default: first of --volumes)")
    ap.add_argument("--no_grid", action="store_true", help="no patch grid overlay in the construction animation")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pose_mapping", default=None, help="validation report.json (flip_rows/flip_cols)")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--n_queries", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", nargs="*", default=None, choices=["orbit", "build", "prediction", "teaser"])
    ap.add_argument("--preprocess", default="both", choices=["public", "legacy", "both"], help="view preprocessing fed to the checkpoint")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    mapping = json.load(open(a.pose_mapping))["pose_mapping"] if a.pose_mapping else None
    scene = Scene(a.cache_dir, device, mapping)
    want = set(a.only or ["orbit", "build", "prediction", "teaser"])
    if "teaser" in want:
        make_teaser(scene, a.volumes, out, a.size)
    if "build" in want:
        make_build(scene, a.build_volume or a.volumes[0], out, a.size, a.fps, grid=not a.no_grid)
    if "orbit" in want:
        for name in (a.orbit_volumes if a.orbit_volumes is not None else a.volumes[:2]):
            make_orbit(scene, name, out, a.frames, a.size, a.fps)
    if "prediction" in want and a.ckpt:
        model = load_model(a.ckpt, device)
        make_prediction(scene, model, a.volumes, out, a.size, a.n_queries, device, a.preprocess)


if __name__ == "__main__":
    main()
