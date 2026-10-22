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


OVERLAY_THR = 0.3  # display cut-off of the correspondence overlays: weights below it are not drawn (CLI --overlay_thr);
#                    0.3 is the positive threshold of the AP / PCK metrics, so the overlays show exactly the counted matches
OVERLAY_RAMP = 0.10  # weights fade in linearly over [thr, thr + ramp] instead of a hard step


def heat_overlay(rgb: np.ndarray, heat: np.ndarray, color, alpha_max=0.85, gamma=0.7, thr=None) -> np.ndarray:
    """heat: (N, N) in [0, 1] (patch grid) or (H, W). Blocky upsampling keeps the patch structure visible.
    Entries below ``thr`` (default: the module-level OVERLAY_THR) are not drawn; the ones above keep their alpha."""
    H, W = rgb.shape[:2]
    if heat.shape != (H, W):
        heat = np.kron(heat, np.ones((H // heat.shape[0], W // heat.shape[1])))
    heat = np.clip(heat, 0, 1)
    thr = OVERLAY_THR if thr is None else thr
    if thr > 0:
        heat = heat * np.clip((heat - thr) / max(OVERLAY_RAMP, 1e-6), 0, 1)
    a = heat ** gamma * alpha_max
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


def content_box(raw: np.ndarray, margin: float = 0.10, min_size: int = 112, thr: float = 0.12):
    """Square box (x0, y0, x1, y1) around the anatomy of a raw DRR, for *display* zoom only."""
    H, W = raw.shape
    m = raw > thr * raw.max()
    if not m.any():
        return (0, 0, W, H)
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin)
    side = int(min(max(side, min_size), min(H, W)))
    x0 = int(min(max(cx - side / 2, 0), W - side))
    y0 = int(min(max(cy - side / 2, 0), H - side))
    return (x0, y0, x0 + side, y0 + side)


def union_box(boxes, H=256, W=256):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    side = max(x1 - x0, y1 - y0)
    side = min(side, min(H, W))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(min(max(cx - side / 2, 0), W - side)); y0 = int(min(max(cy - side / 2, 0), H - side))
    return (x0, y0, x0 + side, y0 + side)


def crop_rgb(rgb: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


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


def panel(rgb: np.ndarray, size: int, caption: str = "", sub: str = "", bg=PAPER, fg=INK, sub2: str = ""):
    """Image panel with up to three caption lines (title, subtitle, optional third line)."""
    im = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    cap_h = (62 if sub2 else 46) if caption else 0
    out = Image.new("RGB", (size, size + cap_h), bg)
    out.paste(im, (0, cap_h))
    if caption:
        d = ImageDraw.Draw(out)
        d.text((6, 4), caption, fill=fg, font=font(17))
        if sub:
            d.text((6, 25), sub, fill=(110, 110, 125), font=font(13))
        if sub2:
            d.text((6, 41), sub2, fill=(110, 110, 125), font=font(13))
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


def save_anim(frames, stem: str, fps: int, gif_width: int = 760, gif_colors: int = 128, hold_last=0, durations_ms=None):
    """mp4 (h264) + gif. ``durations_ms``: optional per-frame display time; the gif uses it natively,
    the mp4 repeats frames accordingly."""
    import imageio

    frames = [np.array(f) for f in frames] + [np.array(frames[-1])] * hold_last
    if durations_ms is not None:
        durations_ms = list(durations_ms) + [durations_ms[-1]] * hold_last
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2  # even dims for yuv420p
    try:
        with imageio.get_writer(stem + ".mp4", fps=fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as wr:
            for i, f in enumerate(frames):
                reps = 1 if durations_ms is None else max(1, int(round(durations_ms[i] / 1000 * fps)))
                for _ in range(reps):
                    wr.append_data(f[:h2, :w2])
        print("  ->", stem + ".mp4")
    except Exception as e:  # noqa: BLE001
        print("  mp4 skipped:", e)
    scale = gif_width / w
    small = [Image.fromarray(f).resize((int(w * scale), int(h * scale)), Image.LANCZOS) for f in frames]
    small[0].save(stem + "_poster.png")  # first frame at gif width, for <video poster>
    small = [im.quantize(colors=gif_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for im in small]
    dur = int(1000 / fps) if durations_ms is None else [int(d) for d in durations_ms]
    small[0].save(stem + ".gif", save_all=True, append_images=small[1:], duration=dur, loop=0, optimize=True)
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

    def pair_csv(self, name, item):
        """Render an acquisition sampled with sample_acquisition() (CSV convention, RTK mapping applied)."""
        return self.r.render_sample(self.cube(name), item["angles"], item["sid"], item["sad"], item["translation"], item.get("tilt"))

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
def _sample_variation(kind: str, rng):
    """Second-view pose for one transformation family (first view = AP unless stated)."""
    th = float(rng.uniform(60, 120))
    sgn = lambda: float(rng.choice([-1, 1]))  # noqa: E731
    if kind == "azimuth":
        return dict(cams=[(0, 0, 0), (th, 0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
    if kind == "roll2":
        return dict(cams=[(0, 0, 0), (th, 0, sgn() * float(rng.uniform(15, 35)))], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
    if kind == "roll12":
        return dict(cams=[(0, 0, sgn() * float(rng.uniform(10, 25))), (th, 0, sgn() * float(rng.uniform(10, 25)))],
                    sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
    if kind == "translation":
        return dict(cams=[(0, 0, 0), (th, 0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=sgn() * float(rng.uniform(25, 50)))
    if kind == "magnification":
        a = rng.random() < 0.5
        return dict(cams=[(0, 0, 0), (th, 0, 0)], sid=(1000, 1100) if a else (1100, 1000), sad=(1200, 800) if a else (800, 1200), tx=0.0)
    if kind == "tilt":
        return dict(cams=[(0, 0, 0), (th, sgn() * 15.0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
    raise ValueError(kind)


VARIATIONS = [
    ("azimuth", "Rotation about the long axis only", "second view at 60–120° (as the other examples)"),
    ("roll2", "+ in-plane roll of view 2", "detector rolled 15–35°: matches leave the row"),
    ("roll12", "both views rolled", "independent rolls of 10–25° on each view"),
    ("translation", "+ translation of the volume", "25–50 voxels along x (10–20 % of the field)"),
    ("magnification", "+ different source distances", "sid 1000/1100, sad 800/1200 mm"),
    ("family", "the paper's full pose family", "angles, rolls, shift, distances as in create_csv.py"),
    ("tilt", "out-of-plane tilt of 15° (not in training)", "gantry tilts never occur in the training poses"),
]


def _fmt_pose(v):
    (t1, p1, r1), (t2, p2, r2) = v["cams"]
    bits = [f"θ₂ = {t2:.0f}°"]
    if abs(r2) > 0.5:
        bits.append(f"roll₂ = {r2:+.0f}°")
    if abs(r1) > 0.5:
        bits.append(f"roll₁ = {r1:+.0f}°")
    if abs(p2) > 0.5:
        bits.append(f"tilt₂ = {p2:+.0f}°")
    if abs(v["tx"]) > 0.5:
        bits.append(f"shift = {v['tx']:+.0f} vox")
    if v["sid"][0] != v["sid"][1] or v["sad"][0] != v["sad"][1]:
        bits.append(f"sid {v['sid'][0]:.0f}/{v['sid'][1]:.0f} · sad {v['sad'][0]:.0f}/{v['sad'][1]:.0f}")
    return " · ".join(bits)


def make_variations(scene: Scene, model, names, out_dir: Path, size: int, device, n_stats: int = 24, seed: int = 3,
                    sweep_volume: str = None, preprocess: str = "legacy", fps: int = 12):
    """Pose-variation grid, statistics per transformation family, and a sweep animation (GT vs prediction)."""
    import random as pyrandom

    from src.data.online_drr import sample_acquisition

    print("[variations]", names)
    ps = 256 // N
    rng = np.random.default_rng(seed)

    def render(kind, name, k):
        if kind == "family":
            item = sample_acquisition(pyrandom.Random(f"{seed}:{name}:{k}"), 2, True)
            views, corr, _, _ = scene.pair_csv(name, item)
            a0, a1 = item["angles"].tolist()
            label = f"a = {a0[0]:.0f}°/{a1[0]:.0f}° , {a0[1]:.0f}°/{a1[1]:.0f}° · shift {item['translation'][0]:+.0f}"
            return views, corr, label
        v = _sample_variation(kind, rng)
        views, corr, _, _ = scene.pair(name, v["cams"], sid=v["sid"], sad=v["sad"], tx=v["tx"])
        return views, corr, _fmt_pose(v)

    # ---- statistics over random pairs ----
    stats = {}
    for kind, title, _ in (VARIATIONS if n_stats > 0 else []):
        aps, pcks = [], []
        for k in range(n_stats):
            name = names[k % len(names)]
            views, corr, _ = render(kind, name, k)
            C = corr[0].cpu().numpy()
            v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
            P = predict(model, v1, v2, device)
            aps.append(_ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32)))
            pcks.append(_pck1(P, C))
        stats[kind] = {"title": title, "n": n_stats, "ap_0.3_mean": float(np.mean(aps)), "ap_0.3_std": float(np.std(aps)),
                       "pck1_mean": float(np.nanmean(pcks)), "pck1_std": float(np.nanstd(pcks))}
        print(f"  {kind:14s} AP {np.mean(aps):.3f} ± {np.std(aps):.3f}   PCK@1 {np.nanmean(pcks):.3f}   (n={n_stats})")
    if stats:
        json.dump(stats, open(out_dir / "variation_stats.json", "w"), indent=1)

    # ---- grid: one row per family, same volume, GT vs prediction ----
    rows = []
    name = names[0]
    base_q = None
    for kind, title, sub in VARIATIONS:
        views, corr, label = render(kind, name, 1000)
        C = corr[0].cpu().numpy()
        v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
        P = predict(model, v1, v2, device)
        ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
        qs = pick_queries(views[0, 0].cpu().numpy(), C, 4)
        if kind == "azimuth" and qs:
            base_q = qs[1] if len(qs) > 1 else qs[0]
        q = base_q if (base_q is not None and kind in ("azimuth", "roll2", "magnification", "tilt") and C[base_q[0] * N + base_q[1]].max() > 0.3) else (qs[1] if len(qs) > 1 else (qs[0] if qs else (8, 8)))
        qi = q[0] * N + q[1]
        rgb1, rgb2 = gray_rgb(v1), gray_rgb(v2)
        rows.append(compose([
            panel(draw_patch(rgb1.copy(), q[0], q[1], CORAL, ps), size, title, label),
            panel(heat_overlay(rgb2, C[qi].reshape(N, N), CORAL), size, "View 2 · ground truth", sub),
            panel(heat_overlay(rgb2, P[qi].reshape(N, N), TEAL), size, "View 2 · prediction", f"AP of this pair {ap:.2f}"),
        ], margin=6))
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows)
    g = Image.new("RGB", (W, H), PAPER)
    y = 0
    for r in rows:
        g.paste(r, (0, y))
        y += r.height
    g.save(out_dir / "pose_variations.png")
    print("  ->", out_dir / "pose_variations.png")

    # ---- sweep animation: roll, then translation, then magnification ----
    sweep_list = [sweep_volume] if isinstance(sweep_volume, str) else (list(sweep_volume) if sweep_volume else [names[0]])
    n_seg = 36
    for sv in sweep_list:
        frames, recs = [], []
        views0, corr0, _, _ = scene.pair(sv, [(0, 0, 0), (90, 0, 0)])
        qs = pick_queries(views0[0, 0].cpu().numpy(), corr0[0].cpu().numpy(), 3)
        q = qs[1] if len(qs) > 1 else (qs[0] if qs else (8, 8))
        for seg in range(3):
            for i in range(n_seg):
                u = math.sin(2 * math.pi * i / n_seg)  # -1..1..-1
                qq = q
                if seg == 0:
                    v = dict(cams=[(0, 0, 0), (90, 0, 35 * u)], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
                    head = f"in-plane roll of view 2: {35 * u:+.0f}°"
                elif seg == 1:
                    v = dict(cams=[(0, 0, 0), (75, 0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=50 * u)
                    head = f"volume translation: {50 * u:+.0f} voxels"
                    # the query follows the anatomy: a shift of tx voxels moves the AP image by ~tx px
                    # (magnification sdd/sid = 1.95 over 2 mm pixels), towards -x after the column flip of the mapping
                    dcol = int(round(-v["tx"] * 0.976 / ps)) if scene.r.mapping.get("flip_cols") else int(round(v["tx"] * 0.976 / ps))
                    qq = (q[0], min(N - 1, max(0, q[1] + dcol)))
                else:
                    v = dict(cams=[(0, 0, 0), (100, 0, 0)], sid=(1050, 1050 + 50 * u), sad=(1000, 1000 - 200 * u), tx=0.0)
                    head = f"source distance of view 2: sid {v['sid'][1]:.0f} · sad {v['sad'][1]:.0f} mm"
                qi = qq[0] * N + qq[1]
                views, corr, _, _ = scene.pair(sv, v["cams"], sid=v["sid"], sad=v["sad"], tx=v["tx"])
                C = corr[0].cpu().numpy()
                v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
                P = predict(model, v1, v2, device)
                ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
                recs.append(dict(seg=seg, qq=qq, qi=qi, head=head, ap=ap, v1=v1, v2=v2, C=C, P=P,
                                 b1=content_box(views[0, 0].cpu().numpy()), b2=content_box(views[1, 0].cpu().numpy())))
        # display zoom: one box per segment and per view (the anatomy moves only in the translation segment)
        boxes = {}
        for seg in range(3):
            rs = [r for r in recs if r["seg"] == seg]
            boxes[seg] = (union_box([r["b1"] for r in rs]), union_box([r["b2"] for r in rs]))
        for r in recs:
            B1, B2 = boxes[r["seg"]]
            rgb1, rgb2 = gray_rgb(r["v1"]), gray_rgb(r["v2"])
            frames.append(compose([
                panel(crop_rgb(draw_patch(rgb1.copy(), r["qq"][0], r["qq"][1], CORAL, ps), B1), size, f"View 1 · {STUDY_NAMES.get(sv, sv)}",
                      "fixed AP view" if r["seg"] != 1 else "shifted with the volume (query follows)"),
                panel(crop_rgb(heat_overlay(rgb2, r["C"][r["qi"]].reshape(N, N), CORAL), B2), size, "View 2 · ground truth", r["head"]),
                panel(crop_rgb(heat_overlay(rgb2, r["P"][r["qi"]].reshape(N, N), TEAL), B2), size, "View 2 · prediction", f"AP of this pair {r['ap']:.2f}"),
            ]))
        save_anim(frames, str(out_dir / f"sweep_{sv}"), fps)


STUDY_NAMES = {"hip": "pelvis and femurs (A)", "hip2": "pelvis and femurs", "head_cube": "skull", "head_bbox": "skull", "ankle_bbox": "feet",
               "ankle_foot": "foot", "ankle_left": "foot", "knee_left": "lower leg", "knee_left_c420": "lower leg", "knee_left_c360": "lower leg", "knee_left_c300": "lower leg", "head_z": "head", "head_z2": "head", "leg_one": "lower leg", "knee2": "knee", "head_nt": "head", "head": "head", "trunk": "abdomen and pelvis",
               "foot_soft_nt": "foot (B)", "arm1": "upper limb", "shoulder2": "thorax",
               "shoulder": "thorax", "shoulder2": "thorax (B)", "foot_bbox": "foot", "knee2_bbox": "knee (patient B)",
               "spine": "cervical spine", "knee_joint": "knees (patient A)", "knee_one": "knee (patient A)", "ankle": "feet",
               "knee": "lower legs"}


def make_tour(scene: Scene, model, volumes, out_dir: Path, size: int, device, seed: int = 11, preprocess: str = "legacy",
              hold_ms: int = 1500, queries_per_example: int = 2, rounds: int = 2, fps: int = 10, name: str = "tour_transformations",
              n_candidates: int = 3):
    """One animation: every transformation family in turn, each time on a different study; GT vs prediction."""
    import random as pyrandom

    from src.data.online_drr import sample_acquisition

    print("[tour]", volumes)
    ps = 256 // N
    rng = np.random.default_rng(seed)
    kinds = [k for k, _, _ in VARIATIONS]
    titles = {k: t for k, t, _ in VARIATIONS}
    frames, durations = [], []
    n_ex = 0
    for r in range(rounds):
        for i, kind in enumerate(kinds):
            vol = volumes[(i + 3 * r) % len(volumes)]
            cands = []
            for c in range(n_candidates):  # median-AP draw: neither a lucky nor an unlucky example
                if kind == "family":
                    item = sample_acquisition(pyrandom.Random(f"{seed}:{vol}:{r}:{c}"), 2, True)
                    views, corr, _, _ = scene.pair_csv(vol, item)
                    a = item["angles"].tolist()
                    label = f"a = {a[0][0]:.0f}°/{a[0][1]:.0f}°, {a[1][0]:.0f}°/{a[1][1]:.0f}° · shift {item['translation'][0]:+.0f}"
                else:
                    v = _sample_variation(kind, rng)
                    views, corr, _, _ = scene.pair(vol, v["cams"], sid=v["sid"], sad=v["sad"], tx=v["tx"])
                    label = _fmt_pose(v)
                C = corr[0].cpu().numpy()
                v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
                P = predict(model, v1, v2, device)
                ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
                cands.append((ap, views, C, v1, v2, P, label))
            cands.sort(key=lambda t: t[0])
            ap, views, C, v1, v2, P, label = cands[len(cands) // 2]
            qs = pick_queries(views[0, 0].cpu().numpy(), C, queries_per_example + 2)
            qs = qs[:queries_per_example] if qs else [(8, 8)]
            rgb1, rgb2 = gray_rgb(v1), gray_rgb(v2)
            b1, b2 = content_box(views[0, 0].cpu().numpy()), content_box(views[1, 0].cpu().numpy())
            study = STUDY_NAMES.get(vol, vol)
            for (qr, qc) in qs:
                qi = qr * N + qc
                frames.append(compose([
                    panel(crop_rgb(draw_patch(rgb1.copy(), qr, qc, CORAL, ps), b1), size, f"View 1 · {study}", titles[kind], sub2=label),
                    panel(crop_rgb(heat_overlay(rgb2, C[qi].reshape(N, N), CORAL), b2), size, "View 2 · ground truth", "analytic correspondence of the query", sub2=" "),
                    panel(crop_rgb(heat_overlay(rgb2, P[qi].reshape(N, N), TEAL), b2), size, "View 2 · prediction", f"AP of this pair {ap:.2f}", sub2=" "),
                ]))
                durations.append(hold_ms)
            n_ex += 1
    print(f"  {n_ex} examples, {len(frames)} frames")
    save_anim(frames, str(out_dir / name), fps, durations_ms=durations)


# ---------------------------------------------------------------------------------------------
# Single-study tour and sweep with a systematic set of transformations
LEG_TOUR = [
    ("Rotation about the long axis", "θ₂ = 30°", dict(cams=[(0, 0, 0), (30, 0, 0)])),
    ("Rotation about the long axis", "θ₂ = 60°", dict(cams=[(0, 0, 0), (60, 0, 0)])),
    ("Rotation about the long axis", "θ₂ = 90° (lateral)", dict(cams=[(0, 0, 0), (90, 0, 0)])),
    ("Rotation about the long axis", "θ₂ = 135°", dict(cams=[(0, 0, 0), (135, 0, 0)])),
    ("Rotation about the long axis", "θ₂ = 180° (PA)", dict(cams=[(0, 0, 0), (180, 0, 0)])),
    ("Rotation about the long axis", "θ₂ = 250°", dict(cams=[(0, 0, 0), (250, 0, 0)])),
    ("In-plane roll of view 2", "θ₂ = 75° · roll₂ = +20°", dict(cams=[(0, 0, 0), (75, 0, 20)])),
    ("In-plane roll of view 2", "θ₂ = 105° · roll₂ = −40°", dict(cams=[(0, 0, 0), (105, 0, -40)])),
    ("Both views rolled", "roll₁ = −15° · θ₂ = 100° · roll₂ = +30°", dict(cams=[(0, 0, -15), (100, 0, 30)])),
    ("Translation of the volume", "θ₂ = 80° · shift −45 voxels", dict(cams=[(0, 0, 0), (80, 0, 0)], tx=-45)),
    ("Translation of the volume", "θ₂ = 110° · shift +30 voxels", dict(cams=[(0, 0, 0), (110, 0, 0)], tx=30)),
    ("Zoom in: source closer to the patient", "view 2 · sid 900 · sad 1300 mm (×2.4)", dict(cams=[(0, 0, 0), (70, 0, 0)], sid=(1050, 900), sad=(1000, 1300))),
    ("Zoom out: source farther away", "view 2 · sid 1100 · sad 800 mm (×1.7)", dict(cams=[(0, 0, 0), (100, 0, 0)], sid=(1050, 1100), sad=(1000, 800))),
    ("Different magnification per view", "view 1 ×2.2 · view 2 ×1.7", dict(cams=[(0, 0, 0), (90, 0, 0)], sid=(1000, 1100), sad=(1200, 800))),
    ("Roll + shift + zoom together", "roll₁ +10° · θ₂ 120° · roll₂ −25° · shift +35 · ×2.3", dict(cams=[(0, 0, 10), (120, 0, -25)], sid=(1000, 950), sad=(1200, 1250), tx=35)),
    ("Paper's full pose family (random)", None, "family"),
    ("Out-of-plane tilt of 15° (not in training)", "θ₂ = 90° · tilt₂ = +15°", dict(cams=[(0, 0, 0), (90, 15, 0)])),
]


def make_leg_tour(scene: Scene, model, vol: str, out_dir: Path, size: int, device, preprocess="legacy", hold_ms=1400,
                  queries_per_example=2, fps=10, seed=5):
    import random as pyrandom

    from src.data.online_drr import sample_acquisition

    print("[leg tour]", vol)
    ps = 256 // N
    frames, durations, stats = [], [], []
    study = STUDY_NAMES.get(vol, vol)
    for k, (title, label, spec) in enumerate(LEG_TOUR):
        if spec == "family":
            cands = []
            for c in range(3):
                item = sample_acquisition(pyrandom.Random(f"{seed}:{vol}:{c}"), 2, True)
                views, corr, _, _ = scene.pair_csv(vol, item)
                a = item["angles"].tolist()
                lab = f"a = {a[0][0]:.0f}°/{a[0][1]:.0f}°, {a[1][0]:.0f}°/{a[1][1]:.0f}° · shift {item['translation'][0]:+.0f}"
                C = corr[0].cpu().numpy()
                v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
                P = predict(model, v1, v2, device)
                ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
                cands.append((ap, views, C, v1, v2, P, lab))
            cands.sort(key=lambda t: t[0])
            ap, views, C, v1, v2, P, label = cands[1]
        else:
            views, corr, _, _ = scene.pair(vol, spec["cams"], sid=spec.get("sid", (1050, 1050)), sad=spec.get("sad", (1000, 1000)), tx=spec.get("tx", 0.0))
            C = corr[0].cpu().numpy()
            v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
            P = predict(model, v1, v2, device)
            ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
        stats.append({"title": title, "label": label, "ap": ap})
        print(f"  {k:2d} {title:42s} {label:50s} AP {ap:.2f}", flush=True)
        qs = pick_queries(views[0, 0].cpu().numpy(), C, queries_per_example + 3)
        # vary the query across examples: rotate the candidate list
        qs = (qs[k % max(1, len(qs)):] + qs[:k % max(1, len(qs))])[:queries_per_example] if qs else [(8, 8)]
        rgb1, rgb2 = gray_rgb(v1), gray_rgb(v2)
        b1, b2 = content_box(views[0, 0].cpu().numpy()), content_box(views[1, 0].cpu().numpy())
        for (qr, qc) in qs:
            qi = qr * N + qc
            frames.append(compose([
                panel(crop_rgb(draw_patch(rgb1.copy(), qr, qc, CORAL, ps), b1), size, f"View 1 · {study}", title, sub2=label),
                panel(crop_rgb(heat_overlay(rgb2, C[qi].reshape(N, N), CORAL), b2), size, "View 2 · ground truth", "analytic correspondence of the query", sub2=" "),
                panel(crop_rgb(heat_overlay(rgb2, P[qi].reshape(N, N), TEAL), b2), size, "View 2 · prediction", f"AP of this pair {ap:.2f}", sub2=" "),
            ]))
            durations.append(hold_ms)
    json.dump(stats, open(out_dir / f"leg_tour_{vol}_stats.json", "w"), indent=1)
    save_anim(frames, str(out_dir / f"tour_{vol}"), fps, durations_ms=durations)


def make_leg_sweep(scene: Scene, model, vol: str, out_dir: Path, size: int, device, preprocess="legacy", fps=10, n_seg=36):
    """Five segments: azimuth 20→160°, roll ±40°, translation ±50 vox, zoom (sid 1100→850, sad 800→1300), all combined."""
    print("[leg sweep]", vol)
    ps = 256 // N
    study = STUDY_NAMES.get(vol, vol)
    views0, corr0, _, _ = scene.pair(vol, [(0, 0, 0), (90, 0, 0)])
    qs = pick_queries(views0[0, 0].cpu().numpy(), corr0[0].cpu().numpy(), 3)
    q = qs[1] if len(qs) > 1 else (qs[0] if qs else (8, 8))
    recs = []
    for seg in range(5):
        for i in range(n_seg):
            t = i / (n_seg - 1)
            u = math.sin(2 * math.pi * i / n_seg)
            qq = q
            if seg == 0:
                th = 20 + 140 * t
                v = dict(cams=[(0, 0, 0), (th, 0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
                head = f"rotation about the long axis: θ₂ = {th:.0f}°"
            elif seg == 1:
                v = dict(cams=[(0, 0, 0), (90, 0, 40 * u)], sid=(1050, 1050), sad=(1000, 1000), tx=0.0)
                head = f"in-plane roll of view 2: {40 * u:+.0f}°"
            elif seg == 2:
                v = dict(cams=[(0, 0, 0), (75, 0, 0)], sid=(1050, 1050), sad=(1000, 1000), tx=50 * u)
                head = f"volume translation: {50 * u:+.0f} voxels"
                dcol = int(round(-v["tx"] * 0.976 / ps)) if scene.r.mapping.get("flip_cols") else int(round(v["tx"] * 0.976 / ps))
                qq = (q[0], min(N - 1, max(0, q[1] + dcol)))
            elif seg == 3:
                w = 0.5 - 0.5 * math.cos(2 * math.pi * i / n_seg)  # 0 → 1 → 0
                sid2, sad2 = 1100 - 250 * w, 800 + 500 * w
                v = dict(cams=[(0, 0, 0), (100, 0, 0)], sid=(1050, sid2), sad=(1000, sad2), tx=0.0)
                head = f"zoom of view 2: sid {sid2:.0f} · sad {sad2:.0f} mm (×{(sid2 + sad2) / sid2:.2f})"
            else:
                th, roll2, tx = 60 + 60 * t, 30 * math.sin(math.pi * t), -40 * t
                sid2, sad2 = 1050 - 100 * t, 1000 + 200 * t
                v = dict(cams=[(0, 0, 8 * t), (th, 0, roll2)], sid=(1050, sid2), sad=(1000, sad2), tx=tx)
                head = f"all at once: θ₂ {th:.0f}° · roll₂ {roll2:+.0f}° · shift {tx:+.0f} · ×{(sid2 + sad2) / sid2:.2f}"
                dcol = int(round(-tx * 0.976 / ps)) if scene.r.mapping.get("flip_cols") else int(round(tx * 0.976 / ps))
                qq = (q[0], min(N - 1, max(0, q[1] + dcol)))
            qi = qq[0] * N + qq[1]
            views, corr, _, _ = scene.pair(vol, v["cams"], sid=v["sid"], sad=v["sad"], tx=v["tx"])
            C = corr[0].cpu().numpy()
            v1, v2 = to_display(views[0, 0], preprocess), to_display(views[1, 0], preprocess)
            P = predict(model, v1, v2, device)
            ap = _ap(P.reshape(-1), (C > 0.3).reshape(-1).astype(np.float32))
            recs.append(dict(seg=seg, qq=qq, qi=qi, head=head, ap=ap, v1=v1, v2=v2, C=C, P=P,
                             b1=content_box(views[0, 0].cpu().numpy()), b2=content_box(views[1, 0].cpu().numpy())))
    boxes = {}
    for seg in range(5):
        rs = [r for r in recs if r["seg"] == seg]
        boxes[seg] = (union_box([r["b1"] for r in rs]), union_box([r["b2"] for r in rs]))
    frames = []
    for r in recs:
        B1, B2 = boxes[r["seg"]]
        rgb1, rgb2 = gray_rgb(r["v1"]), gray_rgb(r["v2"])
        sub1 = "shifted with the volume (query follows)" if r["seg"] in (2, 4) else ("rolled by +8° at the end" if r["seg"] == 4 else "fixed AP view")
        frames.append(compose([
            panel(crop_rgb(draw_patch(rgb1.copy(), r["qq"][0], r["qq"][1], CORAL, ps), B1), size, f"View 1 · {study}", sub1),
            panel(crop_rgb(heat_overlay(rgb2, r["C"][r["qi"]].reshape(N, N), CORAL), B2), size, "View 2 · ground truth", r["head"]),
            panel(crop_rgb(heat_overlay(rgb2, r["P"][r["qi"]].reshape(N, N), TEAL), B2), size, "View 2 · prediction", f"AP of this pair {r['ap']:.2f}"),
        ]))
    aps = [r["ap"] for r in recs]
    print(f"  sweep AP: mean {np.mean(aps):.2f}  min {np.min(aps):.2f}  per segment " + " ".join(f"{np.mean([r['ap'] for r in recs if r['seg'] == s_]):.2f}" for s_ in range(5)))
    save_anim(frames, str(out_dir / f"sweep5_{vol}"), fps)


def main():
    global OVERLAY_THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--volumes", nargs="+", default=["knee", "ankle", "hip", "shoulder"])
    ap.add_argument("--orbit_volumes", nargs="*", default=None)
    ap.add_argument("--build_volume", nargs="*", default=None, help="volume(s) for the construction animation (default: first of --volumes)")
    ap.add_argument("--no_grid", action="store_true", help="no patch grid overlay in the construction animation")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pose_mapping", default=None, help="validation report.json (flip_rows/flip_cols)")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--n_queries", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", nargs="*", default=None, choices=["orbit", "build", "prediction", "teaser", "variations", "tour", "legtour", "legsweep"])
    ap.add_argument("--leg_volume", default="knee_left")
    ap.add_argument("--tour_volumes", nargs="*", default=None)
    ap.add_argument("--tour_hold_ms", type=int, default=1500)
    ap.add_argument("--var_volumes", nargs="*", default=None, help="volumes for the pose-variation grid/statistics (default: --volumes)")
    ap.add_argument("--var_sweep_volume", nargs="*", default=None)
    ap.add_argument("--var_stats_n", type=int, default=24)
    ap.add_argument("--preprocess", default="both", choices=["public", "legacy", "both"], help="view preprocessing fed to the checkpoint")
    ap.add_argument("--overlay_thr", type=float, default=OVERLAY_THR,
                    help="correspondence weights below this value are not drawn in the overlays (GT and prediction)")
    a = ap.parse_args()
    OVERLAY_THR = a.overlay_thr
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    mapping = json.load(open(a.pose_mapping))["pose_mapping"] if a.pose_mapping else None
    scene = Scene(a.cache_dir, device, mapping)
    want = set(a.only or ["orbit", "build", "prediction", "teaser"])
    if ("legtour" in want or "legsweep" in want) and a.ckpt:
        model = load_model(a.ckpt, device)
        pp = "legacy" if a.preprocess == "both" else a.preprocess
        if "legtour" in want:
            make_leg_tour(scene, model, a.leg_volume, out, a.size, device, preprocess=pp, fps=a.fps)
        if "legsweep" in want:
            make_leg_sweep(scene, model, a.leg_volume, out, a.size, device, preprocess=pp, fps=a.fps)
    if "tour" in want and a.ckpt:
        model = load_model(a.ckpt, device)
        make_tour(scene, model, a.tour_volumes or a.volumes, out, a.size, device, hold_ms=a.tour_hold_ms,
                  preprocess="legacy" if a.preprocess == "both" else a.preprocess, fps=a.fps)
    if "variations" in want and a.ckpt:
        model = load_model(a.ckpt, device)
        make_variations(scene, model, a.var_volumes or a.volumes, out, a.size, device, n_stats=a.var_stats_n,
                        sweep_volume=a.var_sweep_volume, preprocess="legacy" if a.preprocess == "both" else a.preprocess, fps=a.fps)
    if "teaser" in want:
        make_teaser(scene, a.volumes, out, a.size)
    if "build" in want:
        for name in (a.build_volume or [a.volumes[0]]):
            make_build(scene, name, out, a.size, a.fps, grid=not a.no_grid)
    if "orbit" in want:
        for name in (a.orbit_volumes if a.orbit_volumes is not None else a.volumes[:2]):
            make_orbit(scene, name, out, a.frames, a.size, a.fps)
    if "prediction" in want and a.ckpt:
        model = load_model(a.ckpt, device)
        make_prediction(scene, model, a.volumes, out, a.size, a.n_queries, device, a.preprocess)


if __name__ == "__main__":
    main()
