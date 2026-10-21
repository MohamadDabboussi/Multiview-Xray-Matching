"""Stage B of the parity check: compare the on-the-fly renderer + analytic correspondence against
the RTK reference of reference_rtk.py.

1. Pose convention.  The RTK geometry of the offline code (gantry 90, out-of-plane a0+90,
   in-plane 90-a1) is identified by a search over mappings of our camera parameters
   (theta = s_t*a0 + o_t, phi = s_p*(a1-180) + o_p, roll, row/column flips), scored by the NCC of
   the DRRs of several settings at mid resolution, then the sign of the volume translation.
2. With the identified mapping: full-resolution DRR NCC per view, and correspondence agreement
   (Pearson r, mean |diff|, IoU of C > 0.3, AP of ours against RTK's labels at 0.3).
3. Timing of rendering + GT on the current device.

    python scripts/online/validate_online.py --ref <dir>/ref_knee.npz --out <dir>/validation_knee
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.online_drr import OnlineDRRRenderer, csv_to_tpr, hu_sigmoid, shift_volume_x  # noqa: E402

N = 16


class _Bank:  # minimal bank: a single in-memory cube
    def __init__(self, cube_xyz):
        self.cube = cube_xyz

    def get(self, i, device):
        return self.cube.to(device)


def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def average_precision(scores, labels):
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / (np.arange(len(lab)) + 1)
    return float((prec * lab).sum() / max(lab.sum(), 1))


def apply_flips_img(img, fr, fc):
    if fr:
        img = img.flip(-2)
    if fc:
        img = img.flip(-1)
    return img


def perm_for_flips(fr, fc):
    perm = np.arange(N * N).reshape(N, N)
    if fr:
        perm = perm[::-1]
    if fc:
        perm = perm[:, ::-1]
    return perm.reshape(-1).copy()


def render_csv(renderer, density, angles_csv, sid, sad, mapping, device):
    """Render CSV-convention views with a candidate mapping (no flips applied)."""
    theta, phi, roll = csv_to_tpr(angles_csv, mapping)
    geom = renderer.geometry_tpr(theta.to(device), phi.to(device), roll.to(device), sid.to(device), sad.to(device))
    return renderer.render(density, geom)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search_dim", type=int, default=128)
    ap.add_argument("--search_points", type=int, default=160)
    ap.add_argument("--search_settings", type=int, nargs="*", default=None, help="settings used for the pose search (default: untranslated ones)")
    ap.add_argument("--footprint_samples", type=int, default=4)
    ap.add_argument("--n_points", type=int, default=300)
    ap.add_argument("--mapping", default=None, help="report.json of a previous run: reuse its pose mapping, skip the search")
    ap.add_argument("--settings", type=int, nargs="*", default=None, help="subset of settings for the full-res comparison")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = np.load(a.ref, allow_pickle=True)
    meta = json.loads(str(ref["meta"]))
    cube_zyx = np.load(str(ref["cube"])).astype(np.float32)
    cube = torch.from_numpy(cube_zyx).permute(2, 1, 0).contiguous().half()  # (X, Y, Z)
    dim = cube.shape[0]

    # ---------------- 1. pose convention search ------------------------------------------
    sd = a.search_dim
    pool = dim // sd
    small = OnlineDRRRenderer(_Bank(None), dim=sd, n_patches=16, delx=2.0 * pool, n_points=a.search_points,
                              footprint_samples=a.footprint_samples, render_chunk=16, voxel_mm=float(pool)).to(device)
    hu_small = F.avg_pool3d(cube.float()[None, None], pool)[0, 0].to(device)
    density_small = hu_sigmoid(hu_small, 1500.0)
    sel = a.search_settings if a.search_settings is not None else [i for i, s in enumerate(meta) if s["tx"] == 0]
    ref_views = torch.from_numpy(ref["views"][sel]).to(device)                       # (S, V, 256, 256)
    ref_small = F.avg_pool2d(ref_views.reshape(-1, 1, dim, dim), pool)[:, 0]        # (S*V, sd, sd)
    angles = torch.tensor([meta[i]["angles"] for i in sel], dtype=torch.float32).reshape(-1, 2)
    sid = torch.tensor([meta[i]["sid"] for i in sel], dtype=torch.float32).reshape(-1)
    sad = torch.tensor([meta[i]["sad"] for i in sel], dtype=torch.float32).reshape(-1)
    if a.mapping:
        best = json.load(open(a.mapping))["pose_mapping"]
        print("reusing mapping", best)
        cands = []
    else:
        cands = [dict(theta_sign=s_t, theta_offset=o_t, roll_sign=s_r, roll_offset=o_r)
                 for s_t, o_t, s_r, o_r in itertools.product([1, -1], [0, 90, 180, 270], [1, -1], [0, 180])]
    t_search = time.time()
    results = []
    with torch.no_grad():
        for mapping in cands:
            img = render_csv(small, density_small, angles, sid, sad, mapping, device)
            for fr, fc in itertools.product([False, True], [False, True]):
                im = apply_flips_img(img, fr, fc)
                score = float(np.mean([ncc(im[k], ref_small[k]) for k in range(im.shape[0])]))
                results.append((score, dict(mapping, flip_rows=fr, flip_cols=fc)))
    if results:
        results.sort(key=lambda r: -r[0])
        best_score, best = results[0]
        print(f"pose search over {len(results)} candidates ({len(sel)} settings, {sd}px) in {time.time() - t_search:.1f}s")
        for sc, m in results[:6]:
            print(f"   NCC={sc:.4f}  {m}")
        best = dict(best, search_ncc=best_score)

    # translation sign (settings with tx != 0), at search resolution
    tx_sets = [i for i, s_ in enumerate(meta) if s_["tx"] != 0]
    if a.mapping:
        pass
    elif tx_sets:
        scores = {}
        with torch.no_grad():
            for sign in (1.0, -1.0):
                vals = []
                for i in tx_sets:
                    hu_s = shift_volume_x(hu_small, sign * meta[i]["tx"] / pool)
                    dens = hu_sigmoid(hu_s, 1500.0)
                    img = render_csv(small, dens, torch.tensor(meta[i]["angles"], dtype=torch.float32),
                                     torch.tensor(meta[i]["sid"], dtype=torch.float32),
                                     torch.tensor(meta[i]["sad"], dtype=torch.float32), best, device)
                    img = apply_flips_img(img, best["flip_rows"], best["flip_cols"])
                    rs = F.avg_pool2d(torch.from_numpy(ref["views"][i]).to(device)[:, None], pool)[:, 0]
                    vals += [ncc(img[k], rs[k]) for k in range(img.shape[0])]
                scores[sign] = float(np.mean(vals))
        best["translation_sign"] = 1.0 if scores[1.0] >= scores[-1.0] else -1.0
        print(f"translation sign: NCC(+)={scores[1.0]:.4f}  NCC(-)={scores[-1.0]:.4f}  ->  {best['translation_sign']:+.0f}")
    else:
        best["translation_sign"] = 1.0

    # ---------------- 2. full-resolution comparison (final API: mapping inside the renderer) -------
    full = OnlineDRRRenderer(_Bank(cube), dim=dim, n_patches=16, delx=2.0, n_points=a.n_points, footprint_samples=a.footprint_samples,
                             render_chunk=2, mapping=best).to(device)
    report = {"pose_mapping": best, "settings": []}
    for i, s_ in enumerate(meta):
        if a.settings is not None and i not in a.settings:
            continue
        angles_i = torch.tensor(s_["angles"], dtype=torch.float32)
        sid_i, sad_i = torch.tensor(s_["sid"], dtype=torch.float32), torch.tensor(s_["sad"], dtype=torch.float32)
        t0 = time.time()
        views, corr, fps, w = full.render_sample(cube.to(device), angles_i, sid_i, sad_i, translation=torch.tensor([float(s_["tx"]), 0, 0]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        v = views[:, 0].cpu().numpy()
        c = corr.cpu().numpy()
        rv, rc = ref["views"][i], ref["corr"][i]
        view_ncc = [ncc(torch.tensor(v[k]), torch.tensor(rv[k])) for k in range(v.shape[0])]
        entry = {"setting": s_, "seconds_online": round(dt, 3), "view_ncc": view_ncc, "pairs": []}
        for p_ in range(c.shape[0]):
            ours, theirs = c[p_].reshape(-1), rc[p_].reshape(-1)
            lab = (theirs > 0.3).astype(np.float32)
            entry["pairs"].append({
                "pearson": float(np.corrcoef(ours, theirs)[0, 1]),
                "mae": float(np.abs(ours - theirs).mean()),
                "iou_0.3": float(((ours > 0.3) & (theirs > 0.3)).sum() / max(((ours > 0.3) | (theirs > 0.3)).sum(), 1)),
                "ap_vs_rtk_0.3": average_precision(ours, lab),
                "ap_rtk_vs_ours_0.3": average_precision(theirs, (ours > 0.3).astype(np.float32)),
                "frac_pos_ours": float((ours > 0.3).mean()), "frac_pos_rtk": float(lab.mean()),
                "max_ours": float(ours.max()), "max_rtk": float(theirs.max()),
            })
        report["settings"].append(entry)
        print(f"setting {i}: NCC views {np.round(view_ncc, 4)}  corr r={entry['pairs'][0]['pearson']:.4f} "
              f"IoU@0.3={entry['pairs'][0]['iou_0.3']:.3f} AP={entry['pairs'][0]['ap_vs_rtk_0.3']:.3f}  ({dt:.2f}s)")
        np.savez_compressed(os.path.join(a.out, f"setting{i}.npz"), ours_views=v, ours_corr=c, rtk_views=rv, rtk_corr=rc)

    # ---------------- 3. timing ------------------------------------------------------------
    s0 = meta[0]
    ang = torch.tensor(s0["angles"], dtype=torch.float32)
    sidt, sadt = torch.tensor(s0["sid"], dtype=torch.float32), torch.tensor(s0["sad"], dtype=torch.float32)
    for _ in range(2):
        full.render_sample(cube.to(device), ang, sidt, sadt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 10
    for _ in range(n):
        full.render_sample(cube.to(device), ang, sidt, sadt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report["timing"] = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                        "seconds_per_pair_render_plus_gt": (time.time() - t0) / n, "n_points": a.n_points,
                        "footprint_samples": a.footprint_samples,
                        "rtk_seconds_per_pair": float(np.mean([m.get("seconds", float("nan")) for m in meta]))}
    print("timing:", report["timing"])
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)

    # ---------------- figure -----------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shown = [i for i in range(len(meta)) if os.path.exists(os.path.join(a.out, f"setting{i}.npz"))]
        n_set = len(shown)
        fig, axs = plt.subplots(n_set, 6, figsize=(18, 3 * n_set))
        axs = np.atleast_2d(axs)
        for row_i, i in enumerate(shown):
            d = np.load(os.path.join(a.out, f"setting{i}.npz"))
            i = row_i
            for k in range(2):
                axs[i, k].imshow(d["rtk_views"][k], cmap="gray"); axs[i, k].set_title(f"RTK view {k + 1}")
                axs[i, 2 + k].imshow(d["ours_views"][k], cmap="gray"); axs[i, 2 + k].set_title(f"online view {k + 1}")
            axs[i, 4].imshow(d["rtk_corr"][0], cmap="magma", vmin=0, vmax=1); axs[i, 4].set_title("RTK corr")
            axs[i, 5].imshow(d["ours_corr"][0], cmap="magma", vmin=0, vmax=1); axs[i, 5].set_title("online corr")
            for ax in axs[i]:
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "comparison.png"), dpi=110)
        print("figure ->", os.path.join(a.out, "comparison.png"))
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
