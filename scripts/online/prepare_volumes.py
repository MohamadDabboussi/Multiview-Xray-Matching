"""Build the HU-cube cache (resampled, cubic, dim^3, float16) for a folder of CT volumes, in parallel.

    python scripts/online/prepare_volumes.py --ct_dir <dir> --cache_dir <dir> [--dim 256] [--workers 8]

The cache is exactly what generate_data.preprocess_volume feeds to the RTK renderer, so the
online pipeline and the offline pipeline start from identical voxels.  Writes manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.online_drr import list_ct_volumes, preprocess_hu_cube, read_any_volume  # noqa: E402


def _one(args):
    path, cache_dir, dim, remove_table, repair, crop, suffix = args
    p = Path(path)
    name = (p.name if p.is_dir() else p.name.split(".")[0]) + suffix
    out = Path(cache_dir) / f"{name}_{dim}.npy"
    t0 = time.time()
    if out.exists():
        return {"name": name, "path": str(p), "cache": str(out), "cached": True}
    spacing, vol = read_any_volume(str(p))
    cube = preprocess_hu_cube(spacing, vol, dim, remove_table=name in remove_table, repair_slices=repair, crop=crop)
    np.save(out, np.clip(cube, -1000, 3000).astype(np.float16))
    return {"name": name, "path": str(p), "cache": str(out), "cached": False, "orig_shape": list(vol.shape),
            "spacing": list(spacing), "hu_min": float(vol.min()), "hu_max": float(vol.max()), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--remove_table", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="volume names to process (default: all)")
    ap.add_argument("--no_repair", action="store_true", help="keep isolated all-air slices (exact offline behaviour)")
    ap.add_argument("--crop", choices=["bbox", "cube"], default=None, help="crop before cubing: body bounding box, or a cube")
    ap.add_argument("--crop_center", type=float, nargs=3, default=None, metavar=("Z", "Y", "X"), help="cube centre as fractions of the body bbox")
    ap.add_argument("--crop_size_mm", type=float, default=None, help="cube edge in mm (default: smallest body extent)")
    ap.add_argument("--crop_center_on", choices=["bone"], default=None, help="cube mode: centre on the bone bounding box instead of the body")
    ap.add_argument("--suffix", default="", help="suffix appended to the cached name (e.g. _knee_joint)")
    a = ap.parse_args()
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    vols = list_ct_volumes(a.ct_dir)
    if a.only:
        vols = [v for v in vols if (v.name if v.is_dir() else v.name.split(".")[0]) in set(a.only)]
    print(f"{len(vols)} volumes")
    results, errors = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        crop = None if a.crop is None else {"mode": a.crop, "center": a.crop_center, "size_mm": a.crop_size_mm, "center_on": a.crop_center_on}
        futs = {ex.submit(_one, (str(v), a.cache_dir, a.dim, set(a.remove_table), not a.no_repair, crop, a.suffix)): v for v in vols}
        for f in as_completed(futs):
            try:
                r = f.result()
                results.append(r)
                print(f"ok  {r['name']}  {'(cached)' if r.get('cached') else r.get('seconds')}", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append({"path": str(futs[f]), "error": repr(e)})
                print(f"ERR {futs[f]}: {e!r}", flush=True)
    manifest = {"dim": a.dim, "ct_dir": a.ct_dir, "volumes": sorted(results, key=lambda r: r["name"]), "errors": errors}
    (Path(a.cache_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(results)} ok, {len(errors)} errors -> {a.cache_dir}/manifest.json")


if __name__ == "__main__":
    main()
