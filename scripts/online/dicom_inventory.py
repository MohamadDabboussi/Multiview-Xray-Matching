"""Resolution inventory of DICOM series folders: slices, pixel spacing, slice spacing, kernel/description.

    python scripts/online/dicom_inventory.py --ct_dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pydicom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_dir", required=True)
    a = ap.parse_args()
    for d in sorted(os.listdir(a.ct_dir)):
        fs = sorted(glob.glob(os.path.join(a.ct_dir, d, "*")))
        if not fs:
            continue
        try:
            h = pydicom.dcmread(fs[0], stop_before_pixels=True)
            step = max(1, len(fs) // 20)
            zs = sorted(float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in fs[::step])
            dz = float(np.median(np.diff(zs))) if len(zs) > 1 else float("nan")
            ps = [float(x) for x in h.PixelSpacing]
            desc = str(getattr(h, "SeriesDescription", ""))[:26]
            kern = str(getattr(h, "ConvolutionKernel", ""))[:8]
            part = str(getattr(h, "BodyPartExamined", ""))[:10]
            print(f"{d[:44]:44s} n={len(fs):4d} px={ps[0]:.2f} dz={abs(dz):.2f} z={len(fs) * abs(dz):4.0f}mm rows={h.Rows} {part:10s} {kern:8s} {desc}")
        except Exception as e:  # noqa: BLE001
            print(d, "ERR", repr(e)[:60])


if __name__ == "__main__":
    main()
