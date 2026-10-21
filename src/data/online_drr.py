"""
On-the-fly DRR rendering + patch-correspondence ground truth on the GPU.

This module replaces the offline data pipeline

    create_csv.py  ->  generate_data.py (ITK-RTK, CPU)  ->  data.h5  ->  DRRDataset

with a renderer that is called *inside the training step*:

    CT volumes (cached once as 256^3 HU cubes)
        -> sample two camera poses            (same distributions as create_csv.py)
        -> render the two DRRs with DiffDRR   (the differentiable renderer behind `xvr`)
        -> compute the many-to-many patch correspondence matrix analytically
           from the camera geometry           (Sec. 3.1 of the paper, max over voxels)
        -> GPU preprocessing / augmentation
        -> model

Nothing is written to disk, so new poses are drawn every epoch.

Geometry
--------
World frame = volume frame (mm, isotropic, voxel size 1 mm as in the offline pipeline, which
treats the 256^3 cube as a 256 mm cube), origin at the volume centre.  The volume tensor is kept
in torchio/DiffDRR order (X, Y, Z) = (columns, rows, slices) of the DICOM series.

Canonical camera (theta = phi = roll = 0): X-ray source on -y at distance ``sid`` from the
isocentre looking along +y, detector plane centre on +y at distance ``sad``;  image columns run
along +x, image rows along -z (head at the top).  ``theta`` rotates the camera about the
cranio-caudal axis z (0 = AP, 90 = lateral), ``phi`` tilts it out of the axial plane (rotation
about x), ``roll`` spins the detector about the viewing axis.  In the offline pipeline (RTK, gantry fixed at
90 deg) the CSV angles ``[a0, a1]`` are an azimuth and an *in-plane* rotation (``a1 ~ 180 +- 15``):
they map to ``theta`` and ``roll`` through ``RTK_MAPPING`` (identified numerically), ``phi = 0``.

Correspondence ground truth (Sec. 3.1)
--------------------------------------
For each coarse voxel v of the N^3 grid (N = 16): its footprint P_k^v on view k is the
normalised set of patches hit by the rays through the voxel (estimated by projecting
``footprint_samples``^3 sub-points per voxel and counting hits per patch, max-normalised exactly
like the offline code normalises the single-voxel Joseph projection);  C^v = P_1^v (x) P_2^v
times the voxel weight sigmoid((HU+1000-850)*0.005) average-pooled to the coarse grid; and
C = max_v C^v.  The scatter-max implementation costs O(V * S^2) instead of 4096 ray-casts.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------------------------
# Pose / acquisition sampling  (verbatim distributions of src/data/create_csv.py)
# ----------------------------------------------------------------------------------------------


def biased_multiview_angle(rng: random.Random) -> float:
    """Random angle biased towards 0/90/180/270 degrees (standard radiographic views)."""
    key_angles = [0, 90, 180, 270]
    probabilities = [0.4, 0.3, 0.2, 0.1]
    chosen = rng.choices(key_angles, probabilities)[0]
    noise = rng.gauss(0, 45)
    return (chosen + noise) % 360


def biased_angle(rng: random.Random, mean, std_dev, lower, upper) -> float:
    return max(lower, min(rng.gauss(mean, std_dev), upper))


def get_random_angle(rng: random.Random, mode: int) -> List[float]:
    if mode == 0:
        return [biased_multiview_angle(rng), biased_angle(rng, 180, 15, 150, 210)]
    if mode == 1:
        return [biased_angle(rng, 0, 15, -20, 20) % 360, biased_angle(rng, 180, 45, 0, 360)]
    return [biased_multiview_angle(rng), biased_angle(rng, 180, 60, 0, 360)]


def get_random_translation(rng: random.Random, mode: int) -> int:
    if mode == 0:
        return 0
    if mode == 1:
        return rng.randint(-50, -1)
    return rng.randint(1, 50)


def get_random_sid_sad(rng: random.Random, which: int) -> int:
    return rng.randint(1000, 1100) if which == 0 else rng.randint(800, 1200)


def sample_acquisition(rng: random.Random, n_views: int = 2, translate: bool = True, tilt_std: float = 0.0) -> Dict[str, torch.Tensor]:
    """One multi-view acquisition: per-view angles / sid / sad and a volume translation (voxels).
    ``tilt_std`` > 0 adds an out-of-plane tilt (degrees) that the offline pipeline did not have."""
    angles = [get_random_angle(rng, rng.randint(0, 2)) for _ in range(n_views)]
    tx = get_random_translation(rng, rng.randint(0, 2)) if translate else 0
    sid = [get_random_sid_sad(rng, 0) for _ in range(n_views)]
    sad = [get_random_sid_sad(rng, 1) for _ in range(n_views)]
    tilt = [max(-45.0, min(45.0, rng.gauss(0, tilt_std))) if tilt_std > 0 else 0.0 for _ in range(n_views)]
    return {
        "angles": torch.tensor(angles, dtype=torch.float32),            # (V, 2)  [a0, a1] CSV convention
        "sid": torch.tensor(sid, dtype=torch.float32),                   # (V,)
        "sad": torch.tensor(sad, dtype=torch.float32),                   # (V,)
        "translation": torch.tensor([float(tx), 0.0, 0.0]),              # (3,) voxels, x only (as offline)
        "tilt": torch.tensor(tilt, dtype=torch.float32),                 # (V,) degrees (0 = offline parity)
    }


class OnlineDRRParams(Dataset):
    """Map-style dataset of *acquisition parameters* (no images).

    Consecutive indices share the same volume (``pairs_per_volume`` in a row) so that a batch
    touches few volumes; the renderer groups samples by volume.  ``set_epoch`` re-draws the
    schedule and the poses (training); with ``fixed=True`` the set is identical every epoch
    (validation).
    """

    def __init__(
        self,
        n_volumes: int,
        samples_per_epoch: int,
        pairs_per_volume: int = 4,
        n_views: int = 2,
        seed: int = 0,
        fixed: bool = False,
        translate: bool = True,
        volume_weights: Optional[Sequence[float]] = None,
        tilt_std: float = 0.0,
    ):
        assert n_volumes > 0
        self.tilt_std = float(tilt_std)
        self.n_volumes = n_volumes
        self.samples_per_epoch = int(samples_per_epoch)
        self.pairs_per_volume = max(1, int(pairs_per_volume))
        self.n_views = n_views
        self.seed = seed
        self.fixed = fixed
        self.translate = translate
        self.volume_weights = None if volume_weights is None else list(volume_weights)
        self.epoch = 0
        self._build_schedule()

    def set_epoch(self, epoch: int):
        if self.fixed:
            return
        self.epoch = int(epoch)
        self._build_schedule()

    def _build_schedule(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        n_blocks = math.ceil(self.samples_per_epoch / self.pairs_per_volume)
        if self.volume_weights is None:
            order: List[int] = []
            while len(order) < n_blocks:
                perm = list(range(self.n_volumes))
                rng.shuffle(perm)
                order += perm
        else:
            order = rng.choices(range(self.n_volumes), weights=self.volume_weights, k=n_blocks)
        schedule = []
        for b in order[:n_blocks]:
            schedule += [b] * self.pairs_per_volume
        self._schedule = schedule[: self.samples_per_epoch]

    def __len__(self):
        return len(self._schedule)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(f"{self.seed}:{self.epoch}:{idx}")
        item = sample_acquisition(rng, self.n_views, self.translate, self.tilt_std)
        item["vol_idx"] = torch.tensor(self._schedule[idx], dtype=torch.long)
        item["sample_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


# ----------------------------------------------------------------------------------------------
# Camera geometry
# ----------------------------------------------------------------------------------------------


def _rot_x(a: torch.Tensor) -> torch.Tensor:
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(*a.shape, 3, 3)


def _rot_y(a: torch.Tensor) -> torch.Tensor:
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(*a.shape, 3, 3)


def _rot_z(a: torch.Tensor) -> torch.Tensor:
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


def camera_frames(theta_deg: torch.Tensor, phi_deg: torch.Tensor, roll_deg: torch.Tensor):
    """Orthonormal camera frames.  Returns (d, u, v): viewing direction (source -> isocentre),
    image column direction and image row direction, each (..., 3)."""
    t, p, r = torch.deg2rad(theta_deg), torch.deg2rad(phi_deg), torch.deg2rad(roll_deg)
    R = _rot_z(t) @ _rot_x(p) @ _rot_y(r)
    d0 = torch.tensor([0.0, 1.0, 0.0], dtype=R.dtype, device=R.device)
    u0 = torch.tensor([1.0, 0.0, 0.0], dtype=R.dtype, device=R.device)
    v0 = torch.tensor([0.0, 0.0, -1.0], dtype=R.dtype, device=R.device)
    return R @ d0, R @ u0, R @ v0


# Mapping from the CSV acquisition parameters of the offline pipeline (ITK-RTK geometry: gantry 90,
# out-of-plane a0 + 90, in-plane 90 - a1) to our camera: theta = theta_sign * a0 + theta_offset,
# roll = roll_sign * (a1 - 180) + roll_offset, no tilt; rendered images are flipped as indicated and
# the volume translation is applied with translation_sign.  Identified numerically against the RTK
# reference by scripts/online/validate_online.py (see the validation report); identity otherwise.
# Identified on 2026-10-21 (knee volume, 5 acquisitions): DRR NCC >= 0.9997 on every view, see
# logs_multiview_xray_matching/results/validation/knee_cpu_v2/report.json.
RTK_MAPPING = dict(theta_sign=-1.0, theta_offset=0.0, roll_sign=1.0, roll_offset=0.0,
                   flip_rows=False, flip_cols=True, translation_sign=1.0)


def csv_to_tpr(angles: torch.Tensor, mapping: Optional[dict] = None, tilt: Optional[torch.Tensor] = None):
    """CSV angles (M, 2) = [a0, a1]  ->  (theta, phi, roll) in degrees under ``mapping``."""
    m = RTK_MAPPING if mapping is None else mapping
    a0, a1 = angles[:, 0].float(), angles[:, 1].float()
    theta = (m["theta_sign"] * a0 + m["theta_offset"]) % 360
    roll = (m["roll_sign"] * (a1 - 180.0) + m["roll_offset"]) % 360
    phi = torch.zeros_like(theta) if tilt is None else tilt.float()
    return theta, phi, roll


class ViewGeometry:
    """A batch of M pinhole X-ray cameras (world coordinates in mm)."""

    def __init__(self, source, center, d, u, v, height, width, delx, dely):
        self.source, self.center, self.d, self.u, self.v = source, center, d, u, v
        self.height, self.width, self.delx, self.dely = int(height), int(width), float(delx), float(dely)

    def __len__(self):
        return self.source.shape[0]

    def to(self, device):
        for k in ("source", "center", "d", "u", "v"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    @classmethod
    def from_tpr(cls, theta, phi, roll, sid, sad, height, width, delx, dely=None) -> "ViewGeometry":
        """theta: azimuth about the cranio-caudal axis (0 = AP, 90 = lateral); phi: out-of-plane tilt;
        roll: in-plane rotation of the detector (all degrees, (M,) tensors); sid/sad in mm."""
        dely = delx if dely is None else dely
        d, u, v = camera_frames(theta.float(), phi.float(), roll.float())
        source = -sid.float()[:, None] * d
        center = sad.float()[:, None] * d
        return cls(source, center, d, u, v, height, width, delx, dely)

    @classmethod
    def build(cls, angles, sid, sad, height, width, delx, dely=None, mapping=None, tilt=None) -> "ViewGeometry":
        """From CSV-convention angles (M, 2) through the RTK mapping."""
        theta, phi, roll = csv_to_tpr(angles, mapping, tilt)
        return cls.from_tpr(theta, phi, roll, sid, sad, height, width, delx, dely)

    def targets(self) -> torch.Tensor:
        """Detector pixel centres, (M, H*W, 3), row-major (row i, column j)."""
        dev = self.source.device
        ii = torch.arange(self.height, device=dev, dtype=torch.float32) - (self.height - 1) / 2
        jj = torch.arange(self.width, device=dev, dtype=torch.float32) - (self.width - 1) / 2
        rows, cols = torch.meshgrid(ii, jj, indexing="ij")
        rows, cols = rows.reshape(1, -1, 1), cols.reshape(1, -1, 1)
        return self.center[:, None, :] + rows * self.dely * self.v[:, None, :] + cols * self.delx * self.u[:, None, :]

    def project(self, pts: torch.Tensor) -> torch.Tensor:
        """Perspective projection of world points.  pts: (P, 3) or (M, P, 3)  ->  (M, P, 2)
        continuous pixel coordinates (row, col); pixel i covers [i - 0.5, i + 0.5)."""
        if pts.dim() == 2:
            pts = pts[None].expand(len(self), -1, -1)
        S = self.source[:, None, :]
        num = ((self.center - self.source) * self.d).sum(-1)[:, None]            # (M, 1)
        den = ((pts - S) * self.d[:, None, :]).sum(-1)                           # (M, P)
        den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
        t = num / den
        Y = S + t[..., None] * (pts - S)
        rel = Y - self.center[:, None, :]
        col = (rel * self.u[:, None, :]).sum(-1) / self.delx + (self.width - 1) / 2
        row = (rel * self.v[:, None, :]).sum(-1) / self.dely + (self.height - 1) / 2
        return torch.stack([row, col], -1)


# ----------------------------------------------------------------------------------------------
# Volumes
# ----------------------------------------------------------------------------------------------

VOLUME_EXTS = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".npy")


def read_any_volume(path: str) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """Return (spacing (x, y, z) mm, HU array (z, y, x)).  DICOM directory, NIfTI/MHA/NRRD or .npy."""
    p = Path(path)
    if p.is_dir():
        from src.utils.utils import read_volume  # DICOM series reader of the offline pipeline

        spacing, vol = read_volume(str(p))
        return tuple(float(s) for s in spacing), vol.astype(np.float32)
    if p.suffix == ".npy":
        vol = np.load(p)
        return (1.0, 1.0, 1.0), vol.astype(np.float32)
    import SimpleITK as sitk

    img = sitk.ReadImage(str(p))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)
    return tuple(float(s) for s in img.GetSpacing()), vol


def repair_blank_slices(volume: np.ndarray, air_hu: float = -950.0, tissue_hu: float = -900.0) -> Tuple[np.ndarray, int]:
    """Replace isolated all-air slices (scanner/export artefacts that render as a black line in the
    DRRs) by the mean of their neighbours.  volume: (z, y, x) HU.  Returns (volume, n_repaired)."""
    means = volume.reshape(volume.shape[0], -1).mean(1)
    n = 0
    for z in range(1, volume.shape[0] - 1):
        if means[z] < air_hu and means[z - 1] > tissue_hu and means[z + 1] > tissue_hu:
            volume[z] = 0.5 * (volume[z - 1] + volume[z + 1])
            n += 1
    return volume, n


def crop_volume(volume: np.ndarray, spacing, mode: str = "bbox", center=None, size_mm: Optional[float] = None,
                body_hu: float = -500.0, margin: float = 0.04) -> np.ndarray:
    """Crop a (z, y, x) HU volume before it is made cubic -- the paper also 'extracts appropriate crops'.
    mode 'bbox': tight bounding box of the body (HU > body_hu) plus a margin.
    mode 'cube': cube of ``size_mm`` (default: the smallest body extent) centred at ``center`` given as
                 fractions (z, y, x) of the body bounding box (default: its centre)."""
    sx, sy, sz = [float(v) for v in spacing]  # spacing is (x, y, z)
    mask = volume > body_hu
    if not mask.any():
        return volume
    zz, yy, xx = np.where(mask)
    lo = np.array([zz.min(), yy.min(), xx.min()])
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    ext = hi - lo
    if mode == "bbox":
        m = (ext * margin).astype(int)
        lo2, hi2 = np.maximum(lo - m, 0), np.minimum(hi + m, volume.shape)
    elif mode == "cube":
        c = np.array([0.5, 0.5, 0.5]) if center is None else np.asarray(center, dtype=float)
        ctr = lo + c * ext
        vox = np.array([sz, sy, sx])
        if size_mm is None:
            size_mm = float(min(ext * vox))
        half = (size_mm / 2) / vox
        lo2 = np.maximum(np.round(ctr - half).astype(int), 0)
        hi2 = np.minimum(np.round(ctr + half).astype(int), volume.shape)
    else:
        raise ValueError(mode)
    return volume[lo2[0]:hi2[0], lo2[1]:hi2[1], lo2[2]:hi2[2]]


def preprocess_hu_cube(spacing, volume: np.ndarray, dim: int = 256, remove_table: bool = False,
                       repair_slices: bool = True, crop: Optional[dict] = None) -> np.ndarray:
    """Identical to generate_data.preprocess_volume (isotropic resampling, pad to a cube, resize to dim^3),
    optionally preceded by the blank-slice repair and by a crop (see crop_volume)."""
    from src.utils.utils import interpolate_and_rescale, pad_volume_to_max_shape, resample_volume

    if repair_slices:
        volume, n_rep = repair_blank_slices(volume)
        if n_rep:
            print(f"[preprocess] repaired {n_rep} blank slice(s)", flush=True)
    if crop:
        before = volume.shape
        volume = crop_volume(volume, spacing, **crop)
        print(f"[preprocess] crop {crop}: {before} -> {volume.shape}", flush=True)
    if remove_table:
        from src.utils.utils_remove_table import apply_remove_table

        volume = apply_remove_table(volume)
    spacing, volume = resample_volume(volume, spacing)
    volume = pad_volume_to_max_shape(volume)
    _, volume = interpolate_and_rescale(volume, target_shape=(dim, dim, dim), original_spacing=spacing)
    return volume.astype(np.float32)


def list_ct_volumes(root: str) -> List[Path]:
    """CT volumes under ``root``: sub-directories (DICOM series) and/or single volume files."""
    root = Path(root)
    if root.is_file() and root.suffix in (".txt", ".json"):
        if root.suffix == ".json":
            items = json.loads(root.read_text())
            items = items if isinstance(items, list) else items.get("volumes", [])
            return [Path(x if isinstance(x, str) else x["path"]) for x in items]
        return [Path(l.strip()) for l in root.read_text().splitlines() if l.strip() and not l.startswith("#")]
    vols: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(True for _ in p.iterdir()):
            vols.append(p)
        elif p.is_file() and p.name.lower().endswith(VOLUME_EXTS):
            vols.append(p)
    return vols


class VolumeBank:
    """Holds the preprocessed HU cubes.  Cubes are cached on disk (``cache_dir/<name>.npy``,
    float16, dim^3), kept in (pinned) CPU memory as float16 and moved to the GPU on demand with
    a small LRU cache (``max_gpu_volumes``)."""

    def __init__(
        self,
        volumes: Sequence[str],
        cache_dir: Optional[str] = None,
        dim: int = 256,
        max_gpu_volumes: int = 16,
        remove_table: Sequence[str] = (),
        pin_memory: bool = True,
        verbose: bool = True,
    ):
        self.paths = [Path(v) for v in volumes]
        self.names = [p.name if p.is_dir() else p.name.split(".")[0] for p in self.paths]
        self.dim = dim
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_gpu_volumes = max_gpu_volumes
        self.remove_table = set(remove_table)
        self._cpu: List[torch.Tensor] = []
        self._gpu: Dict[int, torch.Tensor] = {}
        self._lru: List[int] = []
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(self.paths):
            cube = self._load_or_build(i, p)
            t = torch.from_numpy(cube).permute(2, 1, 0).contiguous().half()  # (z,y,x) -> (X,Y,Z)
            if pin_memory and torch.cuda.is_available():
                t = t.pin_memory()
            self._cpu.append(t)
            if verbose:
                print(f"[VolumeBank] {i + 1}/{len(self.paths)} {self.names[i]} {tuple(t.shape)}", flush=True)

    def __len__(self):
        return len(self._cpu)

    def _cache_path(self, i: int) -> Optional[Path]:
        return None if self.cache_dir is None else self.cache_dir / f"{self.names[i]}_{self.dim}.npy"

    def _load_or_build(self, i: int, path: Path) -> np.ndarray:
        cp = self._cache_path(i)
        if cp is not None and cp.exists():
            return np.load(cp).astype(np.float32)
        spacing, vol = read_any_volume(str(path))
        cube = preprocess_hu_cube(spacing, vol, self.dim, remove_table=self.names[i] in self.remove_table)
        if cp is not None:
            np.save(cp, np.clip(cube, -1000, 3000).astype(np.float16))
        return cube

    def get(self, i: int, device: torch.device) -> torch.Tensor:
        """HU cube (X, Y, Z) float16 on ``device``."""
        i = int(i)
        if device.type != "cuda":
            return self._cpu[i]
        if i in self._gpu:
            self._lru.remove(i)
            self._lru.append(i)
            return self._gpu[i]
        if len(self._gpu) >= self.max_gpu_volumes:
            old = self._lru.pop(0)
            del self._gpu[old]
        t = self._cpu[i].to(device, non_blocking=True)
        self._gpu[i] = t
        self._lru.append(i)
        return t


# ----------------------------------------------------------------------------------------------
# Rendering + correspondence
# ----------------------------------------------------------------------------------------------


def hu_sigmoid(hu: torch.Tensor, x0: float, k: float = 0.005) -> torch.Tensor:
    """sigmoid(k * (clip(HU, -1000, 3000) + 1000 - x0)) -- the two intensity models of the offline code
    (x0 = 1500 for the rendered attenuation, x0 = 850 for the correspondence weights)."""
    return torch.sigmoid(k * (hu.float().clamp(-1000, 3000) + 1000.0 - x0))


def shift_volume_x(vol: torch.Tensor, tx: int, fill: float = -1000.0) -> torch.Tensor:
    """Translate along X (first axis of the (X, Y, Z) cube) by ``tx`` voxels; vacated voxels are air.
    Mirrors scipy.ndimage.shift(volume, (0, 0, tx), cval=-1000) of the offline pipeline."""
    tx = int(round(tx))
    if tx == 0:
        return vol
    out = torch.full_like(vol, fill)
    if tx > 0:
        out[tx:] = vol[:-tx]
    else:
        out[:tx] = vol[-tx:]
    return out


class OnlineDRRRenderer(torch.nn.Module):
    """Renders DRR views and their patch-level correspondence matrices from HU cubes, on the GPU.

    forward(batch) with batch = collated OnlineDRRParams items (on the model device) returns
        views : (B, V, 1, H, W) raw line integrals  (apply ``preprocess_views`` before the model)
        corr  : (B, V(V-1)/2, N*N, N*N) soft correspondence in [0, 1]  (squeeze dim 1 for V = 2)
    """

    def __init__(
        self,
        bank: VolumeBank,
        dim: int = 256,
        n_patches: int = 16,
        delx: float = 2.0,
        n_points: int = 300,
        footprint_samples: int = 4,
        render_chunk: int = 4,
        corr_x0: float = 850.0,
        render_x0: float = 1500.0,
        sigmoid_k: float = 0.005,
        corr_reduce: str = "max",
        voxel_mm: float = 1.0,
        mapping: Optional[dict] = None,
    ):
        super().__init__()
        from diffdrr.data import read as diffdrr_read
        from diffdrr.drr import DRR
        from torchio import ScalarImage

        self.bank = bank
        self.dim, self.N = int(dim), int(n_patches)
        assert self.dim % self.N == 0, "detector size must be a multiple of the patch grid"
        self.ps = self.dim // self.N
        self.delx = float(delx)
        self.n_points = int(n_points)
        self.footprint_samples = int(footprint_samples)
        self.render_chunk = int(render_chunk)
        self.corr_x0, self.render_x0, self.k = corr_x0, render_x0, sigmoid_k
        self.corr_reduce = corr_reduce
        self.voxel_mm = float(voxel_mm)
        self.mapping = dict(RTK_MAPPING if mapping is None else mapping)
        perm = np.arange(self.N * self.N).reshape(self.N, self.N)
        if self.mapping.get("flip_rows"):
            perm = perm[::-1]
        if self.mapping.get("flip_cols"):
            perm = perm[:, ::-1]
        self.register_buffer("flip_perm", torch.from_numpy(perm.reshape(-1).copy()), persistent=False)

        # One DiffDRR module built on a dummy cube: it only provides the (shared) world<->voxel
        # affine of a centred, 1 mm isotropic dim^3 volume and the trilinear ray-caster; the
        # density is swapped per call (same trick as xvr's multi-volume trainer).
        dummy = ScalarImage(tensor=torch.zeros(1, dim, dim, dim), affine=np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0]))
        subject = diffdrr_read(dummy, orientation=None)
        self.drr = DRR(subject, sdd=2000.0, height=dim, delx=delx, renderer="trilinear", reverse_x_axis=False)
        self.drr.density = None  # free the dummy
        self.drr.mask = None

        # Coarse voxel sub-points (for footprints), in world mm, (Nv, S, 3) with Nv = N^3, S = s^3
        s = self.footprint_samples
        cs = self.dim * self.voxel_mm / self.N  # coarse voxel size in mm
        centres = (torch.arange(self.N, dtype=torch.float32) + 0.5) * cs - self.dim * self.voxel_mm / 2
        cx, cy, cz = torch.meshgrid(centres, centres, centres, indexing="ij")  # (X, Y, Z) order
        centres3 = torch.stack([cx, cy, cz], -1).reshape(-1, 3)
        off = ((torch.arange(s, dtype=torch.float32) + 0.5) / s - 0.5) * cs
        ox, oy, oz = torch.meshgrid(off, off, off, indexing="ij")
        offsets = torch.stack([ox, oy, oz], -1).reshape(-1, 3)
        self.register_buffer("voxel_subpoints", centres3[:, None, :] + offsets[None, :, :], persistent=False)

    # ----- rendering ---------------------------------------------------------------------------
    def geometry_tpr(self, theta, phi, roll, sid, sad) -> ViewGeometry:
        return ViewGeometry.from_tpr(theta, phi, roll, sid, sad, self.dim, self.dim, self.delx)

    def geometry(self, angles_csv, sid, sad, tilt=None) -> ViewGeometry:
        theta, phi, roll = csv_to_tpr(angles_csv, self.mapping, tilt)
        return self.geometry_tpr(theta, phi, roll, sid, sad)

    @torch.no_grad()
    def render(self, density: torch.Tensor, geom: ViewGeometry) -> torch.Tensor:
        """density: (X, Y, Z) float32 on device; returns (M, 1, H, W) line integrals."""
        targets = geom.targets()
        source = geom.source[:, None, :]
        outs = []
        for i in range(0, len(geom), self.render_chunk):
            img = self.drr.render(density, source[i:i + self.render_chunk], targets[i:i + self.render_chunk],
                                  n_points=self.n_points)
            outs.append(self.drr.reshape_transform(img, batch_size=img.shape[0]))
        return torch.cat(outs, 0)

    # ----- correspondence ----------------------------------------------------------------------
    @torch.no_grad()
    def footprints(self, geom: ViewGeometry) -> torch.Tensor:
        """Max-normalised patch footprints of every coarse voxel on every view: (M, Nv, N*N)."""
        M = len(geom)
        Nv, S, _ = self.voxel_subpoints.shape
        pix = geom.project(self.voxel_subpoints.reshape(-1, 3))                  # (M, Nv*S, 2)
        cell = torch.floor((pix + 0.5) / self.ps).long()                          # patch (row, col)
        valid = (cell[..., 0] >= 0) & (cell[..., 0] < self.N) & (cell[..., 1] >= 0) & (cell[..., 1] < self.N)
        idx = (cell[..., 0].clamp(0, self.N - 1) * self.N + cell[..., 1].clamp(0, self.N - 1))
        idx = idx.reshape(M, Nv, S)
        hits = torch.zeros(M, Nv, self.N * self.N, device=pix.device, dtype=torch.float32)
        hits.scatter_add_(2, idx, valid.reshape(M, Nv, S).float())
        return hits / hits.amax(dim=2, keepdim=True).clamp_min(1.0)

    @torch.no_grad()
    def correspondence(self, fp1: torch.Tensor, fp2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """C[p, q] = max_v fp1[v, p] * fp2[v, q] * w[v]   (or sum_v if corr_reduce == 'sum').
        fp*: (Nv, P), w: (Nv,)  ->  (P, P)."""
        P = fp1.shape[1]
        K = min(self.footprint_samples ** 3, P)
        v1, i1 = fp1.topk(K, dim=1)                                               # (Nv, K)
        v2, i2 = fp2.topk(K, dim=1)
        vals = (v1[:, :, None] * v2[:, None, :] * w[:, None, None]).reshape(-1)
        flat = (i1[:, :, None] * P + i2[:, None, :]).reshape(-1)
        C = torch.zeros(P * P, device=fp1.device, dtype=torch.float32)
        reduce = "amax" if self.corr_reduce == "max" else "sum"
        C.scatter_reduce_(0, flat, vals, reduce=reduce, include_self=True)
        return C.view(P, P)

    # ----- full sample -------------------------------------------------------------------------
    def _finish(self, views, fps, apply_flips):
        """Apply the orientation flips of the RTK mapping to rendered views and footprints."""
        if not apply_flips:
            return views, fps
        if self.mapping.get("flip_rows"):
            views = views.flip(-2)
        if self.mapping.get("flip_cols"):
            views = views.flip(-1)
        if self.mapping.get("flip_rows") or self.mapping.get("flip_cols"):
            fps = fps[:, :, self.flip_perm]
        return views, fps

    @torch.no_grad()
    def render_tpr(self, hu: torch.Tensor, theta, phi, roll, sid, sad, translation_x: float = 0.0, apply_flips: bool = True):
        """V views of one HU cube (X, Y, Z) from explicit camera angles (degrees, (V,) tensors).
        Returns views (V, 1, H, W), corr (V(V-1)/2, P, P), footprints (V, Nv, P), weights (Nv,)."""
        device = hu.device
        tx = self.mapping.get("translation_sign", 1.0) * float(translation_x)
        if tx != 0.0:
            hu = shift_volume_x(hu, tx)
        density = hu_sigmoid(hu, self.render_x0, self.k)
        w = F.adaptive_avg_pool3d(hu_sigmoid(hu, self.corr_x0, self.k)[None, None], self.N)[0, 0].reshape(-1)
        geom = self.geometry_tpr(theta.to(device), phi.to(device), roll.to(device), sid.to(device), sad.to(device))
        views = self.render(density, geom)
        fps = self.footprints(geom)
        views, fps = self._finish(views, fps, apply_flips)
        V = len(geom)
        corrs = [self.correspondence(fps[i], fps[j], w) for i in range(V) for j in range(i + 1, V)]
        return views, torch.stack(corrs, 0), fps, w

    @torch.no_grad()
    def render_sample(self, hu: torch.Tensor, angles, sid, sad, translation=None, tilt=None):
        """One acquisition in the CSV convention of the offline pipeline (angles (V, 2) = [a0, a1],
        translation (3,) in voxels along x), rendered in the offline orientation."""
        theta, phi, roll = csv_to_tpr(angles, self.mapping, tilt)
        tx = 0.0 if translation is None else float(translation[0])
        return self.render_tpr(hu, theta, phi, roll, sid, sad, translation_x=tx, apply_flips=True)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor], device: Optional[torch.device] = None):
        angles = batch["angles"]
        device = angles.device if device is None else device
        B, V = angles.shape[:2]
        n_pairs = V * (V - 1) // 2
        views = torch.empty(B, V, 1, self.dim, self.dim, device=device)
        corr = torch.empty(B, n_pairs, self.N * self.N, self.N * self.N, device=device)
        translation = batch.get("translation", None)
        tilt = batch.get("tilt", None)
        for b in range(B):
            hu = self.bank.get(int(batch["vol_idx"][b]), device)
            t = None if translation is None else translation[b]
            v, c, _, _ = self.render_sample(hu, angles[b], batch["sid"][b], batch["sad"][b], t, None if tilt is None else tilt[b])
            views[b], corr[b] = v, c
        return views, corr


# ----------------------------------------------------------------------------------------------
# GPU preprocessing (replacement of utils.postprocess_view + radiometric jitter)
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def preprocess_views(
    views: torch.Tensor,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    clahe_clip: float = 1.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    jitter_p: float = 0.0,
    brightness: Tuple[float, float] = (0.8, 1.2),
    contrast: Tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """views (..., 1, H, W) raw DRRs -> [0, 1] images, same recipe as the offline pipeline
    (1-99 percentile clip, min-max, CLAHE with 8x8 tiles) but on the GPU, plus the optional
    brightness/contrast jitter of train.py drawn independently per image."""
    shape = views.shape
    x = views.reshape(-1, 1, shape[-2], shape[-1]).float()
    flat = x.flatten(1)
    lo = torch.quantile(flat, percentiles[0] / 100, dim=1, keepdim=True)
    hi = torch.quantile(flat, percentiles[1] / 100, dim=1, keepdim=True)
    x = ((flat.clamp(lo, hi) - lo) / (hi - lo + 1e-8)).reshape_as(x).clamp(0, 1)
    if clahe_clip and clahe_clip > 0:
        from kornia.enhance import equalize_clahe

        x = equalize_clahe(x, clip_limit=float(clahe_clip), grid_size=tuple(clahe_grid))
    if jitter_p > 0:
        n = x.shape[0]
        apply = (torch.rand(n, device=x.device) < jitter_p).float().view(n, 1, 1, 1)
        bf = torch.empty(n, 1, 1, 1, device=x.device).uniform_(*brightness)
        cf = torch.empty(n, 1, 1, 1, device=x.device).uniform_(*contrast)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        y = ((x * bf).clamp(0, 1) - mean) * cf + mean
        x = apply * y.clamp(0, 1) + (1 - apply) * x
    return x.reshape(shape)


def apply_corr_augmentations(corr: torch.Tensor, views: torch.Tensor, augmentations, probabilities):
    """Per-sample correspondence-space augmentations of src/utils/utils.py on a batch.
    corr (B, P, P), views (B, 2, 1, H, W) -> same shapes."""
    if not augmentations or all(p <= 0 for p in probabilities):
        return corr, views
    out_c, out_v = corr.clone(), views.clone()
    for b in range(corr.shape[0]):
        c, v1, v2 = corr[b], views[b, 0, 0], views[b, 1, 0]
        for fn, p in zip(augmentations, probabilities):
            if random.random() < p:
                c, v1, v2 = fn(c, v1, v2)
        out_c[b], out_v[b, 0, 0], out_v[b, 1, 0] = c, v1, v2
    return out_c, out_v
