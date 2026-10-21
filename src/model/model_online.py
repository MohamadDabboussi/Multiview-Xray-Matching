"""CorrespondenceModel variant whose batches are *acquisition parameters*: the DRR views and the
correspondence matrices are rendered on the GPU inside the training/validation step."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from src.data.online_drr import OnlineDRRRenderer, apply_corr_augmentations, preprocess_views
from src.model.model import CorrespondenceModel


class OnlineCorrespondenceModel(CorrespondenceModel):
    def __init__(self, online_cfg: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.online_cfg = dict(online_cfg or {})
        self.renderer: Optional[OnlineDRRRenderer] = None
        self._corr_augs = ([], [])
        self.last_batch_images = None  # (corr, views) of the last training batch, for logging

    # --------------------------------------------------------------------------------------
    def attach_renderer(self, renderer: OnlineDRRRenderer, corr_augmentations=None, corr_probabilities=None):
        self.renderer = renderer  # nn.Module -> follows .to(device)
        self._corr_augs = (list(corr_augmentations or []), list(corr_probabilities or []))

    def materialize(self, batch, train: bool):
        """dict of acquisition parameters -> (corr_soft (B,P,P), views (B,2,1,H,W)) in [0,1].
        Offline batches (tuple from DRRDataset) pass through unchanged."""
        if not isinstance(batch, dict):
            return batch
        assert self.renderer is not None, "call attach_renderer() first"
        pre = self.online_cfg.get("preprocess", {})
        views, corr = self.renderer(batch)
        corr = corr[:, 0]  # dual-view
        views = preprocess_views(
            views,
            percentiles=tuple(pre.get("clip_percentiles", (1.0, 99.0))),
            clahe_clip=float(pre.get("clahe_clip", 1.0)),
            clahe_grid=tuple(pre.get("clahe_grid", (8, 8))),
            jitter_p=float(self.online_cfg.get("radiometric", 0.0)) if train else 0.0,
        )
        if train:
            corr, views = apply_corr_augmentations(corr, views, *self._corr_augs)
        return corr.float(), views.float()

    # --------------------------------------------------------------------------------------
    def on_train_epoch_start(self):
        ds = getattr(self.trainer.train_dataloader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        corr, views = self.materialize(batch, train=True)
        if batch_idx == 0:
            self.last_batch_images = (corr[:4].detach().cpu(), views[:4].detach().cpu())
        return super().training_step((corr, views), batch_idx)

    def validation_step(self, batch, batch_idx):
        corr, views = self.materialize(batch, train=False)
        return super().validation_step((corr, views), batch_idx)
