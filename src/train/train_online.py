"""End-to-end correspondence pretraining: CT volumes in, trained model out (no DRR dataset on disk).

    python main_online.py --config configs/config_online.yml
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.online_drr import OnlineDRRParams, OnlineDRRRenderer, VolumeBank, list_ct_volumes
from src.model.model_online import OnlineCorrespondenceModel
from src.train.train import SaveBestMetricsCallback, load_config  # noqa: F401  (re-exported)
from src.utils.utils import (
    NamedLambda,
    augment_corr_cutout_in,
    augment_corr_masked_crop,
    augment_corr_shift_pad,
    augment_corr_transformation,
    postprocess_view,
)


def split_volumes(volumes, val_fraction: float, seed: int):
    """Deterministic volume-level train/val split (a volume never appears in both)."""
    vols = list(volumes)
    rng = random.Random(seed)
    rng.shuffle(vols)
    n_val = max(1, int(round(len(vols) * val_fraction))) if val_fraction > 0 and len(vols) > 1 else 0
    return sorted(vols[n_val:]), sorted(vols[:n_val])


def build_model(config):
    m, t = config["model"], config["model"]["transformer"]
    tr = config["training"]
    return OnlineCorrespondenceModel(
        online_cfg={"preprocess": config.get("preprocess", {}), "radiometric": tr["augmentation"]["radiometric"]},
        backbone=m["backbone"],
        freeze_backbone=m["freeze_backbone"],
        optimizer=tr["optimizer"],
        scheduler=tr["scheduler"],
        warmup_epochs=tr["warmup_epochs"],
        loss_mse_weight=tr["losses_weights"]["mse"],
        lr=tr["learning_rate"],
        transformer=t["transformer"],
        pos_enc=t["positional_encoding"],
        transformer_d_model=t["transformer_d_model"],
        transformer_num_heads=t["transformer_num_heads"],
        transformer_hidden_dim=t["transformer_hidden_dim"],
        transformer_num_layers=t["transformer_num_layers"],
        transformer_attention_bias=t["transformer_attention_bias"],
        transformer_message_pass=t["transformer_message_pass"],
        transformer_mlp_input=t["transformer_mlp_input"],
        transformer_mlp_bias=t["transformer_mlp_bias"],
        transformer_mlp_activation=t["transformer_mlp_activation"],
        transformer_norm_type=t["transformer_norm_type"],
        transformer_mlp_layer_norm_input=t["transformer_mlp_layer_norm_input"],
        transformer_mlp_layer_norm_output=t["transformer_mlp_layer_norm_output"],
        transformer_attention_layer_norm_input=t["transformer_attention_layer_norm_input"],
        transformer_attention_layer_norm_output=t["transformer_attention_layer_norm_output"],
        transformer_norm_after_add=t["transformer_norm_after_add"],
    )


def train_model_online(config):
    data, tr, rc = config["data"], config["training"], config.get("render", {})
    pl.seed_everything(tr.get("seed", 42), workers=True)

    # ---- volumes ---------------------------------------------------------------------------
    volumes = list_ct_volumes(data["ct_dir"])
    if data.get("max_volumes"):
        volumes = volumes[: int(data["max_volumes"])]
    if not volumes:
        raise FileNotFoundError(f"no CT volumes found under {data['ct_dir']}")
    train_vols, val_vols = split_volumes(volumes, float(data.get("val_fraction", 0.1)), tr.get("seed", 42))
    print(f"[online] {len(train_vols)} train / {len(val_vols)} val volumes")
    bank_kwargs = dict(cache_dir=data.get("cache_dir"), dim=int(rc.get("dim", 256)),
                       max_gpu_volumes=int(rc.get("max_gpu_volumes", 16)),
                       remove_table=data.get("remove_table", []))
    train_bank = VolumeBank([str(v) for v in train_vols], **bank_kwargs)
    val_bank = VolumeBank([str(v) for v in val_vols], **bank_kwargs) if val_vols else train_bank

    renderer_kwargs = dict(
        dim=int(rc.get("dim", 256)), n_patches=int(rc.get("n_patches", 16)), delx=float(rc.get("delx", 2.0)),
        n_points=int(rc.get("n_points", 300)), footprint_samples=int(rc.get("footprint_samples", 2)),
        render_chunk=int(rc.get("render_chunk", 4)), corr_reduce=rc.get("corr_reduce", "max"),
    )
    renderer = OnlineDRRRenderer(train_bank, **renderer_kwargs)

    # ---- parameter datasets ------------------------------------------------------------------
    n_views = int(data.get("n_views", 2))
    train_ds = OnlineDRRParams(len(train_bank), int(data["samples_per_epoch"]), int(data.get("pairs_per_volume", 4)),
                               n_views=n_views, seed=tr.get("seed", 42), translate=bool(data.get("translate", True)),
                               tilt_std=float(data.get("tilt_std", 0.0)))
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0, drop_last=True)

    if data.get("val_h5"):  # optional: validate on an *offline* dataset generated with the original pipeline
        from src.data.dataloader import DRRDataset

        transform_view = transforms.Compose([NamedLambda(lambda img: postprocess_view(img), name="postprocess_view"),
                                             transforms.Resize(size=(256, 256))])
        val_ds = DRRDataset(data["val_h5"], transform_view=transform_view, transform_corr=None,
                            data_type=data.get("val_h5_split", "val"))
        val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=int(data.get("val_workers", 8)))
    else:
        val_ds = OnlineDRRParams(len(val_bank), int(data.get("val_samples", 512)), int(data.get("pairs_per_volume", 4)),
                                 n_views=n_views, seed=tr.get("seed", 42) + 1000, fixed=True,
                                 translate=bool(data.get("translate", True)), tilt_std=float(data.get("tilt_std", 0.0)))
        val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=0)

    # ---- model ---------------------------------------------------------------------------------
    model = build_model(config)
    aug = tr["augmentation"]
    model.attach_renderer(
        renderer,
        corr_augmentations=[augment_corr_transformation, augment_corr_masked_crop, augment_corr_cutout_in, augment_corr_shift_pad],
        corr_probabilities=[aug.get("geometric", 0.0), aug.get("masked_crop", 0.0), aug.get("cutout", 0.0), aug.get("shift", 0.0)],
    )
    model.val_bank = val_bank  # keep a reference (the renderer swaps banks during validation)

    class SwapBank(pl.Callback):
        """Validation renders from the held-out volumes."""

        def on_validation_start(self, trainer, pl_module):
            pl_module.renderer.bank = val_bank

        def on_validation_end(self, trainer, pl_module):
            pl_module.renderer.bank = train_bank

    # ---- trainer -------------------------------------------------------------------------------
    output_dir = config.get("output_dir", "output")
    save_path = os.path.join(output_dir, tr["name"])
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_ap", dirpath=save_path, filename=tr["name"] + "-{epoch:02d}-{val_ap:.4f}",
                                          save_last=True, save_top_k=1, mode="max")
    metrics_callback = SaveBestMetricsCallback(filepath=os.path.join(output_dir, "results_online.csv"), config=config)
    callbacks = [checkpoint_callback, metrics_callback, SwapBank()]
    from lightning.pytorch.loggers import CSVLogger

    logger = CSVLogger(save_dir=save_path, name="lightning_logs")  # metrics.csv next to the checkpoints
    if tr.get("wandb_project"):
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=tr["wandb_project"], job_type="train-online")
    trainer = pl.Trainer(
        callbacks=callbacks, log_every_n_steps=10, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        max_epochs=tr["max_epochs"], logger=logger, limit_train_batches=tr.get("limit_train_batches", 1.0),
        limit_val_batches=tr.get("limit_val_batches", 1.0), precision=tr.get("precision", "32-true"),
    )
    trainer.fit(model, train_loader, val_loader)
    with open(os.path.join(save_path, tr["name"] + ".yml"), "w") as f:
        yaml.dump(config, f)
    return trainer
