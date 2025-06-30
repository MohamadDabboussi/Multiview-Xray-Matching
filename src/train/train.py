import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.data.dataloader import DRRDataset
from src.model.model import CorrespondenceModel
from src.utils.utils import (
    postprocess_view,
    NamedLambda,
    augment_corr_transformation,
    augment_corr_masked_crop,
    augment_corr_cutout_in,
    augment_corr_shift_pad,
)


# Load the configuration from the YAML file
def load_config(config_path="config.yml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class SaveBestMetricsCallback(pl.Callback):
    def __init__(self, filepath: str, config: dict):
        super().__init__()
        self.config = config
        self.filepath = filepath

    def on_fit_start(self, trainer, pl_module):
        # Inject the config and csv_filepath into the model
        pl_module.config = self.config
        pl_module.csv_filepath = self.filepath


def train_model(config):
    # Set the device
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Define transforms
    transform_view = transforms.Compose(
        [
            NamedLambda(lambda img: postprocess_view(img), name="postprocess_view"),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))],
                p=config["training"]["augmentation"]["radiometric"],
            ),
            transforms.Resize(size=(256, 256)),
        ]
    )
    transform_corr = (
        [
            augment_corr_transformation,
            augment_corr_masked_crop,
            augment_corr_cutout_in,
            augment_corr_shift_pad,
        ],
        [config["training"]["augmentation"]["geometric"],
         config["training"]["augmentation"]["masked_crop"],
         config["training"]["augmentation"]["cutout"],
         config["training"]["augmentation"]["shift"]],
    )

    # Model Setup
    model = CorrespondenceModel(
        backbone=config["model"]["backbone"],
        freeze_backbone=config["model"]["freeze_backbone"],
        optimizer=config["training"]["optimizer"],
        scheduler=config["training"]["scheduler"],
        warmup_epochs=config["training"]["warmup_epochs"],
        loss_mse_weight=config["training"]["losses_weights"]["mse"],
        lr=config["training"]["learning_rate"],
        transformer=config["model"]["transformer"]["transformer"],
        pos_enc=config["model"]["transformer"]["positional_encoding"],
        transformer_d_model=config["model"]["transformer"]["transformer_d_model"],
        transformer_num_heads=config["model"]["transformer"]["transformer_num_heads"],
        transformer_hidden_dim=config["model"]["transformer"]["transformer_hidden_dim"],
        transformer_num_layers=config["model"]["transformer"]["transformer_num_layers"],
        transformer_attention_bias=config["model"]["transformer"][
            "transformer_attention_bias"
        ],
        transformer_message_pass=config["model"]["transformer"][
            "transformer_message_pass"
        ],
        transformer_mlp_input=config["model"]["transformer"]["transformer_mlp_input"],
        transformer_mlp_bias=config["model"]["transformer"]["transformer_mlp_bias"],
        transformer_mlp_activation=config["model"]["transformer"][
            "transformer_mlp_activation"
        ],
        transformer_norm_type=config["model"]["transformer"]["transformer_norm_type"],
        transformer_mlp_layer_norm_input=config["model"]["transformer"][
            "transformer_mlp_layer_norm_input"
        ],
        transformer_mlp_layer_norm_output=config["model"]["transformer"][
            "transformer_mlp_layer_norm_output"
        ],
        transformer_attention_layer_norm_input=config["model"]["transformer"][
            "transformer_attention_layer_norm_input"
        ],
        transformer_attention_layer_norm_output=config["model"]["transformer"][
            "transformer_attention_layer_norm_output"
        ],
        transformer_norm_after_add=config["model"]["transformer"][
            "transformer_norm_after_add"
        ],
    )

    model = model.to(device)

    # Initialize datasets
    train_dataset = DRRDataset(
        config["data"]["data_path"], transform_view=transform_view, transform_corr=transform_corr, data_type="train"
    )

    val_dataset = DRRDataset(
        config["data"]["data_path"], transform_view=transform_view, transform_corr=transform_corr, data_type="val"
    )

    # DataLoader setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )

    # Setup Checkpointing and Logging
    output_dir = "output"
    save_path = f"{output_dir}/{config['training']['name']}"
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ap",
        dirpath=save_path,
        filename=config["training"]["name"] + "-{epoch:02d}-{val_ap:.4f}",
        save_last=True,
        save_top_k=1,
        mode="max",
    )
    metrics_callback = SaveBestMetricsCallback(filepath="output/results.csv", config=config)
    if config["training"].get("wandb_project") is not None:
        wandb_logger = WandbLogger(
            project=config["training"]["wandb_project"], job_type="train"
        )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, metrics_callback],
            log_every_n_steps=10,
            accelerator="gpu",
            devices=1,
            max_epochs=config["training"]["max_epochs"],
            logger=wandb_logger,
        )
    else:
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback, metrics_callback],
            log_every_n_steps=10,
            accelerator="gpu",
            devices=1,
            max_epochs=config["training"]["max_epochs"],
        )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the configuration file as yml
    with open(f"{save_path}/{config['training']['name']}.yml", "w") as file:
        yaml.dump(config, file)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the network")
    config_path = parser.add_argument(
        "--config",
        type=str,
        help="Value for config path",
        default="configs/config.yml",
    )

    config = load_config("config.yml")
    train_model(config)
