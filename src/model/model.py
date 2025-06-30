import os
import csv
import sys
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops.einops import rearrange
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim import lr_scheduler
from torchmetrics import AveragePrecision, Precision, Recall
from torchvision import models
from transformers import ViTImageProcessor, ViTModel

sys.path.append("../../")
from src.model.transformer import TransformerBuilder
from src.utils.eval import evaluate


class CorrespondenceHead(nn.Module):
    def __init__(self):
        super(CorrespondenceHead, self).__init__()

    def forward(self, features_1, features_2):
        view1_embeddings = F.normalize(features_1, p=2, dim=-1)
        view2_embeddings = F.normalize(features_2, p=2, dim=-1)
        correspondence_matrix = torch.matmul(
            view1_embeddings, view2_embeddings.transpose(-1, -2)
        )
        correspondence_matrix = torch.pow(correspondence_matrix, 2)
        return correspondence_matrix

class CorrespondenceModel(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet",
        freeze_backbone=False,
        optimizer="adam",
        scheduler="cosine",
        warmup_epochs=0,
        loss_mse_weight=1.0,
        lr=0.0001,
        transformer=True,
        dropout=0.1,
        pos_enc=None,
        transformer_d_model=512,
        transformer_num_heads=8,
        transformer_hidden_dim=1024,
        transformer_num_layers=8,
        transformer_layers="self_cross_alternate",
        transformer_attention_bias=False,
        transformer_mlp_bias=True,
        transformer_norm_type="layer",
        transformer_split_self_cross=False,
        transformer_parallel_self_cross=False,
        transformer_message_pass=False,
        transformer_mlp_input="last",
        transformer_mlp_activation="gelu",
        transformer_mlp_layer_norm_input=False,
        transformer_mlp_layer_norm_output=True,
        transformer_attention_layer_norm_input=False,
        transformer_attention_layer_norm_output=True,
        transformer_norm_after_add=True,
    ):
        super(CorrespondenceModel, self).__init__()
        self.save_hyperparameters()

        self.criterion = nn.MSELoss()  # nn.L1Loss(reduction='mean')
        self.training_step_loss = []
        self.validation_step_loss = []
        self.validation_mse_loss = []
        self.mse_weight = loss_mse_weight
        self.backbone_name = backbone
        
        # metrics
        self.ap_metric = AveragePrecision(task="binary")
        self.precision_metric = Precision(task="binary", threshold=0.2)
        self.recall_metric = Recall(task="binary", threshold=0.2)
        self.best_val_metrics = None
        self.best_epoch = 0

        # Optimization
        self.lr = lr
        self.optim = optimizer
        self.sched = scheduler
        self.warmup_epochs = warmup_epochs

        # model
        if backbone == "resnet":
            self.backbone = models.resnet50(weights="ResNet50_Weights.DEFAULT")
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
            self.fc_ = nn.Sequential(
                nn.Conv2d(1024, transformer_d_model, kernel_size=1, stride=1),
            )
        elif backbone == "dino":
            self.backbone_processor = ViTImageProcessor.from_pretrained(
                "facebook/dino-vitb16"
            )
            self.backbone_processor.size["height"] = 256
            self.backbone_processor.size["width"] = 256
            self.backbone = ViTModel.from_pretrained("facebook/dino-vitb16")
            self.fc_ = nn.Sequential(nn.Linear(768, transformer_d_model))
        else:
            raise ValueError("Invalid backbone name: {}".format(backbone))

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        if transformer:
            self.transformer = (
                TransformerBuilder()
                .set_d_model(transformer_d_model)
                .set_num_heads(transformer_num_heads)
                .set_hidden_dim(transformer_hidden_dim)
                .set_num_layers(transformer_num_layers)
                .set_dropout(dropout)
                .set_pos_encoding(pos_enc)
                .set_transformer_layers(transformer_layers)
                .set_transformer_norm_type(transformer_norm_type)
                .set_transformer_split_self_cross(transformer_split_self_cross)
                .set_transformer_parallel_self_cross(transformer_parallel_self_cross)
                .set_transformer_message_pass(transformer_message_pass)
                .set_transformer_mlp_input(transformer_mlp_input)
                .set_transformer_mlp_bias(transformer_mlp_bias)
                .set_transformer_mlp_activation(transformer_mlp_activation)
                .set_mlp_layer_norm_input(transformer_mlp_layer_norm_input)
                .set_mlp_layer_norm_output(transformer_mlp_layer_norm_output)
                .set_transformer_attention_bias(transformer_attention_bias)
                .set_attention_layer_norm_input(transformer_attention_layer_norm_input)
                .set_attention_layer_norm_output(
                    transformer_attention_layer_norm_output
                )
                .set_norm_after_add(transformer_norm_after_add)
                .build()
            )
        else:
            self.projection = nn.Linear(512, 512)
        self.correspondence_head = CorrespondenceHead()

    def forward(self, x_1, x_2):
        # VitModel
        device = x_1.device
        if self.backbone_name == "dino":
            x_1 = self.backbone_processor(images=x_1, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            fts_1 = self.backbone(x_1, interpolate_pos_encoding=True)
            fts_1 = fts_1.last_hidden_state[:, 1:, :]
            fts_1 = self.fc_(fts_1)
            x_2 = self.backbone_processor(images=x_2, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            fts_2 = self.backbone(x_2, interpolate_pos_encoding=True)
            fts_2 = fts_2.last_hidden_state[:, 1:, :]
            fts_2 = self.fc_(fts_2)

        # CNN
        elif self.backbone_name == "resnet":
            fts_1 = self.backbone(x_1)
            fts_1 = self.fc_(fts_1)
            fts_2 = self.backbone(x_2)
            fts_2 = self.fc_(fts_2)
            fts_1 = rearrange(fts_1, "n c h w -> n (h w) c")
            fts_2 = rearrange(fts_2, "n c h w -> n (h w) c")

        else:
            raise ValueError("Invalid backbone name: {}".format(self.backbone_name))

        if self.transformer is not None:
            features_1, features_2 = self.transformer(fts_1.clone(), fts_2.clone())
        else:
            # features_1, features_2 = fts_1.clone(), fts_2.clone()
            features_1, features_2 = self.projection(fts_1), self.projection(fts_2)
        correspondence_matrix = self.correspondence_head(features_1, features_2)
        return {
            "correspondence_matrix": correspondence_matrix,
        }

    def training_step(self, batch, batch_idx):
        corr_soft, views = batch
        view_1, view_2 = views[:, 0, ...], views[:, 1, ...]
        target = corr_soft

        view1_rgb = torch.cat([view_1, view_1, view_1], dim=1).float()
        view2_rgb = torch.cat([view_2, view_2, view_2], dim=1).float()
        output_data = self(view1_rgb, view2_rgb)
        output = output_data["correspondence_matrix"]

        # loss
        mse = self.criterion(output.float(), target.float())
        loss = mse * self.mse_weight
        self.training_step_loss.append(loss)
        return {"loss": loss}

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_loss).mean()
        self.training_step_loss.clear()
        self.log("train_loss", epoch_average, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        corr_soft, views = batch
        view_1, view_2 = views[:, 0, ...], views[:, 1, ...]
        target = corr_soft

        view1_rgb = torch.cat([view_1, view_1, view_1], dim=1).float()
        view2_rgb = torch.cat([view_2, view_2, view_2], dim=1).float()
        output_data = self(view1_rgb, view2_rgb)
        output = output_data["correspondence_matrix"]

        mse = self.criterion(output.float(), target.float())
        loss = mse * self.mse_weight

        self.validation_step_loss.append(loss)
        self.validation_mse_loss.append(mse)

        output_flatten = output.flatten()
        target_flatten = target.flatten()

        self.ap_metric.update(output_flatten.cpu(), (target_flatten > 0.2).cpu().int())
        self.precision_metric.update(
            output_flatten.cpu(), (target_flatten > 0.2).cpu().int()
        )
        self.recall_metric.update(
            output_flatten.cpu(), (target_flatten > 0.2).cpu().int()
        )

        return {"val_loss": mse}

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_loss).mean()
        epoch_average_mse = torch.stack(self.validation_mse_loss).mean()

        self.validation_step_loss.clear()
        self.validation_mse_loss.clear()

        self.log("Validation Loss", epoch_average)
        self.log("Validation MSE", epoch_average_mse)

        ap_score = self.ap_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()

        self.log("val_ap", ap_score, prog_bar=True)
        self.log("Validation AP Score", ap_score)
        self.log("Validation Precision", precision)
        self.log("Validation Recall", recall)

        # Reset the metric states for the next epoch
        self.ap_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

        # save metrics in csv
        if (
            self.best_val_metrics is None
            or ap_score > self.best_val_metrics["Validation AP Score"]
        ):
            self.best_val_metrics = {
                "Validation Loss": epoch_average.item(),
                "Validation MSE": epoch_average_mse.item(),
                "Validation Precision": precision.item(),
                "Validation Recall": recall.item(),
                "Validation AP Score": ap_score.item(),
            }
            self.best_epoch = self.current_epoch

            self._update_csv_with_best_metrics()

        return {"val_ap": ap_score}

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        elif self.optim == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        if self.sched == "step":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        elif self.sched == "cosine":
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.trainer.max_epochs,
                cycle_decay=0.66,
                lr_min=1e-6,
                warmup_t=self.warmup_epochs,
                warmup_lr_init=1e-6,
                warmup_prefix=True,
                cycle_limit=1,
                t_in_epochs=True,
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def _update_csv_with_best_metrics(self):
        fieldnames = [
            "name",
            "type",
            "backbone",
            "transformer",
            "message_pass",
            "positional_encoding",
            "epoch",
            "Validation Loss",
            "Validation MSE",
            "Validation Precision",
            "Validation Recall",
            "Validation ROC AUC",
            "Validation AP Score",
        ]

        new_row = {
            "name": self.config["training"]["name"],
            "type": self.config["training"]["type"],
            "backbone": self.config["model"]["backbone"],
            "transformer": self.config["model"]["transformer"]["transformer"],
            "message_pass": self.config["model"]["transformer"][
                "transformer_message_pass"
            ],
            "positional_encoding": self.config["model"]["transformer"][
                "positional_encoding"
            ],
            "epoch": self.best_epoch,
            "Validation Loss": self.best_val_metrics["Validation Loss"],
            "Validation MSE": self.best_val_metrics["Validation MSE"],
            "Validation Precision": self.best_val_metrics["Validation Precision"],
            "Validation Recall": self.best_val_metrics["Validation Recall"],
            "Validation ROC AUC": 0.0,
            "Validation AP Score": self.best_val_metrics["Validation AP Score"],
        }

        # Load existing rows if file exists.
        rows = []
        if os.path.isfile(self.csv_filepath):
            with open(self.csv_filepath, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    rows.append(row)

        # Look for an existing row for this training run.
        run_found = False
        for idx, row in enumerate(rows):
            if row["name"] == self.config["training"]["name"]:
                rows[idx] = new_row  # Update row
                run_found = True
                break

        if not run_found:
            rows.append(new_row)

        # Write all rows back to the CSV file.
        with open(self.csv_filepath, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
