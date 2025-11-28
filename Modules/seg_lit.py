
import math
from typing import Dict, Any

import torch
import torch.nn as nn
import lightning as pl
import timm

from lightning.pytorch.loggers import WandbLogger
import wandb

import torchvision.utils as vutils
from torchvision.utils import make_grid

from Models.models import CXRModel
# --------------------
# Dice helper
# --------------------
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice loss for binary segmentation.
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W] with {0,1}
    """
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)

    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class LungSegmentationModule(pl.LightningModule):
    """
    Lung segmentation on CXR using a frozen ViT/MAE encoder and a trainable decoder head.

    Assumptions:
    - self.backbone is a ViT-like model (timm or MAE) with:
      patch_embed, cls_token, pos_embed, blocks, norm
    - img_size and patch_size are consistent with the backbone.
    """

    def __init__(
        self,
        mode: str,
        backbone_name: str,
        model_checkpoints: str = None,
        img_size: int = 224,
        patch_size: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        betas=(0.9, 0.999),
        unfreeze_backbone: bool = False,
        visualize_n_batches: int = 8,   # how many val batches to visualize
        visualize_n_samples: int = 4,   # how many samples per batch
    ):
        super().__init__()
        # don't try to save the checkpoint object; path is fine
        self.save_hyperparameters(ignore=[])

        self.mode = mode
        self.backbone_name = backbone_name

        # backbone + head
        self.model = CXRModel(
            num_classes=0,
            mode=self.mode,
            backbone_name=self.backbone_name,
            model_checkpoints=model_checkpoints,
            unfreeze_backbone=unfreeze_backbone,
            task="seg"
        )


        self.bce_loss = nn.BCEWithLogitsLoss()

        # for visualization
        self.visualize_n_batches = visualize_n_batches
        self.visualize_n_samples = visualize_n_samples

        # store mean/std to unnormalize for visualization (ImageNet)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # --------------------
    # Loss & metrics
    # --------------------
    def _compute_loss_and_metrics(self, logits, masks) -> Dict[str, Any]:
        bce = self.bce_loss(logits, masks)
        dice = soft_dice_loss(logits, masks)
        loss = self.hparams.bce_weight * bce + self.hparams.dice_weight * dice

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            intersection = (preds * masks).sum(dim=(0, 2, 3))
            union = preds.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
            dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_score = dice_score.mean()

            inter = (preds * masks).sum(dim=(0, 2, 3))
            union_iou = preds.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3)) - inter
            iou = (inter + 1e-6) / (union_iou + 1e-6)
            iou = iou.mean()

        return {
            "loss": loss,
            "bce": bce,
            "dice_loss": dice,
            "dice_score": dice_score,
            "iou": iou,
        }

    # --------------------
    # Visualization helper (val) – first N batches
    # --------------------
    @torch.no_grad()
    def _log_val_visualizations(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        tag: str = "val/segmentation"
    ):
        """
        Log a grid for segmentation sanity check:
        row 1: input images
        row 2: GT overlay
        row 3: Pred overlay

        imgs:  [B,3,H,W] normalized (ImageNet stats)
        masks: [B,1,H,W] {0,1}
        logits:[B,1,H,W]
        """
        if imgs.ndim != 4 or imgs.size(1) != 3:
            return  # only handle RGB 4D tensors

        # how many images per grid
        n = int(getattr(self.hparams, "log_max_images", 8))
        n = max(1, min(n, imgs.size(0)))

        # --- denorm toward [0,1] using stored mean/std buffers ---
        def denorm(x: torch.Tensor) -> torch.Tensor:
            # self.mean/self.std were registered as buffers: [1,3,1,1]
            return (x * self.std + self.mean).clamp(0, 1)

        imgs_vis = denorm(imgs[:n])                  # [n,3,H,W]
        masks_vis = masks[:n]                        # [n,1,H,W]
        preds = torch.sigmoid(logits[:n])            # [n,1,H,W]
        preds_bin = (preds > 0.5).float()            # [n,1,H,W]

        # make overlays: GT (greenish), Pred (reddish)
        masks_3c = masks_vis.repeat(1, 3, 1, 1)      # [n,3,H,W]
        preds_3c = preds_bin.repeat(1, 3, 1, 1)      # [n,3,H,W]

        overlay_gt = (imgs_vis * 0.7 + masks_3c * 0.3).clamp(0, 1)
        overlay_pred = (imgs_vis * 0.7 + preds_3c * 0.3).clamp(0, 1)

        # stack rows: [input | gt overlay | pred overlay] as batch
        grid_src = torch.cat([imgs_vis, overlay_gt, overlay_pred], dim=0)
        grid_src = grid_src.detach().float().cpu()

        # make grid: n images per row
        grid = vutils.make_grid(grid_src, nrow=n, padding=2, pad_value=0.5)

        logger = getattr(self, "logger", None)
        if logger is None or getattr(logger, "experiment", None) is None:
            return

        exp = logger.experiment
        step = int(getattr(self, "global_step", 0))

        # ----- TensorBoard -----
        if hasattr(exp, "add_image"):
            exp.add_image(tag, grid, global_step=step)
            return

        # ----- Weights & Biases -----
        if hasattr(exp, "log"):
            try:
                import numpy as np
                np_img = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
                import wandb
                exp.log({tag: wandb.Image(np_img)}, step=step)
            except Exception:
                # don't crash training if logging fails
                pass

    # --------------------
    # PL steps
    # --------------------
    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self.model.seg_forward(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("train/loss", metrics["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.size(0))
        self.log("train/dice", metrics["dice_score"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("train/iou", metrics["iou"], prog_bar=False, on_step=False, on_epoch=True, batch_size=imgs.size(0))

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self.model.seg_forward(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("val/loss", metrics["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("val/dice", metrics["dice_score"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("val/iou", metrics["iou"], prog_bar=False, on_step=False, on_epoch=True, batch_size=imgs.size(0))

        # visualize only first 8 batches in validation
        if batch_idx < 8:
            self._log_val_visualizations(imgs, masks, logits, tag="val/segmentation")
            
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self.model.seg_forward(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("test/loss", metrics["loss"], prog_bar=True, batch_size=imgs.size(0))
        self.log("test/dice", metrics["dice_score"], prog_bar=True, batch_size=imgs.size(0))
        self.log("test/iou", metrics["iou"], prog_bar=True, batch_size=imgs.size(0))

        return metrics["loss"]

    # --------------------
    # Optimizer + warmup + cosine
    # --------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
        )

        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs

        def lr_lambda(epoch: int):
            # linear warmup
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # cosine decay
            total = max(max_epochs - warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / total
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }