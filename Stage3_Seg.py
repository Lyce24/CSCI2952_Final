# test Seg Downstream Task
import os
from typing import Optional, Callable, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import lightning as pl
import random

from Modules.lightning_modules import MAELightningModule

class RandomFlipRotate:
    """Random horizontal flip + small rotation, applied to image and mask together."""
    def __init__(self, p_flip: float = 0.5, degrees: float = 10.0):
        self.p_flip = p_flip
        self.degrees = degrees

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Horizontal flip
        if random.random() < self.p_flip:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # Random rotation in [-degrees, degrees]
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle, fill=0)
        mask = F.rotate(mask, angle, fill=0)

        return img, mask


class CXLSegLungDataset(Dataset):
    """
    CXLSeg segmentation dataset: returns (image, mask).

    CSV format:
        dicom_id    subject_id  study_id    split
    Paths:
        image: root / subject_id / study_id / dicom_id.jpg
        mask : root / subject_id / study_id / dicom_id-mask.jpg
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_root: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        self.image_size = image_size

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((image_size, image_size))
        self.img_normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.df)

    def _get_paths(self, idx: int) -> Tuple[str, str]:
        row = self.df.iloc[idx]
        dicom_id = str(row["dicom_id"])
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])

        major_dir = "p" + subject_id[:2]
        subject_id = "p" + subject_id
        study_id = "s" + study_id

        rel_dir = os.path.join(major_dir, subject_id, str(study_id))
        img_filename = dicom_id + ".jpg"
        mask_filename = dicom_id + "-mask.jpg"

        img_path = os.path.join(self.images_root, rel_dir, img_filename)
        mask_path = os.path.join(self.images_root, rel_dir, mask_filename)

        return img_path, mask_path

    def __getitem__(self, idx: int):
        img_path, mask_path = self._get_paths(idx)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        # Paired aug
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # To tensor
        img = self.to_tensor(img)          # [3, H, W], 0–1
        mask = self.to_tensor(mask)        # [1, H, W], 0–1
        mask = (mask > 0.5).float()        # binarize

        img = self.img_normalize(img)

        return img, mask


class CXLSegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        split_csv: str = "../../scratch/physionet.org/files/chest-x-ray-segmentation/1.0.0/CXLSeg-split.csv",
        images_root: str = "../../scratch/physionet.org/files/chest-x-ray-segmentation/1.0.0/files/",
        batch_size: int = 16,
        num_workers: int = 8,
        image_size: int = 224,
    ):
        super().__init__()
        self.split_csv = split_csv
        self.images_root = images_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.split_csv, sep="\t" if self.split_csv.endswith(".tsv") else ",")

        # Ensure we have the expected columns
        required_cols = {"dicom_id", "subject_id", "study_id", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        self.train_df = df[df["split"] == "train"].copy()
        self.val_df   = df[df["split"] == "validate"].copy()
        self.test_df  = df[df["split"] == "test"].copy()

        train_transform = RandomFlipRotate(p_flip=0.5, degrees=10.0)
        val_transform = None

        self.train_dataset = CXLSegLungDataset(
            self.train_df,
            images_root=self.images_root,
            transform=train_transform,
            image_size=self.image_size,
        )

        self.val_dataset = CXLSegLungDataset(
            self.val_df,
            images_root=self.images_root,
            transform=val_transform,
            image_size=self.image_size,
        )

        self.test_dataset = CXLSegLungDataset(
            self.test_df,
            images_root=self.images_root,
            transform=val_transform,
            image_size=self.image_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


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


class ViTLungSegmentationModule(pl.LightningModule):
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
        visualize_n_batches: int = 8,   # how many val batches to visualize
        visualize_n_samples: int = 4,   # how many samples per batch
    ):
        super().__init__()
        # don't try to save the checkpoint object; path is fine
        self.save_hyperparameters(ignore=[])

        self.mode = mode
        self.backbone_name = backbone_name

        # --------------------
        # Build backbone
        # --------------------
        if mode == "imagenet":
            # timm ViT
            backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,   # no classifier head
            )
        else:
            # MAE Lightning module -> extract underlying MAE/ViT
            mae_module = MAELightningModule.load_from_checkpoint(
                model_checkpoints,
                strict=False,
            )
            # IMPORTANT: change this line to match your MAE module
            # e.g. backbone = mae_module.mae or mae_module.model or mae_module.encoder
            backbone = mae_module.model  # <-- ADJUST THIS TO YOUR IMPLEMENTATION

        # Remove classifier head if any (safety for timm)
        if hasattr(backbone, "reset_classifier"):
            backbone.reset_classifier(0)
        elif hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        self.backbone = backbone

        # --------------------
        # Freeze backbone
        # --------------------
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # --------------------
        # Patch grid
        # --------------------
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches_per_side = img_size // patch_size  # 224//16 = 14
        self.h = num_patches_per_side
        self.w = num_patches_per_side

        # --------------------
        # Embedding dim
        # --------------------
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            if backbone_name == "vit_huge_patch14_224":
                in_dim = 1280
            elif backbone_name == "vit_large_patch16_224":
                in_dim = 1024
            elif backbone_name == "vit_base_patch16_224":
                in_dim = 768
            else:
                raise ValueError(f"Unknown backbone name: {backbone_name}")

        # --------------------
        # Decoder: [B, D, 14, 14] -> [B, 1, 224, 224]
        # --------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()

        # for visualization
        self.visualize_n_batches = visualize_n_batches
        self.visualize_n_samples = visualize_n_samples

        # store mean/std to unnormalize for visualization (ImageNet)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # --------------------
    # ViT token encoder
    # --------------------
    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Re-implement ViT forward to get all patch tokens.

        Returns:
            tokens: [B, L, D] (patch tokens only, CLS dropped)
        """
        patch_embed = self.backbone.patch_embed
        cls_token = self.backbone.cls_token      # [1,1,D]
        pos_embed = self.backbone.pos_embed      # [1, 1+L, D]
        blocks = self.backbone.blocks
        norm = self.backbone.norm

        # 1) patch embedding
        x = patch_embed(x)                       # [B, L, D]
        # 2) add positional embeddings (skip CLS pos)
        x = x + pos_embed[:, 1:, :]              # [B, L, D]

        # 3) prepend CLS token with its pos embedding
        cls_tok = cls_token + pos_embed[:, :1, :]      # [1,1,D]
        cls_tok = cls_tok.expand(x.shape[0], -1, -1)   # [B,1,D]
        x = torch.cat((cls_tok, x), dim=1)             # [B,1+L,D]

        # 4) transformer blocks
        for blk in blocks:
            x = blk(x)
        x = norm(x)                             # [B,1+L,D]

        # 5) drop CLS → patch tokens
        tokens = x[:, 1:, :]                    # [B,L,D]
        return tokens

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        return: feature map [B, D, H', W'] where H'=W'=img_size/patch_size.
        """
        tokens = self._encode_tokens(x)         # [B, L, D]
        B, L, D = tokens.shape

        expected_L = self.h * self.w
        if L != expected_L:
            raise ValueError(f"Expected {expected_L} patches (h={self.h}, w={self.w}), got {L}")

        feat = tokens.transpose(1, 2).contiguous().view(B, D, self.h, self.w)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        returns logits: [B, 1, 224, 224]
        """
        feat = self._encode(x)                 # [B, D, 14, 14]
        logits = self.decoder(feat)            # [B, 1, 224, 224]
        return logits

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
        logits = self(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("train_loss", metrics["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.size(0))
        self.log("train_dice", metrics["dice_score"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("train_iou", metrics["iou"], prog_bar=False, on_step=False, on_epoch=True, batch_size=imgs.size(0))

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("val_loss", metrics["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("val_dice", metrics["dice_score"], prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log("val_iou", metrics["iou"], prog_bar=False, on_step=False, on_epoch=True, batch_size=imgs.size(0))

        # visualize only first 8 batches in validation
        if batch_idx < 8:
            self._log_val_visualizations(imgs, masks, logits, tag="val/segmentation")
            
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        metrics = self._compute_loss_and_metrics(logits, masks)

        self.log("test_loss", metrics["loss"], prog_bar=True, batch_size=imgs.size(0))
        self.log("test_dice", metrics["dice_score"], prog_bar=True, batch_size=imgs.size(0))
        self.log("test_iou", metrics["iou"], prog_bar=True, batch_size=imgs.size(0))

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


import argparse
import time
from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def main(args):
    # -------------------------
    # Data module
    # -------------------------
    data_module = CXLSegDataModule(
        split_csv=args.split_csv,
        images_root=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # -------------------------
    # Run naming + dirs
    # -------------------------
    current_time = time.strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"seg_cxls_{args.mode}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_ep{args.max_epochs}"
        f"_{current_time}"
    )

    base_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Model
    # -------------------------
    model = ViTLungSegmentationModule(
        mode=args.mode,
        backbone_name=args.backbone_name,
        model_checkpoints=args.ckpt_path,
        img_size=args.image_size,
        patch_size=args.patch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=1.0,
        bce_weight=1.0,
        visualize_n_batches=8,
        visualize_n_samples=4,
    )

    # -------------------------
    # Logger
    # -------------------------
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(run_dir),
        log_model=False,  # don't let wandb store extra model copies
    )

    # -------------------------
    # Checkpointing: monitor val_dice
    # -------------------------
    monitor_metric = "val_dice"
    filename = "epoch{epoch:03d}-valdice{val_dice:.4f}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -------------------------
    # Trainer
    # -------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        check_val_every_n_epoch=1,
        default_root_dir=str(run_dir),
    )

    # -------------------------
    # Fit + Test
    # -------------------------
    trainer.fit(model, datamodule=data_module)

    best_model = ViTLungSegmentationModule.load_from_checkpoint(
        checkpoint_cb.best_model_path
    )

    trainer.test(best_model, datamodule=data_module)

    print(f"\nRun directory: {run_dir}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data / IO
    parser.add_argument(
        "--split_csv",
        type=str,
        default="./src/chest-x-ray-dataset-with-lung-segmentation-1.0.0/chest-x-ray-dataset-with-lung-segmentation-1.0.0/CXLSeg-split.csv",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./src/chest-x-ray-dataset-with-lung-segmentation-1.0.0/chest-x-ray-dataset-with-lung-segmentation-1.0.0/files/",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)

    # Optimizer / schedule
    parser.add_argument("--lr", type=float, default=1e-4)           # AdamW LR
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)

    # Backbone / mode
    parser.add_argument("--mode", type=str, default="imagenet", choices=["mae", "imagenet"])
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/mae/mae_cxr_final.ckpt")

    # Logging / output
    parser.add_argument("--wandb_project", type=str, default="cxls_lung_seg")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../scratch/model_checkpoints/seg_cxls",
    )

    args = parser.parse_args()
    main(args)
