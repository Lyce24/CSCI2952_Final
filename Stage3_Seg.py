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

from Modules.seg_lit import LungSegmentationModule
from Data.data_modules import CXRSegDataModule

import argparse
import time
from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def main(args):
    pl.seed_everything(args.seed, workers=True)

    # -------------------------
    # Data module
    # -------------------------
    data_module = CXRSegDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        images_root=args.root_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        task=args.task,
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
    model = LungSegmentationModule(
        mode=args.mode,
        backbone_name=args.backbone_name,
        model_checkpoints=args.ckpt_path,
        img_size=args.image_size,
        patch_size=args.patch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=1.0,
        bce_weight=1.0,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        betas=(0.9, 0.999),
        unfreeze_backbone=False,
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
    monitor_metric = "val/dice"
    filename = "epoch{epoch:03d}-valdice{val/dice:.4f}"

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

    best_model = LungSegmentationModule.load_from_checkpoint(
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
    parser.add_argument("--mode", type=str, default="mae", choices=["mae", "imagenet"])
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/mae/last.ckpt")

    # Logging / output
    parser.add_argument("--wandb_project", type=str, default="cxls_lung_seg")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/seg_cxr",
    )

    args = parser.parse_args()
    main(args)
