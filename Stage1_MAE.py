import torch

from Modules.data_modules import CXRDataModule
from Modules.lightning_modules import MAELightningModule

from lightning.pytorch.loggers import WandbLogger
import lightning as pl

import argparse
import time

def main(args):
    data_module = CXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task="MAE"
    )

    model = MAELightningModule(
        size="base",
        norm_pix_loss=False,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        warmup_epochs=args.warmup_epochs,
        log_images_every_n_epochs=args.log_images_every_n_epochs,
        log_max_images=args.log_max_images,
    )

    current_time = time.strftime("%Y%m%d_%H%M%S")

    logger = WandbLogger(project=args.wandb_project,
                         name=f"mae_cxr_base_bs{args.batch_size}_mr{args.mask_ratio}_lr{args.lr}_wd{args.weight_decay}_ep{args.max_epochs}_we{args.warmup_epochs}_{current_time}",
                         log_model="all")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",   # AMP
        logger=logger,
        gradient_clip_val=0.0,
        deterministic=False,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="auto",
        default_root_dir=f"{args.checkpoint_dir}/mae_cxr_base_bs{args.batch_size}_mr{args.mask_ratio}_lr{args.lr}_wd{args.weight_decay}_ep{args.max_epochs}_we{args.warmup_epochs}_{current_time}"
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="./src/chexpert_train_split.csv")
    parser.add_argument("--val_csv", type=str, default="./src/chexpert_val_split.csv")
    parser.add_argument("--root_dir", type=str, default="../../scratch/kagglehub_cache/kagglehub/datasets/ashery/chexpert/versions/1")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--mask_ratio", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--log_images_every_n_epochs", type=int, default=10)
    parser.add_argument("--log_max_images", type=int, default=8)

    parser.add_argument("--wandb_project", type=str, default="mae-cxr")
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="../../scratch/model_checkpoints/mae")
    args = parser.parse_args()
    
    main(args)