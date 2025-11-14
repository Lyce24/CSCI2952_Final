import torch

# --- Existing code ---
from Modules.data_modules import CXRDataModule
from Modules.lightning_modules import MAELightningModule

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import argparse

def main(args):
    data_module = CXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    model = MAELightningModule(
        size="base",
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        warmup_epochs=args.warmup_epochs,
        log_images_every_n_epochs=args.log_images_every_n_epochs,
        log_max_images=args.log_max_images,
    )

    # ---------- Test the Model ----------
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = next(iter(train_loader))  # CheXpertDataModule returns only images
    imgs = batch.to(device)           # [B, 3, 224, 224]

    with torch.no_grad():
        out = model(imgs)

    print("Forward output type:", type(out))
    if isinstance(out, tuple) or isinstance(out, list):
        print("Tuple length:", len(out))
        for i, t in enumerate(out):
            if torch.is_tensor(t):
                print(f"  out[{i}] shape:", t.shape)
    else:
        if torch.is_tensor(out):
            print("Output shape:", out.shape)

    logger = WandbLogger(project=args.wandb_project)

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
    )

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_csv", type=str, default="./data/train_split.csv")
    parser.add_argument("--val_csv", type=str, default="./data/val_split.csv")
    parser.add_argument("--root_dir", type=str, default="../../scratch/kagglehub_cache/kagglehub/datasets/ashery/chexpert/versions/1")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--log_images_every_n_epochs", type=int, default=10)
    parser.add_argument("--log_max_images", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default="mae-cxr")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, default="../../scratch/model_checkpoints/mae/mae_cxr_base_2_gpu.ckpt")
    args = parser.parse_args()
    
    main(args)