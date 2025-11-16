import torch

# --- Existing code ---
from Modules.data_modules import CXRDataModule
from Modules.lightning_modules import ClassificationLightningModule

import argparse

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def main(args):
    data_module = CXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    model = ClassificationLightningModule(
        num_classes=1 if args.task == "COVID" else 14,
        model_mode=args.mode,
        model_weights_path=args.ckpt_path,
        freeze_backbone=True,
        pos_weight=float(args.pos_weight) if args.pos_weight is not None else None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        class_names=["COVID"] if args.task == "COVID" else [
                    'Hernia', 'Pneumothorax', 'Nodule', 'Edema', 'Effusion', 
                    'Pleural_Thickening', 'Cardiomegaly', 'Mass', 'Fibrosis', 
                    'Consolidation', 'Pneumonia', 'Infiltration', 'Emphysema', 'Atelectasis'
                ],
        backbone_name="vit_base_patch16_224", # default backbone
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"sl_{args.task.lower()}_{args.mode}_{args.lr}_{args.weight_decay}_bs{args.batch_size}_ep{args.max_epochs}"
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val/auroc" if args.task == "COVID" else "val/auroc_macro",
        mode="max",
        save_top_k=1,
        filename="sl-{epoch:02d}-val/{val/auroc:.4f}" if args.task == "COVID" else "sl-{epoch:02d}-val/{val/auroc_macro:.4f}",
        dirpath=args.checkpoint_dir,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",   # AMP
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_cb.best_model_path
    )

    trainer.test(best_model, dataloaders=test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--mode", type=str, default="mae", choices=["mae", "imagenet"])
    
    parser.add_argument("--task", type=str, default="COVID")
    parser.add_argument("--train_csv", type=str, default="./data/covid_train_split.csv")
    parser.add_argument("--val_csv", type=str, default="./data/covid_val_split.csv")
    parser.add_argument("--test_csv", type=str, default="./data/covid_test_split.csv")
    parser.add_argument("--root_dir", type=str, default="/home/liue/.cache/kagglehub/datasets/andyczhao/covidx-cxr2/versions/9")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/mae/mae_cxr_final.ckpt")

    parser.add_argument("--wandb_project", type=str, default="covid_cxr_ssl_eval")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/sl_covid")

    args = parser.parse_args()
    
    main(args)