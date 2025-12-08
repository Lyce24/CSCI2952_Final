from Data.data_modules import CXRDataModule
from Modules.sl_lit import ClassificationLightningModule

import argparse
import time
from pathlib import Path

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def main(args):
    # SEED FOR REPRODUCIBILITY
    pl.seed_everything(args.seed, workers=True)
    
    # -------------------------
    # Data module
    # -------------------------
    data_module = CXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task,
    )

    # -------------------------
    # Task / classes
    # -------------------------
    if args.task == "COVID":
        class_names = ["COVID"]
        num_classes = 1
        task_type = "binary"
    elif args.task == "PNE":
        class_names = ["PNEUMONIA"]
        num_classes = 1
        task_type = "binary"
    elif args.task == "TB":
        class_names = ["TB"]
        num_classes = 1
        task_type = "binary"
    elif args.task == "chexchonet":
        class_names = ["SLVH_DLV_Positive"]
        num_classes = 1
        task_type = "binary"
    elif args.task == "COVIDQU":
        class_names = ["Normal", "COVID-19", "Non-COVID"]
        num_classes = len(class_names)
        task_type = "multiclass"
    elif args.task == "CHESTX6":
        class_names = ["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"]
        num_classes = len(class_names)
        task_type = "multiclass"
    elif args.task == "NIH":
        class_names = [
            "Hernia", "Pneumothorax", "Nodule", "Edema", "Effusion",
            "Pleural_Thickening", "Cardiomegaly", "Mass", "Fibrosis",
            "Consolidation", "Pneumonia", "Infiltration", "Emphysema", "Atelectasis",
        ]
        num_classes = len(class_names)
        task_type = "multilabel"
    elif args.task == "VINDR":
        class_names = [
            'Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
            'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
            'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
            'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'
        ]
        num_classes = len(class_names)
        task_type = "multilabel"
    elif args.task == "ECHO":
        class_names = [
            'slvh', 'dlv', 
            'heart_transplant', 'lung_transplant', 'pacemaker_or_icd',
        ]
        num_classes = len(class_names)
        task_type = "multilabel"
    else:
        raise ValueError(f"Unsupported task: {args.task}")


    # -------------------------
    # Run naming + dirs
    # -------------------------
    current_time = time.strftime("%Y%m%d_%H%M%S")

    probe_mode = "ft" if args.unfreeze_backbone else "lp"  # linear probe vs fine-tune

    run_name = (
        f"sl_{args.task.lower()}_{args.mode}_{probe_mode}"
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
    model = ClassificationLightningModule(
        num_classes=num_classes,
        model_mode=args.mode,
        model_weights_path=args.ckpt_path,
        unfreeze_backbone=args.unfreeze_backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        betas=(0.9, 0.999),
        class_names=class_names,
        backbone_name="vit_base_patch16_224",
        task_type=task_type
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(run_dir),
        log_model=False,  # don't let wandb store extra model copies
    )

    monitor_metric = "val/auroc" if num_classes == 1 else "val/auroc_macro"

    if num_classes == 1:
        filename = "epoch{epoch:03d}-valauroc{val/auroc:.4f}"
    else:
        filename = "epoch{epoch:03d}-valauroc_macro{val/auroc_macro:.4f}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,          # Lightning will fill {epoch}, {val/...}
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    best_model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_cb.best_model_path
    )

    trainer.test(best_model, datamodule=data_module)

    print(f"\nRun directory: {run_dir}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-3) # 3e-3 for linear probing / 3e-4 for fine-tuning
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=40) # fine-tune for 40 epochs/ linear probe for 30 epochs
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--mode", type=str, default="imagenet")
    
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="If set, unfreeze the backbone (fine-tuning). Default (not set): freeze backbone (linear probing).",
    )
    parser.add_argument("--task", type=str, default="COVID")
    parser.add_argument("--train_csv", type=str, default="./src/covid_train_split.csv")
    parser.add_argument("--val_csv", type=str, default="./src/covid_val_split.csv")
    parser.add_argument("--test_csv", type=str, default="./src/covid_test_split.csv")
    parser.add_argument("--root_dir", type=str, default="/home/liue/.cache/kagglehub/datasets/andyczhao/covidx-cxr2/versions/9")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/mae/mae_cxr_final.ckpt")

    # Logging / output
    parser.add_argument("--wandb_project", type=str, default="covid_cxr_ssl_eval")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../scratch/model_checkpoints/sl_cxr",
    )
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    main(args)