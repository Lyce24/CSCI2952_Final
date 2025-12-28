from Data.data_modules import LowDataCXRDataModule
from Modules.cl_lit import ClassificationLightningModule

import argparse
import time
from pathlib import Path
import math

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

def main(args):
    # SEED FOR REPRODUCIBILITY
    pl.seed_everything(args.seed, workers=True)
    
    # -------------------------
    # Data module
    # -------------------------
    data_module = LowDataCXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task,
        subset_fraction=args.subset_fraction,
        seed=args.seed
    )

    # -------------------------
    # Pre-calculate Steps for Logging
    # -------------------------
    # We manually trigger setup here to inspect the dataset size
    data_module.setup("fit")
    
    # Access the train dataset length directly
    try:
        # Assuming your DataModule saves the dataset as self.cxr_train or self.train_dataset
        # We try standard attributes:
        if hasattr(data_module, 'cxr_train'):
            train_len = len(data_module.cxr_train)
        elif hasattr(data_module, 'train_dataset'):
            train_len = len(data_module.train_dataset)
        else:
            # Fallback: estimate based on batch retrieval (slightly slower but safe)
            train_len = len(data_module.train_dataloader().dataset)
    except Exception as e:
        print(f"Warning: Could not auto-detect dataset size ({e}). Defaulting log freq to 10.")
        train_len = 100 # arbitrary small number fallback

    # Calculate effective batch size (considering multi-gpu)
    eff_batch_size = args.batch_size * args.devices * args.num_nodes
    
    # Steps per epoch = Total samples / effective batch size
    steps_per_epoch = math.ceil(train_len / eff_batch_size)
    
    # Dynamic Logic:
    # 1. If steps_per_epoch is tiny (e.g., 5), log every 1 step.
    # 2. Otherwise, log every ~50 steps or every epoch, whichever is smaller.
    log_freq = min(50, steps_per_epoch)
    
    # Safety clamp: never 0
    log_freq = max(1, log_freq)

    print(f"--- Dataset Stats ---")
    print(f"Train Samples: {train_len}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    print(f"Log Frequency: Every {log_freq} steps")
    print(f"---------------------")

    # -------------------------
    # Task / classes
    if args.task == "chexchonet":
        class_names = ["composite_slvh_dlv"]
        num_classes = 1
        task_type = "binary"
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
    elif args.task == "COVID":
        class_names = ["COVID"]
        num_classes = 1
        task_type = "binary"
    elif args.task == "CHESTX6":
        class_names = ["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"]
        num_classes = len(class_names)
        task_type = "multiclass"
    elif args.task == "MEDMOD-PHYS":
        class_names = [
            "Acute and unspecified renal failure", "Acute cerebrovascular disease", "Acute myocardial infarction", "Cardiac dysrhythmias",
            "Chronic kidney disease", "Chronic obstructive pulmonary disease and bronchiectasis", "Complications of surgical procedures or medical care",
            "Conduction disorders", "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
            "Diabetes mellitus with complications", "Diabetes mellitus without complication", "Disorders of lipid metabolism",
            "Essential hypertension", "Fluid and electrolyte disorders",
            "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
            "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
            "Pleurisy; pneumothorax; pulmonary collapse", "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)", "Respiratory failure; insufficiency; arrest (adult)",
            "Septicemia (except in labor)", "Shock"
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
    pct_str = f"{int(args.subset_fraction * 100)}pct"
    
    run_name = (
        f"cl_{args.task.lower()}_{args.mode}_{probe_mode}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_ep{args.max_epochs}"
        f"_{pct_str}"
        f"_{current_time}"
    )

    base_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Low-Data Experiment ---")
    print(f"Task: {args.task}")
    print(f"Data Usage: {args.subset_fraction * 100}%")
    print(f"Run Name: {run_name}")
    print(f"------------------------------------")

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
        log_every_n_steps=log_freq,
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
    parser.add_argument(
        "--subset_fraction", 
        type=float, 
        default=0.1, 
        help="Fraction of training data to use (0.0 to 1.0). Default 1.0."
    )
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
        default="../../scratch/model_checkpoints/cl_cxr",
    )
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    main(args)