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
    # Task / classes
    # -------------------------
    if args.task == "COVID":
        class_names = ["COVID"]
        num_classes = 1
        task_type = "binary"
        train_csv= "./src/covid_train_split.csv"
        val_csv= "./src/covid_val_split.csv"
        test_csv= "./src/covid_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/andyczhao/covidx-cxr2/versions/9"
        wandb_project= "covid_cxr_ssl_eval_final_v2"
    elif args.task == "PNE":
        class_names = ["PNEUMONIA"]
        num_classes = 1
        task_type = "binary"
        train_csv= "./src/pneumonia_train_split.csv"
        val_csv= "./src/pneumonia_val_split.csv"
        test_csv= "./src/pneumonia_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2"
        wandb_project= "pneumonia_cxr_ssl_eval_final_v2"
    elif args.task == "TB":
        class_names = ["TB"]
        num_classes = 1
        task_type = "binary"
        train_csv= "./src/tb_train_split.csv"
        val_csv= "./src/tb_val_split.csv"
        test_csv= "./src/tb_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/raddar/tuberculosis-chest-xrays-shenzhen/versions/1/images"
        wandb_project= "tb_cxr_ssl_eval_final_v2"
    elif args.task == "chexchonet":
        class_names = ["SLVH_DLV_Positive"]
        num_classes = 1
        task_type = "binary"
        train_csv= "./src/chexchonet_train.csv"
        val_csv= "./src/chexchonet_val.csv"
        test_csv= "./src/chexchonet_test.csv"
        root_dir= "./src/chex"
        wandb_project= "chexchonet_cxr_ssl_eval_final_v2"
    elif args.task == "COVIDQU":
        class_names = ["Normal", "COVID-19", "Non-COVID"]
        num_classes = len(class_names)
        task_type = "multiclass"
        train_csv= "./src/covidqu_train_split.csv"
        val_csv= "./src/covidqu_val_split.csv"
        test_csv= "./src/covidqu_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/anasmohammedtahir/covidqu/versions/7"
        wandb_project= "covidqu_cxr_ssl_eval_final_v2"
    elif args.task == "CHESTX6":
        class_names = ["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"]
        num_classes = len(class_names)
        task_type = "multiclass"
        train_csv= "./src/chestx6_train_split.csv"
        val_csv= "./src/chestx6_val_split.csv"
        test_csv= "./src/chestx6_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/mohamedasak/chest-x-ray-6-classes-dataset/versions/1/chest-xray"
        wandb_project= "chestx6_cxr_ssl_eval_final_v2"
    elif args.task == "NIH":
        class_names = [
            "Hernia", "Pneumothorax", "Nodule", "Edema", "Effusion",
            "Pleural_Thickening", "Cardiomegaly", "Mass", "Fibrosis",
            "Consolidation", "Pneumonia", "Infiltration", "Emphysema", "Atelectasis",
        ]
        num_classes = len(class_names)
        task_type = "multilabel"
        train_csv= "./src/nih_train_split.csv"
        val_csv= "./src/nih_val_split.csv"
        test_csv= "./src/nih_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3"
        wandb_project= "nih_cxr_ssl_eval_final_v2"
    elif args.task == "VINDR":
        class_names = [
            'Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
            'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
            'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
            'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'
        ]
        num_classes = len(class_names)
        task_type = "multilabel"
        train_csv= "./src/vindr_train_split.csv"
        val_csv= "./src/vindr_val_split.csv"
        test_csv= "./src/vindr_test_split.csv"
        root_dir= "/users/yliu802/.cache/kagglehub/datasets/awsaf49/vinbigdata-512-image-dataset/versions/1/vinbigdata"
        wandb_project= "vindr_cxr_ssl_eval_final_v2"
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    
    batch_size = 512
    lr = 3e-3
    if args.task in ["CHESTX6", "COVIDQU", "VINDR"]:
        batch_size = 256  # reduce batch size for multiclass tasks due to memory constraints
        lr = 2e-3
    elif args.task == "PNE":
        batch_size = 128
        lr = 1e-3
    elif args.task == "TB":
        batch_size = 32
        lr = 1e-3
    
    # -------------------------
    # Data module
    # -------------------------
    data_module = CXRDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task,
    )

    # -------------------------
    # Run naming + dirs
    # -------------------------
    current_time = time.strftime("%Y%m%d_%H%M%S")

    probe_mode = "lp"  # linear probe vs fine-tune

    run_name = (
        f"sl_{args.task.lower()}_{args.mode}_{probe_mode}"
        f"_bs{batch_size}"
        f"_lr{lr}"
        f"_wd{args.weight_decay}"
        f"_ep{args.max_epochs}"
        f"_{current_time}"
    )

    base_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "imagenet":
        model_weights_path = None
    elif args.mode == "mae":
        model_weights_path = "../../scratch/checkpoints/mae/last.ckpt"
    elif args.mode == "pacx":
        model_weights_path = "../../scratch/checkpoints/PACX-Sweep-LRWD/Run1_BS500_LR1e-4_WD0.05.ckpt"

    # -------------------------
    # Model
    # -------------------------
    model = ClassificationLightningModule(
        num_classes=num_classes,
        model_mode=args.mode,
        model_weights_path=model_weights_path, #no more passing through args.ckpt_path
        unfreeze_backbone=False,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        betas=(0.9, 0.999),
        class_names=class_names,
        backbone_name="vit_base_patch16_224",
        task_type=task_type
    )

    wandb_logger = WandbLogger(
        project=wandb_project,
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
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--mode", type=str, default="imagenet")
    
    parser.add_argument("--task", type=str, default="COVID")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../scratch/model_checkpoints/sl_cxr",
    )
    
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    main(args)