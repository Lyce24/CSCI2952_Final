"""
PACX Physio-Aware Pipeline
==========================
Single-stage training pipeline for aligning MAE-CXR with Frozen Physiology.

Architecture:
- Input: CXR Images
- Model: MAE ViT (Unfrozen + LLRD) -> Projection Heads
- Targets: Frozen ECG & Lab Embeddings (from pre-computed dataloader)
- Loss: CLIP (Alignment) + MSE (Reconstruction)
"""

import os
import time
import warnings
from typing import Optional
from pathlib import Path
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

# ----------------------------------------------------------------------------
# Import your model and data modules here
# ----------------------------------------------------------------------------
from Data.data_modules import SymileMIMICDataModule, SymileMIMICDataModuleV2
from Modules.pacx_lit import CrossModalCXRDistillation

pl.seed_everything(42, workers=True)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Logging Helper
# ============================================================================

try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    WandbLogger = None

def build_logger(project_name: str, save_dir: Path, run_name: Optional[str] = None):
    """Build a single WandbLogger instance."""
    if WandbLogger is None:
        print("⚠ wandb not available, running without external logger.")
        return None

    if run_name is None:
        run_name = f"pacx_physio_{time.strftime('%Y%m%d_%H%M%S')}_DA"

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=str(save_dir),
        log_model=False,
    )
    return logger

# ============================================================================
# Training Stage
# ============================================================================

def train_model(
    data_module: SymileMIMICDataModule,
    args: argparse.Namespace,
    save_dir: Path,
    logger=None,
) -> Path:
    """
    Train the CrossModalCXRDistillation model using Unfreeze All + LLRD.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 
    # Initialize Model with new LLRD args
    model = CrossModalCXRDistillation(
        mae_checkpoint_path=args.mae_checkpoint,
        ablate_mode=args.ablate_mode,
        ablate_loss=args.ablate_loss,
        # Modalities Dimensions
        cxr_dim=768,
        ecg_dim=1024,
        labs_dim=256,
        # Loss Weights
        clip_weight=args.clip_weight,
        regression_weight=args.regression_weight,
        tuning_strategy=args.tuning_strategy,
        # Optimization
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size
    )
    
    # Checkpointing: Save based on Validation Loss (Total)
    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="pacx-{epoch:02d}-{val/loss_total:.4f}",
        monitor="val/loss_total", # Using train loss if val is infrequent, otherwise 'val/loss_total'
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True), 
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(save_dir),
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=data_module)

    # Resolve best checkpoint path
    best_ckpt = ckpt_cb.best_model_path
    if not best_ckpt:
        best_ckpt = str(save_dir / "last.ckpt")

    if trainer.is_global_zero:
        print(f"✓ Best checkpoint: {best_ckpt}")

    return Path(best_ckpt)

# ============================================================================
# Main Orchestrator
# ============================================================================

def main(args: argparse.Namespace):
    # 1. Setup dirs
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup Data
    actual_batch_size = int(args.batch_size // (args.devices * args.num_nodes))
    print(f"⚙ Using actual batch size of {actual_batch_size} per device.")
    
    data_module = SymileMIMICDataModuleV2(
        data_dir=args.data_dir,
        batch_size=actual_batch_size,
        use_data_aug=True
    )

    # 3. Setup logger
    logger = build_logger(args.project_name, save_dir)

    # 4. Train
    if args.ckpt_path is not None and Path(args.ckpt_path).is_file():
        target_ckpt = Path(args.ckpt_path)
        print(f"⚙ Using provided checkpoint: {target_ckpt}")
    else:
        target_ckpt = train_model(
            data_module=data_module,
            args=args,
            save_dir=save_dir / "train",
            logger=logger,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACX Physio-Aware Training")

    # --- Hardware ---
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")

    # --- Data ---
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy",
        help="Path to Symile-MIMIC data",
    )
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default="../../scratch/checkpoints/mae/last.ckpt",
        help="Path to MAE pretrained weights",
    )
    parser.add_argument("--save_dir", type=str, default="../../scratch/checkpoints/PACX-Physio")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Skip training, test this ckpt")
    parser.add_argument("--project_name", type=str, default="PACX-Physio-V7")

    # --- Training Params ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--ablate_mode", type=str, default=None, choices=[None, "ecg", "labs"], help="Modality to ablate: 'ecg' or 'labs'")
    parser.add_argument("--ablate_loss", type=str, default=None, choices=[None, "clip", "reg"], help="Loss to ablate: 'clip' or 'reg'")

    # --- LLRD & Unfreezing ---
    parser.add_argument(
        "--tuning_strategy", 
        type=str, 
        default="lora"
    )

    # --- Loss Weights ---
    parser.add_argument("--clip_weight", type=float, default=0.5)
    parser.add_argument("--regression_weight", type=float, default=1.0)

    args = parser.parse_args()
    main(args)