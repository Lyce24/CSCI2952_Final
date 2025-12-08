"""
PACX Teacher-Student Pipeline
==============================
Robust teacher-student distillation for physiology-aware CXR encoding.

Architecture:
- Teacher: Fusion([CXR, ECG, Labs]) → Rich multi-modal representation
- Student: CXR-only → Matches teacher's representation

Training:
- Stage 1 (Teacher): Learn multi-modal fusion with reconstruction
- Stage 2 (Student): Distill physiology knowledge to CXR encoder
"""

import os
import time
import warnings
from typing import Optional
from pathlib import Path
import lightning.pytorch as pl

from Data.data_modules import SymileMIMICDataModule
from Modules.pacx_lit import TeacherModule, StudentModule

pl.seed_everything(42, workers=True)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Training Pipeline
# ============================================================================
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    WandbLogger = None

def build_logger(
    project_name: str,
    save_dir: Path,
    run_name: Optional[str] = None,
):
    """
    Build a single (optional) WandbLogger instance shared by teacher + student.

    If wandb is not installed, returns None and Lightning will just use CSV/progress bar.
    """
    if WandbLogger is None:
        print("⚠ wandb not available, running without external logger.")
        return None

    if run_name is None:
        run_name = f"pacx_teacher_student_{time.strftime('%Y%m%d_%H%M%S')}"

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=str(save_dir),
        log_model=False,   # we're managing checkpoints ourselves
    )
    print(f"✓ Using WandbLogger run: {project_name}/{run_name}")
    return logger

def train_teacher(
    data_module: SymileMIMICDataModule,
    mae_checkpoint: str,
    save_dir: Path,
    max_epochs: int = 10,
    logger = None,
):
    """Stage 1: Train Teacher (multi-modal fusion with CLIP-style loss)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TeacherModule(mae_checkpoint_path=mae_checkpoint)

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="teacher-{epoch:02d}-{teacher/val/loss_total:.4f}",
        monitor="teacher/val/loss_total",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(save_dir),
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("STAGE 1: Training Teacher (Multi-Modal Fusion / CLIP-style)")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=data_module)

    # Prefer best checkpoint if available, else fallback to last
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt is None or best_ckpt == "":
        best_ckpt = os.path.join(str(save_dir), "last.ckpt")

    print(f"\n✓ Teacher training complete")
    print(f"✓ Teacher checkpoint: {best_ckpt}\n")

    return Path(best_ckpt)


def train_student(
    data_module: SymileMIMICDataModule,
    mae_checkpoint: str,
    teacher_checkpoint: Path,
    save_dir: Path,
    max_epochs: int = 20,
    logger = None,
):
    """Stage 2: Train Student (CXR-only distillation from teacher fused embedding)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    model = StudentModule(
        mae_checkpoint_path=mae_checkpoint,
        teacher_checkpoint_path=str(teacher_checkpoint),
        data_dir=str(data_module.data_dir),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="student-{epoch:02d}-{student/val/distill_loss_total:.4f}",
        monitor="student/val/distill_loss_total",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(save_dir),
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("STAGE 2: Training Student (Feature-level Distillation)")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=data_module)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt is None or best_ckpt == "":
        best_ckpt = os.path.join(str(save_dir), "last.ckpt")

    print(f"\n✓ Student training complete")
    print(f"✓ Student checkpoint: {best_ckpt}\n")

    return Path(best_ckpt)

import argparse

def main(args):
    # ------------------------------------------------------------------
    # Paths / dirs
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("✓ Initializing Symile-MIMIC data module...")
    data_module = SymileMIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    data_module.setup("fit")

    # ------------------------------------------------------------------
    # Logger (single run for both stages)
    # ------------------------------------------------------------------
    logger = build_logger(
        project_name=args.project_name,
        save_dir=save_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1: Teacher (optional)
    # ------------------------------------------------------------------
    if args.teacher_ckpt is not None and Path(args.teacher_ckpt).is_file():
        # Use existing teacher checkpoint
        teacher_ckpt = Path(args.teacher_ckpt)
        print("\n" + "=" * 80)
        print("USING EXISTING TEACHER CHECKPOINT")
        print("=" * 80)
        print(f"✓ teacher_ckpt: {teacher_ckpt}")
        print("=" * 80 + "\n")
    else:
        # Train teacher from MAE checkpoint
        print("\n" + "=" * 80)
        print("STAGE 1: TRAINING TEACHER")
        print("=" * 80 + "\n")

        teacher_ckpt = train_teacher(
            data_module=data_module,
            mae_checkpoint=args.mae_checkpoint,
            save_dir=save_dir / "teacher",
            max_epochs=args.max_epochs_teacher,
            logger=logger,
        )

        print("\n" + "=" * 80)
        print("TEACHER TRAINING COMPLETE")
        print("=" * 80)
        print(f"✓ Best teacher checkpoint: {teacher_ckpt}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Stage 2: Student
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING STUDENT (DISTILLATION)")
    print("=" * 80 + "\n")

    student_ckpt = train_student(
        data_module=data_module,
        mae_checkpoint=args.mae_checkpoint,
        teacher_checkpoint=str(teacher_ckpt),
        save_dir=save_dir / "student",
        max_epochs=args.max_epochs_student,
        logger=logger,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Final student checkpoint: {student_ckpt}")
    print("\nTo extract encoder:")
    print("  from train_teacher_student import StudentModule")
    print(f"  model = StudentModule.load_from_checkpoint('{student_ckpt}')")
    print("  encoder = model.get_encoder()")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PACX teacher+student")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy",
        help="Root directory of Symile-MIMIC npy files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=250,
        help="Batch size for teacher & student training",
    )

    # Checkpoints
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default="../../scratch/checkpoints/mae/last.ckpt",
        help="Path to MAE pretraining checkpoint for CXR encoder",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help="Optional: existing teacher checkpoint. "
             "If provided, teacher training is skipped.",
    )

    # Training config
    parser.add_argument(
        "--max_epochs_teacher",
        type=int,
        default=40,
        help="Number of training epochs for teacher",
    )
    parser.add_argument(
        "--max_epochs_student",
        type=int,
        default=40,
        help="Number of training epochs for student",
    )

    # Logging / saving
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../../scratch/checkpoints/teacher_student",
        help="Base directory to save teacher and student checkpoints",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="PACX-Teacher-Student",
        help="W&B / logger project name",
    )

    args = parser.parse_args()
    main(args)
