from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from Models.models import CXRModel

import torch
import lightning as pl
from torch.optim import AdamW

import math
from typing import Optional, List
import torch.nn as nn

class ClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_mode: str = "imagenet",
        model_weights_path: Optional[str] = None,
        unfreeze_backbone: bool = False,
        lr: float = 1e-4,
        backbone_lr: Optional[float] = None,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        betas=(0.9, 0.95),
        class_names: Optional[List[str]] = None,
        backbone_name: str = "vit_base_patch16_224",
        task_type: str = "multilabel",
    ):
        """
        task_type:
            - "binary"    : ignored here; num_classes == 1 forces binary
            - "multilabel": multi-label (CheXpert-style)
            - "multiclass": single-label multi-class (Normal / Pneumonia / COVID, etc.)
        """
        super().__init__()
        self.save_hyperparameters()

        assert task_type in {"multilabel", "multiclass", "binary"}, f"Invalid task_type: {task_type}"

        self.num_classes = num_classes

        if num_classes == 1:
            self.task_type = "binary"
        else:
            self.task_type = task_type

        self.is_binary = self.task_type == "binary"
        self.is_multilabel = self.task_type == "multilabel"
        self.is_multiclass = self.task_type == "multiclass"

        # backbone + head
        self.model = CXRModel(
            num_classes=num_classes,
            mode=model_mode,
            backbone_name=backbone_name,
            model_checkpoints=model_weights_path,
            unfreeze_backbone=unfreeze_backbone,
        )

        # Loss
        if self.is_binary or self.is_multilabel:
            # sigmoid + BCE for binary/multilabel
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # softmax + CE for multiclass
            self.criterion = nn.CrossEntropyLoss()

        # Class names
        if class_names is None:
            if self.is_binary:
                self.class_names = ["class_0"]
            else:
                self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            assert len(class_names) == num_classes
            self.class_names = class_names

        # ---------------- METRICS ----------------
        if self.is_binary:
            # Binary metrics
            # val
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
            self.val_prec = BinaryPrecision()
            self.val_rec = BinaryRecall()
            self.val_f1 = BinaryF1Score()
            # test
            self.test_auroc = BinaryAUROC()
            self.test_acc = BinaryAccuracy()
            self.test_prec = BinaryPrecision()
            self.test_rec = BinaryRecall()
            self.test_f1 = BinaryF1Score()

        elif self.is_multilabel:
            # Multilabel global metrics (macro)
            self.val_auroc_macro_ml = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.val_acc_macro_ml = MultilabelAccuracy(
                num_labels=num_classes, average="macro"
            )
            self.val_prec_macro_ml = MultilabelPrecision(
                num_labels=num_classes, average="macro"
            )
            self.val_rec_macro_ml = MultilabelRecall(
                num_labels=num_classes, average="macro"
            )
            self.val_f1_macro_ml = MultilabelF1Score(
                num_labels=num_classes, average="macro"
            )

            self.test_auroc_macro_ml = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.test_acc_macro_ml = MultilabelAccuracy(
                num_labels=num_classes, average="macro"
            )
            self.test_prec_macro_ml = MultilabelPrecision(
                num_labels=num_classes, average="macro"
            )
            self.test_rec_macro_ml = MultilabelRecall(
                num_labels=num_classes, average="macro"
            )
            self.test_f1_macro_ml = MultilabelF1Score(
                num_labels=num_classes, average="macro"
            )

        elif self.is_multiclass:
            # Multiclass global metrics (macro)
            self.val_auroc_macro_mc = MulticlassAUROC(
                num_classes=num_classes, average="macro"
            )
            self.val_acc_macro_mc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            )
            self.val_prec_macro_mc = MulticlassPrecision(
                num_classes=num_classes, average="macro"
            )
            self.val_rec_macro_mc = MulticlassRecall(
                num_classes=num_classes, average="macro"
            )
            self.val_f1_macro_mc = MulticlassF1Score(
                num_classes=num_classes, average="macro"
            )

            self.test_auroc_macro_mc = MulticlassAUROC(
                num_classes=num_classes, average="macro"
            )
            self.test_acc_macro_mc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            )
            self.test_prec_macro_mc = MulticlassPrecision(
                num_classes=num_classes, average="macro"
            )
            self.test_rec_macro_mc = MulticlassRecall(
                num_classes=num_classes, average="macro"
            )
            self.test_f1_macro_mc = MulticlassF1Score(
                num_classes=num_classes, average="macro"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        imgs, targets = batch
        logits = self(imgs)

        # ------------- LOSS + PROBS / PREDS -------------
        if self.is_binary:
            # logits: [B,1] or [B] -> [B]
            logits = logits.view(-1)
            targets = targets.float().view(-1)

            loss = self.criterion(logits, targets)

            probs = torch.sigmoid(logits)          # [B]
            preds = (probs >= 0.5).int()          # [B]

        elif self.is_multilabel:
            # multilabel: logits [B,C], targets [B,C]
            targets = targets.float()

            loss = self.criterion(logits, targets)

            probs = torch.sigmoid(logits)         # [B,C]
            preds = (probs >= 0.5).int()          # [B,C]

        else:  # multiclass
            # logits [B,C], targets [B] or [B,1]
            if targets.ndim == 2:
                if targets.size(-1) == 1:
                    # shape [B, 1] with class indices -> flatten
                    targets = targets.view(-1)
                else:
                    # shape [B, C] one-hot or probabilities -> take argmax
                    # assume row-wise one-hot / soft labels
                    targets = targets.argmax(dim=-1)
            # else: already [B]
            targets = targets.long()  # class indices [B]

            loss = self.criterion(logits, targets)

            probs = torch.softmax(logits, dim=-1)  # [B,C]
            preds = torch.argmax(probs, dim=-1)    # [B]

        # ------------- METRICS UPDATE -------------
        if self.is_binary:
            if stage == "val":
                self.val_auroc.update(probs, targets.int())
                self.val_acc.update(preds, targets.int())
                self.val_prec.update(preds, targets.int())
                self.val_rec.update(preds, targets.int())
                self.val_f1.update(preds, targets.int())
            elif stage == "test":
                self.test_auroc.update(probs, targets.int())
                self.test_acc.update(preds, targets.int())
                self.test_prec.update(preds, targets.int())
                self.test_rec.update(preds, targets.int())
                self.test_f1.update(preds, targets.int())

        elif self.is_multilabel:
            if stage == "val":
                self.val_auroc_macro_ml.update(probs, targets.int())
                self.val_acc_macro_ml.update(preds, targets.int())
                self.val_prec_macro_ml.update(preds, targets.int())
                self.val_rec_macro_ml.update(preds, targets.int())
                self.val_f1_macro_ml.update(preds, targets.int())
            elif stage == "test":
                self.test_auroc_macro_ml.update(probs, targets.int())
                self.test_acc_macro_ml.update(preds, targets.int())
                self.test_prec_macro_ml.update(preds, targets.int())
                self.test_rec_macro_ml.update(preds, targets.int())
                self.test_f1_macro_ml.update(preds, targets.int())

        else:  # multiclass
            if stage == "val":
                # AUROC: probs [B,C], targets [B]
                self.val_auroc_macro_mc.update(probs, targets)
                # others use preds [B], targets [B]
                self.val_acc_macro_mc.update(preds, targets)
                self.val_prec_macro_mc.update(preds, targets)
                self.val_rec_macro_mc.update(preds, targets)
                self.val_f1_macro_mc.update(preds, targets)
            elif stage == "test":
                self.test_auroc_macro_mc.update(probs, targets)
                self.test_acc_macro_mc.update(preds, targets)
                self.test_prec_macro_mc.update(preds, targets)
                self.test_rec_macro_mc.update(preds, targets)
                self.test_f1_macro_mc.update(preds, targets)

        return loss

    # ---------- TRAIN ----------
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # ---------- VAL ----------
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        if self.is_binary:
            auroc = self.val_auroc.compute()
            acc = self.val_acc.compute()
            prec = self.val_prec.compute()
            rec = self.val_rec.compute()
            f1 = self.val_f1.compute()

            self.log("val/auroc", auroc, prog_bar=True, sync_dist=True)
            self.log("val/acc", acc, sync_dist=True)
            self.log("val/precision", prec, sync_dist=True)
            self.log("val/recall", rec, sync_dist=True)
            self.log("val/f1", f1, sync_dist=True)

            self.val_auroc.reset()
            self.val_acc.reset()
            self.val_prec.reset()
            self.val_rec.reset()
            self.val_f1.reset()

        elif self.is_multilabel:
            auroc = self.val_auroc_macro_ml.compute()
            acc = self.val_acc_macro_ml.compute()
            prec = self.val_prec_macro_ml.compute()
            rec = self.val_rec_macro_ml.compute()
            f1 = self.val_f1_macro_ml.compute()

            self.log("val/auroc_macro", auroc, prog_bar=True, sync_dist=True)
            self.log("val/acc_macro", acc, sync_dist=True)
            self.log("val/precision_macro", prec, sync_dist=True)
            self.log("val/recall_macro", rec, sync_dist=True)
            self.log("val/f1_macro", f1, sync_dist=True)

            self.val_auroc_macro_ml.reset()
            self.val_acc_macro_ml.reset()
            self.val_prec_macro_ml.reset()
            self.val_rec_macro_ml.reset()
            self.val_f1_macro_ml.reset()

        else:  # multiclass
            auroc = self.val_auroc_macro_mc.compute()
            acc = self.val_acc_macro_mc.compute()
            prec = self.val_prec_macro_mc.compute()
            rec = self.val_rec_macro_mc.compute()
            f1 = self.val_f1_macro_mc.compute()

            self.log("val/auroc_macro", auroc, prog_bar=True, sync_dist=True)
            self.log("val/acc_macro", acc, sync_dist=True)
            self.log("val/precision_macro", prec, sync_dist=True)
            self.log("val/recall_macro", rec, sync_dist=True)
            self.log("val/f1_macro", f1, sync_dist=True)

            self.val_auroc_macro_mc.reset()
            self.val_acc_macro_mc.reset()
            self.val_prec_macro_mc.reset()
            self.val_rec_macro_mc.reset()
            self.val_f1_macro_mc.reset()

    # ---------- TEST ----------
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="test")
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        if self.is_binary:
            auroc = self.test_auroc.compute()
            acc = self.test_acc.compute()
            prec = self.test_prec.compute()
            rec = self.test_rec.compute()
            f1 = self.test_f1.compute()

            self.log("test/auroc", auroc, sync_dist=True)
            self.log("test/acc", acc, sync_dist=True)
            self.log("test/precision", prec, sync_dist=True)
            self.log("test/recall", rec, sync_dist=True)
            self.log("test/f1", f1, sync_dist=True)

            self.test_auroc.reset()
            self.test_acc.reset()
            self.test_prec.reset()
            self.test_rec.reset()
            self.test_f1.reset()

        elif self.is_multilabel:
            auroc = self.test_auroc_macro_ml.compute()
            acc = self.test_acc_macro_ml.compute()
            prec = self.test_prec_macro_ml.compute()
            rec = self.test_rec_macro_ml.compute()
            f1 = self.test_f1_macro_ml.compute()

            self.log("test/auroc_macro", auroc, sync_dist=True)
            self.log("test/acc_macro", acc, sync_dist=True)
            self.log("test/precision_macro", prec, sync_dist=True)
            self.log("test/recall_macro", rec, sync_dist=True)
            self.log("test/f1_macro", f1, sync_dist=True)

            self.test_auroc_macro_ml.reset()
            self.test_acc_macro_ml.reset()
            self.test_prec_macro_ml.reset()
            self.test_rec_macro_ml.reset()
            self.test_f1_macro_ml.reset()

        else:  # multiclass
            auroc = self.test_auroc_macro_mc.compute()
            acc = self.test_acc_macro_mc.compute()
            prec = self.test_prec_macro_mc.compute()
            rec = self.test_rec_macro_mc.compute()
            f1 = self.test_f1_macro_mc.compute()

            self.log("test/auroc_macro", auroc, sync_dist=True)
            self.log("test/acc_macro", acc, sync_dist=True)
            self.log("test/precision_macro", prec, sync_dist=True)
            self.log("test/recall_macro", rec, sync_dist=True)
            self.log("test/f1_macro", f1, sync_dist=True)

            self.test_auroc_macro_mc.reset()
            self.test_acc_macro_mc.reset()
            self.test_prec_macro_mc.reset()
            self.test_rec_macro_mc.reset()
            self.test_f1_macro_mc.reset()

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        # 1. Separate groups for Weight Decay (WD) vs No-WD
        #    and Backbone vs Head
        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []

        # Define what constitutes the "head" (Adjust this string to match your model!)
        # Common names: "head"
        head_name_identifiers = ["head"] 
        
        # Helper to check if a param belongs to the head
        def is_head(name):
            return any(id in name for id in head_name_identifiers)

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Check if parameter is bias or normalization (No WD)
            if p.ndim < 2 or "bias" in name or "norm" in name or "bn" in name:
                if is_head(name):
                    head_no_decay.append(p)
                else:
                    backbone_no_decay.append(p)
            else:
                # Weights (Apply WD)
                if is_head(name):
                    head_decay.append(p)
                else:
                    backbone_decay.append(p)

        # 2. Determine Learning Rates
        # If backbone is frozen, these lists are empty, so the specific LR value won't matter
        bb_lr = self.hparams.backbone_lr or (self.hparams.lr * 0.1)
        
        # 3. Create Optimizer Groups
        optim_groups = [
            {"params": backbone_decay,    "lr": bb_lr,           "weight_decay": self.hparams.weight_decay},
            {"params": backbone_no_decay, "lr": bb_lr,           "weight_decay": 0.0},
            {"params": head_decay,        "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            {"params": head_no_decay,     "lr": self.hparams.lr, "weight_decay": 0.0},
        ]
        
        # Safety check: Did we actually find head params?
        total_head_params = len(head_decay) + len(head_no_decay)
        if total_head_params == 0:
            print(f"WARNING: No parameters matched '{head_name_identifiers}'. Check your model architecture names!")

        optimizer = AdamW(optim_groups, betas=self.hparams.betas)

        # 4. Scheduler (unchanged, your logic was good)
        def lr_lambda(current_epoch):
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            
            total = max(self.trainer.max_epochs - self.hparams.warmup_epochs, 1)
            # Clamp progress to [0, 1] to prevent math domain errors if training overruns
            progress = min(1.0, max(0.0, (current_epoch - self.hparams.warmup_epochs) / total))
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }