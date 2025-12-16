"""
Cross-Modal Knowledge Distillation for CXR Encoder

Goal: Fine-tune MAE ViT (CXR) so its embeddings capture physiological 
information from frozen Labs and ECG encoders.

Training: CXR + ECG + Labs available
Inference: CXR only → embeddings contain "phantom" physiology knowledge

Architecture:
- MAE ViT (768 dim) - TRAINABLE
- Labs Encoder (256 dim) - FROZEN
- ECG Encoder (1024 dim) - FROZEN

Losses:
- CLIP loss - align CXR with ECG/Labs in shared space
- Regression loss - predict ECG/Labs embeddings from CXR
- Regularization Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchvision.transforms as T
from typing import Optional, Dict, Tuple, List
from torchmetrics import MeanMetric, CosineSimilarity
import math
from torch.distributed.nn.functional import all_gather as all_gather_with_grad
from peft import LoraConfig, get_peft_model

# Assuming this import exists in your codebase
from Models.models import MAECXREncoder
import numpy as np

# ============================================================================
# Projection Heads (Discarded at Inference)
# ============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim = 256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

# ============================================================================
# Loss Module
# ============================================================================

class CLIPRegressionLoss(nn.Module):
    """
    Combined CLIP + Regression loss for cross-modal distillation.
    
    CLIP Loss: Contrastive alignment in shared embedding space
    Regression Loss: Direct prediction of target embeddings
    """
    
    def __init__(
        self,
        clip_weight: float = 1.0,
        regression_weight: float = 0.1,
        temperature: float = 0.07,
        label_smoothing: float = 0.02,
        use_ecg: bool = True,
        use_labs: bool = True,
    ):
        super().__init__()
        self.clip_weight = clip_weight
        self.regression_weight = regression_weight
        self.temperature = temperature
        self.label_smoothing=label_smoothing
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.use_ecg = use_ecg
        self.use_labs = use_labs
    
    def clip_loss(self, local_z, global_z, local_targets):
        """
        local_z: [B_local, Dim]
        global_z: [B_total, Dim] (Where B_total = B_local * World_Size)
        local_targets: [B_local] - Indices indicating where the positives are in global_z
        """
        z1 = F.normalize(local_z, dim=-1, eps=1e-6)
        z2 = F.normalize(global_z, dim=-1, eps=1e-6)

        logit_scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        
        # Logits shape: [B_local, B_total]
        logits = logit_scale * (z1 @ z2.t())
        
        return F.cross_entropy(logits, local_targets, label_smoothing=self.label_smoothing)

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(pred, p=2, dim=-1, eps=1e-6)
        target_norm = F.normalize(target.detach(), p=2, dim=-1, eps=1e-6)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        return 1.0 - cosine_sim
    
    def forward(self, 
                cxr_local, ecg_local, labs_local,       # Local embeddings (Batch B)
                cxr_global, ecg_global, labs_global,    # Global embeddings (Batch B * WorldSize)
                cxr_to_ecg_pred, cxr_to_labs_pred,      # Regression preds (Local)
                ecg_target, labs_target,                # Regression targets (Local)
                clip_targets):                          # Targets adjusted for Rank
        
        # ---- CLIP Losses (Symmetric Local->Global) ----
        zero = cxr_local.new_tensor(0.0)
        loss_dict: Dict[str, torch.Tensor] = {}
        clip_terms: List[torch.Tensor] = []

        # 1. CXR <-> ECG
        if self.use_ecg:
            loss_cxr_ecg = self.clip_loss(cxr_local, ecg_global, clip_targets)
            loss_ecg_cxr = self.clip_loss(ecg_local, cxr_global, clip_targets)
            clip_cxr_ecg = 0.5 * (loss_cxr_ecg + loss_ecg_cxr)
            loss_dict["clip_cxr_ecg"] = clip_cxr_ecg
            clip_terms.append(clip_cxr_ecg)

        # 2. CXR <-> Labs
        if self.use_labs:
            loss_cxr_labs = self.clip_loss(cxr_local, labs_global, clip_targets)
            loss_labs_cxr = self.clip_loss(labs_local, cxr_global, clip_targets)
            clip_cxr_labs = 0.5 * (loss_cxr_labs + loss_labs_cxr)
            loss_dict["clip_cxr_labs"] = clip_cxr_labs
            clip_terms.append(clip_cxr_labs)

        clip_total = torch.stack(clip_terms).mean() if clip_terms else zero
        loss_dict["clip_total"] = clip_total

        reg_terms: List[torch.Tensor] = []
        if self.use_ecg and cxr_to_ecg_pred is not None and ecg_target is not None:
            reg_ecg = self.regression_loss(cxr_to_ecg_pred, ecg_target)
            loss_dict["reg_ecg"] = reg_ecg
            reg_terms.append(reg_ecg)

        if self.use_labs and cxr_to_labs_pred is not None and labs_target is not None:
            reg_labs = self.regression_loss(cxr_to_labs_pred, labs_target)
            loss_dict["reg_labs"] = reg_labs
            reg_terms.append(reg_labs)

        reg_total = torch.stack(reg_terms).mean() if reg_terms else zero
        loss_dict["reg_total"] = reg_total

        total_loss = self.clip_weight * clip_total + self.regression_weight * reg_total
        loss_dict["total"] = total_loss
        loss_dict["logit_scale"] = self.logit_scale.exp()

        return total_loss, loss_dict


# ============================================================================
# Main Model
# ============================================================================

class CrossModalCXRDistillation(pl.LightningModule):
    """
    Cross-modal distillation: Train CXR encoder to capture ECG/Labs knowledge.
    
    Training: All three modalities available
    Inference: CXR only, but embeddings contain physiological information
    """
    
    def __init__(
        self,
        # Encoders
        mae_checkpoint_path: str,
        ablation: Optional[str], # None, "ecg", "labs"

        # Dimensions
        cxr_dim: int = 768,
        ecg_dim: int = 1024,
        labs_dim: int = 256,
        shared_dim: int = 256,

        tuning_strategy: str = "lora",  # "lora", "last_n_blocks", "full"
        
        # LoRA Params (used if tuning_strategy="lora")
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,

        # Last-N Params (used if tuning_strategy="last_n_blocks")
        train_last_n_blocks: int = 4,

        # Loss weights
        clip_weight: float = 1.0,
        regression_weight: float = 0.1,
        temperature: float = 0.07,
        label_smoothing: float = 0.02,

        # Optimizer
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 50,
        warmup_epochs = 5,
        batch_size = 128
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store dimensions
        self.cxr_dim = cxr_dim
        self.ecg_dim = ecg_dim
        self.labs_dim = labs_dim
        self.shared_dim = shared_dim
        
        # ---- CXR Encoder (TRAINABLE) ----
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        # Unfreeze ALL layers
        for param in self.cxr_encoder.parameters():
            param.requires_grad = True

        self._apply_tuning_strategy()

        self.ablation = ablation
        self.use_ecg = True
        self.use_labs = True
        if self.ablation:
            if self.ablation == "ecg":
                self.use_ecg = False
            elif self.ablation == "labs":
                self.use_labs = False
                self._val_labs_updated = False


        # ---- Projection Heads for CLIP (all trainable, discarded at inference) ----
        self.cxr_to_shared = ProjectionHead(cxr_dim, shared_dim)
        self.ecg_to_shared = ProjectionHead(ecg_dim, shared_dim)
        self.labs_to_shared = ProjectionHead(labs_dim, shared_dim)
        
        # ---- Regression Heads (trainable, discarded at inference) ----
        self.cxr_to_ecg = ProjectionHead(cxr_dim, ecg_dim)
        self.cxr_to_labs = ProjectionHead(cxr_dim, labs_dim)
        
        # ---- Loss Module ----
        self.loss_module = CLIPRegressionLoss(
            clip_weight=clip_weight,
            regression_weight=regression_weight,
            temperature=temperature,
            label_smoothing=label_smoothing,
            use_ecg=self.use_ecg,
            use_labs=self.use_labs,
        )

        self.val_cos_ecg = CosineSimilarity(reduction='mean')
        self.val_cos_labs = CosineSimilarity(reduction='mean')
        self._val_ecg_updated = True if self.use_ecg else False
        self._val_labs_updated = True if self.use_labs else False

        self.train_loss_tracker = MeanMetric()

        
        self._print_model_info()
    
    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------
    
    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder."""
        encoder = MAECXREncoder(embed_dim=768, depth=12, num_heads=12)
        print(f"Loading MAE checkpoint from: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("model.") and not k.startswith("model.decoder"):
                new_key = k.replace("model.", "")
                encoder_state[new_key] = v
        
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        print(f"✓ MAE loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return encoder

    def _apply_tuning_strategy(self):
        strategy = self.hparams.tuning_strategy
        print(f"\nApplying Fine-Tuning Strategy: {strategy.upper()}")

        # Strategy A: LoRA
        if strategy == "lora":
            config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                target_modules=["qkv", "fc1", "fc2"],
                lora_dropout=self.hparams.lora_dropout,
                bias="none",
            )

            self.cxr_encoder = get_peft_model(self.cxr_encoder, config)

            # HARD guarantee: only LoRA params trainable
            for n, p in self.cxr_encoder.named_parameters():
                p.requires_grad = ("lora_" in n)

            for n, p in self.cxr_encoder.named_parameters():
                if n.startswith("base_model.model.norm.") or n.endswith(".norm.weight") or n.endswith(".norm.bias"):
                    p.requires_grad = True

            self.cxr_encoder.print_trainable_parameters()

        # Strategy B: Last N Blocks (Partial Freezing)
        elif strategy == "last_n_blocks":
            total = len(self.cxr_encoder.blocks)
            n_train = int(self.hparams.train_last_n_blocks)
            n_train = max(0, min(n_train, total))
            cutoff = total - n_train

            print(f"Freezing first {cutoff} blocks. Training last {n_train}.")

            # Freeze everything
            for p in self.cxr_encoder.parameters():
                p.requires_grad = False

            # Unfreeze last N blocks
            for i in range(cutoff, total):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True

            # Unfreeze final norm
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True

            trainable = sum(p.numel() for p in self.cxr_encoder.parameters() if p.requires_grad)
            totalp = sum(p.numel() for p in self.cxr_encoder.parameters())
            print(f"Trainable Params: {trainable:,} ({100*trainable/totalp:.2f}%)")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _print_model_info(self):
        def count_params(m, trainable=False):
            if trainable:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            return sum(p.numel() for p in m.parameters())
        
        print("\n" + "=" * 70)
        print("Cross-Modal CXR Distillation Model")
        print("=" * 70)
        print(f"  CXR Encoder (trainable):     {count_params(self.cxr_encoder, True):>12,}")
        print("-" * 70)
        print(f"  CLIP Projection Heads:       {count_params(self.cxr_to_shared) + count_params(self.ecg_to_shared) + count_params(self.labs_to_shared):>12,}")
        print(f"  Regression Heads:            {count_params(self.cxr_to_ecg) + count_params(self.cxr_to_labs):>12,}")
        print("-" * 70)
        print(f"  Loss weights: CLIP={self.loss_module.clip_weight}, Regression={self.loss_module.regression_weight}")
        print(f"  Tuning Strategy: {self.hparams.tuning_strategy} (LoRA r={self.hparams.lora_r} alpha={self.hparams.lora_alpha} dropout={self.hparams.lora_dropout})" if self.hparams.tuning_strategy=="lora" else f"  Tuning Strategy: {self.hparams.tuning_strategy} (Last {self.hparams.train_last_n_blocks} blocks)")
        print(f"  Total Params: {count_params(self):>12,}")
        print("=" * 70 + "\n")
    
    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    
    def encode_cxr(self, cxr: torch.Tensor) -> torch.Tensor:
        """
        Encode CXR images. Use this at inference time.
        Returns enriched embeddings (768 dim).
        """
        return self.cxr_encoder(cxr)

    def forward(
        self,
        cxr: torch.Tensor,
        ecg_emb: Optional[torch.Tensor] = None,
        labs_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cxr: Raw CXR images [B, 3, H, W]
            is_training: Whether in training mode
        
        Returns:
            Dictionary with all embeddings and projections
        """
        # ---- Encode CXR (trainable) ----
        cxr_emb = self.cxr_encoder(cxr)  # [B, 768]
        labs_emb = labs_emb if self.use_labs else None
        ecg_emb = ecg_emb if self.use_ecg else None
        
        outputs = {"cxr_emb": cxr_emb}
        
        # ---- CLIP projections ----
        outputs["cxr_shared"] = self.cxr_to_shared(cxr_emb)  # [B, shared_dim]
        
        # ---- Regression predictions ----
        if self.use_ecg:
            outputs["cxr_to_ecg_pred"] = self.cxr_to_ecg(cxr_emb)   # [B, ecg_dim]
        if self.use_labs:
            outputs["cxr_to_labs_pred"] = self.cxr_to_labs(cxr_emb) # [B, labs_dim]
        
        # ---- Encode ECG/Labs (frozen) ----
        # inference-safe: only compute these when provided
        if self.use_ecg and ecg_emb is not None:
            if ecg_emb.ndim != 2 or ecg_emb.size(-1) != self.ecg_dim:
                raise ValueError(f"ecg_emb must be [B,{self.ecg_dim}], got {tuple(ecg_emb.shape)}")
            ecg_emb = ecg_emb.detach()
            outputs["ecg_emb"] = ecg_emb
            outputs["ecg_shared"] = self.ecg_to_shared(ecg_emb)

        if self.use_labs and labs_emb is not None:
            if labs_emb.ndim != 2 or labs_emb.size(-1) != self.labs_dim:
                raise ValueError(f"labs_emb must be [B,{self.labs_dim}], got {tuple(labs_emb.shape)}")
            labs_emb = labs_emb.detach()
            outputs["labs_emb"] = labs_emb
            outputs["labs_shared"] = self.labs_to_shared(labs_emb)

        return outputs
    
    # -------------------------------------------------------------------------
    # Training & Validation Steps
    # -------------------------------------------------------------------------   

    def training_step(self, batch, batch_idx):
        cxr, ecg_emb, labs_emb = batch
        if not self.use_ecg:
            ecg_emb = None
        if not self.use_labs:
            labs_emb = None

        B = cxr.size(0)

        # Forward
        outputs = self(cxr, ecg_emb, labs_emb)
        if self.trainer.world_size > 1:
            # all_gather returns [WorldSize, B, Dim] -> Flatten to [WorldSize*B, Dim]
            cxr_global = self.all_gather(outputs["cxr_shared"], sync_grads=True).flatten(0, 1)
            ecg_global = self.all_gather(outputs["ecg_shared"], sync_grads=True).flatten(0, 1) if self.use_ecg else None
            labs_global = self.all_gather(outputs["labs_shared"], sync_grads=True).flatten(0, 1) if self.use_labs else None
        else:
            # Single GPU fallback
            cxr_global = outputs["cxr_shared"]
            ecg_global = outputs["ecg_shared"] if self.use_ecg else None
            labs_global = outputs["labs_shared"] if self.use_labs else None

        targets = torch.arange(B, device=self.device) + (self.global_rank * B)

        # Compute loss
        loss, loss_dict = self.loss_module(
            # Local Queries
            cxr_local=outputs["cxr_shared"], 
            ecg_local=outputs.get("ecg_shared"), 
            labs_local=outputs.get("labs_shared"),
            # Global Keys
            cxr_global=cxr_global,
            ecg_global=ecg_global,
            labs_global=labs_global,
            # Regression (always local)
            cxr_to_ecg_pred=outputs.get("cxr_to_ecg_pred"),
            cxr_to_labs_pred=outputs.get("cxr_to_labs_pred"),
            ecg_target=outputs.get("ecg_emb"),
            labs_target=outputs.get("labs_emb"),
            # Targets
            clip_targets=targets,
        )
        
        N_ecg  = int(ecg_global.size(0)) if self.use_ecg and ecg_global is not None else None
        N_labs = int(labs_global.size(0)) if self.use_labs and labs_global is not None else None
        # total uses same candidate count; pick one (or average if you ever change it)
        N_total = N_ecg if N_ecg else N_labs

        def _log_clip_adjusted(tag: str, loss_tensor: torch.Tensor, N: int):
            if loss_tensor is None or N is None:
                return
            logN = loss_tensor.new_tensor(math.log(N))
            mi_lb = (logN - loss_tensor).detach()      # higher is better
            norm  = (loss_tensor / logN).detach()      # random ~1, perfect -> 0

            self.log(f"{tag}", loss_tensor.detach(), batch_size=B, sync_dist=True)
            self.log(f"{tag}_mi_lb", mi_lb, batch_size=B, sync_dist=True)
            self.log(f"{tag}_norm", norm, batch_size=B, sync_dist=True)
            self.log(f"{tag}_logN", logN.detach(), batch_size=B, sync_dist=True)

        # --- Per-loss logging ---
        if self.use_ecg:
            _log_clip_adjusted("train/clip_cxr_ecg",  loss_dict.get("clip_cxr_ecg"),  N_ecg)
        if self.use_labs:
            _log_clip_adjusted("train/clip_cxr_labs", loss_dict.get("clip_cxr_labs"), N_labs)
        _log_clip_adjusted("train/clip_total",    loss_dict["clip_total"],    N_total)

        # Logging
        self.log("train/loss_total", loss_dict["total"], prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train/loss_reg", loss_dict["reg_total"], batch_size=B, sync_dist=True)
        if "reg_ecg" in loss_dict:
            self.log("train/loss_reg_ecg", loss_dict["reg_ecg"], batch_size=B, sync_dist=True)
        if "reg_labs" in loss_dict:
            self.log("train/loss_reg_labs", loss_dict["reg_labs"], batch_size=B, sync_dist=True)
        self.log("train/logit_scale", loss_dict["logit_scale"], batch_size=B, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        cxr, ecg_emb, labs_emb = batch
        if not self.use_ecg:
            ecg_emb = None
        if not self.use_labs:
            labs_emb = None

        B = cxr.size(0)

        outputs = self(cxr, ecg_emb, labs_emb)

        if self.trainer.world_size > 1:
            # all_gather returns [WorldSize, B, Dim] -> Flatten to [WorldSize*B, Dim]
            cxr_global = self.all_gather(outputs["cxr_shared"], sync_grads=False).flatten(0, 1)
            ecg_global = self.all_gather(outputs["ecg_shared"], sync_grads=False).flatten(0, 1) if self.use_ecg else None
            labs_global = self.all_gather(outputs["labs_shared"], sync_grads=False).flatten(0, 1) if self.use_labs else None
        else:
            # Single GPU fallback
            cxr_global = outputs["cxr_shared"]
            ecg_global = outputs.get("ecg_shared") if self.use_ecg else None
            labs_global = outputs.get("labs_shared") if self.use_labs else None

        targets = torch.arange(B, device=self.device) + (self.global_rank * B)

        # Compute loss
        loss, loss_dict = self.loss_module(
            # Local Queries
            cxr_local=outputs["cxr_shared"], 
            ecg_local=outputs.get("ecg_shared"), 
            labs_local=outputs.get("labs_shared"),
            # Global Keys
            cxr_global=cxr_global,
            ecg_global=ecg_global,
            labs_global=labs_global,
            # Regression (always local)
            cxr_to_ecg_pred=outputs.get("cxr_to_ecg_pred"),
            cxr_to_labs_pred=outputs.get("cxr_to_labs_pred"),
            ecg_target=outputs.get("ecg_emb"),
            labs_target=outputs.get("labs_emb"),
            # Targets
            clip_targets=targets,
        )


        with torch.no_grad():
            # ---- CLIP retrieval metrics (within-batch) ----
            cxr_norm = F.normalize(outputs["cxr_shared"], dim=-1, eps=1e-6)
            targets = torch.arange(B, device=cxr_norm.device)

            k = min(5, B)  # top-k for retrieval

            if self.use_ecg:
                ecg_norm = F.normalize(outputs["ecg_shared"], dim=-1, eps=1e-6)
                sim_cxr_ecg = cxr_norm @ ecg_norm.t()

                top1_cxr_ecg = sim_cxr_ecg.argmax(1).eq(targets).float().mean()
                top5_cxr_ecg = sim_cxr_ecg.topk(k, dim=1).indices.eq(targets[:, None]).any(1).float().mean()

                pos_ecg = sim_cxr_ecg.diag().mean()

                if B > 1:
                    neg_ecg = (sim_cxr_ecg.sum() - sim_cxr_ecg.diag().sum()) / (B * (B - 1))
                else:
                    neg_ecg = sim_cxr_ecg.new_tensor(0.0)

                # ---- Regression cosine (epoch-aggregated) ----
                pred_ecg_norm = F.normalize(outputs["cxr_to_ecg_pred"], dim=-1, eps=1e-6)
                tgt_ecg_norm = F.normalize(outputs["ecg_emb"], dim=-1, eps=1e-6)
                cosine_ecg = (pred_ecg_norm * tgt_ecg_norm).sum(dim=-1).mean()

                self.val_cos_ecg(outputs["cxr_to_ecg_pred"], outputs["ecg_emb"]) 
                self._val_ecg_updated = True

                ecg_shared_norm = outputs["ecg_shared"].norm(dim=-1).mean()

            if self.use_labs:
                labs_norm = F.normalize(outputs["labs_shared"], dim=-1, eps=1e-6)
                sim_cxr_labs = cxr_norm @ labs_norm.t()

                top1_cxr_labs = sim_cxr_labs.argmax(1).eq(targets).float().mean()
                top5_cxr_labs = sim_cxr_labs.topk(k, dim=1).indices.eq(targets[:, None]).any(1).float().mean()

                # pos/neg similarity diagnostics (detect collapse / shortcutting)
                pos_labs = sim_cxr_labs.diag().mean()

                if B > 1:
                    neg_labs = (sim_cxr_labs.sum() - sim_cxr_labs.diag().sum()) / (B * (B - 1))
                else:
                    neg_labs = sim_cxr_labs.new_tensor(0.0)

                # ---- Regression cosine (epoch-aggregated) ----
                pred_labs_norm = F.normalize(outputs["cxr_to_labs_pred"], dim=-1, eps=1e-6)
                tgt_labs_norm = F.normalize(outputs["labs_emb"], dim=-1, eps=1e-6)
                cosine_labs = (pred_labs_norm * tgt_labs_norm).sum(dim=-1).mean()

                self.val_cos_labs(outputs["cxr_to_labs_pred"], outputs["labs_emb"])
                self._val_labs_updated = True

                labs_shared_norm = outputs["labs_shared"].norm(dim=-1).mean()

            # embedding health checks
            cxr_emb_norm = outputs["cxr_emb"].norm(dim=-1).mean()
            cxr_shared_norm = outputs["cxr_shared"].norm(dim=-1).mean()
            cxr_shared_std = outputs["cxr_shared"].std(dim=0).mean()

        N_ecg  = int(ecg_global.size(0)) if self.use_ecg and ecg_global is not None else None
        N_labs = int(labs_global.size(0)) if self.use_labs and labs_global is not None else None
        # total uses same candidate count; pick one (or average if you ever change it)
        N_total = N_ecg if N_ecg else N_labs

        def _log_clip_adjusted(tag: str, loss_tensor: torch.Tensor, N: int):
            if loss_tensor is None or N is None:
                return
            logN = loss_tensor.new_tensor(math.log(N))
            mi_lb = (logN - loss_tensor).detach()      # higher is better
            norm  = (loss_tensor / logN).detach()      # random ~1, perfect -> 0

            self.log(f"{tag}", loss_tensor.detach(), batch_size=B, sync_dist=True)
            self.log(f"{tag}_mi_lb", mi_lb, batch_size=B, sync_dist=True)
            self.log(f"{tag}_norm", norm, batch_size=B, sync_dist=True)
            self.log(f"{tag}_logN", logN.detach(), batch_size=B, sync_dist=True)

        # --- Per-loss logging ---
        if self.use_ecg:
            _log_clip_adjusted("val/clip_cxr_ecg",  loss_dict.get("clip_cxr_ecg"),  N_ecg)
        if self.use_labs:
            _log_clip_adjusted("val/clip_cxr_labs", loss_dict.get("clip_cxr_labs"), N_labs)
        _log_clip_adjusted("val/clip_total",    loss_dict["clip_total"],    N_total)

        # ---- Logging (epoch-level for val) ----
        self.log("val/loss_total", loss_dict["total"], prog_bar=True, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/loss_reg", loss_dict["reg_total"], batch_size=B, sync_dist=True)
        if "reg_ecg" in loss_dict:
            self.log("val/loss_reg_ecg", loss_dict["reg_ecg"], batch_size=B, sync_dist=True)
        if "reg_labs" in loss_dict:
            self.log("val/loss_reg_labs", loss_dict["reg_labs"], batch_size=B, sync_dist=True)
        self.log("val/logit_scale", loss_dict["logit_scale"], batch_size=B, sync_dist=True)

        if self.use_ecg:
            self.log("val/top1_cxr_ecg", top1_cxr_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
            self.log("val/top5_cxr_ecg", top5_cxr_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        if self.use_labs:
            self.log("val/top1_cxr_labs", top1_cxr_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
            self.log("val/top5_cxr_labs", top5_cxr_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        if self.use_ecg:
            self.log("val/pos_sim_ecg", pos_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
            self.log("val/neg_sim_ecg", neg_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        if self.use_labs:
            self.log("val/pos_sim_labs", pos_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
            self.log("val/neg_sim_labs", neg_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        self.log("val/cxr_emb_norm", cxr_emb_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/cxr_shared_norm", cxr_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        if self.use_ecg:
            self.log("val/ecg_shared_norm", ecg_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        if self.use_labs:
            self.log("val/labs_shared_norm", labs_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/cxr_shared_std", cxr_shared_std, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # epoch-aggregated regression cosines
        if self.use_ecg and self._val_ecg_updated:
            self.log("val/cosine_sim_ecg", self.val_cos_ecg.compute(), prog_bar=True, sync_dist=True)
        if self.use_labs and self._val_labs_updated:
            self.log("val/cosine_sim_labs", self.val_cos_labs.compute(), prog_bar=True, sync_dist=True)
        self.val_cos_ecg.reset()
        self.val_cos_labs.reset()
        # NOTE: May not be necessary to track these flags, but just in case
        self._val_ecg_updated = False
        self._val_labs_updated = False

    def configure_optimizers(self):
        """
        Optimizer with linear warmup + cosine decay.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Warmup + Cosine Decay scheduler
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs
        
        def lr_lambda(epoch):
            # Linear warmup
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # Cosine decay after warmup
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }