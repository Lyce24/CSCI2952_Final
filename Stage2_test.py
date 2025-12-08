import os
import sys
import math
import time
import warnings
from typing import Optional, Dict, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
from torchvision import transforms as T
from functools import partial
import itertools

from timm.models.vision_transformer import Block, PatchEmbed
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


# Seed for reproducibility
pl.seed_everything(42, workers=True)

warnings.filterwarnings("ignore", category=UserWarning)

################
# Symile-MIMIC #
################

class SymileMIMICRetrievalDataset(Dataset):
    """Retrieval dataset for validation"""
    def __init__(self, data_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self.cxr = torch.load(self.data_dir / f"{split}/cxr_{split}.pt")
        self.ecg = torch.load(self.data_dir / f"{split}/ecg_{split}.pt")
        self.labs_percentiles = torch.load(self.data_dir / f"{split}/labs_percentiles_{split}.pt")
        self.labs_missingness = torch.load(self.data_dir / f"{split}/labs_missingness_{split}.pt")
        self.hadm_id = torch.load(self.data_dir / f"{split}/hadm_id_{split}.pt")
        self.label_hadm_id = torch.load(self.data_dir / f"{split}/label_hadm_id_{split}.pt")
        self.label = torch.load(self.data_dir / f"{split}/label_{split}.pt")

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {
            "cxr": self.cxr[idx],
            "ecg": self.ecg[idx],
            "labs_percentiles": self.labs_percentiles[idx],
            "labs_missingness": self.labs_missingness[idx],
            "hadm_id": self.hadm_id[idx],
            "label_hadm_id": self.label_hadm_id[idx],
            "label": self.label[idx]
        }

class SymileMIMICDataModule(pl.LightningDataModule):
    """Data module for Symile-MIMIC dataset"""
    def __init__(
        self,
        data_dir: str,
        batch_sz_train: int = 128,
        batch_sz_val: int = 256,
        batch_sz_test: int = 256,
        drop_last: bool = True,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_sz_train = batch_sz_train
        self.batch_sz_val = batch_sz_val
        self.batch_sz_test = batch_sz_test
        self.drop_last = drop_last
        
        # Auto-detect num workers
        if num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_workers = 4  # Default for non-Linux systems
        else:
            self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Load data for training/validation/test"""
        if stage == "fit" or stage is None:
            # Training data
            cxr_train = torch.load(self.data_dir / "train/cxr_train.pt")
            ecg_train = torch.load(self.data_dir / "train/ecg_train.pt")
            labs_percentiles_train = torch.load(self.data_dir / "train/labs_percentiles_train.pt")
            labs_missingness_train = torch.load(self.data_dir / "train/labs_missingness_train.pt")
            hadm_id_train = torch.load(self.data_dir / "train/hadm_id_train.pt")

            # Validation data
            cxr_val = torch.load(self.data_dir / "val/cxr_val.pt")
            ecg_val = torch.load(self.data_dir / "val/ecg_val.pt")
            labs_percentiles_val = torch.load(self.data_dir / "val/labs_percentiles_val.pt")
            labs_missingness_val = torch.load(self.data_dir / "val/labs_missingness_val.pt")
            hadm_id_val = torch.load(self.data_dir / "val/hadm_id_val.pt")

            # Sanity checks
            assert torch.unique(hadm_id_train).numel() == hadm_id_train.numel(), "Duplicate hadm_ids in train"
            assert torch.unique(hadm_id_val).numel() == hadm_id_val.numel(), "Duplicate hadm_ids in val"

            self.ds_train = TensorDataset(
                cxr_train, ecg_train, labs_percentiles_train,
                labs_missingness_train, hadm_id_train
            )
            self.ds_val = TensorDataset(
                cxr_val, ecg_val, labs_percentiles_val,
                labs_missingness_val, hadm_id_val
            )

        if stage == "test" or stage is None:
            # Dummy test dataset (required by Lightning)
            self.ds_test = TensorDataset(torch.zeros(1))

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_sz_train,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_sz_val,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_sz_test,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        
# ============================================================================
# Encoder Modules
# ============================================================================

class ECGEncoder(nn.Module):
    """ResNet18-based ECG encoder for 12-lead ECG signals"""
    def __init__(self, output_dim: int = 512, pretrained: bool = False):
        super().__init__()
        from torchvision import models
        
        if pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(pretrained=False)
        
        # Modify first conv for single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Project to output_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 5000, 12) ECG tensor
        Returns:
            (B, output_dim) features
        """
        x = self.resnet(x)
        x = self.projection(x)
        return x

class LabsEncoder(nn.Module):
    """3-layer MLP for lab values"""
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 100) concatenated [percentiles | missingness]
        Returns:
            (B, output_dim) features
        """
        return self.network(x)

class MAECXRModel(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

    def forward(self, x):
        """
        Encoder forward for downstream tasks: 
        no masking, use full image, return CLS token representation.
        """
        # embed patches
        x = self.patch_embed(x)                           # [N, L, D]
        x = x + self.pos_embed[:, 1:, :]                  # add pos embed (no cls yet)

        # add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)             # [N, 1+L, D]

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                  # [N, 1+L, D]

        # return CLS embedding
        return x[:, 0]                                    # [N, D]

# ============================================================================
# Projection and Prediction Heads
# ============================================================================

class ProjectionHead(nn.Module):
    """Projects embeddings to normalized contrastive space"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)

class PredictionHead(nn.Module):
    """Predicts target modality embeddings from CXR features"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ------------------------------------
# Teacher–Student PACX-MAE Lightning
# ------------------------------------
class PACXMAEModule(pl.LightningModule):
    def __init__(
        self,
        # Architecture
        cxr_embed_dim: int = 768,          # ViT-Base default
        ecg_output_dim: int = 512,
        labs_output_dim: int = 256,
        proj_dim: int = 256,               # Contrastive projection dimension
        proj_hidden_dim: int = 2048,
        pred_hidden_dim: int = 1024,
        
        # Backbone config
        init: str = "mae",  # "mae" or "imagenet" or "random"
        backbone_name: str = "vit_base_patch16_224",  # default timm ViT backbone name
        mae_checkpoint_path: Optional[str] = None,   # for MAELightningModule
        
        # Training stages
        stage = "warmup", # "warmup", "contrastive", "prediction", "finetune"
        
        # Data paths
        data_dir: Optional[Path] = None,  # For retrieval evaluation
        
        # Loss weights
        lambda_cxr_ecg: float = 1.0,
        lambda_cxr_lab: float = 1.0,
        lambda_ecg_lab: float = 0.5,
        lambda_pred_ecg: float = 0.3,
        lambda_pred_lab: float = 0.3,
        
        # Contrastive learning
        temperature: float = 0.07,
        
        # Optimizer (stage-specific)
        lr_warmup: float = 3e-4,           # Higher LR for training from scratch
        lr_contrastive: float = 1e-4,      # Lower LR when unfreezing ViT
        lr_prediction: float = 5e-5,       # Even lower for prediction phase
        lr_finetune: float = 1e-5,         # Minimal LR for supervised fine-tuning
        weight_decay: float = 0.05,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 500,
        
        # ViT unfreezing strategy
        unfreeze_strategy = "all", # "all", "last_n_blocks", or "gradual"
        unfreeze_last_n: int = 4,          # Unfreeze last 4 transformer blocks
        
        eval_retrieval_every_n_epochs: int = 5,  # Expensive, do less frequently
        compute_embedding_stats: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.init = init
        self.backbone_name = backbone_name
        
        self.stage = stage
        self.data_dir = data_dir
        self.cxr_embed_dim = cxr_embed_dim
        self.ecg_output_dim = ecg_output_dim
        self.labs_output_dim = labs_output_dim
        self.proj_dim = proj_dim
        self.eval_retrieval_every_n_epochs = eval_retrieval_every_n_epochs
        self.compute_embedding_stats = compute_embedding_stats
        
        # Loss weights
        self.lambda_cxr_ecg = lambda_cxr_ecg
        self.lambda_cxr_lab = lambda_cxr_lab
        self.lambda_ecg_lab = lambda_ecg_lab
        self.lambda_pred_ecg = lambda_pred_ecg
        self.lambda_pred_lab = lambda_pred_lab
        self.temperature = temperature
        
        # Stage-specific LRs
        self.lr_warmup = lr_warmup
        self.lr_contrastive = lr_contrastive
        self.lr_prediction = lr_prediction
        self.lr_finetune = lr_finetune
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        
        # Unfreezing config
        self.unfreeze_strategy = unfreeze_strategy
        self.unfreeze_last_n = unfreeze_last_n
        
        # CXR Encoder (MAE-pretrained ViT)
        self.cxr_encoder = self._build_cxr_encoder(self.init, self.backbone_name, mae_checkpoint_path)
        
        # ECG Encoder (ResNet18)
        self.ecg_encoder = ECGEncoder(output_dim=ecg_output_dim, pretrained=False)
        
        # Labs Encoder (MLP)
        self.labs_encoder = LabsEncoder(output_dim=labs_output_dim)
        
        # ============================================================
        # Projection Heads (for contrastive learning)
        # ============================================================
        
        self.cxr_proj = ProjectionHead(cxr_embed_dim, proj_hidden_dim, proj_dim)
        self.ecg_proj = ProjectionHead(ecg_output_dim, proj_hidden_dim, proj_dim)
        self.labs_proj = ProjectionHead(labs_output_dim, proj_hidden_dim, proj_dim)
        
        # ============================================================
        # Prediction Heads (CXR → ECG/Labs)
        # ============================================================
        
        self.pred_ecg = PredictionHead(cxr_embed_dim, pred_hidden_dim, ecg_output_dim)
        self.pred_labs = PredictionHead(cxr_embed_dim, pred_hidden_dim, labs_output_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # ============================================================
        # Data Augmentation
        # ============================================================
        
        # Training augmentation
        self.train_transform = T.Compose([
            T.Resize(256),  # Resize to slightly larger
            T.RandomCrop(224),  # Random crop to 224x224
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation augmentation (no augmentation, just resize)
        self.val_transform = T.Compose([
            T.Resize(256),  # Resize to slightly larger
            T.CenterCrop(224),  # Center crop to 224x224
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        
        # ============================================================
        # Initialize stage-specific settings
        # ============================================================
        self._configure_stage()

    # -----------------------
    # Backbone builder helper
    # -----------------------
    def _build_cxr_encoder(self, mode: str, backbone_name: str, mae_checkpoint_path: Optional[str]):
        """
        Returns a module where forward(x) -> (B, dim).
        For MAE, we assume MAELightningModule.load_from_checkpoint and that
        it exposes a .backbone or .encoder with a feature method.

        You must adapt this to YOUR MAE implementation if needed.
        """
        if mode in ["imagenet", "random"]:
            backbone = timm.create_model(
                backbone_name,
                pretrained=True if mode == "imagenet" else False,
                num_classes=0,   # feature extractor
            )

            # Remove classifier head if any
            if hasattr(backbone, "reset_classifier"):
                backbone.reset_classifier(0)
            elif hasattr(backbone, "head"):
                backbone.head = nn.Identity()
                
        elif mode == "mae":
            # Load the checkpoint
            ckpt = torch.load(
                mae_checkpoint_path,
                map_location="cpu",
                weights_only=False
            )

            state = ckpt["state_dict"]
            encoder_state = {k: v for k, v in state.items() if k.startswith("model.") and not k.startswith("model.decoder")}

            encoder_state_stripped = {}
            for k, v in encoder_state.items():
                new_key = k.replace("model.", "")   # Encoder expects keys without the "model." prefix
                encoder_state_stripped[new_key] = v
            
            # ViT Base backbone
            backbone = MAECXRModel(
                        patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    )
            
            missing, unexpected = backbone.load_state_dict(encoder_state_stripped, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return backbone
    
    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in module"""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
    
    def _unfreeze_module(self, module: nn.Module):
        """Unfreeze all parameters in module"""
        for param in module.parameters():
            param.requires_grad = True
        module.train()
        
    def _unfreeze_cxr_encoder(self):
        """Selectively unfreeze CXR encoder based on strategy"""
        if self.unfreeze_strategy == "all":
            # Unfreeze entire encoder
            self._unfreeze_module(self.cxr_encoder)
            print("  → Unfroze entire CXR encoder")
        
        elif self.unfreeze_strategy == "last_n_blocks":
            # Freeze patch embedding and early blocks, unfreeze last N transformer blocks
            self._freeze_module(self.cxr_encoder.patch_embed)
            self.cxr_encoder.pos_embed.requires_grad = False
            self.cxr_encoder.cls_token.requires_grad = False
            
            # Freeze early blocks
            total_blocks = len(self.cxr_encoder.blocks)
            freeze_until = total_blocks - self.unfreeze_last_n
            
            for i in range(freeze_until):
                self._freeze_module(self.cxr_encoder.blocks[i])
            
            # Unfreeze last N blocks
            for i in range(freeze_until, total_blocks):
                self._unfreeze_module(self.cxr_encoder.blocks[i])
            
            # Unfreeze norm layer
            self._unfreeze_module(self.cxr_encoder.norm)
            
            print(f"  → Unfroze last {self.unfreeze_last_n} blocks (out of {total_blocks})")
        
        elif self.unfreeze_strategy == "gradual":
            # Start with last block, gradually unfreeze more
            # This is handled externally via stage transitions
            self._unfreeze_module(self.cxr_encoder.blocks[-1])
            self._unfreeze_module(self.cxr_encoder.norm)
            print("  → Gradual unfreezing: last block only")
                    
    def _configure_stage(self):
        """Configure model freezing/unfreezing based on current stage"""
        if self.stage == "warmup":
            # Stage 1: Freeze CXR, train ECG + Labs only
            self._freeze_module(self.cxr_encoder)
            self._freeze_module(self.cxr_proj)
            print("Stage 1 (Warmup): CXR frozen, ECG/Labs trainable")
        
        elif self.stage == "contrastive":
            # Stage 2: Unfreeze CXR (selectively)
            self._unfreeze_cxr_encoder()
            print(f"Stage 2 (Contrastive): CXR unfrozen ({self.unfreeze_strategy})")
        
        elif self.stage == "prediction":
            # Stage 3: All encoders + prediction heads trainable
            self._unfreeze_module(self.cxr_encoder)
            print("Stage 3 (Prediction): All modules trainable")
        
        elif self.stage == "finetune":
            # Stage 4: Fine-tune everything with minimal LR
            self._unfreeze_module(self.cxr_encoder)
            print("Stage 4 (Fine-tuning): Supervised fine-tuning")
            
    # ============================================================
    # Forward Methods
    # ============================================================
    
    def forward_cxr(self, cxr: torch.Tensor, transform: bool = True) -> torch.Tensor:
        """Forward CXR through encoder"""
        if transform:
            if self.training:
                cxr = torch.stack([self.train_transform(img) for img in cxr])
            else:
                cxr = torch.stack([self.val_transform(img) for img in cxr])
        return self.cxr_encoder(cxr)
    
    def forward_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """Forward ECG through encoder"""
        return self.ecg_encoder(ecg)
    
    def forward_labs(self, labs: torch.Tensor) -> torch.Tensor:
        """Forward Labs through encoder"""
        return self.labs_encoder(labs)

    # ============================================================
    # Loss Functions
    # ============================================================
    
    @staticmethod
    def info_nce_loss(
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss (NT-Xent)
        Args:
            z1, z2: (B, D) normalized embeddings
        Returns:
            scalar loss
        """
        batch_size = z1.size(0)
        
        # Cosine similarity
        sim_matrix = torch.matmul(z1, z2.T) / temperature  # (B, B)
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=z1.device)
        
        # Cross-entropy loss (symmetric)
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_1 + loss_2) / 2
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step with stage-dependent behavior"""
        # Unpack batch (handle both dict and tuple formats)
        if isinstance(batch, dict):
            cxr, ecg, labs = batch["cxr"], batch["ecg"], batch["labs"]
        else:
            cxr, ecg, labs_pct, labs_miss, _ = batch
            labs = torch.cat([labs_pct, labs_miss], dim=1)
        
        batch_size = cxr.size(0)
        
        # Extract features
        z_cxr = self.forward_cxr(cxr, transform=True)
        z_ecg = self.forward_ecg(ecg)
        z_labs = self.forward_labs(labs)
        
        # Project to contrastive space
        z_cxr_proj = self.cxr_proj(z_cxr)
        z_ecg_proj = self.ecg_proj(z_ecg)
        z_labs_proj = self.labs_proj(z_labs)
        
        loss_dict = {}
        
        # ============================================================
        # Stage 1: Warmup (ECG + Labs contrastive only)
        # ============================================================
        if self.stage == "warmup":
            loss_ecg_lab = self.info_nce_loss(z_ecg_proj, z_labs_proj, self.temperature)
            loss = self.lambda_ecg_lab * loss_ecg_lab
            
            loss_dict["train/loss_ecg_lab"] = loss_ecg_lab
        
        # ============================================================
        # Stage 2: Joint Contrastive (all modalities)
        # ============================================================
        elif self.stage == "contrastive":
            loss_cxr_ecg = self.info_nce_loss(z_cxr_proj, z_ecg_proj, self.temperature)
            loss_cxr_lab = self.info_nce_loss(z_cxr_proj, z_labs_proj, self.temperature)
            loss_ecg_lab = self.info_nce_loss(z_ecg_proj, z_labs_proj, self.temperature)
            
            loss = (
                self.lambda_cxr_ecg * loss_cxr_ecg + 
                self.lambda_cxr_lab * loss_cxr_lab + 
                self.lambda_ecg_lab * loss_ecg_lab
            )
            
            loss_dict["train/loss_cxr_ecg"] = loss_cxr_ecg
            loss_dict["train/loss_cxr_lab"] = loss_cxr_lab
            loss_dict["train/loss_ecg_lab"] = loss_ecg_lab
        
        # ============================================================
        # Stage 3: Prediction (contrastive + cross-modal prediction)
        # ============================================================
        elif self.stage == "prediction":
            # Contrastive losses
            loss_cxr_ecg = self.info_nce_loss(z_cxr_proj, z_ecg_proj, self.temperature)
            loss_cxr_lab = self.info_nce_loss(z_cxr_proj, z_labs_proj, self.temperature)
            loss_ecg_lab = self.info_nce_loss(z_ecg_proj, z_labs_proj, self.temperature)
            
            loss_contrastive = (
                self.lambda_cxr_ecg * loss_cxr_ecg + 
                self.lambda_cxr_lab * loss_cxr_lab + 
                self.lambda_ecg_lab * loss_ecg_lab
            )
            
            # Prediction losses (CXR → ECG/Labs)
            z_ecg_pred = self.pred_ecg(z_cxr)
            z_labs_pred = self.pred_labs(z_cxr)
            
            loss_pred_ecg = F.mse_loss(z_ecg_pred, z_ecg.detach())
            loss_pred_lab = F.mse_loss(z_labs_pred, z_labs.detach())
            
            loss_prediction = (
                self.lambda_pred_ecg * loss_pred_ecg + 
                self.lambda_pred_lab * loss_pred_lab
            )
            
            loss = loss_contrastive + loss_prediction
            
            loss_dict["train/loss_cxr_ecg"] = loss_cxr_ecg
            loss_dict["train/loss_cxr_lab"] = loss_cxr_lab
            loss_dict["train/loss_ecg_lab"] = loss_ecg_lab
            loss_dict["train/loss_pred_ecg"] = loss_pred_ecg
            loss_dict["train/loss_pred_lab"] = loss_pred_lab
            loss_dict["train/loss_contrastive"] = loss_contrastive
            loss_dict["train/loss_prediction"] = loss_prediction
        
        # ============================================================
        # Stage 4: Fine-tuning (supervised - requires labels)
        # ============================================================
        elif self.stage == "finetune":
            # This stage requires task-specific implementation
            # Example: diagnosis classification
            # logits = self.classifier(z_cxr)
            # loss = F.cross_entropy(logits, batch["labels"])
            raise NotImplementedError("Supervised fine-tuning requires task-specific head")
        
        # Log all losses
        loss_dict["train/loss"] = loss
        for key, val in loss_dict.items():
            self.log(key, val, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        # Unpack batch
        if isinstance(batch, dict):
            cxr, ecg, labs = batch["cxr"], batch["ecg"], batch["labs"]
        else:
            cxr, ecg, labs_pct, labs_miss, _ = batch
            labs = torch.cat([labs_pct, labs_miss], dim=1)
        
        batch_size = cxr.size(0)
        
        # Extract features (no augmentation)
        z_cxr = self.forward_cxr(cxr, transform=True)
        z_ecg = self.forward_ecg(ecg)
        z_labs = self.forward_labs(labs)
        
        # Project to contrastive space
        z_cxr_proj = self.cxr_proj(z_cxr)
        z_ecg_proj = self.ecg_proj(z_ecg)
        z_labs_proj = self.labs_proj(z_labs)
        
        loss_dict = {}
        
        # Compute relevant losses based on stage
        if self.stage == "warmup":
            loss_ecg_lab = self.info_nce_loss(z_ecg_proj, z_labs_proj, self.temperature)
            loss = loss_ecg_lab
            loss_dict["val/loss_ecg_lab"] = loss_ecg_lab
        
        elif self.stage in ["contrastive", "prediction"]:
            loss_cxr_ecg = self.info_nce_loss(z_cxr_proj, z_ecg_proj, self.temperature)
            loss_cxr_lab = self.info_nce_loss(z_cxr_proj, z_labs_proj, self.temperature)
            loss_ecg_lab = self.info_nce_loss(z_ecg_proj, z_labs_proj, self.temperature)
            
            loss = loss_cxr_ecg + loss_cxr_lab + 0.5 * loss_ecg_lab
            
            loss_dict.update({
                "val/loss_cxr_ecg": loss_cxr_ecg,
                "val/loss_cxr_lab": loss_cxr_lab,
                "val/loss_ecg_lab": loss_ecg_lab,
            })
            
            # Compute embedding statistics
            if self.compute_embedding_stats:
                stats = self._compute_embedding_stats(z_cxr_proj, z_ecg_proj, z_labs_proj)
                loss_dict.update(stats)
            
            # Add prediction losses if in prediction stage
            if self.stage == "prediction":
                z_ecg_pred = self.pred_ecg(z_cxr)
                z_labs_pred = self.pred_labs(z_cxr)
                
                loss_pred_ecg = F.mse_loss(z_ecg_pred, z_ecg)
                loss_pred_lab = F.mse_loss(z_labs_pred, z_labs)
                
                loss_dict.update({
                    "val/loss_pred_ecg": loss_pred_ecg,
                    "val/loss_pred_lab": loss_pred_lab,
                })
        
        loss_dict["val/loss"] = loss
        for key, val in loss_dict.items():
            self.log(key, val, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Run expensive retrieval evaluation at end of validation epoch"""
        # Only run retrieval eval every N epochs
        if (self.current_epoch + 1) % self.eval_retrieval_every_n_epochs != 0:
            return
        
        # Only run if data_dir is provided and stage is appropriate
        if self.data_dir is None or self.stage == "warmup":
            return
        
        print(f"\n{'='*80}")
        print(f"Running retrieval evaluation at epoch {self.current_epoch}")
        print(f"{'='*80}\n")
        
        try:
            # Run zero-shot retrieval
            retrieval_acc = self.zeroshot_retrieval(split="val_retrieval")
            self.log("val/retrieval_acc", retrieval_acc, prog_bar=True)
            
            # Run top-K retrieval
            topk_accs = self.topk_retrieval(split="val_retrieval", k_values=[1, 5, 10])
            for k, acc in topk_accs.items():
                self.log(f"val/retrieval_top{k}", acc, prog_bar=True)
            
            print(f"\nRetrieval Results:")
            print(f"  Zero-shot accuracy: {retrieval_acc:.4f}")
            for k, acc in topk_accs.items():
                print(f"  Top-{k} accuracy: {acc:.4f}")
            print()
            
        except Exception as e:
            print(f"Warning: Retrieval evaluation failed: {e}")
            
    def _compute_embedding_stats(
        self, 
        z_cxr: torch.Tensor, 
        z_ecg: torch.Tensor, 
        z_labs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute embedding quality metrics:
        - Alignment: Average cosine similarity of positive pairs
        - Uniformity: How uniformly embeddings are distributed on hypersphere
        - Cross-modal similarity statistics
        """
        stats = {}
        
        # Alignment (higher is better, range [0, 1])
        cxr_ecg_sim = (z_cxr * z_ecg).sum(dim=1).mean()
        cxr_lab_sim = (z_cxr * z_labs).sum(dim=1).mean()
        ecg_lab_sim = (z_ecg * z_labs).sum(dim=1).mean()
        
        stats["val/alignment_cxr_ecg"] = cxr_ecg_sim
        stats["val/alignment_cxr_lab"] = cxr_lab_sim
        stats["val/alignment_ecg_lab"] = ecg_lab_sim
        stats["val/alignment_mean"] = (cxr_ecg_sim + cxr_lab_sim + ecg_lab_sim) / 3
        
        # Uniformity (lower is better, measures clustering vs. uniform distribution)
        # Computed as log of average pairwise similarity
        uniformity_cxr = torch.pdist(z_cxr, p=2).pow(2).mul(-2).exp().mean().log()
        uniformity_ecg = torch.pdist(z_ecg, p=2).pow(2).mul(-2).exp().mean().log()
        uniformity_labs = torch.pdist(z_labs, p=2).pow(2).mul(-2).exp().mean().log()
        
        stats["val/uniformity_cxr"] = uniformity_cxr
        stats["val/uniformity_ecg"] = uniformity_ecg
        stats["val/uniformity_labs"] = uniformity_labs
        
        # Standard deviation of embeddings (should be ~1 for normalized)
        stats["val/std_cxr"] = z_cxr.std()
        stats["val/std_ecg"] = z_ecg.std()
        stats["val/std_labs"] = z_labs.std()
        
        return stats
    
    # ============================================================
    # Retrieval Evaluation (from official implementation)
    # ============================================================
    
    def get_retrieval_dataset(self, split: str) -> Dict[str, torch.Tensor]:
        """Extract embeddings for retrieval evaluation"""
        retrieval_ds = SymileMIMICRetrievalDataset(self.data_dir, split)
        
        batch_size = 128
        
        r_c, r_e, r_l = [], [], []
        hadm_id, label_hadm_id, label = [], [], []
        
        self.eval()
        with torch.no_grad():
            for batch in DataLoader(
                retrieval_ds, 
                batch_size=batch_size, 
                shuffle=False,
                drop_last=False
            ):
                # Extract features
                cxr = batch["cxr"].to(self.device)
                ecg = batch["ecg"].to(self.device)
                labs = torch.cat([
                    batch["labs_percentiles"], 
                    batch["labs_missingness"]
                ], dim=1).to(self.device)
                
                z_cxr = self.forward_cxr(cxr, transform=True)
                z_ecg = self.forward_ecg(ecg)
                z_labs = self.forward_labs(labs)
                
                r_c.append(z_cxr)
                r_e.append(z_ecg)
                r_l.append(z_labs)
                hadm_id.append(batch["hadm_id"])
                label_hadm_id.append(batch["label_hadm_id"])
                label.append(batch["label"])
        
        return {
            "r_c": torch.cat(r_c, dim=0),
            "r_e": torch.cat(r_e, dim=0),
            "r_l": torch.cat(r_l, dim=0),
            "hadm_id": torch.cat(hadm_id, dim=0),
            "label_hadm_id": torch.cat(label_hadm_id, dim=0),
            "label": torch.cat(label, dim=0),
        }
    
    def zeroshot_retrieval(self, split: str) -> float:
        """
        Zero-shot retrieval: Given ECG + Labs, retrieve correct CXR
        Returns accuracy
        """
        retrieval_ds = self.get_retrieval_dataset(split)
        
        # Get queries (positive samples)
        mask = retrieval_ds["label"] == 1
        query_r_c = retrieval_ds["r_c"][mask]
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        query_hadm_id = retrieval_ds["hadm_id"][mask]
        
        correct_pred = 0
        
        for ix, true_hadm_id in enumerate(query_hadm_id):
            r_c = query_r_c[ix]
            r_e = query_r_e[ix]
            r_l = query_r_l[ix]
            
            # Get negative candidates
            neg_mask = (retrieval_ds["label_hadm_id"] == true_hadm_id) & (retrieval_ds["label"] == 0)
            neg_r_c = retrieval_ds["r_c"][neg_mask]
            r_c_candidates = torch.cat([r_c.unsqueeze(0), neg_r_c], dim=0)
            
            # Compute logits
            logits = self.zeroshot_retrieval_logits(
                r_c_candidates, 
                [r_e, r_l], 
                self.logit_scale.exp()
            ).cpu()
            
            # Predict
            pred_ix = logits.argmax(dim=1).item()
            
            if pred_ix == 0:  # First candidate is always the correct one
                correct_pred += 1
        
        return correct_pred / len(query_hadm_id)
    
    def topk_retrieval(self, split: str, k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """
        Top-K retrieval accuracy
        Returns dict of {k: accuracy}
        """
        retrieval_ds = self.get_retrieval_dataset(split)
        
        mask = retrieval_ds["label"] == 1
        query_r_c = retrieval_ds["r_c"][mask]
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        query_hadm_id = retrieval_ds["hadm_id"][mask]
        
        topk_correct = {k: 0 for k in k_values}
        
        for ix, true_hadm_id in enumerate(query_hadm_id):
            r_c = query_r_c[ix]
            r_e = query_r_e[ix]
            r_l = query_r_l[ix]
            
            neg_mask = (retrieval_ds["label_hadm_id"] == true_hadm_id) & (retrieval_ds["label"] == 0)
            neg_r_c = retrieval_ds["r_c"][neg_mask]
            r_c_candidates = torch.cat([r_c.unsqueeze(0), neg_r_c], dim=0)
            
            logits = self.zeroshot_retrieval_logits(
                r_c_candidates, 
                [r_e, r_l], 
                self.logit_scale.exp()
            ).cpu()
            
            # Check top-K
            for k in k_values:
                if k <= len(logits[0]):
                    topk_indices = logits.topk(k, dim=1).indices[0]
                    if 0 in topk_indices:
                        topk_correct[k] += 1
        
        return {k: topk_correct[k] / len(query_hadm_id) for k in k_values}
    
    def zeroshot_retrieval_logits(
        self, 
        r_x: torch.Tensor, 
        rep_list: List[torch.Tensor], 
        logit_scale_exp: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits for retrieval based on loss function type
        """
        # Pairwise sum
        for i in range(len(rep_list)):
            rep_list[i] = rep_list[i].unsqueeze(0) if rep_list[i].dim() == 1 else rep_list[i]
        
        pairwise_sum_with_r_x = torch.zeros_like(rep_list[0] @ torch.t(r_x))
        for r in rep_list:
            pairwise_sum_with_r_x += r @ torch.t(r_x)
        
        pairwise_sum_without_r_x = torch.zeros((rep_list[0].shape[0], 1), device=rep_list[0].device)
        for x, y in itertools.combinations(rep_list, 2):
            pairwise_sum_without_r_x += torch.diagonal(x @ torch.t(y)).unsqueeze(dim=1)
        
        logits = pairwise_sum_with_r_x + pairwise_sum_without_r_x
        
        return logit_scale_exp * logits
    # ============================================================
    # Optimizer Configuration
    # ============================================================
    
    def configure_optimizers(self):
        """Stage-dependent optimizer configuration"""
        
        # Determine learning rate based on stage
        if self.stage == "warmup":
            lr = self.lr_warmup
        elif self.stage == "contrastive":
            lr = self.lr_contrastive
        elif self.stage == "prediction":
            lr = self.lr_prediction
        elif self.stage == "finetune":
            lr = self.lr_finetune
        else:
            lr = self.lr_contrastive
        
        # Collect trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Stage: {self.stage}, LR: {lr}, Trainable params: {sum(p.numel() for p in params):,}")
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.weight_decay,
            betas=self.betas
        )
        
        # Cosine annealing with linear warmup
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / float(
                max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    # ============================================================
    # Public API
    # ============================================================
    
    def get_cxr_encoder(self) -> nn.Module:
        """Return the physiology-aware CXR encoder for downstream use"""
        return self.cxr_encoder
    
    def extract_cxr_features(self, cxr: torch.Tensor) -> torch.Tensor:
        """
        Extract CXR features for downstream tasks
        Args:
            cxr: (B, 3, H, W) raw CXR images
        Returns:
            (B, cxr_embed_dim) features
        """
        return self.forward_cxr(cxr, transform=True)


# ============================================================================
# Training Script
# ============================================================================
def train_stage(
    stage: str,
    data_module: pl.LightningDataModule,
    model=None,
    mae_checkpoint: Optional[str] = None,
    prev_checkpoint: Optional[str] = None,
    max_epochs: int = 10,
    project_name: str = "PACX-MAE",
    save_dir: Optional[Path] = None,
    logger: Optional[WandbLogger] = None,
):
    """Train a single stage of the hybrid SSL pipeline."""
    save_dir = Path(save_dir) if save_dir is not None else Path("./checkpoints/pacx_mae")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Just use a simple per-stage subfolder for checkpoints
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"stage_{stage}_{timestamp}"
    run_dir = save_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # If no logger is passed, create one (e.g., in unit tests)
    if logger is None:
        logger = WandbLogger(
            project=project_name,
            name=run_name,
            save_dir=str(run_dir),
            log_model=False,
        )

    # Log/update stage information as hyperparam/tag
    logger.log_hyperparams({"stage": stage})

    # Initialize or load model
    if model is None:
        if prev_checkpoint is not None:
            print(f"Loading from previous checkpoint: {prev_checkpoint}")
            model = PACXMAEModule.load_from_checkpoint(
                prev_checkpoint,
                stage=stage,
                strict=False,
            )
            model.stage = stage
            model._configure_stage()
        else:
            print(f"Initializing new model for stage: {stage}")
            model = PACXMAEModule(
                mae_checkpoint_path=mae_checkpoint,
                data_dir=data_module.data_dir,
                stage=stage,
            )
    else:
        # If you reuse the in-memory model between stages, make sure stage is updated
        model.stage = stage
        model._configure_stage()

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision="16-mixed" if accelerator == "gpu" else 32,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        default_root_dir=str(run_dir),
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, datamodule=data_module)

    return model, ckpt_dir / "last.ckpt"


# ============================================================================
# Main Training Pipeline
# ============================================================================

if __name__ == "__main__":
    # Initialize data module
    data_module = SymileMIMICDataModule(
        data_dir="../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy",
        batch_sz_train=256,
    )
    data_module.setup("fit")

    # Paths
    mae_checkpoint = "./checkpoints/mae/last.ckpt"
    save_dir = Path("./checkpoints/pacx_mae")
    
    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}"
    
    wandb_logger = WandbLogger(
        project="PACX-MAE",
        name=run_name,
        save_dir=str(save_dir),
        log_model=False,
    )
    
    # ========================================================================
    # Stage 1: Warmup (5 epochs) - Train ECG + Labs encoders
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 1: WARMUP - Training ECG + Labs encoders (CXR frozen)")
    print("="*80 + "\n")
    
    model_stage1, ckpt_stage1 = train_stage(
        stage="warmup",
        data_module=data_module,
        mae_checkpoint=mae_checkpoint,
        max_epochs=10,
        save_dir=save_dir,
        logger=wandb_logger,
    )

    
    # ========================================================================
    # Stage 2: Joint Contrastive (15 epochs) - Unfreeze CXR
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 2: JOINT CONTRASTIVE - Multi-modal contrastive learning")
    print("="*80 + "\n")
    
    model_stage2, ckpt_stage2 = train_stage(
        stage="contrastive",
        data_module=data_module,
        prev_checkpoint=str(ckpt_stage1),
        max_epochs=50,
        save_dir=save_dir,
        logger=wandb_logger,
    )
    
    # ========================================================================
    # Stage 3: Add Prediction (20 epochs) - Cross-modal prediction
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 3: PREDICTION - Adding cross-modal prediction heads")
    print("="*80 + "\n")
    
    model_stage3, ckpt_stage3 = train_stage(
        stage="prediction",
        data_module=data_module,
        prev_checkpoint=str(ckpt_stage2),
        max_epochs=40,
        save_dir=save_dir,
        logger=wandb_logger,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Final checkpoint: {ckpt_stage3}")
    print("="*80 + "\n")
    
    # ========================================================================
    # Extract final encoder for downstream use
    # ========================================================================
    final_encoder = model_stage3.get_cxr_encoder()
    print(f"Physiology-aware CXR encoder ready for downstream tasks")
    print(f"Output dimension: {model_stage3.cxr_embed_dim}")