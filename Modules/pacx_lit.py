import math
import warnings
from typing import Optional, Tuple, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from torchvision import transforms as T
from torchvision import models

from Models.models import MAECXREncoder, ECGEncoder, LabsEncoder

# ============================================================================
# Encoder Modules
# ============================================================================

class CXRAnchoredFusion(nn.Module):
    """CXR queries ECG/Labs - asymmetric design for better student transfer"""
    def __init__(self, cxr_dim, ecg_dim, labs_dim, hidden_dim, output_dim, dropout=0.1, num_heads=8):
        super().__init__()
        self.cxr_proj = nn.Linear(cxr_dim, hidden_dim)
        self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
        self.labs_proj = nn.Linear(labs_dim, hidden_dim)
        
        # CXR attends to ECG/Labs (asymmetric)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=num_heads, 
                                                dropout=dropout, 
                                                batch_first=True)        
        # Gated fusion to control contribution
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, z_c, z_e, z_l, return_attn_weights=False):
        """
        Args:
            modality_mask: dict with 'ecg': bool tensor, 'labs': bool tensor
                          True = keep, False = mask out (for modality dropout)
        """
        B = z_c.size(0)
        
        h_c = self.cxr_proj(z_c).unsqueeze(1)  # (B, 1, H)
        h_e = self.ecg_proj(z_e).unsqueeze(1)  # (B, 1, H)
        h_l = self.labs_proj(z_l).unsqueeze(1)  # (B, 1, H)
        
        physio = torch.cat([h_e, h_l], dim=1)  # (B, 2, H)
        
        # CXR queries ECG and Labs separately
        h_physio, attn_weights = self.cross_attn(h_c, physio, physio)
        
        x = self.norm1(h_c + h_physio)   # inject physio into CXR space
        x = self.norm2(x).squeeze(1)
        out = self.mlp(x)              # (B, output_dim)

        if return_attn_weights:
            return out, attn_weights
        return out

class ReconstructionHeads(nn.Module):
    """
    Auxiliary heads to reconstruct modality features from fused representation.
    Ensures the fused embedding preserves information from all modalities.
    """
    def __init__(self, fused_dim: int, ecg_dim: int, labs_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # ECG reconstruction
        self.ecg_recon = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ecg_dim),
        )
        
        # Labs reconstruction
        self.labs_recon = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, labs_dim),
        )
        
    def forward(self, z_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_fused: (B, fused_dim) fused representation
        Returns:
            ecg_pred: (B, ecg_dim) predicted ECG features
            labs_pred: (B, labs_dim) predicted Labs features
        """
        return self.ecg_recon(z_fused), self.labs_recon(z_fused)

# ============================================================================
# Teacher Module
# ============================================================================

class TeacherLossModule(nn.Module):
    """
    Minimal, robust loss design.
    
    Core insight: Only 2 losses are truly necessary:
    1. Reconstruction - ensures physiology is encoded
    2. Alignment - ensures semantic structure
    
    Everything else is a METRIC for monitoring, not a loss to optimize.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        recon_weight: float = 1.0,
        align_ecg_weight: float = 1.0,
        align_labs_weight: float = 1.0,
        align_cxr_weight: float = 0.3,  # smaller!
    ):
        super().__init__()
        self.temperature = temperature
        self.recon_weight = recon_weight
        self.align_ecg_weight = align_ecg_weight
        self.align_labs_weight = align_labs_weight
        self.align_cxr_weight = align_cxr_weight
    
    def forward(
        self,
        z_fused: torch.Tensor,      # (B, D) fused representation
        z_c_proj: torch.Tensor,     # (B, D) CXR projected
        z_e_proj: torch.Tensor,     # (B, D) ECG projected
        z_l_proj: torch.Tensor,     # (B, D) Labs projected
        z_e: torch.Tensor,          # (B, d_e) ECG raw features
        z_l: torch.Tensor,          # (B, d_l) Labs raw features
        ecg_pred: torch.Tensor,     # (B, d_e) reconstructed from fused
        labs_pred: torch.Tensor,    # (B, d_l) reconstructed from fused
        modality_mask: dict = None,
    ) -> dict:
        
        B = z_fused.size(0)
        device = z_fused.device
        
        if modality_mask is None:
            ecg_mask = torch.ones(B, dtype=torch.bool, device=device)
            labs_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            ecg_mask = modality_mask["ecg"]
            labs_mask = modality_mask["labs"]
        
        losses = {}
        
        # ================================================================
        # LOSS 1: Reconstruction (Primary)
        # ================================================================
        # PURPOSE: z_fused must contain enough info to recover physiology.
        #
        # This is the ONLY direct constraint that physiology is encoded.
        # Without it, z_fused could align well in CLIP space but lose
        # actual physiological detail.
        #
        # CRITICAL: Detach targets to prevent encoders from learning
        #           "easy to predict" features.
        # ================================================================
        
        loss_recon_ecg = self._masked_mse(ecg_pred, z_e.detach(), ecg_mask)
        loss_recon_labs = self._masked_mse(labs_pred, z_l.detach(), labs_mask)
        loss_recon = loss_recon_ecg + loss_recon_labs
        
        losses["recon_total"] = loss_recon
        losses["recon_ecg"] = loss_recon_ecg
        losses["recon_labs"] = loss_recon_labs
        
        # ================================================================
        # LOSS 2: Contrastive Alignment
        # ================================================================
        # PURPOSE: Create structured embedding space.
        #
        # We align fused with ALL modalities including CXR:
        #   - fused ↔ ECG: semantic link to cardiac signal
        #   - fused ↔ Labs: semantic link to metabolic signal  
        #   - fused ↔ CXR: keeps z_fused "reachable" from CXR space
        #
        # WHY include CXR alignment?
        #   The student only has CXR. If z_fused is orthogonal to z_cxr,
        #   distillation becomes impossible. We WANT them related.
        #   The reconstruction loss ensures physiology is still there.
        # ================================================================
        
        loss_align_ecg = self._masked_clip_loss(z_fused, z_e_proj, ecg_mask)
        loss_align_labs = self._masked_clip_loss(z_fused, z_l_proj, labs_mask)
        loss_align_cxr = self._clip_loss(z_fused, z_c_proj)  # Always available
        
        loss_align = (
            self.align_ecg_weight  * loss_align_ecg +
            self.align_labs_weight * loss_align_labs +
            self.align_cxr_weight  * loss_align_cxr
        )
        
        losses["align_total"] = loss_align
        losses["align_ecg"] = loss_align_ecg
        losses["align_labs"] = loss_align_labs
        losses["align_cxr"] = loss_align_cxr
        
        # ================================================================
        # TOTAL LOSS
        # ================================================================
        
        loss_total = (
            self.recon_weight * loss_recon +
            loss_align  # already weighted components
        )
        losses["total"] = loss_total
        
        # ================================================================
        # METRICS (for monitoring, NOT backpropagated)
        # ================================================================
        
        return losses
    
    def _masked_mse(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.zeros((), device=pred.device)
        mse = (pred - target).pow(2).mean(dim=-1)
        return (mse * mask.float()).sum() / mask.float().sum()
    
    def _clip_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + 
                      F.cross_entropy(logits.t(), labels))
    
    def _masked_clip_loss(self, z1, z2, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return torch.zeros((), device=z1.device)
        return self._clip_loss(z1[idx], z2[idx])

class TeacherModule(pl.LightningModule):
    """
    Teacher Module: Learns to fuse CXR + ECG + Labs via CLIP-style alignment

    Training objective:
    - Cross-modal contrastive alignment:
        L_align = L_fused-cxr + L_fused-ecg + L_fused-labs
      where each term is an InfoNCE loss between the fused embedding and
      the corresponding modality embedding.

    Validation metrics:
    - Contrastive losses
    - Cosine similarity between fused and each modality
    - Basic feature stats (mean, std)
    """

    def __init__(
        self,
        # Architecture
        mae_checkpoint_path: str,
        cxr_dim: int = 768,
        ecg_dim: int = 256,
        labs_dim: int = 256,
        fusion_hidden_dim: int = 1024,
        fusion_output_dim: int = 768,
        # Contrastive
        temperature: float = 0.07,
        # Optimizer
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        cxr_unfreeze: str = "none",   # "none", "last_n", or "all"
        cxr_unfreeze_last_n: int = 4,
        recon_weight: float = 1.0,
        align_ecg_weight: float = 1.0,
        align_labs_weight: float = 1.0,
        align_cxr_weight: float = 0.3,  # smaller!
        modality_dropout_prob: float = 0.3, # Probability to zero-out ECG/Labs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cxr_dim = cxr_dim
        self.ecg_dim = ecg_dim
        self.labs_dim = labs_dim
        self.fusion_output_dim = fusion_output_dim

        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.modality_dropout_prob = modality_dropout_prob

        self.recon_weight = recon_weight
        self.align_ecg_weight = align_ecg_weight
        self.align_labs_weight = align_labs_weight
        self.align_cxr_weight = align_cxr_weight

        # -----------------
        # Encoders
        # -----------------
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        self.ecg_encoder = ECGEncoder(output_dim=ecg_dim)
        self.labs_encoder = LabsEncoder(output_dim=labs_dim)

        # CXR unfreezing strategy
        self.cxr_unfreeze = cxr_unfreeze
        self.cxr_unfreeze_last_n = cxr_unfreeze_last_n
        self._set_cxr_trainability()

        # -----------------
        # Fusion module
        # -----------------
        self.fusion = CXRAnchoredFusion(
            cxr_dim=cxr_dim,
            ecg_dim=ecg_dim,
            labs_dim=labs_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim
        )

        # -----------------
        # Projection heads for contrastive alignment
        # All mapped into fusion_output_dim
        # -----------------
        self.cxr_proj = nn.Sequential(
            nn.Linear(cxr_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )
        self.labs_proj = nn.Sequential(
            nn.Linear(labs_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )

        # The fused embedding must be able to regenerate the physiology features
        self.recon_heads = ReconstructionHeads(
            fused_dim=fusion_output_dim,
            ecg_dim=ecg_dim,
            labs_dim=labs_dim,
        )
        
        self.loss_module = TeacherLossModule(
            temperature=temperature,
            recon_weight=recon_weight,
            align_ecg_weight=align_ecg_weight,
            align_labs_weight=align_labs_weight,
            align_cxr_weight=align_cxr_weight,
        )
        
        # Transforms
        self.val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self._print_model_info()

    # ---------------------------------------------------------------------
    # Init helpers
    # ---------------------------------------------------------------------

    def _print_model_info(self):
        def count_params(m, trainable=False):
            if trainable:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            return sum(p.numel() for p in m.parameters())
        
        print("\n" + "=" * 60)
        print("Teacher Module Initialized")
        print("=" * 60)
        print(f"  CXR encoder (trainable):  {count_params(self.cxr_encoder, True):>10,}")
        print(f"  ECG encoder:              {count_params(self.ecg_encoder):>10,}")
        print(f"  Labs encoder:             {count_params(self.labs_encoder):>10,}")
        print(f"  Fusion:                   {count_params(self.fusion):>10,}")
        proj_params = count_params(self.cxr_proj) + count_params(self.ecg_proj) + count_params(self.labs_proj)
        print(f"  Projections:              {proj_params:>10,}")
        print(f"  Reconstruction heads:     {count_params(self.recon_heads):>10,}")
        print("-" * 60)
        print(f"  Loss weights: recon={self.loss_module.recon_weight}, align_ecg={self.loss_module.align_ecg_weight}, align_labs={self.loss_module.align_labs_weight}, align_cxr={self.loss_module.align_cxr_weight}")
        print(f"  Modality dropout: {self.modality_dropout_prob}")
        print("=" * 60 + "\n")

    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder"""
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
    
    def _set_cxr_trainability(self):
        # freeze everything first
        for p in self.cxr_encoder.parameters():
            p.requires_grad = False

        if self.cxr_unfreeze == "all":
            for p in self.cxr_encoder.parameters():
                p.requires_grad = True

        elif self.cxr_unfreeze == "last_n":
            total_blocks = len(self.cxr_encoder.blocks)
            start = max(0, total_blocks - self.cxr_unfreeze_last_n)

            # unfreeze last N transformer blocks
            for i in range(start, total_blocks):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True

            # usually also unfreeze final norm + cls_token
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True
            self.cxr_encoder.cls_token.requires_grad = True

        # eval mode is still fine; gradients will flow if requires_grad=True
        self.cxr_encoder.eval()

        trainable = sum(p.numel() for p in self.cxr_encoder.parameters() if p.requires_grad)
        print(f"✓ Teacher CXR trainable params: {trainable:,}")

    # ---------------------------------------------------------------------
    # Modality dropout
    # ---------------------------------------------------------------------

    def _sample_modality_mask(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Returns bool masks where True = keep, False = drop.

        We never drop BOTH ECG and Labs for a given sample.
        """
        p = self.modality_dropout_prob
        if p <= 0.0:
            return {
                "ecg": torch.ones(batch_size, dtype=torch.bool, device=device),
                "labs": torch.ones(batch_size, dtype=torch.bool, device=device),
            }

        drop_ecg = (torch.rand(batch_size, device=device) < p)
        drop_labs = (torch.rand(batch_size, device=device) < p)

        # Prevent dropping both: if both would be dropped, keep ECG
        both = drop_ecg & drop_labs
        drop_ecg = drop_ecg & ~both  # keep labs-only when collision

        return {
            "ecg": ~drop_ecg,
            "labs": ~drop_labs,
        }

   
    def forward(
        self,
        cxr: torch.Tensor,
        ecg: torch.Tensor,
        labs: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass through teacher

        Args:
            cxr: (B, 3, H, W) - raw CXR, will be transformed to 224x224
            ecg: (B, 1, H_ecg, W_ecg) or similar
            labs: (B, 100)

        Returns:
            z_fused: (B, fusion_output_dim) - teacher fused representation
            z_e:     (B, ecg_dim) - ECG features (unprojected)
            z_l:     (B, labs_dim) - Labs features (unprojected)
            z_c:     (B, cxr_dim) - CXR features (unprojected)
        """
        B = cxr.size(0)

        # Transform CXR
        cxr = torch.stack([self.val_transform(img) for img in cxr])

        # Encode
        z_c = self.cxr_encoder(cxr)   # (B, cxr_dim)
        z_e = self.ecg_encoder(ecg)   # (B, ecg_dim)
        z_l = self.labs_encoder(labs) # (B, labs_dim)

        # Apply modality dropout (zero features) *only* on physio inputs
        if modality_mask is not None:
            ecg_keep = modality_mask["ecg"].view(B, 1).float()
            labs_keep = modality_mask["labs"].view(B, 1).float()
            z_e = z_e * ecg_keep
            z_l = z_l * labs_keep

        # Fuse (using unprojected features)
        z_fused = self.fusion(z_c, z_e, z_l)  # (B, fusion_output_dim)

        return z_fused, z_e, z_l, z_c
    
    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)
        device = cxr.device

        # Sample modality mask for this batch
        modality_mask = self._sample_modality_mask(B, device)

        # Forward
        z_fused, z_e, z_l, z_c = self(cxr, ecg, labs, modality_mask)
        z_c_proj = self.cxr_proj(z_c)
        z_e_proj = self.ecg_proj(z_e)
        z_l_proj = self.labs_proj(z_l)

        # 1) CLIP-style contrastive
        ecg_pred, labs_pred = self.recon_heads(z_fused)

        losses = self.loss_module(
            z_fused=z_fused,
            z_c_proj=z_c_proj,
            z_e_proj=z_e_proj,
            z_l_proj=z_l_proj,
            z_e=z_e,
            z_l=z_l,
            ecg_pred=ecg_pred,
            labs_pred=labs_pred,
            modality_mask=modality_mask,
        )

        # ----------------- Logging (TRAIN) -----------------
        # Main objective
        self.log("teacher/train/loss_total", losses["total"],
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        # Decomposed losses
        self.log("teacher/train/loss_recon_total", losses["recon_total"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_recon_ecg", losses["recon_ecg"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_recon_labs", losses["recon_labs"],
                 on_epoch=True, batch_size=B)

        self.log("teacher/train/loss_align_total", losses["align_total"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_cxr", losses["align_cxr"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_ecg", losses["align_ecg"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_labs", losses["align_labs"],
                 on_epoch=True, batch_size=B)

        # Modality dropout stats
        self.log("teacher/train/modality_keep_ecg",
                 modality_mask["ecg"].float().mean(),
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/modality_keep_labs",
                 modality_mask["labs"].float().mean(),
                 on_epoch=True, batch_size=B)

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)
        device = cxr.device
        
        # No modality dropout in validation
        z_fused, z_e, z_l, z_c = self(cxr, ecg, labs, modality_mask=None)
        
        # Project
        z_c_proj = self.cxr_proj(z_c)
        z_e_proj = self.ecg_proj(z_e)
        z_l_proj = self.labs_proj(z_l)
        
        # Reconstruct from fused
        ecg_pred, labs_pred = self.recon_heads(z_fused)
        
        # ─────────────────────────────────────────────────────────────
        # Compute losses
        # ─────────────────────────────────────────────────────────────
        losses = self.loss_module(
            z_fused=z_fused,
            z_c_proj=z_c_proj,
            z_e_proj=z_e_proj,
            z_l_proj=z_l_proj,
            z_e=z_e,
            z_l=z_l,
            ecg_pred=ecg_pred,
            labs_pred=labs_pred,
            modality_mask=None,
        )
        
        # ─────────────────────────────────────────────────────────────
        # Key Metrics: Physiological Gain
        # ─────────────────────────────────────────────────────────────
        # Does z_fused reconstruct physiology better than z_cxr alone?
        # This is THE metric that tells us if fusion adds value.
        
        with torch.no_grad():
            ecg_pred_cxr, labs_pred_cxr = self.recon_heads(z_c_proj)
            
            # MSE from fused (should be lower)
            mse_fused_ecg = F.mse_loss(ecg_pred, z_e)
            mse_fused_labs = F.mse_loss(labs_pred, z_l)
            
            # MSE from CXR only (should be higher)
            mse_cxr_ecg = F.mse_loss(ecg_pred_cxr, z_e)
            mse_cxr_labs = F.mse_loss(labs_pred_cxr, z_l)
            
            # Gain = improvement of fused over CXR (positive = good)
            gain_ecg = mse_cxr_ecg - mse_fused_ecg
            gain_labs = mse_cxr_labs - mse_fused_labs
            
            # Relative gain (percentage improvement)
            rel_gain_ecg = gain_ecg / (mse_cxr_ecg + 1e-8)
            rel_gain_labs = gain_labs / (mse_cxr_labs + 1e-8)
        
        # ─────────────────────────────────────────────────────────────
        # Retrieval Metrics
        # ─────────────────────────────────────────────────────────────
        
        z_f_norm = F.normalize(z_fused, dim=-1)
        z_c_norm = F.normalize(z_c_proj, dim=-1)
        z_e_norm = F.normalize(z_e_proj, dim=-1)
        z_l_norm = F.normalize(z_l_proj, dim=-1)
        
        targets = torch.arange(B, device=device)
        
        # Top-1 retrieval accuracy
        top1_fused_ecg = (z_f_norm @ z_e_norm.t()).argmax(1).eq(targets).float().mean()
        top1_fused_labs = (z_f_norm @ z_l_norm.t()).argmax(1).eq(targets).float().mean()
        top1_fused_cxr = (z_f_norm @ z_c_norm.t()).argmax(1).eq(targets).float().mean()
        top1_cxr_fused = (z_c_norm @ z_f_norm.t()).argmax(1).eq(targets).float().mean()
        
        # ─────────────────────────────────────────────────────────────
        # Representation Quality Metrics
        # ─────────────────────────────────────────────────────────────
        
        with torch.no_grad():
            # CXR-fused similarity (want 0.6-0.9: related but not identical)
            cxr_similarity = (z_f_norm * z_c_norm).sum(dim=-1).mean()
            
            # Collapse detection
            gram = z_f_norm @ z_f_norm.t()
            off_diag = gram - torch.eye(B, device=device)
            collapse_score = off_diag.abs().mean()
            
            # Feature stats
            fused_std = z_fused.std()
            fused_mean = z_fused.mean()
            
            # Attention weights distribution
            _, attn_weights = self.fusion(z_c, z_e, z_l, return_attn_weights=True)
            attn = attn_weights.squeeze(1)  # (B, 2)
            attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
            attn_to_ecg = attn[:, 0].mean()
            attn_to_labs = attn[:, 1].mean()
        
        # ─────────────────────────────────────────────────────────────
        # Logging
        # ─────────────────────────────────────────────────────────────
        self.log("teacher/val/loss_total", losses["total"],
                 prog_bar=True, batch_size=B)

        # Decomposed losses
        self.log("teacher/val/loss_recon_total", losses["recon_total"],
                 batch_size=B)
        self.log("teacher/val/loss_align_total", losses["align_total"],
                 batch_size=B)

        # Physiological gain – key metrics
        self.log("teacher/val/physio_gain_ecg", gain_ecg,
                 prog_bar=True, batch_size=B)
        self.log("teacher/val/physio_gain_labs", gain_labs,
                 prog_bar=True, batch_size=B)
        self.log("teacher/val/physio_rel_gain_ecg", rel_gain_ecg,
                 batch_size=B)
        self.log("teacher/val/physio_rel_gain_labs", rel_gain_labs,
                 batch_size=B)

        self.log("teacher/val/mse_fused_ecg", mse_fused_ecg, batch_size=B)
        self.log("teacher/val/mse_cxr_ecg", mse_cxr_ecg, batch_size=B)
        self.log("teacher/val/mse_fused_labs", mse_fused_labs, batch_size=B)
        self.log("teacher/val/mse_cxr_labs", mse_cxr_labs, batch_size=B)

        # Retrieval
        self.log("teacher/val/top1_fused_ecg", top1_fused_ecg, batch_size=B)
        self.log("teacher/val/top1_fused_labs", top1_fused_labs, batch_size=B)
        self.log("teacher/val/top1_fused_cxr", top1_fused_cxr, batch_size=B)
        self.log("teacher/val/top1_cxr_fused", top1_cxr_fused, batch_size=B)

        # Representation quality
        self.log("teacher/val/cxr_similarity", cxr_similarity, batch_size=B)
        self.log("teacher/val/collapse_score", collapse_score, batch_size=B)
        self.log("teacher/val/fused_std", fused_std, batch_size=B)
        self.log("teacher/val/fused_mean", fused_mean, batch_size=B)

        # Attention
        self.log("teacher/val/attn_entropy", attn_entropy, batch_size=B)
        self.log("teacher/val/attn_to_ecg", attn_to_ecg, batch_size=B)
        self.log("teacher/val/attn_to_labs", attn_to_labs, batch_size=B)

        return losses["total"]

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------

    def configure_optimizers(self):
        base_params = (
            list(self.ecg_encoder.parameters())
            + list(self.labs_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.cxr_proj.parameters())
            + list(self.ecg_proj.parameters())
            + list(self.labs_proj.parameters())
            + list(self.recon_heads.parameters())
        )

        param_groups = [{"params": base_params, "lr": self.lr}]

        cxr_trainable = [p for p in self.cxr_encoder.parameters() if p.requires_grad]
        if len(cxr_trainable) > 0:
            param_groups.append({"params": cxr_trainable, "lr": self.lr * 0.1})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
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
            },
        }

    # ---------------------------------------------------------------------
    # Public API for student
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_distillation_targets(
        self, 
        cxr: torch.Tensor, 
        ecg: torch.Tensor, 
        labs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get targets for student distillation.
        
        Args:
            cxr, ecg, labs: Input data
            
        Returns:
            z_fused: Target representation (contains CXR + physiology)
            z_c_proj: CXR-only projection (for comparison/debugging)
        """
        self.eval()
        z_fused, _, _, z_c = self(cxr, ecg, labs, modality_mask=None)
        z_c_proj = self.cxr_proj(z_c)
        return z_fused, z_c_proj
    
    @torch.no_grad()
    def encode_cxr_only(self, cxr: torch.Tensor) -> torch.Tensor:
        """
        Encode CXR without fusion (for student baseline comparison).
        """
        self.eval()
        cxr = torch.stack([self.val_transform(img) for img in cxr])
        z_c = self.cxr_encoder(cxr)
        return self.cxr_proj(z_c)


# ============================================================================
# Student Module
# ============================================================================


class StudentLossModule(nn.Module):
    """
    Loss for CXR-only student distilling a multimodal teacher.

    Components:
      - L_mse:    MSE between student and teacher fused embeddings
      - L_clip:   CLIP-style cosine-contrastive distillation
      - L_anchor: small MSE to teacher's CXR-only projection (stability)
      - L_mae:    optional MAE-style reconstruction loss on patches

    Only L_mse + L_clip are required. Others are optional / small-weight.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mse_weight: float = 1.0,
        clip_weight: float = 1.0,
        anchor_weight: float = 0.1,   # small, can set 0.0 to disable
    ):
        super().__init__()
        self.temperature = temperature
        self.mse_weight = mse_weight
        self.clip_weight = clip_weight
        self.anchor_weight = anchor_weight

    def forward(
        self,
        z_s: torch.Tensor,             # (B, D) student embedding
        z_fused_t: torch.Tensor,       # (B, D) teacher fused embedding
        z_cxr_t: torch.Tensor = None
    ) -> dict:
        """
        Returns:
            dict with 'total' and individual loss components / metrics.
        """
        losses = {}

        # ---------------------------------------------
        # 1) Distillation to teacher fused embedding
        # ---------------------------------------------
        z_t = z_fused_t.detach()

        loss_mse = F.mse_loss(z_s, z_t)
        loss_clip = self._clip_loss(z_s, z_t)

        losses["distill_mse"] = loss_mse
        losses["distill_clip"] = loss_clip

        # ---------------------------------------------
        # 2) CXR anchor (optional, small weight)
        # ---------------------------------------------
        if z_cxr_t is not None and self.anchor_weight > 0.0:
            z_c = z_cxr_t.detach()
            loss_anchor = F.mse_loss(z_s, z_c)
        else:
            loss_anchor = torch.zeros((), device=z_s.device)

        losses["distill_anchor"] = loss_anchor

        # ---------------------------------------------
        # Total
        # ---------------------------------------------
        loss_total = (
            self.mse_weight * loss_mse +
            self.clip_weight * loss_clip +
            self.anchor_weight * loss_anchor
        )
        losses["distill_total"] = loss_total

        return losses

    # ---------------------------------------------
    # Helper: CLIP-style loss
    # ---------------------------------------------
    def _clip_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        )


class StudentModule(pl.LightningModule):
    """
    Student Module: CXR-only encoder that matches teacher
    
    Training objective:
    - Distillation: Match teacher's fused representation using only CXR
    - This injects physiology knowledge into CXR encoder
    
    Validation metrics:
    - Distillation loss (MSE + cosine)
    - Alignment with teacher
    - Retrieval accuracy (if data_dir provided)
    """
    
    def __init__(
        self,
        # Architecture
        mae_checkpoint_path: str,
        teacher_checkpoint_path: str,
        cxr_dim: int = 768,
        student_hidden_dim: int = 768,
        
        # Unfreezing strategy
        unfreeze_strategy: Literal["all", "last_n_blocks"] = "last_n_blocks",
        unfreeze_last_n: int = 4,
        
        # Loss weights
        mse_weight: float = 1.0,
        cos_weight: float = 1.0,
        
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        
        # Validation
        eval_retrieval_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.cxr_dim = cxr_dim
        self.unfreeze_strategy = unfreeze_strategy
        self.unfreeze_last_n = unfreeze_last_n
        
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.eval_retrieval_every_n_epochs = eval_retrieval_every_n_epochs
        
        # Build student encoder
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        
        # Unfreeze strategy
        self._apply_unfreezing()
        
        # Student prediction head
        self.student_head = nn.Sequential(
            nn.Linear(cxr_dim, student_hidden_dim),
            nn.GELU(),
            nn.Linear(student_hidden_dim, cxr_dim)
        )
        
        # Load teacher (frozen)
        print(f"Loading teacher from: {teacher_checkpoint_path}")
        self.teacher = TeacherModule.load_from_checkpoint(teacher_checkpoint_path)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        print(f"✓ Teacher loaded and frozen")
        
        # Transforms
        self.train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder"""
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
    
    def _apply_unfreezing(self):
        """Apply unfreezing strategy to CXR encoder"""
        if self.unfreeze_strategy == "all":
            # Unfreeze everything
            for param in self.cxr_encoder.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.cxr_encoder.parameters())
            print(f"✓ Unfroze entire CXR encoder ({trainable:,} params)")
        
        elif self.unfreeze_strategy == "last_n_blocks":
            for p in self.cxr_encoder.parameters():
                p.requires_grad = False

            total_blocks = len(self.cxr_encoder.blocks)
            start = max(0, total_blocks - self.unfreeze_last_n)
            for i in range(start, total_blocks):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True

            num_trainable = sum(p.numel() for p in self.cxr_encoder.parameters()
                                if p.requires_grad)
            print(f"✓ Unfroze last {self.unfreeze_last_n}/{total_blocks} blocks "
                  f"({num_trainable:,} params)")
            
    def forward(self, cxr, transform: bool = True):
        """
        Forward through student
        
        Args:
            cxr: (B, 3, H, W)
            transform: If True, apply augmentation/normalization
        
        Returns:
            z_student: (B, cxr_dim) - student's prediction
        """
        if transform:
            if self.training:
                cxr = torch.stack([self.train_transform(img) for img in cxr])
            else:
                cxr = torch.stack([self.val_transform(img) for img in cxr])
        
        z = self.cxr_encoder(cxr)
        z_student = self.student_head(z)
        return z_student
    
    def training_step(self, batch, batch_idx):
        """Training step with distillation objective"""
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        
        B = cxr.size(0)
        
        # Get teacher's target (frozen)
        # Teacher targets
        with torch.no_grad():
            z_fused_t, _, _, z_c_t = self.teacher(cxr, ecg, labs, modality_mask=None)
            z_cxr_t = self.teacher.cxr_proj(z_c_t)
        
        z_s = self(cxr, transform=True)

        # Losses
        losses = self.loss_module(
            z_s=z_s,
            z_fused_t=z_fused_t,
            z_cxr_t=z_cxr_t,
        )
        loss = losses["distill_total"]

        # Main distillation objective
        self.log("student/train/distill_loss_total", loss,
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        # Components
        self.log("student/train/distill_loss_mse", losses["distill_mse"],
                 on_epoch=True, batch_size=B)
        self.log("student/train/distill_loss_clip", losses["distill_clip"],
                 on_epoch=True, batch_size=B)
        self.log("student/train/distill_loss_anchor", losses["distill_anchor"],
                 on_epoch=True, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)

        with torch.no_grad():
            z_fused_t, _, _, z_c_t = self.teacher(cxr, ecg, labs, modality_mask=None)
            z_cxr_t = self.teacher.cxr_proj(z_c_t)

        z_s = self(cxr, transform=True)

        losses = self.loss_module(
            z_s=z_s,
            z_fused_t=z_fused_t,
            z_cxr_t=z_cxr_t,
        )
        loss = losses["distill_total"]

        with torch.no_grad():
            cos_align = F.cosine_similarity(z_s, z_fused_t).mean()

        self.log("student/val/distill_loss_total", loss,
                 on_epoch=True, prog_bar=True, batch_size=B)
        self.log("student/val/distill_loss_mse", losses["distill_mse"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_loss_clip", losses["distill_clip"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_loss_anchor", losses["distill_anchor"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_alignment_cos", cos_align,
                 on_epoch=True, prog_bar=True, batch_size=B)
        return loss
    
    # ========================================================================
    # Optimizer
    # ========================================================================
    def configure_optimizers(self):
        cxr_params = [p for p in self.cxr_encoder.parameters() if p.requires_grad]
        params = cxr_params + list(self.student_head.parameters())

        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        print(f"✓ Student trainable params: {n_trainable:,}")

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
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
            },
        }

    # ========================================================================
    # Public API
    # ========================================================================
    
    def get_encoder(self):
        """Return physiology-aware CXR encoder"""
        return self.cxr_encoder