import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import timm
from torchvision import transforms as T

from Modules.lightning_modules import MAELightningModule
import torch.nn as nn
from torchvision import models


class ECGEncoder(nn.Module):
    def __init__(self, d, pretrained):
        """
        Initialize the ECGEncoder, which encodes ECG data using a modified
        ResNet-18 architecture.

        If `args.pretrained` is True, the ResNet-18 model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V1"). The
        first convolutional layer of ResNet-18 is modified to accept single-
        channel input by changing the number of input channels to 1. The fully
        connected layer (fc) of ResNet-18 is replaced with a new Linear layer to
        match the desired output dimensionality (`args.d`). A LayerNorm layer is
        added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for
                              the model.
        """
        super().__init__()

        if pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, d, bias=True)

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): ECG data (batch_sz, 1, 5000, 12).
        Returns:
            x (torch.Tensor): learned ECG representation (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x


class LabsEncoder(nn.Module):
    def __init__(self, d):
        """
        Initialize the LabsEncoder, which encodes laboratory test results using
        a multi-layer perceptron (MLP) architecture.

        The encoder consists of three fully connected layers (fc1, fc2, fc3) with
        GELU activation functions. A LayerNorm layer is added to normalize the
        output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): concatenated laboratory percentiles and missingness
                              data (batch_sz, 100).
        Returns:
            x (torch.Tensor): learned labs representation (batch_sz, d)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


class SymileMIMICModel(nn.Module):
    def __init__(self, d, pretrained):
        """
        Initialize the PyTorch Lightning module, which learns CXR, ECG, and labs
        representations using either the Symile or CLIP loss.

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.ecg_encoder = ECGEncoder(d, pretrained)
        self.labs_encoder = LabsEncoder(d)

    def forward_ecg_features(self, x):
        return self.ecg_encoder(x)

    def forward_labs_features(self, x):
        return self.labs_encoder(x)


# ------------------------------------
# Teacher–Student PACX-MAE Lightning
# ------------------------------------
class PACXTeacherStudentModule(pl.LightningModule):
    """
    Teacher-student module:

    Teacher:
      - Frozen MAE/ViT CXR encoder
      - Frozen ECG encoder
      - Frozen Labs encoder
      - Fusion MLP -> teacher physiology-aware embedding

    Student:
      - Trainable MAE/ViT CXR encoder (initialized from MAE)
      - Projection head for image-image contrastive
      - Distillation head to match teacher embedding

    Batch is assumed to be: (cxr, ecg, labs)
      - cxr: (B, 3, H, W)
      - ecg: (B, 1, H_ecg, W_ecg)
      - labs: (B, 100)
    Adapt to your actual DataModule as needed.
    """

    def __init__(
        self,
        # Backbone config
        mode: str = "mae",  # "mae" or "imagenet"
        backbone_name: str = "vit_base_patch16_224",
        mae_checkpoint_path: Optional[str] = None,   # for MAELightningModule
        encoders_checkpoint_path = None,
        # Dimensionalities
        pretrained_symile_dim: int = 8192, # MUST match the checkpoint's output dim
        distill_embed_dim: int = 512,     # The dimension we distill into
        proj_dim: int = 256,              # For image-image contrastive head
        # Loss weights
        lambda_contrast: float = 1.0,
        lambda_distill: float = 1.0,
        temperature: float = 0.1,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas=(0.9, 0.95),
        warmup_epochs: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mode = mode
        self.lambda_contrast = lambda_contrast
        self.lambda_distill = lambda_distill
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs

        # --------------------------------
        # 1. Build Teacher (Frozen)
        # --------------------------------
        # A. Image Encoder
        self.teacher_cxr_encoder = self._build_cxr_encoder(
            mode, backbone_name, mae_checkpoint_path
        )
        
        # Get output dim dynamically
        dummy_in = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.teacher_cxr_encoder(dummy_in)
            teacher_out_dim = out.shape[1]

        # B. Physiology Encoders (ECG + Labs)
        # We initialize with the DIM used during Pretraining
        self.teacher_symile_model = self._build_symile_encoder(
            d=pretrained_symile_dim, 
            ckpt_path=encoders_checkpoint_path
        )
        self.ecg_encoder = self.teacher_symile_model.ecg_encoder
        self.labs_encoder = self.teacher_symile_model.labs_encoder

        # C. Fusion Module
        # Inputs: CXR (ViT dim) + ECG (Symile dim) + Labs (Symile dim)
        fusion_input_dim = teacher_out_dim + pretrained_symile_dim + pretrained_symile_dim
        
        self.teacher_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, distill_embed_dim) # Projects to shared space
        )

        self._freeze_module(self.teacher_cxr_encoder)
        self._freeze_module(self.ecg_encoder)
        self._freeze_module(self.labs_encoder)

        # --------------------------------
        # 2. Build Student (Trainable)
        # --------------------------------
        self.student_cxr_encoder = self._build_cxr_encoder(
            mode, backbone_name, mae_checkpoint_path
        )
        
        self.student_distill_head = nn.Sequential(
            nn.Linear(teacher_out_dim, distill_embed_dim),
            nn.LayerNorm(distill_embed_dim)
        )

        self.student_proj_head = nn.Sequential(
            nn.Linear(teacher_out_dim, teacher_out_dim),
            nn.GELU(),
            nn.Linear(teacher_out_dim, proj_dim)
        )

        # --------------------------------
        # 3. Augmentations
        # --------------------------------
        # Ideally move this to DataModule, but defining here for clarity
        self.aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            # Add ColorJitter if CXR allows, usually irrelevant for grayscale
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_aug = T.Compose([
             T.Resize(256),
             T.CenterCrop(224),
             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False
        module.eval()

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
        if mode == "imagenet":
            backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,  # feature extractor
            )
        elif mode == "mae":
            backbone = MAELightningModule.load_from_checkpoint(mae_checkpoint_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # If backbone has classifier, remove it
        if hasattr(backbone, "reset_classifier"):
            backbone.reset_classifier(0)
        elif hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        return backbone

    def _build_symile_encoder(self, d, encoders_checkpoint_path):
        ckpt = torch.load(encoders_checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        model = SymileMIMICModel(d = d, pretrained = False)
        model.load_state_dict(state, strict=False)
        return model

    # -----------------------
    # Core forward helpers
    # -----------------------
    def forward_student_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward through student backbone to get feature vector.
        Assumes backbone(x) returns pooled features.
        """
        feats = self.student_cxr_encoder(imgs) if self.mode == "imagenet" else self.student_cxr_encoder.encode(imgs)
        return feats

    def forward_teacher_embedding(
        self, cxr: torch.Tensor, ecg: torch.Tensor, labs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute teacher embedding from frozen encoders.
        """
        with torch.no_grad():
            img_feat = self.teacher_cxr_encoder(cxr) if self.mode == "imagenet" else self.teacher_cxr_encoder.encode(cxr)
            ecg_feat = self.ecg_encoder(ecg)            # (B, d)
            labs_feat = self.labs_encoder(labs)         # (B, d)

        fused = torch.cat([img_feat, ecg_feat, labs_feat], dim=1)
        t = self.teacher_fusion(fused)                  # (B, d)
        return t

    # -----------------------
    # Contrastive + distill
    # -----------------------
    @staticmethod
    def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Symmetric NT-Xent (SimCLR-style) between two batches of embeddings.
        z1, z2: (B, D)
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = z1 @ z2.T / temperature  # (B, B)
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_12 + loss_21)

    @staticmethod
    def _cosine_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        1 - cosine similarity, averaged.
        """
        s = F.normalize(student, dim=-1)
        t = F.normalize(teacher, dim=-1)
        return 1.0 - (s * t).sum(dim=-1).mean()

    # -----------------------
    # Lightning hooks
    # -----------------------
    def training_step(self, batch, batch_idx):
        """
        Expect batch to be (cxr, ecg, labs) or dict with these keys.
        Adapt this unpacking to your DataModule.
        """
        if isinstance(batch, dict):
            cxr = batch["cxr"]
            ecg = batch["ecg"]
            labs = batch["labs"]
        else:
            cxr, ecg, labs = batch  # type: ignore

        # Two augmented CXR views
        imgs_1 = torch.stack([self.aug(img) for img in cxr], dim=0)
        imgs_2 = torch.stack([self.aug(img) for img in cxr], dim=0)

        # Student forward
        f1 = self.forward_student_features(imgs_1)
        f2 = self.forward_student_features(imgs_2)

        z1 = self.student_proj_head(f1)
        z2 = self.student_proj_head(f2)

        contrast_loss = self._nt_xent_loss(
            z1, z2, temperature=self.temperature
        )

        # Distillation: student (view1) -> teacher embedding
        student_for_distill = self.student_distill_head(f1)  # (B, d)
        teacher_embedding = self.forward_teacher_embedding(cxr, ecg, labs)  # (B, d)

        distill_loss = self._cosine_distill_loss(
            student_for_distill, teacher_embedding
        )

        loss = self.lambda_contrast * contrast_loss + self.lambda_distill * distill_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=cxr.size(0))
        self.log("train_contrast_loss", contrast_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=cxr.size(0))
        self.log("train_distill_loss", distill_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=cxr.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            cxr = batch["cxr"]
            ecg = batch["ecg"]
            labs = batch["labs"]
        else:
            cxr, ecg, labs = batch  # type: ignore

        imgs_1 = torch.stack([self.val_aug(img) for img in cxr], dim=0)
        imgs_2 = torch.stack([self.val_aug(img) for img in cxr], dim=0)

        f1 = self.forward_student_features(imgs_1)
        f2 = self.forward_student_features(imgs_2)

        z1 = self.student_proj_head(f1)
        z2 = self.student_proj_head(f2)

        contrast_loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)

        student_for_distill = self.student_distill_head(f1)
        teacher_embedding = self.forward_teacher_embedding(cxr, ecg, labs)

        distill_loss = self._cosine_distill_loss(
            student_for_distill, teacher_embedding
        )

        loss = self.lambda_contrast * contrast_loss + self.lambda_distill * distill_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cxr.size(0))
        self.log("val_contrast_loss", contrast_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=cxr.size(0))
        self.log("val_distill_loss", distill_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=cxr.size(0))

        return loss

    def configure_optimizers(self):
        # AdamW with warmup + cosine decay, same style you used before
        params = (
            list(self.student_cxr_encoder.parameters())
            + list(self.student_proj_head.parameters())
            + list(self.student_distill_head.parameters())
            + list(self.teacher_fusion.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)

        def lr_lambda(current_epoch):
            # linear warmup
            if current_epoch < self.warmup_epochs:
                return float(current_epoch + 1) / float(
                    max(1, self.warmup_epochs)
                )
            # cosine decay
            total = max(self.trainer.max_epochs - self.warmup_epochs, 1)
            progress = (current_epoch - self.warmup_epochs) / total
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # -----------------------
    # Public: get final encoder
    # -----------------------
    def get_pacx_encoder(self) -> nn.Module:
        """
        Return the CXR encoder to use downstream (PACX-MAE encoder).
        You probably want self.student_backbone.
        """
        return self.student_cxr_encoder
