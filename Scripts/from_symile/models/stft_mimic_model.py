from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm

from torchvision.transforms import Compose
from torch.distributions.uniform import Uniform
import random
import math
from . import simsiam

import os
from collections import OrderedDict
import torch.nn.functional as F

from datasets import SymileMIMICRetrievalDataset
from losses import infonce, clip, symile, zeroshot_retrieval_logits
from utils import PathToStrEncoder

# ViT-b-16 CXREncoder from Kenichi Maeda
def _load_state_dict_maybe_lightning(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
    return sd

def _strip_prefix(sd, prefixes=("model.")):
    out = OrderedDict()
    for k, v in sd.items():
        newk = k
        for pref in prefixes:
            if newk.startswith(pref):
                newk = newk[len(pref):]
        out[newk] = v
    return out


class StudentCXREncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the CXREncoder, which encodes chest X-ray (CXR) images using
        a modified ViT-B-16 architecture.

        If `args.pretrained` is True, the ViT model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V2"). The
        fully connected layer (fc) of ViT-B-16 is replaced with a new Linear
        layer to match the desired output dimensionality (`args.d`). A LayerNorm
        layer is added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()

        self.transform = simsiam.ToSimSiam()

        if args.pretrained:
            self.vit = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0 
            )
        else:
            self.vit = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0 
            )

        embed_dim = self.vit.num_features

        # Map ViT embedding -> desired dim d
        self.proj = nn.Linear(embed_dim, args.d, bias=True) if args.d != embed_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(args.d)

        # optional: load custom weights
        if getattr(args, "cxr_weights_path", None):
            sd = _load_state_dict_maybe_lightning(args.cxr_weights_path)
            sd = _strip_prefix(sd, prefixes=("model.",)) 
            missing, unexpected = self.vit.load_state_dict(sd, strict=False)
            print(f"[timm ViT] missing={len(missing)}, unexpected={len(unexpected)}")

    def apply_cxr_aug(self, x, transform=None):
        if transform is None:
            transform = self.transform
            
        N = x.shape[0]

        # apply SimSiam augmentation per image tensor
        view = torch.empty_like(x)
        for i in range(N):
            aug_x = transform(x[i])
            view[i] = aug_x
        return view

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
        Returns:
            x (torch.Tensor): learned CXR representation (batch_sz, d)
        """
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        v1 = self.apply_cxr_aug(x)
        v2 = self.apply_cxr_aug(x)
        feats1 = self.vit(v1)            # (B, 768) because heads=Identity()
        feats2 = self.vit(v2)
        z1 = self.proj(feats1)           # (B, d)
        z2 = self.proj(feats2)
        return self.layer_norm(z1), self.layer_norm(z2)      # (B, d)
    
class TeacherCXREncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the CXREncoder, which encodes chest X-ray (CXR) images using
        a modified ViT-B-16 architecture.

        If `args.pretrained` is True, the ViT model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V2"). The
        fully connected layer (fc) of ViT-B-16 is replaced with a new Linear
        layer to match the desired output dimensionality (`args.d`). A LayerNorm
        layer is added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()

        self.transform = simsiam.ToSimSiam()

        if args.pretrained:
            self.vit = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0 
            )
        else:
            self.vit = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0 
            )

        embed_dim = self.vit.num_features

        # Map ViT embedding -> desired dim d
        self.proj = nn.Linear(embed_dim, args.d, bias=True) if args.d != embed_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(args.d)

        # optional: load custom weights
        if getattr(args, "cxr_weights_path", None):
            sd = _load_state_dict_maybe_lightning(args.cxr_weights_path)
            sd = _strip_prefix(sd, prefixes=("model.",)) 
            missing, unexpected = self.vit.load_state_dict(sd, strict=False)
            print(f"[timm ViT] missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
        Returns:
            x (torch.Tensor): learned CXR representation (batch_sz, d)
        """
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        feats = self.vit(x)            # (B, 768) because heads=Identity()
        z= self.proj(feats)           # (B, d)
        return self.layer_norm(z)


class ECGEncoder(nn.Module):
    def __init__(self, args):
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

        self.transform = "smd-ssl"

        if args.pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, args.d, bias=True)

        self.layer_norm = nn.LayerNorm(args.d)

        if getattr(args, "symile_mimic_weights_path", None):
            sd = _load_state_dict_maybe_lightning(args.symile_mimic_weights_path)
            sd = _strip_prefix(sd, prefixes=("ecg_encoder.",)) 
            missing, unexpected = self.resnet.load_state_dict(sd, strict=False)
            missing2, unexpected2 = self.load_state_dict(sd, strict=False)
            print(f"[Teacher ECG loaded] missing={len(missing+missing2)}, unexpected={len(unexpected+unexpected2)}")

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
    def __init__(self, args):
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
        self.fc3 = nn.Linear(1024, args.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.d)

        if getattr(args, "symile_mimic_weights_path", None):
            sd = _load_state_dict_maybe_lightning(args.symile_mimic_weights_path)
            sd = _strip_prefix(sd, prefixes=("labs_encoder.",)) 
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[Teacher ECG loaded] missing={len(missing)}, unexpected={len(unexpected)}")

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
    
class TeacherEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize ...

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.args = args

        self.loss_fn = infonce if self.args.loss_fn == "infonce" else symile

        self.ecg_encoder = ECGEncoder(self.args)
        self.cxr_encoder = TeacherCXREncoder(self.args)
        self.labs_encoder = LabsEncoder(self.args)

        # load weights
        if getattr(args, "teacher_ecg_path", None):
            sd = torch.load(args.teacher_ecg_path, map_location="cpu")
            missing, unexpected = self.ecg_encoder.load_state_dict(sd, strict=False)
            print("Teacher ECG loaded:", len(missing), "missing,", len(unexpected), "unexpected")

        if getattr(args, "teacher_lab_path", None):
            sd = torch.load(args.teacher_lab_path, map_location="cpu")
            missing, unexpected = self.labs_encoder.load_state_dict(sd, strict=False)
            print("Teacher LAB loaded:", len(missing), "missing,", len(unexpected), "unexpected")

        # freeze encoder
        self.freeze_module(self.cxr_encoder)
        self.freeze_module(self.ecg_encoder)
        self.freeze_module(self.labs_encoder)
        
        # Fusion MLP projection after concatenation
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.args.d * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.args.d)
        )

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging attributes and metrics
        self.run_info = {}

    def freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False
        module.eval()

    def forward(self, x):
        """
        Forward pass through the SymileMIMICModel. `x` is a list representing
        the training or validation dataset.

        Args:
            x (list): A list of length 5 with the following elements:
                - cxr (torch.Tensor): CXR training data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG training data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles training data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory training data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the training data (batch_sz,).
        """
        cxr = x[0]                           # (B, 3, 320, 320)
        ecg = x[1]                           # (B, 1, 5000, 12)
        labs = torch.cat([x[2], x[3]], dim=1)  # (B, 100)

        r_c = self.cxr_encoder(cxr) 
        r_e = self.ecg_encoder(ecg)
        r_l = self.labs_encoder(labs)

        fused = torch.cat([r_c, r_e, r_l], dim=1)  
        t = self.fusion_proj(fused) 

        return t
    

class STFTModel(pl.LightningModule):
    def __init__(self, **args):
        """
        Initialize the PyTorch Lightning module, which learns CXR, ECG, and labs
        representations using either the Symile or CLIP loss.

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.teacher = TeacherEncoder(self.args)
        self.student = StudentCXREncoder(self.args)

        self.loss_ssl = infonce
        self.loss_clip = clip
        self.loss_distill = nn.CosineSimilarity(dim=-1)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging attributes and metrics
        self.run_info = {}

    def forward(self, x):
        """
        Forward pass through the SymileMIMICModel. `x` is a list representing
        the training or validation dataset.

        Args:
            x (list): A list of length 5 with the following elements:
                - cxr (torch.Tensor): CXR training data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG training data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles training data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory training data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the training data (batch_sz,).
        """

        cxr = x[0]
        ecg = x[1]

        t = self.teacher([cxr, ecg, x[2], x[3], None])

        se1, se2 = self.student(cxr)

        return se1, se2, t

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch (list): A list of length 5 representing the training batch with elements:
                - cxr (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the data (batch_sz,).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        se1, se2, t = self(batch)

        # SSL loss
        L_ssl = self.loss_ssl(se1, se2, self.logit_scale.exp())

        # distillation loss
        L_distill = (1 - self.loss_distill(se1, t).mean()) + \
                    (1 - self.loss_distill(se2, t).mean())
        
        # clip loss
        L_clip = infonce(se1, t, self.teacher.logit_scale.exp()) + \
                infonce(se2, t, self.teacher.logit_scale.exp())

        # total loss
        total_loss = L_ssl + L_distill + L_clip

        # tracking to help evaluate optimization (given total correlation lower bound established in paper)
        log_n = np.log(len(batch[0]))

        self.log_dict(
            {
                "train_loss": total_loss, 
                "ssl": L_ssl,
                "distill": L_distill,
                "clip": L_clip,
                "log_n": log_n
            },
                on_step=True, 
                on_epoch=True, 
                sync_dist=False, 
                prog_bar=True
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch (list): A list of length 5 representing the validation batch.
                          Refer to the `training_step` method for detailed
                          descriptions of the elements and their shapes.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        se1, se2, t = self(batch)

        # SSL loss
        L_ssl = self.loss_ssl(se1, se2)

        # distillation loss
        L_distill = (1 - self.loss_distill(se1, t).mean()) + \
                    (1 - self.loss_distill(se2, t).mean())
        
        # clip loss
        L_clip = self.loss_clip(se1, t, self.teacher.logit_scale.exp()) + \
                 self.loss_clip(se2, t, self.teacher.logit_scale.exp())
        
        # total loss
        total_loss = L_ssl + L_distill + L_clip

        self.log("val_loss", total_loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_end(self):
        """
        Calculates and logs zeroshot retrieval accuracy for the validation set,
        and updates the `run_info` dictionary with the current epoch's metrics.
        """
        acc = self.zeroshot_retrieval("val_retrieval")

        self.log("val_acc", acc, sync_dist=True, prog_bar=False)

        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item(),
            "val_acc": acc
        }

        self.run_info.setdefault("validation_metrics", []).append(val_metrics)

    def on_train_end(self):
        """
        Stores the arguments and logging information in the `run_info` attribute,
        which is then saved to a JSON file in the specified directory.
        """
        self.run_info["args"] = self.args

        try:
            self.run_info["wandb"] = self.trainer.logger.experiment.url
        except AttributeError:
            self.run_info["wandb"] = None

        with open(self.args.save_dir / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=4, cls=PathToStrEncoder)

        # --- export ViT weights ---
        if self.trainer.is_global_zero: 
            vit_sd = self.student.vit.state_dict()
            torch.save(vit_sd, self.args.save_dir / "cxr_ViT2_stft.pt")

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        acc = self.zeroshot_retrieval("test", self.args.bootstrap)

        self.log("test_acc", acc, sync_dist=True, prog_bar=False)
