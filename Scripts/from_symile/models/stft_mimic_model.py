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

        # self.transform = Compose([
        #     RotateTransform(angle=45),
        #     ScaleTimeTransform(scale=1.5, orig_time=5000),
        #     TimeMaskTransform(max_mask_size=100)
        # ])

        if args.pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, args.d, bias=True)

        self.layer_norm = nn.LayerNorm(args.d)

    # ECG Signal augmentation from SMD-SSL ICML paper
    def mask_augmentation(self, signal, crop_rate=0.25):
        # (1, 5000, 12) for signal x[i]
        signal = signal.clone()
        if crop_rate == 0: return signal

        C, S, L = signal.shape
        crop_len = int(crop_rate * S)

        # mask random start position per lead
        for l in range(L):
            crop_start = np.random.randint(0, S - crop_len)
            # fill with Gaussian noise
            stdval = 0.5
            noise = 0.5 * stdval * np.random.randn(crop_len)
            if crop_start + crop_len <= S:
                signal[0, crop_start:crop_start+crop_len, l] = torch.tensor(noise)
            else:
                remainder = crop_len - (S-crop_start)
                signal[0, crop_start:S, l] = torch.tensor(noise[:S-crop_start])
                signal[0, 0:remainder, l] = torch.tensor(noise[S-crop_start:])
        return signal

    def apply_ecg_aug(self, x, simsiam_transforms=True, ssl_mask=True):
        N = x.shape[0]
        view = torch.empty_like(x)
        for i in range(N):
            aug_x = self.mask_augmentation(x[i])
            view[i] = aug_x
        return view

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): ECG data (batch_sz, 1, 5000, 12).
        Returns:
            x (torch.Tensor): learned ECG representation (batch_sz, d)
        """
        v1 = self.apply_ecg_aug(x)
        v2 = self.apply_ecg_aug(x)
        z1 = self.resnet(v1)
        z2 = self.resnet(v2)
        return self.layer_norm(z1), self.layer_norm(z2)

# From SCARF by Google Research team in 2021
class LabSCARFTransform:
    def __init__(self, corruption_rate=0.2): # slightly less harsh than original SCARF 
        self.corruption_rate = corruption_rate
    
    def __call__(self, x):
        N, _ = x.size()

        features_low = x.min(dim=0).values
        features_high = x.max(dim=0).values

        eps = 1e-6
        same_mask = (features_low >= features_high)
        features_high = torch.where(same_mask, features_low + eps, features_high)
        marginals = Uniform(features_low, features_high)


        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true
        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = marginals.sample(torch.Size((N,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        return x_corrupted

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

        self.transform = LabSCARFTransform(corruption_rate=0.15)

        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, args.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.d)

    def _encode(self, x):
        """
        MLP encoder for lab features
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): concatenated laboratory percentiles and missingness
                              data (batch_sz, 100).
        Returns:
            x (torch.Tensor): learned labs representation (batch_sz, d)
        """
        v1 = self.transform(x)
        v2 = self.transform(x)
        z1 = self._encode(v1)
        z2 = self._encode(v2)
        return z1, z2
    
class TeacherEncoder(nn.Module):
    def __init__(self, **args):
        """
        Initialize ...

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

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

        self.cxr_attends_ecg = nn.MultiheadAttention(
            self.args.d, self.args.num_heads, batch_first=True
        )
        # CXR attends to Labs
        self.cxr_attends_lab = nn.MultiheadAttention(
            self.args.d, self.args.num_heads, batch_first=True
        )
        # ECG attends to Labs
        self.ecg_attends_lab = nn.MultiheadAttention(
            self.args.d, self.args.num_heads, batch_first=True
        )
        
        # Fusion MLP projection after concatenation
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.args.d * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.args.d)
        )

        self.freeze_module(self.cxr_attends_ecg)
        self.freeze_module(self.cxr_attends_lab)
        self.freeze_module(self.ecg_attends_lab)
        self.freeze_module(self.fusion_proj)


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
        r_e, _ = self.ecg_encoder(ecg)
        r_l, _ = self.labs_encoder(labs)

        r_c = r_c.unsqueeze(1)               # (B, 1, d)
        r_e = r_e.unsqueeze(1)
        r_l = r_l.unsqueeze(1)

        r_c_from_e, _ = self.cxr_attends_ecg(
            query=r_c, key=r_e, value=r_e
        )
        r_c_from_l, _ = self.cxr_attends_lab(
            query=r_c, key=r_l, value=r_l
        )
        r_e_from_l, _ = self.ecg_attends_lab(
            query=r_e, key=r_l, value=r_l
        )

        r_c_upd = r_c + r_c_from_e + r_c_from_l    # (B, 1, d)
        r_e_upd = r_e + r_e_from_l
        r_l_upd = r_l

        r_c_upd = r_c_upd.squeeze(1)          # (B, d)
        r_e_upd = r_e_upd.squeeze(1)
        r_l_upd = r_l_upd.squeeze(1)

        fused = torch.cat([r_c_upd, r_e_upd, r_l_upd], dim=1)  

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

        self.teacher = TeacherEncoder(**args)
        self.student = StudentCXREncoder(**args)

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
        L_ssl = self.loss_ssl(se1, se2)

        # distillation loss
        L_distill = (1 - self.loss_distill(se1, t).mean()) + \
                    (1 - self.loss_distill(se2, t).mean())
        
        # clip loss
        L_clip = self.loss_clip(se1, t, self.teacher.logit_scale.exp()) + \
                 self.loss_clip(se2, t, self.teacher.logit_scale.exp())
        
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
