import lightning as pl

from Data.create_datasets import ChestXrayDataset, CXRSegDataset
from torchvision import transforms as T

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import os
from typing import Optional

from PIL import Image
import torch
import torchvision.transforms.functional as F
import random
from pathlib import Path

class CXRDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, root_dir, test_csv=None,
                 batch_size=64, num_workers=4, image_size=224, task="MAE"):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)

        self.task = task
        
        if self.task != "MAE":
            self.test_csv = test_csv
            self.test_df = pd.read_csv(test_csv)

        # MAE uses simple augmentations: Resize, Horizontal Flip, ToTensor, Normalize
        self.mae_train = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.sl_train = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if self.task == "MAE":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/train",
                transform=self.mae_train,
                labels=[],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/valid",
                transform=self.val_transform,
                labels=[],
                path_index="Path"
            )
        elif self.task == "COVID":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=["Label"],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=["Label"],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=["Label"],
                path_index="Path"
            )
        elif self.task == "NIH":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=[
                    'Hernia', 'Pneumothorax', 'Nodule', 'Edema', 'Effusion', 
                    'Pleural_Thickening', 'Cardiomegaly', 'Mass', 'Fibrosis', 
                    'Consolidation', 'Pneumonia', 'Infiltration', 'Emphysema', 'Atelectasis'
                ],
                path_index="Image Index"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=[
                    'Hernia', 'Pneumothorax', 'Nodule', 'Edema', 'Effusion',
                    'Pleural_Thickening', 'Cardiomegaly', 'Mass', 'Fibrosis', 
                    'Consolidation', 'Pneumonia', 'Infiltration', 'Emphysema', 'Atelectasis'
                ],
                path_index="Image Index"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=[
                   'Hernia', 'Pneumothorax', 'Nodule', 'Edema', 'Effusion', 
                   'Pleural_Thickening', 'Cardiomegaly', 'Mass', 'Fibrosis', 
                   'Consolidation', 'Pneumonia', 'Infiltration', 'Emphysema', 'Atelectasis'
                ],
                path_index="Image Index"
            )
        elif self.task == "PNE":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=['Pneumonia'],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Pneumonia'],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Pneumonia'],
                path_index="Path"
            )
        elif self.task == "COVIDQU":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=['Label'],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Label'],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Label'],
                path_index="Path"
            )
        elif self.task == "VINDR":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/train",
                transform=self.sl_train,
                labels=['Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
                        'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
                        'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
                        'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'],
                path_index="image_id"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/train",
                transform=self.val_transform,
                labels=['Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
                        'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
                        'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
                        'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'],
                path_index="image_id"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/test",
                transform=self.val_transform,
                labels=['Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
                        'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
                        'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
                        'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'],
                path_index="image_id"
            )
        elif self.task == "TB":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=['Label'],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Label'],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['Label'],
                path_index="Path"
            )
        elif self.task == "CHESTX6":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir,
                transform=self.sl_train,
                labels=["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir,
                transform=self.val_transform,
                labels=["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir,
                transform=self.val_transform,
                labels=["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"],
                path_index="Path"
            )
        elif self.task == "ECHO":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/train",
                transform=self.sl_train,
                labels=["slvh","dlv","heart_transplant","lung_transplant","pacemaker_or_icd"],
                path_index="cxr_filename"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/val",
                transform=self.val_transform,
                labels=["slvh","dlv","heart_transplant","lung_transplant","pacemaker_or_icd"],
                path_index="cxr_filename"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/test",
                transform=self.val_transform,
                labels=["slvh","dlv","heart_transplant","lung_transplant","pacemaker_or_icd"],
                path_index="cxr_filename"
            )
        elif self.task == "chexchonet":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/images",
                transform=self.sl_train,
                labels=['composite_slvh_dlv'],
                path_index="cxr_filename"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['composite_slvh_dlv'],
                path_index="cxr_filename"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/images",
                transform=self.val_transform,
                labels=['composite_slvh_dlv'],
                path_index="cxr_filename"
            )
        elif self.task == "MEDMOD-PHYS":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/train/",
                transform=self.sl_train,
                labels=[
                    "Acute and unspecified renal failure", "Acute cerebrovascular disease", "Acute myocardial infarction", "Cardiac dysrhythmias",
                    "Chronic kidney disease", "Chronic obstructive pulmonary disease and bronchiectasis", "Complications of surgical procedures or medical care",
                    "Conduction disorders", "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
                    "Diabetes mellitus with complications", "Diabetes mellitus without complication", "Disorders of lipid metabolism",
                    "Essential hypertension", "Fluid and electrolyte disorders",
                    "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
                    "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
                    "Pleurisy; pneumothorax; pulmonary collapse", "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)", "Respiratory failure; insufficiency; arrest (adult)",
                    "Septicemia (except in labor)", "Shock"
                ],
                path_index="cxr_path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/val/",
                transform=self.val_transform,
                labels=[
                    "Acute and unspecified renal failure", "Acute cerebrovascular disease", "Acute myocardial infarction", "Cardiac dysrhythmias",
                    "Chronic kidney disease", "Chronic obstructive pulmonary disease and bronchiectasis", "Complications of surgical procedures or medical care",
                    "Conduction disorders", "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
                    "Diabetes mellitus with complications", "Diabetes mellitus without complication", "Disorders of lipid metabolism",
                    "Essential hypertension", "Fluid and electrolyte disorders",
                    "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
                    "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
                    "Pleurisy; pneumothorax; pulmonary collapse", "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)", "Respiratory failure; insufficiency; arrest (adult)",
                    "Septicemia (except in labor)", "Shock"
                ],
                path_index="cxr_path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/test/",
                transform=self.val_transform,
                labels=[
                    "Acute and unspecified renal failure", "Acute cerebrovascular disease", "Acute myocardial infarction", "Cardiac dysrhythmias",
                    "Chronic kidney disease", "Chronic obstructive pulmonary disease and bronchiectasis", "Complications of surgical procedures or medical care",
                    "Conduction disorders", "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
                    "Diabetes mellitus with complications", "Diabetes mellitus without complication", "Disorders of lipid metabolism",
                    "Essential hypertension", "Fluid and electrolyte disorders",
                    "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
                    "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
                    "Pleurisy; pneumothorax; pulmonary collapse", "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)", "Respiratory failure; insufficiency; arrest (adult)",
                    "Septicemia (except in labor)", "Shock"
                ],
                path_index="cxr_path"
            )
        elif self.task == "MEDMOD-MORT":
            self.train_dataset = ChestXrayDataset(
                df=self.train_df,
                root_dir=self.root_dir + "/train/",
                transform=self.sl_train,
                labels=[
                    "mortality_inunit","mortality","mortality_inhospital"
                ],
                path_index="cxr_path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/val/",
                transform=self.val_transform,
                labels=[
                    "mortality_inunit","mortality","mortality_inhospital"
                ],
                path_index="cxr_path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/test/",
                transform=self.val_transform,
                labels=[
                    "mortality_inunit","mortality","mortality_inhospital"
                ],
                path_index="cxr_path"
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.task == "MAE":
            return

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True
        )
        

class RandomFlipRotate:
    """Random horizontal flip + small rotation, applied to image and mask together."""
    def __init__(self, p_flip: float = 0.5, degrees: float = 10.0):
        self.p_flip = p_flip
        self.degrees = degrees

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Horizontal flip
        if random.random() < self.p_flip:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # Random rotation in [-degrees, degrees]
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle, fill=0)
        mask = F.rotate(mask, angle, fill=0)

        return img, mask

class CXRSegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        images_root: str,
        batch_size: int = 256,
        image_size: int = 224,
        task : str = "seg",
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.images_root = images_root
        self.batch_size = batch_size
        self.num_workers = len(os.sched_getaffinity(0))
        self.image_size = image_size

        self.train_df = pd.read_csv(self.train_csv, sep="\t" if self.train_csv.endswith(".tsv") else ",")
        self.val_df   = pd.read_csv(self.val_csv, sep="\t" if self.val_csv.endswith(".tsv") else ",")
        self.test_df  = pd.read_csv(self.test_csv, sep="\t" if self.test_csv.endswith(".tsv") else ",")
        self.task = task

    def setup(self, stage: Optional[str] = None):
        train_transform = RandomFlipRotate(p_flip=0.5, degrees=10.0)
        val_transform = None

        self.train_dataset = CXRSegDataset(
            self.train_df,
            images_root=self.images_root,
            transform=train_transform,
            image_size=self.image_size,
        )

        self.val_dataset = CXRSegDataset(
            self.val_df,
            images_root=self.images_root,
            transform=val_transform,
            image_size=self.image_size,
        )

        self.test_dataset = CXRSegDataset(
            self.test_df,
            images_root=self.images_root,
            transform=val_transform,
            image_size=self.image_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

import numpy as np
from torch.utils.data import Dataset, DataLoader

class SymileMIMICDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.cxr = np.load(data_dir / split / f"cxr_{split}.npy", mmap_mode="r")
        self.ecg = np.load(data_dir / split / f"ecg_{split}.npy", mmap_mode="r")
        self.labs_pct = np.load(data_dir / split / f"labs_percentiles_{split}.npy", mmap_mode="r")
        self.labs_miss = np.load(data_dir / split / f"labs_missingness_{split}.npy", mmap_mode="r")

        assert len(self.cxr) == len(self.ecg) == len(self.labs_pct) == len(self.labs_miss)
        self.N = len(self.cxr)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        cxr = torch.from_numpy(self.cxr[idx]).float()      # or half
        ecg = torch.from_numpy(self.ecg[idx]).float()
        labs_pct = torch.from_numpy(self.labs_pct[idx]).float()
        labs_miss = torch.from_numpy(self.labs_miss[idx]).float()
        return cxr, ecg, labs_pct, labs_miss

class SymileMIMICDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=256, num_workers=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        if num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_workers = 4
        else:
            self.num_workers = num_workers

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.ds_train = SymileMIMICDatasetV2(self.data_dir, "train")
            self.ds_val   = SymileMIMICDatasetV2(self.data_dir, "val")

        if stage in ("test", None):
            self.ds_test  = SymileMIMICDatasetV2(self.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple
import random
import math


class GammaCorrection(nn.Module):
    """
    Simulates X-ray exposure variations via gamma correction.
    More realistic than linear brightness for radiographs.
    """
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        super().__init__()
        self.gamma_range = gamma_range
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = random.uniform(*self.gamma_range)
        return x.clamp(min=1e-8).pow(gamma)


class LocalContrastNoise(nn.Module):
    """
    Adds spatially-varying noise to simulate detector variations.
    """
    def __init__(self, noise_std: float = 0.02):
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x.clamp(0, 1)


class GridDistortion(nn.Module):
    """
    Subtle elastic-like distortion via grid warping.
    Simulates patient positioning variations and anatomical differences.
    """
    def __init__(self, distort_limit: float = 0.05, p: float = 0.3):
        super().__init__()
        self.distort_limit = distort_limit
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        # Simple implementation: small random affine as approximation
        # For true elastic, use kornia.augmentation.RandomElasticTransform
        angle = random.uniform(-2, 2)
        scale = random.uniform(0.98, 1.02)
        shear = random.uniform(-self.distort_limit * 10, self.distort_limit * 10)
        
        return TF.affine(
            x, 
            angle=angle, 
            translate=[0, 0], 
            scale=scale, 
            shear=shear,
            fill=0
        )


# ============================================================================
# 1. Dataset with Transform Support
# ============================================================================

class SymileMIMICDatasetV2(Dataset):
    def __init__(
        self, 
        data_dir: Path, 
        split: str, 
        transform: Optional[Callable] = None
    ):
        self.split = split
        # Using mmap_mode="r" keeps memory usage low
        self.cxr = np.load(data_dir / split / f"cxr_{split}.npy", mmap_mode="r")
        self.ecg = np.load(data_dir / split / "ecg_features.npy", mmap_mode="r")
        self.labs = np.load(data_dir / split / "labs_features.npy", mmap_mode="r")
        
        self.transform = transform
        self.N = len(self.cxr)

        assert len(self.cxr) == len(self.ecg) == len(self.labs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1. Load data
        # Note: Copying avoids stride issues with negative indexing in transforms
        cxr_img = self.cxr[idx].copy() 
        
        # 2. Apply Transforms (Augmentation/Normalization)
        if self.transform:
            # Transforms usually expect Tensor [C, H, W] or PIL
            # Assuming cxr_img is [C, H, W] or [H, W]. If [H,W], add channel dim.
            cxr_tensor = torch.from_numpy(cxr_img).float()
            if cxr_tensor.ndim == 2:
                cxr_tensor = cxr_tensor.unsqueeze(0) # Add channel dim if missing
            
            cxr = self.transform(cxr_tensor)
        else:
            # Fallback if no transform provided
            cxr = torch.from_numpy(cxr_img).float()

        # 3. Load Features (Frozen embeddings)
        ecg_features = torch.from_numpy(self.ecg[idx]).float()   # [1024]
        labs_features = torch.from_numpy(self.labs[idx]).float() # [256]

        return cxr, ecg_features, labs_features

# ============================================================================
# 2. DataModule with Regularization Transforms
# ============================================================================
class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def forward(self, tensor):
        # x = (x - mean) / std  <-- Original Normalization
        # x * std + mean        <-- Undo
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)

class RobustTrainTransforms(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # 1. The Fixers
        self.un_norm = UnNormalize(self.mean, self.std)
        self.re_norm = T.Normalize(self.mean, self.std)
        
        # 2. The Augmentations (Operate in [0, 1] space)
        self.aug_pipeline = T.Compose([
            # Strong Geometric Augmentation
            T.RandomResizedCrop(
                size=img_size,
                scale=(0.75, 1.0),      # Safe for 10k regime
                ratio=(0.85, 1.15),
                antialias=True,
            ),
            
            T.RandomApply([
                T.RandomAffine(
                    degrees=10, 
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    fill=0,
                )
            ], p=0.6),

            # Intensity Augmentation (Now mathematically valid)
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0, 
                    hue=0
                )
            ], p=0.3),

            # Blur
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.15),
        ])
        
        # 3. Erasing (Best done after re-normalization)
        self.eraser = T.RandomErasing(
            p=0.25,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0, # 0 is roughly mean in normalized space
        )

    def forward(self, x):
        # x is Normalized Tensor (CheXpert Preprocessing)
        
        # 1. Un-Normalize -> [0, 1]
        x = self.un_norm(x)
        
        # 2. Apply Augmentations
        x = self.aug_pipeline(x)
        
        # 3. Re-Normalize -> Gaussian Distribution
        x = self.re_norm(x)
        
        # 4. Random Erasing
        x = self.eraser(x)
        
        return x

class SymileMIMICDataModuleV2(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 256, 
        num_workers: Optional[int] = None,
        img_size: int = 224,
        use_data_aug = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_data_aug = use_data_aug

        if num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_workers = 4
        else:
            self.num_workers = num_workers
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # 2. The "Fixer" layers
        self.un_norm = UnNormalize(self.mean, self.std)
        self.re_norm = T.Normalize(self.mean, self.std)

    def get_train_transforms(self):
            # Returns the custom nn.Module class that handles the un-norm/re-norm sandwich
            return RobustTrainTransforms(self.img_size)

    def get_val_transforms(self):
        """
        Deterministic preprocessing for val/test.
        From 320x320 -> 224x224 (no Normalize again).
        """
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
        ])

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.ds_train = SymileMIMICDatasetV2(
                self.data_dir, 
                split="train", 
                transform=self.get_train_transforms() if self.use_data_aug else self.get_val_transforms()
            )
            self.ds_val = SymileMIMICDatasetV2(
                self.data_dir, 
                split="val", 
                transform=self.get_val_transforms()   # <--- Deterministic
            )
        
        if stage in ("test", None):
            self.ds_test = SymileMIMICDatasetV2(
                self.data_dir, 
                split="test", 
                transform=self.get_val_transforms()   # <--- Deterministic
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

import lightning as pl
from Data.create_datasets import ChestXrayDataset
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import DataLoader
import random
import numpy as np

class LowDataCXRDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, root_dir, test_csv=None,
                 batch_size=64, num_workers=4, image_size=224, task="MAE",
                 subset_fraction=1.0, seed=42):
        """
        Args:
            subset_fraction (float): Percentage of training data to use (0.0 to 1.0).
                                     Default is 1.0 (100% data).
            seed (int): Random seed for reproducible subsampling.
        """
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Low Data Settings
        self.subset_fraction = subset_fraction
        self.seed = seed
        
        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)

        self.task = task
        
        if self.task != "MAE":
            self.test_csv = test_csv
            self.test_df = pd.read_csv(test_csv)

        # Transforms
        self.mae_train = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sl_train = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_stratified_subset(self, df, labels):
        """
        Creates a subset of the dataframe based on self.subset_fraction.
        Ensures at least one positive sample for every label exists in the subset.
        """
        # Return full dataset if fraction is 100%
        if self.subset_fraction >= 1.0:
            return df

        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_total = len(df)
        n_subset = int(n_total * self.subset_fraction)
        
        # Handle edge case where subset is too small
        if n_subset < len(labels):
            n_subset = len(labels) + 1

        selected_indices = set()

        # 1. Guarantee at least 1 positive sample for each label
        if labels: # Only if labels are provided (not MAE)
            for label in labels:
                # Assuming labels are columns with 0/1 or True/False
                if label in df.columns:
                    pos_rows = df[df[label] == 1].index.tolist()
                    if pos_rows:
                        # Pick one random positive sample
                        selected = random.choice(pos_rows)
                        selected_indices.add(selected)

        # 2. Fill the remaining quota randomly from the rest of the dataset
        num_selected = len(selected_indices)
        remaining_slots = n_subset - num_selected

        if remaining_slots > 0:
            all_indices = set(df.index.tolist())
            # Exclude already selected indices to avoid duplicates
            candidates = list(all_indices - selected_indices)
            
            if len(candidates) >= remaining_slots:
                random_extras = random.sample(candidates, remaining_slots)
                selected_indices.update(random_extras)
            else:
                # If we asked for more data than exists (unlikely in this logic branch), take all
                selected_indices.update(candidates)

        # Return the subset dataframe
        print(f"Low-Data Experiment: Using {len(selected_indices)}/{n_total} samples ({self.subset_fraction*100}%).")
        return df.loc[list(selected_indices)].reset_index(drop=True)

    def setup(self, stage=None):
        # We define the labels locally for each task so we can pass them to the subsampler
        
        task_labels = []
        path_idx = "Path" # default
        
        # --- Define Task Specific Configs ---
        if self.task == "MAE":
            task_labels = []
            path_idx = "Path"
            
        elif self.task == "COVID":
            task_labels = ["Label"]
            path_idx = "Path"
            
        elif self.task == "NIH":
            task_labels = [
                'Hernia', 'Pneumothorax', 'Nodule', 'Edema', 'Effusion', 
                'Pleural_Thickening', 'Cardiomegaly', 'Mass', 'Fibrosis', 
                'Consolidation', 'Pneumonia', 'Infiltration', 'Emphysema', 'Atelectasis'
            ]
            path_idx = "Image Index"

        elif self.task == "PNE":
            task_labels = ['Pneumonia']
            path_idx = "Path"
            
        elif self.task == "COVIDQU":
            task_labels = ['Label']
            path_idx = "Path"
            
        elif self.task == "VINDR":
            task_labels = [
                'Pneumothorax', 'Atelectasis', 'Mediastinal shift', 'Consolidation', 
                'Lung tumor', 'ILD', 'Calcification', 'Infiltration', 'Other lesion', 
                'Nodule/Mass', 'Pneumonia', 'Tuberculosis', 'Lung Opacity', 'Pleural effusion', 
                'Pleural thickening', 'Pulmonary fibrosis', 'Cardiomegaly', 'Aortic enlargement', 'Other diseases'
            ]
            path_idx = "image_id"
            
        elif self.task == "TB":
            task_labels = ['Label']
            path_idx = "Path"
            
        elif self.task == "CHESTX6":
            task_labels = ["Covid-19","Emphysema","Normal","Pneumonia-Bacterial","Pneumonia-Viral","Tuberculosis"]
            path_idx = "Path"
            
        elif self.task == "ECHO":
            task_labels = ["slvh","dlv","heart_transplant","lung_transplant","pacemaker_or_icd"]
            path_idx = "cxr_filename"
            
        elif self.task == "chexchonet":
            task_labels = ['composite_slvh_dlv']
            path_idx = "cxr_filename"
            
        elif self.task == "MEDMOD-PHYS":
            task_labels = [
                "Acute and unspecified renal failure", "Acute cerebrovascular disease", "Acute myocardial infarction", "Cardiac dysrhythmias",
                "Chronic kidney disease", "Chronic obstructive pulmonary disease and bronchiectasis", "Complications of surgical procedures or medical care",
                "Conduction disorders", "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
                "Diabetes mellitus with complications", "Diabetes mellitus without complication", "Disorders of lipid metabolism",
                "Essential hypertension", "Fluid and electrolyte disorders",
                "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
                "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
                "Pleurisy; pneumothorax; pulmonary collapse", "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)", "Respiratory failure; insufficiency; arrest (adult)",
                "Septicemia (except in labor)", "Shock"
            ]
            path_idx = "cxr_path"
            
        elif self.task == "MEDMOD-MORT":
            task_labels = ["mortality_inunit","mortality","mortality_inhospital"]
            path_idx = "cxr_path"
            
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        # --- Apply Low Data Logic ---
        # We assume self.train_df is what needs subsampling. 
        # Val/Test are kept full for consistent evaluation.
        train_df_subset = self._get_stratified_subset(self.train_df, task_labels)
        
        # --- Instantiate Datasets ---
        # Determine root directory logic (some use /train, some use /images)
        if self.task in ["MAE", "VINDR", "ECHO"]:
            train_dir = self.root_dir + "/train"
            val_dir = self.root_dir + "/valid" if self.task == "MAE" else self.root_dir + "/val" # Handle inconsistent naming
            if self.task == "VINDR": val_dir = self.root_dir + "/train" # Vindr split is often in csv but same folder
        elif self.task in ["CHESTX6"]:
             train_dir = self.root_dir
             val_dir = self.root_dir
        else:
            # Standard structure for NIH, COVID, etc.
            train_dir = self.root_dir + "/images"
            val_dir = self.root_dir + "/images"

        self.train_dataset = ChestXrayDataset(
            df=train_df_subset,
            root_dir=train_dir,
            transform=self.mae_train if self.task == "MAE" else self.sl_train,
            labels=task_labels,
            path_index=path_idx
        )
        
        self.val_dataset = ChestXrayDataset(
            df=self.val_df,
            root_dir=val_dir,
            transform=self.val_transform,
            labels=task_labels,
            path_index=path_idx
        )
        
        if self.task != "MAE":
             # Test Dir Logic
            if self.task in ["VINDR", "ECHO", "MEDMOD-PHYS", "MEDMOD-MORT"]:
                 test_dir = self.root_dir + "/test" if self.task != "MEDMOD-PHYS" else self.root_dir + "/test/"
            elif self.task in ["CHESTX6"]:
                 test_dir = self.root_dir
            else:
                 test_dir = self.root_dir + "/images"

            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=test_dir,
                transform=self.val_transform,
                labels=task_labels,
                path_index=path_idx
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.task == "MAE":
            return
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True
        )