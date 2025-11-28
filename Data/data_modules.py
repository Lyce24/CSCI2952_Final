import lightning as pl

from Data.create_datasets import ChestXrayDataset, CXRSegDataset
from torchvision import transforms as T

import pandas as pd
from torch.utils.data import DataLoader

import os
from typing import Optional

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as F
import lightning as pl
import random

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