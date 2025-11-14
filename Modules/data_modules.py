import pytorch_lightning as pl

from Data.create_datasets import ChestXrayDataset
from torchvision import transforms as T

import pandas as pd
from torch.utils.data import DataLoader

class CheXpertDataModule(pl.LightningDataModule):
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
        self.train_transform = T.Compose([
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
                transform=self.train_transform,
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
                root_dir=self.root_dir + "/train",
                transform=self.val_transform,
                labels=["Label"],
                path_index="Path"
            )
            self.val_dataset = ChestXrayDataset(
                df=self.val_df,
                root_dir=self.root_dir + "/val",
                transform=self.val_transform,
                labels=["Label"],
                path_index="Path"
            )            
            self.test_dataset = ChestXrayDataset(
                df=self.test_df,
                root_dir=self.root_dir + "/test",
                transform=self.val_transform,
                labels=["Label"],
                path_index="Path"
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