import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, labels = [], path_index = "Path"):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.path_index = path_index

        if labels is not None and len(labels) > 0:
            self.label_cols = labels
            self.with_labels = True

            # Ensure labels are numeric float (this fixes the numpy.object_ / TypeError)
            self.df[self.label_cols] = self.df[self.label_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            # Optional: if any NaNs creep in, treat them as 0
            self.df[self.label_cols] = self.df[self.label_cols].fillna(0.0)
        else:
            self.with_labels = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row[self.path_index])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not self.with_labels:
            return image

        labels_np = row[self.label_cols].to_numpy(dtype="float32")
        labels = torch.from_numpy(labels_np)

        return image, labels