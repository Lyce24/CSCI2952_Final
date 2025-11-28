import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as T

class ChestXrayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, labels = [], path_index = "Path"):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.path_index = path_index

        if labels is not None and len(labels) > 0:
            self.label_cols = labels
            self.with_labels = True

            # Convert to numeric
            self.df[self.label_cols] = self.df[self.label_cols].apply(
                pd.to_numeric, errors="coerce"
            )

            # Detect NaNs BEFORE filling
            for col in self.label_cols:
                nan_idx = self.df[self.df[col].isna()].index.tolist()
                if len(nan_idx) > 0:
                    print(f"[WARNING] Column '{col}' contains NaN at indices: {nan_idx[:10]} "
                        f"(showing first 10). Total = {len(nan_idx)}")

            # Fill NaNs
            self.df[self.label_cols] = self.df[self.label_cols].fillna(0).astype("int64")

            # Verify again after fill
            for col in self.label_cols:
                if self.df[col].isna().any():
                    raise ValueError(f"[ERROR] Column '{col}' STILL contains NaNs after fill().")
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
    
class CXRSegDataset(Dataset):
    """
    CXLSeg segmentation dataset: returns (image, mask).

    CSV format:
        dicom_id    subject_id  study_id    split
    Paths:
        image: root / subject_id / study_id / dicom_id.jpg
        mask : root / subject_id / study_id / dicom_id-mask.jpg
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_root: str,
        transform=None,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        self.image_size = image_size

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((image_size, image_size))
        self.img_normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.df)

    def _get_paths(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["img_path"])
        mask_path = str(row["mask_path"])

        img_path = os.path.join(self.images_root, img_path)
        mask_path = os.path.join(self.images_root, mask_path)

        return img_path, mask_path

    def __getitem__(self, idx: int):
        img_path, mask_path = self._get_paths(idx)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        # Paired aug
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # To tensor
        img = self.to_tensor(img)          # [3, H, W], 0–1
        mask = self.to_tensor(mask)        # [1, H, W], 0–1
        mask = (mask > 0.5).float()        # binarize

        img = self.img_normalize(img)

        return img, mask
