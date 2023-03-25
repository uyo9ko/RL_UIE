import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from PIL import Image
import numpy as np
import torch
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, data_dir='data/'):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, stage=None):
        self.train_dataset = MyDataset(os.path.join(self.data_dir, 'train'), transform=self._train_transforms())
        self.val_dataset = MyDataset(os.path.join(self.data_dir, 'val'), transform=self._val_transforms())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def _train_transforms(self):
        return A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])

    def _val_transforms(self):
        return A.Compose([
            ToTensorV2()
        ])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_name_list = os.listdir(os.path.join(self.data_dir, 'raw'))

    def __getitem__(self, idx):
        # Load raw image
        raw_image_path = os.path.join(self.data_dir, 'raw', self.image_name_list[idx])
        raw_image = np.array(Image.open(raw_image_path))

        # Load reference image (select random folder)
        ref_image_folder = np.random.choice(['acce', 'dive+', 'my_model', 'mlle', 'fusion', 'two_step', 'reference'])
        ref_image_path = os.path.join(self.data_dir, ref_image_folder, f'{idx}.jpg')
        ref_image = np.array(Image.open(ref_image_path))

        # Apply transforms (if any)
        if self.transform:
            transformed = self.transform(image=raw_image, image_ref=ref_image)
            raw_image = transformed['image']
            ref_image = transformed['image_ref']

        # Return as tensor
        return raw_image, ref_image, os.path.basename(raw_image_path)

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_dir, 'raw')))

