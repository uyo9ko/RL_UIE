import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from PIL import Image
import numpy as np
import torch
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, data_dir='data/'):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, stage=None):
        self.train_dataset = MyDataset(os.path.join(self.data_dir ), transform=self._train_transforms(),is_train =True)
        self.val_dataset = MyDataset(os.path.join(self.data_dir), transform=self._val_transforms())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=16)

    def _train_transforms(self):
        return A.Compose([
            # A.Resize(512,512),
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()]
            , additional_targets={'image_ref': 'image'})

    def _val_transforms(self):
        return A.Compose([
            ToTensorV2()
        ])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None,is_train=False):
        self.data_dir = data_dir
        self.transform = transform
        self.istrain = is_train
        if is_train:
            # read train_list.txt
            with open(os.path.join(self.data_dir, 'train_list.txt'), 'r') as f:
                self.image_name_list = f.read().splitlines()
        else:
            # read test_list.txt
            with open(os.path.join(self.data_dir, 'test_list.txt'), 'r') as f:
                self.image_name_list = f.read().splitlines()

    def __getitem__(self, idx):
        # Load raw image
        raw_image_path = os.path.join(self.data_dir, 'raw', self.image_name_list[idx])
        raw_image = Image.open(raw_image_path)

        # Load reference image (select random folder)
        ref_image_folder = np.random.choice(['acce', 'dive+', 'my_model', 'mlle', 'fusion', 'two_step', 'reference'])
        ref_image_path = os.path.join(self.data_dir, ref_image_folder, self.image_name_list[idx])
        if not os.path.exists(ref_image_path):
            ref_image_path = os.path.join(self.data_dir, 'reference', self.image_name_list[idx])
        ref_image = Image.open(ref_image_path)

        (ih, iw) = raw_image.size
        if ih<256 or iw<256:
            raw_image = raw_image.resize((256,256))
            ref_image = ref_image.resize((256,256))

        if not self.istrain:
            dh = ih % 8
            dw = iw % 8
            new_h, new_w = ih - dh, iw - dw
            raw_image = raw_image.resize((new_h,new_w))
            ref_image = ref_image.resize((new_h,new_w))

        raw_image = np.array(raw_image)
        ref_image = np.array(ref_image)
        # Apply transforms (if any)
        if self.transform:
            transformed = self.transform(image=raw_image, image_ref=ref_image)
            raw_image = transformed['image']
            ref_image = transformed['image_ref']

        # Return as tensor
        return raw_image, ref_image, os.path.basename(raw_image_path)

    def __len__(self):
        return len(self.image_name_list)

