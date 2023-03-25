import torch
import torchvision

from dataset import MyDataModule

data_dir = '/mnt/epnfs/zhshen/RL_UIE/uieb/BIG_UIEB'
batch_size = 4

# Initialize datamodule
dm = MyDataModule(batch_size=batch_size, data_dir=data_dir)
dm.setup()

# Test dataloader
dataloader = dm.train_dataloader()
batch = next(iter(dataloader))

# Get images and labels from batch
raw_images, ref_images, names = batch

# Convert tensor images to numpy images and show them
raw_images = torchvision.utils.make_grid(raw_images, nrow=4, normalize=True)
ref_images = torchvision.utils.make_grid(ref_images, nrow=4, normalize=True)
raw_images = raw_images.permute(1, 2, 0).numpy()
ref_images = ref_images.permute(1, 2, 0).numpy()
print(f'Raw images: shape={raw_images.shape}, dtype={raw_images.dtype}')
print(f'Ref images: shape={ref_images.shape}, dtype={ref_images.dtype}')
