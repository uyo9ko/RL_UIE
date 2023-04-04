import torch
from x_unet import XUnet

unet = XUnet(
    dim = 64,
    channels = 3,
    dim_mults = (1, 2, 4, 8),
    nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
)

img = torch.randn(1, 3, 256, 256)
out = unet(img) # (1, 3, 256, 256)

# calculate the total number of parameters
total_params = sum(p.numel() for p in unet.parameters())

# print the total number of parameters in human-readable format
print(f"Total number of parameters: {total_params / (1024*1024):.2f} Mb")