import pytorch_lightning as pl
import pyiqa
import time
import torch
import torchvision
from model import mynet
import os
from x_unet import XUnet
from vit_model.model import Estimation

class MyModel(pl.LightningModule):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        # Define your PyTorch model here
        self.opt = opt
        # self.net = XUnet(
        #         dim = 64,
        #         channels = 3,
        #         dim_mults = (1, 2, 4, 8),
        #         nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
        #         consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
        #     )
        self.net = Estimation()
        self.loss_func = pyiqa.create_metric('lpips',as_loss=True)
        self.l1_loss =  torch.nn.L1Loss()
        assert self.loss_func.lower_better == True
    
    def forward(self, x):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        input, target, _ = batch

        pred = self(input)
        lpip_loss = self.loss_func(pred, target).mean()
        l1_loss = self.l1_loss(pred, target).mean()
        loss = l1_loss+lpip_loss

        self.log('train_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
        
        return {'optimizer': optimizer , 'lr_scheduler': scheduler}
    
    def validation_step(self, batch, batch_idx):
        input, target, name = batch
        pred = self(input)
        loss = self.loss_func(pred, target)
        self.log('val_loss', loss)
        # save each image of pred
        for i in range(pred.shape[0]):
            torchvision.utils.save_image(pred[i], os.path.join(self.opt.val_img_folder,self.opt.log_name,name[i]))

        

