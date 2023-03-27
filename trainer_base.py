import pytorch_lightning as pl
import pyiqa
import time
import torch
import torchvision
from model import mynet
import os



class MyModel(pl.LightningModule):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        # Define your PyTorch model here
        self.opt = opt
        self.iqa_metric = pyiqa.create_metric('musiq-spaq')
        self.net = mynet()
        self.test_musiq_spaq = torch.tensor(0.0,device='cuda')
     

    
    # def forward(self, x):
    #     return self.net(x)
        
    def training_step(self, batch, batch_idx):
        input, target, _ = batch

        self.net.forward(input, target, training=True)
        loss = self.net.elbo(target)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
        
        return {'optimizer': optimizer , 'lr_scheduler': scheduler}
    
    def validation_step(self, batch, batch_idx):
        input, target, name = batch

        self.net.forward(input, input, training=False)
        avg_pre = 0
        for i in range(self.opt.num_samples):
            t0 = time.time()
            prediction = self.net.sample(testing=True)
            t1 = time.time()
            avg_pre = avg_pre + prediction / self.opt.num_samples
        self.test_musiq_spaq += self.iqa_metric(avg_pre)
        # save_img = torchvision.utils.make_grid(avg_pre, normalize=True)
        torchvision.utils.save_image(avg_pre, os.path.join(self.opt.val_img_folder,name[0]))

        
    def on_validation_epoch_end(self):
        self.log('val_musiq_spaq', self.test_musiq_spaq/len(self.trainer.datamodule.val_dataloader()))
        self.test_musiq_spaq = 0
        

