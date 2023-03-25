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

    
    def forward(self, x):
        return mynet(x)
        
    def training_step(self, batch, batch_idx):
        input, target = batch

        t0 = time.time()        
        self.forward(input, input, training=False)
        y_sample = self.sample(testing=True)
        y_baseline = self.sample(testing=True)
        musiq_spaq_sample = self.iqa_metric(y_sample)
        musiq_spaq_baseline = self.iqa_metric(y_baseline)
        r = musiq_spaq_sample - musiq_spaq_baseline
        self.forward(input, y_sample, training=True)
        loss = self.elbo_r(y_sample, r)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.decoder.compute_z_pr.parameters()},
            {'params': self.decoder.compute_z_po.parameters()},
            {'params': self.decoder.conv_u.parameters()},
            {'params': self.decoder.conv_s.parameters()},
            ],
            lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8)
        
        warmup_epochs = 5
        num_training_steps = len(self.training_data_loader) * self.opt.epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.opt.lr,
                steps_per_epoch=len(self.training_data_loader),
                epochs=self.opt.epochs,
                anneal_strategy='linear',
                pct_start=num_warmup_steps/num_training_steps,
                div_factor=25.0,
                final_div_factor=10000.0,
            ),
            'interval': 'step',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def val_step(self, batch, batch_idx):
        input, target, name = batch

        self.forward(input, input, training=False)
        avg_pre = 0
        for i in range(self.opt.num_samples):
            t0 = time.time()
            prediction = self.sample(testing=True)
            t1 = time.time()
            avg_pre = avg_pre + prediction / self.num_samples
        test_musiq_spaq = self.iqa_metric(avg_pre)
        save_img = torchvision.utils.make_grid(avg_pre, normalize=True)
        torchvision.utils.save_image(save_img, os.path.join(self.opt.save_folder,name[0]))

        self.log('val_musiq_spaq', test_musiq_spaq, on_step=True, on_epoch=True)

