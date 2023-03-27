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
        weights = torch.load('/mnt/epnfs/zhshen/RL_UIE/base_check_point/model_epoch=359_val_loss=0.00.ckpt')['state_dict']
        weights = {k:v for k, v in weights.items() if 'net' in k }
        weights = {k.replace('net.', ''): v for k, v in weights.items()}
        self.net.load_state_dict(weights)
        self.test_musiq_spaq = torch.tensor(0.0,device='cuda')
        if not os.path.exists(self.opt.val_img_folder):
            os.makedirs(self.opt.val_img_folder)
     

    
    # def forward(self, x):
    #     return self.net(x)
        
    def training_step(self, batch, batch_idx):
        input, target, _ = batch

        self.net.forward(input, input, training=False)
        y_sample = self.net.sample(testing=True)
        y_baseline = self.net.sample(testing=True)
        musiq_spaq_sample = self.iqa_metric(y_sample)
        musiq_spaq_baseline = self.iqa_metric(y_baseline)
        r = musiq_spaq_sample - musiq_spaq_baseline
        self.net.forward(input, y_sample, training=True)
        loss = self.net.elbo_r(y_sample, r)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.net.decoder.compute_z_pr.parameters()},
            {'params': self.net.decoder.compute_z_po.parameters()},
            {'params': self.net.decoder.conv_u.parameters()},
            {'params': self.net.decoder.conv_s.parameters()},
            ],
            lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8)
        
        warmup_epochs = 5
        step_len = len(self.trainer.datamodule.val_dataloader())
        num_training_steps = step_len * self.opt.epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.opt.lr,
                steps_per_epoch=step_len,
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
        

