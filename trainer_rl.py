import pytorch_lightning as pl
import pyiqa
import time
import torch
import torchvision
from model import mynet
import os
import sys
from utils import uciqe_and_uiqm
from uwranker import ranker_utils
import numpy as np


def cal_reward_uciqe_uiqm(y_sample,y_baseline,reward_func=uciqe_and_uiqm):
    batch_size = y_sample.shape[0]

    rl_metric_sample = torch.zeros(batch_size)
    rl_metric_baseline = torch.zeros(batch_size)

    # loop through each element in the batch
    for i in range(batch_size):
        # calculate RL metric value for sample input
        y_sample_np = y_sample[i].detach().cpu().numpy()
        rl_metric_sample[i] = torch.tensor(uciqe_and_uiqm(y_sample_np)[0]).cuda()

        # calculate RL metric value for baseline input
        y_baseline_np = y_baseline[i].detach().cpu().numpy()
        rl_metric_baseline[i] = torch.tensor(uciqe_and_uiqm(y_baseline_np)[0]).cuda()

    # calculate the mean of the RL metric values
    rl_metric_sample_mean = rl_metric_sample.mean()
    rl_metric_baseline_mean = rl_metric_baseline.mean()

    r = (rl_metric_sample_mean - rl_metric_baseline_mean)
    return r

def cal_reward_pyiqa(y_sample,y_baseline,rl_metric_func):
    with torch.no_grad():
        rl_metric_func.eval()
        rl_metric_sample_mean =  rl_metric_func(y_sample).mean()
        rl_metric_baseline_mean = rl_metric_func(y_baseline).mean()
    r = (rl_metric_sample_mean - rl_metric_baseline_mean)
    return r

def cal_reward_uranker(y_sample,y_baseline,rl_metric_func):
    with torch.no_grad():
        rl_metric_func.eval()
        inputs_sample = ranker_utils.preprocessing(y_sample)
        rl_metric_sample_mean = rl_metric_func(**inputs_sample)['final_result'].mean()

        inputs_baseline = ranker_utils.preprocessing(y_baseline)
        rl_metric_baseline_mean = rl_metric_func(**inputs_baseline)['final_result'].mean()
    r = (rl_metric_sample_mean - rl_metric_baseline_mean)
    return r




class MyModel(pl.LightningModule):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        # Define your PyTorch model here
        self.opt = opt
        self.net = mynet()
        weights = torch.load('/mnt/epnfs/zhshen/RL_UIE/base_check_point/model_epoch=359_val_loss=0.00.ckpt')['state_dict']
        weights = {k:v for k, v in weights.items() if 'iqa_metric' not in k }
        weights = {k.replace('net.', ''): v for k, v in weights.items()}
        self.net.load_state_dict(weights)
        # self.rl_metric_func = pyiqa.create_metric('musiq')
        self.u_metric_func =  uciqe_and_uiqm
        self.niqe_func = pyiqa.create_metric('niqe',as_loss=True)
        self.rl_metric = torch.tensor(0.0,device='cuda')
        self.uciqe = torch.tensor(0.0,device='cuda')
        self.uiqm = torch.tensor(0.0,device='cuda')

        options = ranker_utils.get_option('/mnt/epnfs/zhshen/RL_UIE/uwranker/URanker.yaml')
        self.rl_metric_func = ranker_utils.build_model(options['model'])

    
    # def forward(self, x):
    #     return self.net(x)
        
    def training_step(self, batch, batch_idx):
        input, target, _ = batch

        self.net.forward(input, input, training=False)
        y_sample = self.net.sample(testing=True)
        y_baseline = self.net.sample(testing=True)
        # r_uciqe = cal_reward_uciqe_uiqm(y_sample,y_baseline,self.rl_metric_func)
        r_uranker = cal_reward_uranker(y_sample,y_baseline,self.rl_metric_func)
        NIQE = cal_reward_pyiqa(y_sample,y_baseline,self.niqe_func)
        if NIQE > 3.0:
            penalty = 5 * (NIQE - 3.0)**2
        else:
            penalty = 0
        r = r_uranker - penalty
                                             
        self.net.forward(input, y_sample, training=True)
        loss = self.net.elbo_r(y_sample,r)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.net.decoder.compute_z_pr.parameters()},
            {'params': self.net.decoder.compute_z_po.parameters()},
            {'params': self.net.decoder.conv_u.parameters()},
            {'params': self.net.decoder.conv_s.parameters()},
            # {'params': self.net.decoder.pr_UpConv1.parameters()},
            # {'params': self.net.decoder.out_conv.parameters()},
            ],
            lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8)
        
        # warmup_epochs = 5
        # step_len = len(self.trainer.datamodule.val_dataloader())
        # num_training_steps = step_len * self.opt.epochs
        # num_warmup_steps = int(num_training_steps * 0.1)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.OneCycleLR(
        #         optimizer,
        #         max_lr=self.opt.lr,
        #         steps_per_epoch=step_len,
        #         epochs=self.opt.epochs,
        #         anneal_strategy='linear',
        #         pct_start=num_warmup_steps/num_training_steps,
        #         div_factor=25.0,
        #         final_div_factor=10000.0,
        #     ),
        #     'interval': 'step',
        #     'frequency': 1
        # }
        return {'optimizer': optimizer}
    
    def validation_step(self, batch, batch_idx):
        input, target, name = batch

        self.net.forward(input, input, training=False)
        avg_pre = 0
        for i in range(self.opt.num_samples):
            t0 = time.time()
            prediction = self.net.sample(testing=True)
            t1 = time.time()
            avg_pre = avg_pre + prediction / self.opt.num_samples
        # self.rl_metric += self.rl_metric_func(avg_pre)
        inputs = ranker_utils.preprocessing(avg_pre)
        self.rl_metric +=  self.rl_metric_func(**inputs)['final_result'].mean()
        # self.rl_metric += self.rl_metric_func(avg_pre.detach().cpu().numpy())[0]
        self.uciqe += self.u_metric_func(avg_pre.detach().cpu().numpy())[1]
        self.uiqm += self.u_metric_func(avg_pre.detach().cpu().numpy())[2]
        torchvision.utils.save_image(avg_pre, os.path.join(self.opt.val_img_folder,self.opt.log_name,name[0]))

        
    def on_validation_epoch_end(self):
        self.log('rl_metric', self.rl_metric/len(self.trainer.datamodule.val_dataloader()))
        self.log('uciqe', self.uciqe/len(self.trainer.datamodule.val_dataloader()))
        self.log('uiqm', self.uiqm/len(self.trainer.datamodule.val_dataloader()))
        print('rl_metric:' , self.rl_metric/len(self.trainer.datamodule.val_dataloader()))
        print('uciqe:' , self.uciqe/len(self.trainer.datamodule.val_dataloader()))
        print('uiqm:' , self.uiqm/len(self.trainer.datamodule.val_dataloader()))
        self.rl_metric = 0
        self.uciqe = 0
        self.uiqm = 0
        

