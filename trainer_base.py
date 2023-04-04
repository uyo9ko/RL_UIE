import pytorch_lightning as pl
import pyiqa
import time
import torch
import torchvision
from HPU_net import *
import os



class MyModel(pl.LightningModule):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        self.opt = opt
        self.net = HPUNet(in_ch=opt.in_ch, out_ch=opt.out_ch, chs=opt.intermediate_ch,
                latent_num=opt.latent_num, latent_channels=opt.latent_chs, latent_locks=opt.latent_locks,
                scale_depth=opt.scale_depth, kernel_size=opt.kernel_size, dilation=opt.dilation,
                padding_mode=opt.padding_mode ).double()
        reconstruction_loss = MSELossWrapper()
        beta_scheduler = BetaConstant(self.opt.beta)
        self.criterion = ELBOLoss(reconstruction_loss=reconstruction_loss, beta=beta_scheduler)

    def record_history(self,loss_dict,type='train'):

        loss_per_pixel = loss_dict['loss'].item() / self.opt.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / self.opt.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / self.opt.pixels
        kl_per_pixel = [ loss_dict['kls'][v].item() / self.opt.pixels for v in range(self.opt.latent_num) ]

        # Total Loss
        if type == 'train':
            _dict = {   'total': loss_per_pixel,
                        'kl term': kl_term_per_pixel, 
                        'reconstruction': reconstruction_per_pixel  }
        else:
            _dict ={ 'val_toal': loss_per_pixel,
                    'val_kl term': kl_term_per_pixel,
                    'val_reconstruction': reconstruction_per_pixel}
        
        self.log(_dict)

        # KL Term Decomposition
        if type == 'train':
            _dict = { 'sum': sum(kl_per_pixel) }
            _dict.update( { 'scale {}'.format(v): kl_per_pixel[v] for v in range(self.opt.latent_num) } )
        else:
            _dict = { 'val_sum': sum(kl_per_pixel) }
            _dict.update( { 'val_scale {}'.format(v): kl_per_pixel[v] for v in range(self.opt.latent_num) } )
        self.log(_dict)
    
        
    def training_step(self, batch, batch_idx):
        images, truths, _ = batch

        preds, infodicts = self.net(images, truths)
        preds, infodict = preds[:,0], infodicts[0]

        loss = self.criterion(preds, truths, kls=infodict['kls'])
        self.criterion.beta_scheduler.step()

        loss_dict = self.criterion.last_loss.copy()
        loss_dict.update( { 'kls': infodict['kls'] } )
        
        self.record_history(loss_dict)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.epochs)
        
        return {'optimizer': optimizer , 'lr_scheduler':lr_scheduler}
    
    def validation_step(self, batch, batch_idx):
        val_images, val_truths, name = batch
        val_minibatches = val_images.shape[0]

        mean_val_loss, mean_val_reconstruction_term, mean_val_kl_term = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        mean_val_kl = torch.zeros(self.opt.latent_num, device=device)

        preds, infodicts = self.net(val_images, val_truths)
        preds, infodict = preds[:,0], infodicts[0]
        ## Calculate Loss
        loss = self.criterion(preds, val_truths, kls=infodict['kls'])

        mean_val_loss += loss
        mean_val_reconstruction_term += self.criterion.last_loss['reconstruction_term']
        mean_val_kl_term += self.criterion.last_loss['kl_term']
        mean_val_kl += infodict['kls']
        mean_val_loss /= val_minibatches
        mean_val_reconstruction_term /= val_minibatches
        mean_val_kl_term /= val_minibatches
        mean_val_kl /= val_minibatches

                # Record Validation History
        loss_dict = {
            'loss': mean_val_loss,
            'reconstruction_term': mean_val_reconstruction_term,
            'kl_term': mean_val_kl_term,
            'kls': mean_val_kl
        }
        self.record_history(loss_dict, type='val')




        

