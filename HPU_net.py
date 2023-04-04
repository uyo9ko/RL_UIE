import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
import sys

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="ReLU", kernel_size=3, dilation=1, padding_mode='circular', conv_class=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_class(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.conv2 = conv_class(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        f = self.activation(self.conv1(x))
        f = self.activation(self.conv2(f))

        return f


class PreResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, activation="ReLU", kernel_size=3, dilation=1, padding_mode='circular', conv_class=nn.Conv2d):
        super().__init__()
        
        if out_ch is None:
            out_ch = in_ch
            self.skipconv = nn.Identity()
        else:
            self.skipconv = conv_class(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()

        med_ch = in_ch//2 if in_ch > 1 else 1

        self.bn1 = nn.BatchNorm2d(num_features=in_ch) if conv_class == nn.Conv2d else nn.BatchNorm1d(num_features=in_ch)
        self.bn2 = nn.BatchNorm2d(num_features=med_ch) if conv_class == nn.Conv2d else nn.BatchNorm1d(num_features=med_ch)
        self.conv1 = conv_class(in_channels=in_ch, out_channels=med_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.conv2 = conv_class(in_channels=med_ch, out_channels=med_ch, kernel_size=kernel_size, dilation=dilation, padding='same', padding_mode=padding_mode)
        self.outconv = conv_class(in_channels=med_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        f = self.conv1(self.activation(self.bn1(x)))
        f = self.conv2(self.activation(self.bn2(f)))
        f = self.outconv(f)

        return f + self.skipconv(x)


class ScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="ReLU", scale_depth=1, kernel_size=3, dilation=1, padding_mode='circular', conv_class=nn.Conv2d):
        super().__init__()
        
        self.first_preres_blocks = nn.ModuleList( [ PreResBlock(in_ch, None, activation, kernel_size, dilation, padding_mode=padding_mode, conv_class=conv_class) for _ in range(scale_depth-1) ] )
        self.last_preres_block = PreResBlock(in_ch, out_ch, activation, kernel_size, dilation, padding_mode=padding_mode, conv_class=conv_class)

    def forward(self, x):
        f = x
        if len(self.first_preres_blocks) > 0:
            for block in self.first_preres_blocks:
                f = block(f)
        f = self.last_preres_block(f)

        return f


class Encoder(nn.Module):
    def __init__(self, chs, activation="ReLU", scale_depth=1, kernel_size=None, dilation=None, padding_mode='circular', conv_class=nn.Conv2d):
        super().__init__()
        self.downsampling = nn.AvgPool2d(kernel_size=2) if conv_class == nn.Conv2d else nn.AvgPool1d(kernel_size=2)
        self.encoder_blocks = nn.ModuleList( [ ScaleBlock(chs[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode, conv_class=conv_class) for i in range(len(chs)-1) ] )

    def forward(self, x):
        encoder_feature_maps = [x]
        f = x
        for block in self.encoder_blocks:
            f = self.downsampling(f)
            f = block(f)
            encoder_feature_maps.append(f)

        encoder_feature_maps.reverse()
        return encoder_feature_maps


class Decoder(nn.Module):
    def __init__(self, chs, latent_num=0, activation="ReLU", scale_depth=1, kernel_size=None, dilation=None, padding_mode='circular', latent_channels=None, latent_locks=None, conv_class=nn.Conv2d):
        super().__init__()
        self.depth = len(chs) - 1
        self.latent_num = latent_num
        self.latent_channels = latent_channels
        self.latent_locks = latent_locks

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        # Prior Net
        self.latent_mean_convs = nn.ModuleList(
            [ conv_class(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ] 
        )

        self.latent_std_convs = nn.ModuleList(
            [ conv_class(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.decoder_blocks = nn.ModuleList(
            [ ScaleBlock(chs[i] + chs[i+1] + latent_channels[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode, conv_class=conv_class) if i < latent_num
            else ScaleBlock(chs[i] + chs[i+1], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode, conv_class=conv_class) for i in range(self.depth) ]
        )

        # Posterior Net
        self.post_latent_mean_convs = nn.ModuleList(
            [ conv_class(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.post_latent_std_convs = nn.ModuleList(
            [ conv_class(in_channels=chs[i], out_channels=latent_channels[i], kernel_size=1) for i in range(latent_num) ]
        )
        self.post_decoder_blocks = nn.ModuleList(
            [ ScaleBlock(chs[i] + chs[i+1] + latent_channels[i], chs[i+1], activation, scale_depth[i+1], kernel_size[i+1], dilation[i+1], padding_mode=padding_mode, conv_class=conv_class) for i in range(latent_num - 1) ]
        )


    def sample_latent(self, means, log_stds, latent_lock):
        if latent_lock is True:
            latent = means
        else:
            rands = torch.normal( mean=torch.zeros_like(means), std=torch.ones_like(log_stds) )
            latent = rands * torch.exp(log_stds) + means

        return latent
    

    def forward(self, feature_maps, post_feature_maps=None, insert_from_postnet=False):
        if post_feature_maps is None:  # Not Using Posterior Net
            prior_means, prior_stds = [], []
            prior_latents = []
            f = feature_maps[0]
            for i in range(self.depth):
                if i < self.latent_num:
                    means, log_stds = self.latent_mean_convs[i](f), self.latent_std_convs[i](f)
                    prior_means.append(means)
                    prior_stds.append(torch.exp(log_stds))

                    latent = self.sample_latent(means, log_stds, self.latent_locks[i])
                    prior_latents.append(latent)

                    f = torch.cat([f, latent], dim=1)

                f = self.upsampling(f)
                # print(f.shape)
                # print(feature_maps[i+1].shape)
                f = torch.cat([ f, feature_maps[i+1] ], dim=1)
                f = self.decoder_blocks[i](f)

            if self.latent_num == self.depth + 1:
                means, log_stds = self.latent_mean_convs[self.depth](f), self.latent_std_convs[self.depth](f)
                latent = self.sample_latent(means, log_stds, self.latent_locks[self.depth])

                f = torch.cat([f, latent], dim=1)


            # Items to return as well as the network's output
            infodict = {
                'prior_latents': prior_latents,
                'prior_means': prior_means,
                'prior_stds': prior_stds
            }

            return f, infodict
        
        
        else:  # Using Posterior Net
            post_means, post_stds = [], []
            post_latents = []
            l = post_feature_maps[0]
            for i in range(self.latent_num):
                means, log_stds = self.post_latent_mean_convs[i](l), self.post_latent_std_convs[i](l)
                post_means.append(means)
                post_stds.append(torch.exp(log_stds))
                
                post_latent = self.sample_latent(means, log_stds, self.latent_locks[i])

                post_latents.append(post_latent)

                if i < self.latent_num - 1:
                    l = torch.cat([l, post_latent], dim=1)
                    l = self.upsampling(l)

                    l = torch.cat([ l, post_feature_maps[i+1] ], dim=1)
                    l = self.post_decoder_blocks[i](l)

            prior_means, prior_stds = [], []
            prior_latents = []
            f = feature_maps[0]
            for i in range(self.depth):
                if i < self.latent_num:
                    means, log_stds = self.latent_mean_convs[i](f), self.latent_std_convs[i](f)
                    prior_means.append(means)
                    prior_stds.append(torch.exp(log_stds))
                    
                    if (self.training is True and self.latent_locks[i] is False) or (insert_from_postnet is True):  # Insert Latents from Posterior Net
                        latent = post_latents[i]
                    else:  # Insert Latents from Prior Net
                        latent = self.sample_latent(means, log_stds, self.latent_locks[i])

                    prior_latents.append(latent)

                    f = torch.cat([f, latent], dim=1)

                f = self.upsampling(f)
                f = torch.cat([ f, feature_maps[i+1] ], dim=1)
                f = self.decoder_blocks[i](f)
            
            if self.latent_num == self.depth + 1:
                means, log_stds = self.latent_mean_convs[self.depth](f), self.latent_std_convs[self.depth](f)
                prior_means.append(means)
                prior_stds.append(torch.exp(log_stds))
                
                if (self.training is True and self.latent_locks[self.depth] is False) or (insert_from_postnet is True):  # Insert Latents from Posterior Net
                    latent = post_latents[self.depth]
                else:  # Insert Latents from Prior Net
                    latent = self.sample_latent(means, log_stds, self.latent_locks[self.depth])

                prior_latents.append(latent)

                f = torch.cat([f, latent], dim=1)


            # Calculate kl divergence between posterior net and prior net latents
            kls = torch.zeros(self.latent_num, device=f.device) if self.latent_num > 0 else torch.zeros(1, device=f.device)  # next(self.parameters()).device

            for i in range(self.latent_num):
                if self.latent_locks[i] is False:
                    kl = torch.log( prior_stds[i] / post_stds[i] )                             \
                        + (    post_stds[i] ** 2  +  (post_means[i] - prior_means[i]) ** 2)     \
                            / 2 / prior_stds[i] ** 2                                         \
                        - 1 / 2
                    
                    kls[i] = kl.reshape(feature_maps[0].shape[0],-1).sum(dim=1).mean()

            
            # Items to return as well as the network's output
            infodict = {
                'kls': kls,
                'post_latents': post_latents,
                'prior_latents': prior_latents,
                'post_means': post_means,
                'post_stds': post_stds,
                'prior_means': prior_means,
                'prior_stds': prior_stds
            }
       
            return f, infodict


class HPUNet(nn.Module):
    def __init__(self, in_ch, chs, latent_num=0, out_ch=1, activation="ReLU", scale_depth=None, kernel_size=None, dilation=None, padding_mode='circular', latent_channels=None, latent_locks=None, conv_dim=2):
        super().__init__()
        if latent_locks is None:
            latent_locks = [False for _ in range(latent_num)]
        if latent_channels is None:
            latent_channels = [1 for _ in range(latent_num)]
        assert len(latent_channels) == latent_num
        assert latent_num <= len(chs)
        assert len(scale_depth) == len(chs)
        assert len(kernel_size) == len(chs)
        assert len(dilation) == len(chs)

        self.conv_dim = conv_dim
        self.conv_class = nn.Conv2d if self.conv_dim == 2 else nn.Conv1d
        decoder_head_in_channels = chs[0] + (0 if latent_num < len(chs) else latent_channels[-1])

        self.encoder_head = ConvBlock(in_ch, chs[0], activation, kernel_size[0], dilation[0], padding_mode='zeros', conv_class=self.conv_class)
        self.encoder = Encoder(chs, activation, scale_depth, kernel_size, dilation, padding_mode='zeros', conv_class=self.conv_class)
        self.decoder = Decoder(list(reversed(chs)), latent_num, activation, list(reversed(scale_depth)), list(reversed(kernel_size)), list(reversed(dilation)), padding_mode=padding_mode, latent_channels=latent_channels, latent_locks=latent_locks, conv_class=self.conv_class)
        self.decoder_head = ScaleBlock(decoder_head_in_channels, out_ch, activation, scale_depth[0], kernel_size[0], dilation[0], padding_mode=padding_mode, conv_class=self.conv_class)

        self.posterior_encoder_head = ConvBlock(in_ch*2, chs[0], activation, kernel_size[0], dilation[0], padding_mode='zeros', conv_class=self.conv_class)
        self.posterior_encoder = Encoder(chs, activation, scale_depth, kernel_size, dilation, padding_mode='zeros', conv_class=self.conv_class)

    def forward(self, x, y=None, times=1, first_channel_only=True, insert_from_postnet=False):
        f = self.encoder_head(x)
        f = self.encoder(f)
        
        outs, infodicts = [], []

        if y is None:  # Not Using Posterior Net
            for _ in range(times):
                o, infodict = self.decoder(f)
                o = self.decoder_head(o)
                outs.append(o)
                infodicts.append(infodict)

        else:  # Using Posterior Net
            l = self.posterior_encoder_head(torch.cat([x, y], dim=1))
            l = self.posterior_encoder(l)

            for _ in range(times):
                o, infodict = self.decoder(f, l, insert_from_postnet)
                o = self.decoder_head(o)
                outs.append(o)
                infodicts.append(infodict)

        output = torch.stack(outs, dim=1)

        return output, infodicts


# Optimization
class BetaConstant(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def step(self):
        return


class BetaLinearScheduler(nn.Module):
    def __init__(self, ascending_steps, constant_steps=0, max_beta=1.0, saturation_step=None):
        super().__init__()
        self.ascending_steps = ascending_steps
        self.constant_steps = constant_steps
        self.max_beta = max_beta
        if saturation_step is not None:
            self.saturation_gap = saturation_step
        else:
            self.saturation_gap = -1

        self.increment = max_beta / ascending_steps

        self.beta = 0.0
        self.state = 'ascend'
        self.s = 0

    def step(self):
        if self.state == 'ascend':
            self.beta += self.increment
            self.s += 1
            if self.s == self.ascending_steps:
                self.state = 'constant'
                self.s = 0
        
        elif self.state == 'constant':
            self.s += 1
            if self.s == self.constant_steps:
                self.state = 'ascend'
                self.s = 0
                self.beta = 0.0
        
        self.saturation_gap -= 1
        if self.saturation_gap == 0:
            self.state = 'saturated'
            self.beta = self.max_beta

        return


# Loss Functions & Utils
class MSELossWrapper(MSELoss):
    def __init__(self):
        super().__init__(reduction='none')
        self.last_loss = None

    def forward(self, yhat, y, **kwargs):
        loss = super().forward(yhat, y)

        self.last_loss = {
            'expanded_loss': loss
        }

        return loss


class ELBOLoss(nn.Module):
    def __init__(self, reconstruction_loss, beta=None, conv_dim=2):
        super().__init__()
        self.conv_dim = conv_dim
        self.reconstruction_loss = reconstruction_loss
        
        if beta is None:
            beta = BetaConstant(1.0)
        self.beta_scheduler = beta
        
        self.last_loss = None

 
    def forward(self, yhat, y, kls, **kwargs):
        rec_loss_before_mean = self.reconstruction_loss(yhat, y, **kwargs).sum( dim=tuple(range(1,self.conv_dim)) )
        rec_term = rec_loss_before_mean.mean()
        kl_term = self.beta_scheduler.beta * torch.sum(kls)
        loss = rec_term + kl_term

        self.last_loss = {
            'reconstruction_loss_before_mean': rec_loss_before_mean,
            'reconstruction_term': rec_term,
            'kl_term': kl_term,
            'loss': loss,
            'reconstruction_internal': self.reconstruction_loss.last_loss
        }

        return loss


class GECOLoss(nn.Module):
    def __init__(self, reconstruction_loss, kappa, decay=0.9, update_rate=0.01, device='cpu', log_inv_function='exp', conv_dim=2):
        super(GECOLoss, self).__init__()
        self.conv_dim = conv_dim
        self.reconstruction_loss = reconstruction_loss
        
        self.kappa = kappa
        self.decay = decay
        self.update_rate = update_rate

        if log_inv_function == 'exp':
            self.log_inv_function = torch.exp
        elif log_inv_function == 'softplus':
            self.log_inv_function = nn.functional.softplus
        
        self.device = device

        self.log_lamda = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=False)
        self.rec_constraint_ma = None

        self.last_loss = None


    def update_rec_constraint_ma(self, cons):
        if self.rec_constraint_ma is None:
            self.rec_constraint_ma = torch.FloatTensor([cons]).to(self.device)
        else:
            self.rec_constraint_ma = self.decay * self.rec_constraint_ma.detach() + (1-self.decay) * cons
    

    def forward(self, yhat, y, kls, **kwargs):
        rec_loss_before_mean = self.reconstruction_loss(yhat, y, **kwargs).sum( dim=tuple(range(1,self.conv_dim)) )
        rec_loss = rec_loss_before_mean.mean()
        rec_constraint = rec_loss - self.kappa

        # Update EMA
        if self.training is True:
            self.update_rec_constraint_ma(rec_constraint)

        # Calculate Loss
        rec_constraint_ma = rec_constraint + (self.rec_constraint_ma - rec_constraint).detach()
        lamda = self.log_inv_function(self.log_lamda)

        rec_term = lamda * rec_constraint_ma
        kl_term = torch.sum(kls)
        loss = rec_term + kl_term

        self.last_loss = {
            'reconstruction_loss_before_mean': rec_loss_before_mean,
            'reconstruction_term': rec_term,
            'kl_term': kl_term,
            'loss': loss,
            'reconstruction_internal': self.reconstruction_loss.last_loss
        }

        # Step Lambda
        if self.training is True:
            with torch.no_grad():
                self.log_lamda += self.update_rate * kwargs['lr'] * rec_constraint_ma

        return loss
