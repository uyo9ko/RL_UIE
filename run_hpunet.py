import os
import argparse
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from dataset import MyDataModule

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Lightning Training')

# Model

parser.add_argument("--in_ch", type=int, default=1, help="# Input Channels")
parser.add_argument("--out_ch", type=int, default=1, help="# Output Channels")
parser.add_argument("--intermediate_ch", type=int, nargs='+', help="<Required> Intermediate Channels", required=True)
parser.add_argument("--kernel_size", type=int, nargs='+', default=[3], help="Kernel Size of the Convolutional Layers at Each Scale")
parser.add_argument("--scale_depth", type=int, nargs='+', default=[1], help="Number of Residual Blocks at Each Scale")
parser.add_argument("--dilation", type=int, nargs='+', default=[1], help="Dilation at Each Scale")
parser.add_argument("--padding_mode", default='zeros', help="Padding Mode in the Decoder's Convolutional Layers")

parser.add_argument("--latent_num", type=int, default=0, help="Number of Latent Scales (Setting to zero results in a deterministic U-Net)")
parser.add_argument("--latent_chs", type=int, nargs='+', help="Number of Latent Channels at Each Latent Scale (Setting to None results in 1 channel per scale)")
parser.add_argument("--latent_locks", type=int, nargs='+', help="Whether Latent Space in Locked at Each Latent Scale (Setting to None makes all scales unlocked)")


# Loss

parser.add_argument("--rec_type", help="Reconstruction Loss Type", required=True)
parser.add_argument("--loss_type", default="ELBO", help="Loss Function Type (ELBO/GECO)")

parser.add_argument("--beta", type=float, default=1.0, help="(If Using ELBO Loss) Beta Parameter")
parser.add_argument("--beta_asc_steps", type=int, help="(If Using ELBO Loss with Beta Scheduler) Number of Ascending Steps (If Not Provided, Beta Will be Constant)")
parser.add_argument("--beta_cons_steps", type=int, default=1, help="(If Using ELBO Loss with Beta Scheduler) Number of Constant Steps")
parser.add_argument("--beta_saturation_step", type=int, help="(If Using ELBO Loss with Beta Scheduler) The Step at Which Beta Becomes Permanently 1")

parser.add_argument("--kappa", type=float, default=1.0, help="(If Using GECO Loss) Kappa Parameter")
parser.add_argument("--kappa_px", action='store_true', help="(If Using GECO Loss) Kappa Parameter Type (If true, Kappa should be provided per pixel)")
parser.add_argument("--decay", type=float, default=0.9, help="(If Using GECO Loss) EMA Decay Rate/Smoothing Factor")
parser.add_argument("--update_rate", type=float, default=0.01, help="(If Using GECO Loss) Lagrange Multiplier Update Rate")


# Training

parser.add_argument("--epochs", type=int, help="<Required> Number of Epochs", required=True)
parser.add_argument("--bs", type=int, help="<Required> Batch Size", required=True)

parser.add_argument("--optimizer", default="adam", help="Optimizer")
parser.add_argument("--wd", type=float, default=0.0, help="Weight Decay Parameter")

parser.add_argument("--lr", type=float, help="<Required> (Initial) Learning Rate", required=True)
parser.add_argument("--scheduler_type", default='cons', help="Scheduler Type (cons/step/milestones)")
parser.add_argument("--scheduler_step_size", type=int, default=128, help="Learning Rate Scheduler Step Size (If type is step)")
parser.add_argument("--scheduler_milestones", type=int, nargs='+', help="Learning Rate Scheduler Milestones (If type is milestones)")
parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Learning Rate Scheduler Gamma")

parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=16, help='Number of data loader workers')
parser.add_argument('--log_name', type=str, default='default', help='Name of the experiment in WandB')
parser.add_argument('--data_dir', type=str, default='data', help='Path to data')
parser.add_argument('--val_img_folder', type=str, default='/mnt/epnfs/zhshen/RL_UIE/val_imgs/', help='Path to validation images')
parser.add_argument('--data_name', type=str,default='uieb') 
parser.add_argument('--rl',action='store_true',default=False)
args = parser.parse_args()

if not os.path.exists(os.path.join(args.val_img_folder,args.log_name)):
    os.makedirs(os.path.join(args.val_img_folder,args.log_name))

# Initialize WandB logger
wandb_logger = WandbLogger(project='RL_UIE', name=args.log_name)
# torch.use_deterministic_algorithms(False)
# torch.use_deterministic_algorithms(True,warn_only=True)
torch.backends.cudnn.benchmark = True
# Initialize checkpoint and early stopping callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.val_img_folder,args.log_name),
    filename='model_{epoch:02d}_{val_loss:.2f}',
    save_top_k=2,
    monitor='rl_metric',
    mode='max'
)

# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     mode='min'
# )

# Initialize data module and model
dm = MyDataModule(batch_size=args.bs, data_dir = args.data_dir,data_name=args.data_name)
if args.rl:
    from trainer_rl import MyModel
else:
    from trainer_base import MyModel
model = MyModel(args)

# Initialize trainer and train the model
trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
    logger=wandb_logger,
    num_sanity_val_steps=0,
    check_val_every_n_epoch = 20
)
trainer.fit(model, datamodule=dm)
# trainer.validate(model, datamodule=dm)
