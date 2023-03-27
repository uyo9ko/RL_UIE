import os
import argparse
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from dataset import MyDataModule
from trainer_base import MyModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=16, help='Number of data loader workers')
parser.add_argument('--log_name', type=str, default='default', help='Name of the experiment in WandB')
parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate')
parser.add_argument('--data_dir', type=str, default='data', help='Path to data')
parser.add_argument('--val_img_folder', type=str, default='data/val', help='Path to validation images')
args = parser.parse_args()

# Initialize WandB logger
wandb_logger = WandbLogger(project='RL_UIE', name=args.log_name)
# torch.use_deterministic_algorithms(False)
# torch.use_deterministic_algorithms(True,warn_only=True)
torch.backends.cudnn.benchmark = True
# Initialize checkpoint and early stopping callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='model_{epoch:02d}_{val_loss:.2f}',
    save_top_k=2,
    monitor='val_musiq_spaq',
    mode='min'
)
# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     mode='min'
# )

# Initialize data module and model
dm = MyDataModule(batch_size=args.batch_size, data_dir = args.data_dir)
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
