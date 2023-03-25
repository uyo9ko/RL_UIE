import os
import argparse
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset import MyDataModule
from trainer import MyModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
parser.add_argument('--log_name', type=str, default='default', help='Name of the experiment in WandB')
args = parser.parse_args()

# Initialize WandB logger
wandb_logger = WandbLogger(project='my_project', name=args.log_name)

# Initialize checkpoint and early stopping callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='model_{epoch:02d}_{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min'
)

# Initialize data module and model
dm = MyDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
model = MyModel(lr=args.lr)

# Initialize trainer and train the model
trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    num_sanity_val_steps=0,
    deterministic=True,
)
trainer.fit(model, datamodule=dm)
