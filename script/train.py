import os
import argparse

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from shape_assembly.config import get_cfg_defaults
from shape_assembly.datasets.baseline.dataloader import ShapeAssemblyDataset
from shape_assembly.models.baseline.network import ShapeAssemblyNet

def main(cfg):
    # Initialize model
    model = ShapeAssemblyNet(cfg=cfg).cuda()

    # Initialize train dataloader
    train_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        data_csv_file=cfg.data.train_csv_file,
        num_points=cfg.data.num_pc_points
    ) 
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    # Initialize test dataloader
    test_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        data_csv_file=cfg.data.test_csv_file,
        num_points=cfg.data.num_pc_points
    ) 
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])

    # Create checkpoint directory
    checkpoint_dir  = os.path.join(cfg.exp.checkpoint_dir, cfg.exp.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/total loss",
        mode='min',
        save_top_k=1,
        dirpath=log_dir,
        filename=os.path.join('weights', 'best'),
        save_last=True
    )

    trainer = pl.Trainer(
        gpus=list(cfg.exp.gpus),
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        max_epochs=cfg.exp.num_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        default_root_dir=checkpoint_dir
    )

    trainer.fit(model, train_loader, test_loader)

    print("Done training...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    if args.gpus == -1:
        args.gpus = [0, ]
    cfg.gpus = args.gpus

    cfg.freeze()
    print(cfg)
    main(cfg)
