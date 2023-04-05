import argparse
import datetime
import os

import numpy as np
import torch
import yaml
from torch import nn, optim

import wandb
from utils import train as train_utils


def train(config):
    wandb.init(project='vkr2', config=config['model'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = "mps"

    np.random.seed(0)
    torch.backends.cudnn.deterministic = True

    patch = config['model']['patch_size']

    dataloaders = train_utils.create_dataloaders(config['datasets'], patch=patch, stride=patch)
    model = train_utils.create_model(config['model']).to(device)

    training_params = config['training_params']

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloaders['train']) * training_params['num_epochs'],
        eta_min=3e-6,
    )

    now = str(datetime.datetime.now()).replace(' ', '_')
    model_dir = f'models_{now}'
    os.mkdir(model_dir)

    best_value = 0

    trainer = train_utils.Trainer(
        train_dataloader=dataloaders['train'],
        valid_dataloader=dataloaders['valid'],
        optimizer=optimizer,
        model=model,
        patch_size=patch,
        criterion=criterion,
        scheduler=scheduler,
        clip=1e-3,
        device=device
    )

    for epoch in range(training_params['num_epochs']):
        if epoch == training_params['num_epochs'] // 2:
            trainer.criterion = nn.MSELoss()

        train_metrics = trainer.train_epoch()
        for name, meter in train_metrics.items():
            wandb.log({f'train/{name}': meter.get_mean()}, step=trainer.wandb_step, commit=False)
        trainer.wandb_step += 1

        if epoch % 5 == 0 or epoch == training_params['num_epochs'] - 1:
            valid_metrics = trainer.valid_epoch()
            main_metric = valid_metrics['dpsnr'].get_mean()
            for name, meter in valid_metrics.items():
                wandb.log({f'valid/{name}': meter.get_mean()}, step=trainer.wandb_step, commit=False)
            trainer.wandb_step += 1

            if main_metric > best_value:
                best_value = main_metric
                model_path = f'{model_dir}/model_{epoch}_{best_value:.5f}.pth'
                print(f'Saving to {model_path}, dpsnr={best_value:.6f}')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
        print(f'Epoch={epoch + 1}, lr={scheduler.get_last_lr()[0]:.6f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Run train')
    parser.add_argument('--dropout', type=float, default=None, help='dropout')
    parser.add_argument('--num_blocks', type=int, default=None, help='num_blocks')
    parser.add_argument('--num_cnn_layers', type=int, default=None, help='num_cnn_layers')
    parser.add_argument('--num_features', type=int, default=None, help='num_features')
    parser.add_argument('--patch_size', type=int, default=None, help='patch_size')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open('train_42.yaml') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.dropout is not None:
        yaml_config['model']['dropout'] = args.dropout
    if args.num_blocks is not None:
        yaml_config['model']['num_blocks'] = args.num_blocks
    if args.num_cnn_layers is not None:
        yaml_config['model']['num_cnn_layers'] = args.num_cnn_layers
    if args.num_features is not None:
        yaml_config['model']['num_features'] = args.num_features
    if args.patch_size is not None:
        yaml_config['model']['patch_size'] = args.patch_size

    train(yaml_config)
