import collections
from typing import Dict

import numpy as np
import torch
import tqdm
import wandb
from skimage.metrics import peak_signal_noise_ratio
from torch.utils import data

import network
from data import dataset
from data import utils as data_utils
from utils import metrics


def calc_grad(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def create_dataloaders(dataset_config, patch, stride):
    dataloaders = {}
    for mode in [
        'train',
        'valid',
    ]:
        cur_config = dataset_config[mode]
        cur_dataset = dataset.get_dataset(cur_config, patch if mode == 'train' else -1, stride)
        dataloaders[mode] = data.DataLoader(
            dataset=cur_dataset,
            batch_size=cur_config['batch_size'],
            shuffle=(mode == 'train'),
            num_workers=2,
            pin_memory=True,
            drop_last=(mode == 'train'),
        )
    return dataloaders


def create_model(model_config):
    model = network.CombinedNet(
        channels=model_config['channels'],
        patch_size=model_config['patch_size'],
        num_features=model_config['num_features'],
        num_cnn_layers=model_config['num_cnn_layers'],
        num_blocks=model_config['num_blocks'],
        dropout=model_config['dropout'],
    )
    print(model)
    return model


def create_patches(tensor, shape, patch_size, stride, return_indices=False):
    patches = []
    indices = []
    for i in range(0, shape[0] - patch_size + 1, stride):
        for j in range(0, shape[1] - patch_size + 1, stride):
            patches.append(tensor[..., i: i + patch_size, j: j + patch_size])
            indices.append((i, j))
    patches = torch.stack(patches).squeeze()
    if return_indices:
        return patches, indices
    return patches


def construct_array(shape, indices, patch_size, patches, weights=None):
    assert len(shape) == 2
    if weights is None:
        weights = [1] * len(indices)
    counts = np.zeros(shape)
    values = np.zeros(shape)
    for (i, j), patch, weight in zip(indices, patches, weights):
        values[i: i + patch_size, j: j + patch_size] += patch * weight
        counts[i: i + patch_size, j: j + patch_size] += weight
    return values / counts


class Trainer:
    def __init__(self,
                 train_dataloader: data.DataLoader,
                 valid_dataloader: data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 model: network.CombinedNet,
                 patch_size: int,
                 criterion,
                 scheduler,
                 clip,
                 device):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.model = model
        self.patch_size = patch_size

        self.criterion = criterion
        self.scheduler = scheduler
        self.clip = clip
        self.device = device

        self.wandb_step = 0

    def train_epoch(self) -> Dict:
        train_metrics = collections.defaultdict(metrics.Meter)
        self.model.train()

        tqdm_loader = tqdm.tqdm(self.train_dataloader)
        for features, target in tqdm_loader:
            self.optimizer.zero_grad()

            batch_size = features.shape[0]

            features = features.to(self.device)
            target = target.to(self.device)
            combined = self.model(features)
            loss = self.criterion(combined, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            train_metrics['loss'].update(loss.item(), batch_size)

            combined = torch.clip(combined, -1, 1)
            train_dpsnr, train_dpsnr_percent = metrics.calc_dpsnr(features, combined, target, return_percent=True)

            train_metrics['dpsnr'].update(train_dpsnr, batch_size)
            train_metrics['dpsnr_percent'].update(train_dpsnr_percent, batch_size)

            grad_l2 = calc_grad(self.model)
            train_metrics['grad'].update(grad_l2, 1)

            description = ''
            for name, meter in train_metrics.items():
                wandb.log({f'running_train/{name}': meter.get_mean()}, step=self.wandb_step, commit=False)
                description += f'{name}={meter.get_mean():.6f}, '
            tqdm_loader.set_description(description)
            self.wandb_step += 1
        return train_metrics

    def valid_epoch(self):
        valid_metrics = collections.defaultdict(metrics.Meter)
        self.model.eval()
        with torch.no_grad():
            for features_frame, (target_name, target_frame) in tqdm.tqdm(self.valid_dataloader):
                target_name = target_name[0]
                n_images = features_frame.shape[1]

                patches, indices = create_patches(features_frame, target_frame.shape[1:], self.patch_size,
                                                  stride=self.patch_size,
                                                  return_indices=True)

                target_patches = create_patches(target_frame, target_frame.shape[1:], self.patch_size,
                                                stride=self.patch_size)

                patches = patches.to(self.device)
                target_patches = target_patches.to(self.device)

                enhanced_patches = []
                for perm_it in range(1):
                    # permutation = np.random.permutation(n_images - 1) + 1
                    # permutation = np.append([0], permutation)
                    # patches = patches[:, permutation]

                    enhanced_patches.append(self.model(patches))
                enhanced_patches = torch.mean(torch.stack(enhanced_patches), dim=0)

                loss_val = self.criterion(target_patches, enhanced_patches)
                enhanced_patches = torch.clip(enhanced_patches, -1, 1).detach().cpu().numpy()
                valid_metrics[f'{target_name}/loss'].update(loss_val.item(), len(patches))

                enhanced = construct_array(
                    shape=target_frame.shape[1:],
                    indices=indices,
                    patch_size=self.patch_size,
                    patches=enhanced_patches)

                target_numpy = data_utils.to_numpy(target_frame.squeeze())
                enhanced = data_utils.to_numpy(enhanced)

                max_feature_psnr = 0
                for ind in range(n_images):
                    feature_numpy = data_utils.to_numpy(features_frame.squeeze()[ind])
                    psnr = peak_signal_noise_ratio(target_numpy, feature_numpy)
                    max_feature_psnr = max(psnr, max_feature_psnr)
                enhanced_psnr = peak_signal_noise_ratio(target_numpy, enhanced, data_range=1.0)

                valid_metrics[f'psnr'].update(enhanced_psnr, 1)
                valid_metrics[f'{target_name}/psnr'].update(enhanced_psnr, 1)

                dpsnr = enhanced_psnr - max_feature_psnr
                valid_metrics[f'dpsnr'].update(dpsnr, 1)
                valid_metrics[f'{target_name}/dpsnr'].update(dpsnr, 1)
                valid_metrics[f'dpsnr_percent'].update(dpsnr / max_feature_psnr, 1)
                valid_metrics[f'{target_name}/dpsnr_percent'].update(dpsnr / max_feature_psnr, 1)

                for name, meter in valid_metrics.items():
                    wandb.log({f'running_valid/{name}': meter.get_mean()}, step=self.wandb_step, commit=False)
                self.wandb_step += 1

        for name in sorted(list(valid_metrics.keys())):
            print(f'{name}:\t{valid_metrics[name].get_mean():.10f}')

        return valid_metrics


def enhance(feature_datasets, model, device, patch_size, stride):
    enhanced_frames = []
    model.eval()

    weights = None

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(feature_datasets[0]))):
            features = [feature_dataset[i] for feature_dataset in feature_datasets]
            features = torch.stack(features)

            patches, indices = create_patches(features, features.shape[1:], patch_size, stride, return_indices=True)
            patches = patches.to(device)

            enhanced_patches = model(patches)
            enhanced_patches = torch.clip(enhanced_patches, -1, 1).detach().cpu()

            enhanced_patches = data_utils.to_numpy(enhanced_patches)

            if i == 0:
                weight_matrix = np.zeros((patch_size, patch_size))
                center = (patch_size - 1) / 2

                for x in range(patch_size):
                    for y in range(patch_size):
                        dist = abs(x - center) + abs(y - center)
                        weight_matrix[x, y] = dist
                # weight_matrix = np.max(weight_matrix) + 1 - weight_matrix
                weight_matrix = np.exp(-weight_matrix)
                # print(weight_matrix)
                weights = [weight_matrix for _ in range(len(indices))]

            enhanced = construct_array(
                shape=features.shape[1:],
                indices=indices,
                patch_size=patch_size,
                patches=enhanced_patches,
                weights=weights)

            enhanced_frames.append(enhanced)
    return np.array(enhanced_frames)
