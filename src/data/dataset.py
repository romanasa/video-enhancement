import glob
import os

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataset import T_co

import data.utils as data_utils


class YuvDataset(data.Dataset):
    def __init__(self, path, size, fmt, return_name=False):
        self.basename = os.path.basename(path)

        self.frames = data_utils.read_yuv(path, size, fmt)
        self.frames = torch.from_numpy(self.frames.copy())

        self.return_name = return_name

    def __getitem__(self, index) -> T_co:
        if self.return_name:
            return self.basename, data_utils.to_tensor(self.frames[index])
        return data_utils.to_tensor(self.frames[index])

    def __len__(self):
        return len(self.frames)


class PatchYuvDataset(YuvDataset):
    def __init__(self, path, size, fmt, patch_size, stride):
        super().__init__(path, size, fmt)

        assert size[0] % patch_size == 0
        assert size[1] % patch_size == 0

        self.indices = []
        for i in range(0, size[0] - patch_size + 1, stride):
            for j in range(0, size[1] - patch_size + 1, stride):
                self.indices.append((i, j))

        self.patch_size = patch_size
        self.patch_cnt = len(self.indices)

    def __getitem__(self, index) -> T_co:
        frame_id = index // self.patch_cnt
        patch_id = index % self.patch_cnt
        return self.crop_patch(self.frames[frame_id], *self.indices[patch_id])

    def crop_patch(self, tensor, i, j):
        return data_utils.to_tensor(tensor[i: i + self.patch_size, j: j + self.patch_size])

    def __len__(self):
        return len(self.frames) * self.patch_cnt


def get_names_in_folder(folder):
    filenames = sorted(glob.glob(f'{folder}/*.yuv'))
    return set([os.path.basename(x) for x in filenames])


class MultiAlgoDataset(data.Dataset):
    def __init__(self, algo_infos, target_info, size, patch_size, stride, augment_func):
        filenames = self.get_common_filenames(algo_infos, target_info)
        print(f'Loading {filenames}')

        self.augment_func = augment_func

        self.feature_datasets = []
        self.target_dataset = []

        for algo_info in algo_infos + [target_info]:
            single_video_datasets = []
            is_target = algo_info['folder'] == target_info['folder']
            for filename in filenames:
                if patch_size > 0:
                    single_video_datasets.append(
                        PatchYuvDataset(os.path.join(algo_info['folder'], filename), size, algo_info['fmt'],
                                        patch_size, stride))
                else:
                    single_video_datasets.append(
                        YuvDataset(os.path.join(algo_info['folder'], filename), size, algo_info['fmt'],
                                   return_name=is_target))

            multi_video_dataset = data.ConcatDataset(single_video_datasets)

            if is_target:
                self.target_dataset = multi_video_dataset
            else:
                self.feature_datasets.append(multi_video_dataset)

        for feature_dataset in self.feature_datasets:
            if len(feature_dataset) != len(self.target_dataset):
                print(algo_infos, target_info, patch_size, filenames)

            assert len(feature_dataset) == len(
                self.target_dataset), f'{len(feature_dataset)=}, {len(self.target_dataset)=}'

    @staticmethod
    def get_common_filenames(algo_infos, target_info):
        filenames = get_names_in_folder(target_info['folder'])
        for algo_info in algo_infos:
            filenames &= get_names_in_folder(algo_info['folder'])
        assert len(filenames) > 0
        return filenames

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, index):
        feature = torch.stack([dataset[index] for dataset in self.feature_datasets])
        target = self.target_dataset[index]
        if self.augment_func is not None:
            feature, target = self.augment_func(feature, target)
        return feature, target


def augment(features, target):
    def rand():
        return np.random.randint(0, 2, 1)[0] > 0

    if rand():  # rotate
        step = np.random.randint(1, 4, 1)[0]
        features = torch.rot90(features, k=step, dims=[1, 2])
        target = torch.rot90(target, k=step, dims=[0, 1])

    if rand():  # swap channels
        permutation = np.random.permutation(len(features) - 1) + 1
        permutation = np.append([0], permutation)
        features = features[permutation]

    if rand():  # flip_horizontal
        features = torch.flip(features, dims=[1])
        target = torch.flip(target, dims=[0])

    if rand():  # flip_vertical
        features = torch.flip(features, dims=[2])
        target = torch.flip(target, dims=[1])

    return features, target


def get_dataset(opt, patch_size, stride):
    dataset_type = opt['type']

    if dataset_type == "MultiAlgoDataset":
        augment_func = None
        if opt['aug']:
            augment_func = augment

        size = (opt['height'], opt['width'])

        algo_infos = list(opt['algos'].values())
        target_info = opt['raw']

        return MultiAlgoDataset(algo_infos, target_info, size, patch_size, stride, augment_func)
    else:
        raise ValueError(f'Unknown {dataset_type=}')
