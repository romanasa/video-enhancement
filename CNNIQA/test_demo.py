from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from scipy.signal import convolve2d
from torchvision.transforms.functional import to_tensor


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    h = im.shape[1]
    w = im.shape[2]
    patches = ()
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            patch = to_tensor(im[i: i + patch_size, j: j + patch_size])
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q


def get_scores(img, model_file, patch_size):
    im = np.dstack((img, img, img))
    patches = NonOverlappingCropPatches(im, patch_size, patch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        scores = patch_scores.numpy()
    return scores


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test demo')
    parser.add_argument("--im_path", type=str, default='data/I03_01_1.bmp',
                        help="image path")
    parser.add_argument("--model_file", type=str, default='models/CNNIQA-LIVE',
                        help="model file (default: models/CNNIQA-LIVE)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    im = Image.open(args.im_path).convert('L')
    patches = NonOverlappingCropPatches(im, 16, 16)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        print(patch_scores.mean().item())
