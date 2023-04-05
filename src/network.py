import torch

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, skip: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.skip = skip

    def forward(self, x):
        identity = x

        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.skip:
            x = x + identity
        return x


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, stride=1, downsample=None, ):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, dropout):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(in_channels, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.layers = nn.Sequential()
        mult = 16
        for i, layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            cur_dropout = dropout / max(len(layers) - 1, 1) * i
            self.layers.add_module(f'layer{i}', self.make_layer(block, mult, cur_dropout, layer, stride))
            mult *= 2

    def make_layer(self, block, out_channels, dropout, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, dropout, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers(out)
        return out


class CombinedNet(nn.Module):
    def __init__(self, channels, patch_size, num_features, num_cnn_layers, num_blocks, dropout):
        super().__init__()

        self.resnet = ResNet(ResidualBlock, [2] * num_cnn_layers, channels, dropout)
        out_size = patch_size // (2 ** (num_cnn_layers - 1)) * patch_size // (2 ** (num_cnn_layers - 1))
        out_channels = 16 * 2 ** (num_cnn_layers - 1)

        blocks = [
            BasicBlock(out_size * out_channels, num_features, dropout=dropout, skip=False)]

        for i in range(num_blocks):
            blocks.append(BasicBlock(num_features, num_features, dropout=dropout, skip=True))
        blocks.append(BasicBlock(num_features, channels, dropout=0, skip=False))
        self.blocks = nn.Sequential(*blocks)
        self.tanh = nn.Tanh()

        self.resnet.apply(self.init_weights)
        self.blocks.apply(self.init_weights)
        self.tanh.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=1.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=1.)
            nn.init.normal_(m.bias.data, mean=0, std=1)

    def forward(self, x):
        compressed = x[:, :1]
        imgs = x[:, 1:]

        differences = imgs - compressed

        out = self.resnet(imgs)
        out = torch.flatten(out, start_dim=1)
        out = self.blocks(out)
        out = self.tanh(out)

        differences = differences * out[:, :, None, None]
        differences = torch.sum(differences, dim=1)

        result = differences + compressed.squeeze()
        return result
