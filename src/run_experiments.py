import random

import yaml
import train

if __name__ == "__main__":

    with open('train_42.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    patch_sizes = [8, 16, 32]
    num_features = [50, 100, 200]
    num_cnn_layers = [1, 2, 3, 4]
    num_blocks = [1, 2, 3, 4, 5]
    dropout = [0, 0.1, 0.2, 0.5]

    ind = 0

    while True:
        ind += 1

        patch_size = random.choice(patch_sizes)
        config['datasets']['train']['patch_size'] = patch_size
        config['model']['patch_size'] = patch_size

        config['model']['num_features'] = random.choice(num_features)
        config['model']['num_cnn_layers'] = random.choice(num_cnn_layers)
        config['model']['num_blocks'] = random.choice(num_blocks)
        config['model']['dropout'] = random.choice(dropout)

        print(ind, config)
        train.train(config)

