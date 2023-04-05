import os
import shutil
from sklearn import model_selection

if __name__ == "__main__":
    raw_path = '/root/data/raw'

    names = sorted(os.listdir(raw_path))
    names = [os.path.splitext(name)[0] for name in names]

    train_names, test_names = model_selection.train_test_split(names,
                                                               test_size=0.2,
                                                               random_state=42)

    train_names, valid_names = model_selection.train_test_split(train_names,
                                                                test_size=0.2,
                                                                random_state=42)

    for name in names:
        if name in train_names:
            mode = 'train'
        elif name in valid_names:
            mode = 'valid'
        else:
            mode = 'test'

        print(name, mode)
        old_path = f'{raw_path}/{name}.yuv'
        new_path = f'{raw_path}_{mode}/{name}.yuv'
        shutil.copy(old_path, new_path)
