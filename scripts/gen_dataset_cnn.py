import numpy as np
import pandas as pd
import os

from PIL import Image

##### Parameters

CSV_PATH = 'book_data.csv'
IMG_PATH = 'images'
DATASET_DIR = 'dataset_cnn'
ALL_PATH = os.path.join(DATASET_DIR, 'all.csv')
TRAIN_PATH = os.path.join(DATASET_DIR, 'train.csv')
VALID_PATH = os.path.join(DATASET_DIR, 'valid.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'test.csv')
LEVELS = 5
SPLITS = [0.8, 0.1, 0.1]
RANDOM_STATE = 123

if not os.path.isdir(DATASET_DIR):
    os.mkdir(DATASET_DIR)


##### Helpers

def get_filename_from_index(index):
    img = str(index) + '.jpg'
    return os.path.join(IMG_PATH, img)


def get_list_of_levels(length, levels):
    chunk_size = length / levels
    if chunk_size % 1:
        raise ValueError('length is not divisible by levels')
    chunk_size = int(chunk_size)

    ret = []
    for i in range(levels):
        lst = [levels - 1 - i] * chunk_size
        ret.extend(lst)

    return ret


def split_train_valid_test(df_, splits):
    df_ = df_.sample(frac=1, random_state=RANDOM_STATE)
    size = df_.shape[0]
    splits = np.cumsum(splits)
    train = df_[:int(size * splits[0])]
    valid = df_[int(size * splits[0]):int(size * splits[1])]
    test = df_[int(size * splits[1]):int(size * splits[2])]
    return train, valid, test


##### Main

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(index=str, columns={"book_title": "title"})
    df['filename'] = df.index.map(get_filename_from_index)
    df = df[['title', 'filename']]

    # filter inaccessible images
    idx = []
    for i, row in df.iterrows():
        fname = row['filename']
        try:
            Image.open(fname)
        except (FileNotFoundError, OSError):
            idx.append(int(i))

    df = df.drop(df.index[idx])
    print("Rows kept: {}".format(df.shape[0]))

    # make dataset multiple of `LEVELS`
    delete_idx = df.shape[0] % LEVELS
    print("Deleting last {} rows".format(delete_idx))
    df = df[:-delete_idx]
    print("Final shape: {}".format(df.shape[0]))

    # generate labels
    df['label'] = get_list_of_levels(df.shape[0], LEVELS)

    df.to_csv(ALL_PATH, index=None)

    # split df into train, valid, and test
    split_dfs = [None, None, None]
    for lvl in range(LEVELS):
        df_lvl = df[df['label'] == lvl]
        ret = split_train_valid_test(df_lvl, SPLITS)
        for i, split in enumerate(ret):
            if split_dfs[i] is None:
                split_dfs[i] = split
            else:
                split_dfs[i] = pd.concat([split_dfs[i], split])

    split_dfs[0].to_csv(TRAIN_PATH, index=None)
    split_dfs[1].to_csv(VALID_PATH, index=None)
    split_dfs[2].to_csv(TEST_PATH, index=None)


if __name__ == '__main__':
    main()
