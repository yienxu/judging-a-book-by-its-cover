import pandas as pd
import os

from PIL import Image

##### Parameters

CSV_PATH = 'book_data.csv'
IMG_PATH = 'images'
DATASET_PATH = 'all.csv'

LEVELS = 10


##### Helpers

def get_list_of_levels(length, levels):
    chunk_size = length / levels
    if chunk_size % 1:
        raise ValueError('length is not divisible by levels')
    chunk_size = int(chunk_size)

    ret = []
    for i in range(levels):
        lst = [i] * chunk_size
        ret.extend(lst)

    return ret


def get_filename_from_index(index):
    img = str(index) + '.jpg'
    return os.path.join(IMG_PATH, img)


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

    df.to_csv(DATASET_PATH, index=None)


if __name__ == '__main__':
    main()
