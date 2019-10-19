import random

import numpy as np
from PIL import Image
import glob


def combine_and_merge(exist_folder, none_folder):
    a, b = get_data(exist_folder, 1)
    c, d = get_data(none_folder, 0)
    e = list(zip(np.concatenate([a, c], axis=0), np.concatenate([b, d], axis=0)))
    random.shuffle(e)
    f = np.array(e).T

    return f


def get_data(folder, label):
    ims = []
    for f in glob.glob('data/' + folder + '/**/*'):
        with Image.open(f.replace('\\', '/')) as im:
            ims.append(np.array(list(im.getdata())).reshape((32,32,3)))
    return ims, np.zeros(len(ims)) + label
