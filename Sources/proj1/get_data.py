import random

import numpy as np
import cv2
import glob
from scipy.io import wavfile

# the bee2 has trouble with amounts of images when excluding 150s
# so, rotate and add doubling the test,train, etc vars
def get_data(folder, folder_none, dim):
    x = []
    y = []
    l = int(np.sqrt(dim))
    d = (int(l/2), int(l/2))
    # get the rotation matrices
    M = cv2.getRotationMatrix2D(d, 90, 1.0)
    M2 = cv2.getRotationMatrix2D(d, 180, 1.0)
    for f in glob.glob('data/' + folder + '/**/*'):
        im = cv2.imread(f.replace('\\', '/'))
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
        grey.resize((l, l), refcheck=False)
        # NAN check
        if grey.all() != grey.all():
            print('bad')
        else:
            # append date
            x.append(grey.reshape(1, dim))
            y.append(1)

            # append 90 rotation
            grey2 = cv2.warpAffine(grey, M, (l, l))
            x.append(grey2.reshape(1, dim))
            y.append(1)

            # append 180 rotation
            grey3 = cv2.warpAffine(grey, M2, (l, l))
            x.append(grey3.reshape(1, dim))
            y.append(1)

    # same as above, but for the non bee data
    for f in glob.glob('data/' + folder_none + '/**/*'):
        im = cv2.imread(f.replace('\\', '/'))
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
        grey.resize((l, l), refcheck=False)
        # NAN check
        if grey.all() != grey.all():
            print('bad')
        else:
            x.append(grey.reshape(1, dim))
            y.append(0)
            grey2 = cv2.warpAffine(grey, M, (l, l))
            x.append(grey2.reshape(1, dim))
            y.append(0)
            grey3 = cv2.warpAffine(grey, M2, (l, l))
            x.append(grey3.reshape(1, dim))
            y.append(0)

    e = list(zip(x, y))
    random.shuffle(e)
    x, y = zip(*e)
    return np.asarray(x).reshape(-1, dim), np.asarray(y).reshape(-1, 1)


def get_wav(folder):
    x = []
    for f in glob.glob('data/' + folder + '/*.wav'):
        sr, a = wavfile.read(f)
        x.append([sr,a])