import sys

import imageio as iio
import numpy as np


def raw1bpp(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    # 16 rows of 16 columns of 8x8 tiles
    unpacked = np.unpackbits(data).reshape(16, 16, 8, 8)
    # group vertical and horizontal dimensions, then flatten
    # adapted from https://stackoverflow.com/questions/9071446
    result = unpacked.transpose((0, 2, 1, 3)).reshape((128, 128)) * 255
    out = filename.removesuffix('.bin') + '.png'
    iio.imwrite(out, result)


if __name__ == '__main__':
    raw1bpp(sys.argv[1])
