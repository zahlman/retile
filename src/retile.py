import sys

import imageio as iio
import numpy as np


def raw1bpp(filename, *, c, r, tw, th):
    data = np.fromfile(filename, dtype=np.uint8)
    unpacked = np.unpackbits(data).reshape(r, c, th, tw)
    # group vertical and horizontal dimensions, then flatten
    # adapted from https://stackoverflow.com/questions/9071446
    result = unpacked.transpose((0, 2, 1, 3)).reshape((r*th, c*tw)) * 255
    out = filename.removesuffix('.bin') + '.png'
    iio.imwrite(out, result)
