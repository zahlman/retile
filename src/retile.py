import sys

import imageio as iio
import numpy as np


_byte_values = np.arange(0, 256, dtype=np.uint8)


def _unpack_shift(bpp):
    low = 0 if bpp > 0 else 8 + bpp 
    high = 8 if bpp > 0 else None 
    return _byte_values[low:high:bpp]


def _unpack_mask(bpp):
    return (1 << abs(bpp)) - 1


def _default_palette(bpp):
    return _byte_values[::255//_unpack_mask(bpp)]


def load_raw(filename, *, c, r, tw, th, bpp):
    data = np.fromfile(filename, dtype=np.uint8)
    # compute all the shifted, masked and scaled results for each byte
    # along a second dimension, then ravel it back into 1d data.
    shift, mask = _unpack_shift(bpp), _unpack_mask(bpp)
    # 1bpp -> 255, 2bpp -> 85, 4bpp -> 17, 8bpp -> 1
    data = ((data[..., None] >> shift) & mask).reshape(-1)
    # Reinterpret as a grid of tiles which are each grids of pixels.
    unpacked = data.reshape(r, c, th, tw)
    # group vertical and horizontal dimensions, then flatten
    # adapted from https://stackoverflow.com/questions/9071446
    return unpacked.transpose((0, 2, 1, 3)).reshape((r*th, c*tw))


def raw_to_png(filename, *, c, r, tw, th, bpp):
    raw = load_raw(filename, c=c, r=r, tw=tw, th=th, bpp=bpp)
    result = _default_palette(bpp)[raw]
    out = filename.removesuffix('.bin') + '.png'
    iio.imwrite(out, result)
