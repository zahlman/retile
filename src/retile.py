import sys
from tomllib import load as load_toml
from types import SimpleNamespace as Namespace

import imageio as iio
import numpy as np


_byte_values = np.arange(0, 256, dtype=np.uint8)


def _unpack_shift(bpp):
    low = 0 if bpp > 0 else 8 + bpp 
    high = 8 if bpp > 0 else None 
    return _byte_values[low:high:bpp]


def _unpack_mask(bpp):
    return (1 << abs(bpp)) - 1


def _default_palette(config):
    return _byte_values[::255//_unpack_mask(config.input_format.bpp)]


def _make_namespace(obj):
    if not isinstance(obj, dict):
        return obj
    return Namespace(**{
        k: _make_namespace(v)
        for k, v in obj.items()
    })


def _get_config(config_file):
    with open(config_file, 'rb') as f: # why does tomllib expect binary?
        return _make_namespace(load_toml(f))


def load_raw(filename, fmt):
    data = np.fromfile(filename, dtype=np.uint8)
    # compute all the shifted, masked and scaled results for each byte
    # along a second dimension, then reshape to the appropriate width.
    shift, mask = _unpack_shift(fmt.bpp), _unpack_mask(fmt.bpp)
    return ((data[..., None] >> shift) & mask).reshape(-1, fmt.width)


def raw_to_png(image_filename, config_filename):
    config = _get_config(config_filename)
    raw = load_raw(image_filename, config.input_format)
    result = _default_palette(config)[raw]
    out = image_filename.removesuffix('.bin') + '.png'
    iio.imwrite(out, result)
