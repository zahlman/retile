import sys
from tomllib import load as load_toml
from types import SimpleNamespace as Namespace

# Ensure forwards compatibility as much as possible
from imageio.v3 import imread as read_image, imwrite as write_image
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


def load_raw(filename, raw_format):
    width, bpp = raw_format.input_width, raw_format.input_bpp
    data = np.fromfile(filename, dtype=np.uint8)
    # compute all the shifted, masked and scaled results for each byte
    # along a second dimension, then reshape to the appropriate width.
    shift, mask = _unpack_shift(bpp), _unpack_mask(bpp)
    return ((data[..., None] >> shift) & mask).reshape(-1, width)


def _ensure(data, min_height, min_width):
    height, width = data.shape
    height_padding = max(0, min_height - height)
    width_padding = max(0, min_width - width)
    return np.pad(data, ((0, height_padding), (0, width_padding)))


def get_tiles(data, tile_config):
    x, y = tile_config.source.offset
    pw, ph = tile_config.source.padding
    tw, th = tile_config.size
    c, r = tile_config.source.count
    w, h = c * (tw + pw), r * (th + ph)
    # Extract the region where the tiles are.
    data = _ensure(data, y+h, x+w)[y:y+h, x:x+w]
    # Rearrange into tiles (https://stackoverflow.com/questions/9071446)
    data = data.reshape(r, th+ph, c, tw+pw).transpose((0, 2, 1, 3))
    # Trim padding and produce a 1d array of tiles
    return data[:, :, :th, :tw].reshape(-1, th, tw)


def arrange_tiles(tiles, tile_config):
    x, y = tile_config.result.offset
    pw, ph = tile_config.result.padding
    tw, th = tile_config.size
    ew, eh = tile_config.result.extra
    bg = tile_config.result.bg
    # add per-tile padding on right and bottom
    tiles = np.pad(tiles, ((0, 0), (0, ph), (0, pw)), constant_values=bg)
    # Rearrange into 16-wide tile grid, then into bitmap
    c = 16
    r, extra = divmod(len(tiles), c)
    assert not extra # tile count not divisible
    tiles = tiles.reshape(r, c, th+ph, tw+pw).transpose((0, 2, 1, 3))
    # Flatten to 2d and fix padding
    result = tiles.reshape(r * (th+ph), c * (tw+pw))
    # FIXME: does this work with negative values?
    return np.pad(result, ((y, eh), (x, ew)), constant_values=bg)


def load(image_filename, config):
    if config.format.force_raw:
        return load_raw(image_filename, config.format), True
    try:
        return read_image(image_filename), False
    except FileNotFoundError:
        raise # this is a OSError subtype so handle it separately
    except OSError:
        return load_raw(image_filename, config.format), True


def convert(image_filename, config_filename):
    config = _get_config(config_filename)
    try:
        image, is_raw = load(image_filename, config)
    except FileNotFoundError as e:
        raise e from None
    if is_raw:
        image = _default_palette(config.format.input_bpp)[image]
    image = arrange_tiles(get_tiles(image, config.tiles), config.tiles)
    base, _, _ = image_filename.rpartition('.')
    write_image(f'{base}.{config.format.output_extension}', image)
