# `retile` - simple image conversion and tilemapping 

`retile` can read and write images in either any format understood by `imageio`, or else in a "raw" format described in a TOML configuration file.

The config file also describes a tile-mapping operation that extracts tiles from the source image, rearranges them as desired, and saves the result - as if repeatedly copying small rectangular regions out of the source image and pasting them onto an output canvas. The dimensions of the output are automatically determined, and the output may be saved in any format.

Detailed explanation pending, as the interface is still being worked out.

`retile` is offered under a permissive MIT license as described in LICENSE.txt.
