import numpy as np
from time import time

from .configuration import *
from .resize import resize


def assemble(mosaic_map, source, gain=1):
    source = np.load(source)
    print(source.keys())
    source = source['data']*gain

    source_h, source_w = source.shape[1:3]
    mosaic_h, mosaic_w = mosaic_map.shape
    h, w = source_h*mosaic_h, source_w*mosaic_w

    res = np.zeros((h, w, 3))

    for i in range(mosaic_h):
        for j in range(mosaic_w):
            x = i*source_h
            y = j*source_w
            res[x:x+source_h, y:y+source_w] = source[int(mosaic_map[i, j])]

            progress = f"{(i*mosaic_h*j)/mosaic_h*mosaic_w*100:.1f}% ({i*mosaic_h*j}/{mosaic_h*mosaic_w})"
            print('\r' + progress, end='', flush=True)
    
    print()
    return res.astype("uint8")



