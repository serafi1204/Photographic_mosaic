import numpy as np
import torch

from .resize import resize
from .configuration import * 
from .LPIPS import LPIPS
from .spiral_from_center import spiral_from_center


def makeMosaicMap(target, source, resolution, label_color=None, reuse=False):
    # init
    output_size = (MOSAIC_SIZE[0]*resolution[0], MOSAIC_SIZE[1]*resolution[1])
    dy, dx = MOSAIC_SIZE

    target = torch.from_numpy((resize(target, output_size)-128)/256); target = torch.permute(target, (2, 0, 1))

    def reset():
        np_loaded = np.load(source)
        data = torch.from_numpy(np_loaded['data']); data = torch.permute(data, (0, 3, 1, 2))    
        label = np_loaded['label']
        index = [i for i in range(label.shape[0])]

        return data, label, index

    data, label, index = reset()
    log = "# {len(index)} memories loaded."

    mosaic_map = np.zeros(resolution)
    label_map = np.zeros(resolution)
    loss_map = np.zeros(resolution)

    model = LPIPS()

    # make mosaic map
    order = []
    reuse = 0
    for x in range(resolution[0]):
        for y in range(resolution[1]):
            order.append((x, y))

    for n, (x, y) in enumerate(order):
        if (len(index) == 0):
            data, label, index = reset()
            reuse += 1
    
        partial_target = target[:, y:y+dy, x:x+dx]
        best, loss = model(partial_target, data)
        
        i = index[best]
        mosaic_map[x, y] = i
        label_map[x, y] = label[i]
        loss_map[x, y] = loss

        if (not reuse):
            index.pop(best)
            data = torch.cat((data[:best], data[best+1:]))
        
        clear()
        print(log)
        print(f"reuse: {reuse}")
        print(f"{(n+1)/len(order)*100:.1f}% ({n+1}/{len(order)})")
    
    return mosaic_map, label_map, loss_map