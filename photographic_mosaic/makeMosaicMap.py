import numpy as np
import torch
from time import time

from .resize import resize
from .configuration import * 
from .LPIPS import LPIPS


def makeMosaicMap(target, source, resolution, reuse=False, lossFunction=LPIPS, device=None, color=None):
    # Select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # output size = block_size * resolution
    dy, dx = MOSAIC_SIZE
    output_size = (dy * resolution[0], dx * resolution[1])

    # prepare target tensor
    target = torch.from_numpy(resize(target, output_size).astype(np.float32) / 255.0)
    target = torch.permute(target, (2, 0, 1)).to(device, dtype=torch.float32)

    # dataset loader
    def reset():
        np_loaded = np.load(source)
        data = torch.from_numpy(np_loaded['data'])
        data = torch.permute(data, (0, 3, 1, 2)).to(device, dtype=torch.float32)
        label = torch.from_numpy(np_loaded['label']).to(device)
        index = torch.arange(label.shape[0], device=device)
        active_mask = torch.ones_like(index, dtype=torch.bool)
        return data, label, index, active_mask

    data, label, index, active_mask = reset()
    model = lossFunction()

    clear_cmd()
    print(f"♠♥♣◆ {len(index)} memories loaded ◆♣♥♠")

    # result maps
    mosaic_map = np.zeros(resolution, dtype=np.int32)
    label_map = np.zeros(resolution, dtype=np.int32) if (color  is None) else np.zeros((*resolution, 3), dtype=np.uint8)
    loss_map = np.zeros(resolution, dtype=np.float32)

    # iterate pixels
    order = [(y, x) for y in range(resolution[0]) for x in range(resolution[1])]
    reuse_cnt = 0
    st = time()

    for n, (y, x) in enumerate(order):
        # reset if all used
        if not active_mask.any():
            data, label, index, active_mask = reset()
            reuse_cnt += 1

        # target patch
        partial_target = target[:, y*dy:(y+1)*dy, x*dx:(x+1)*dx]

        # candidates only from active data
        candidates = data[active_mask]
        cand_index = index[active_mask]

        # compute loss
        best, loss = model(partial_target, candidates)

        # map result
        chosen_idx = cand_index[best].item()
        mosaic_map[y, x] = chosen_idx
        label_map[y, x] = label[chosen_idx].item() if (color is None) else color[int(label[chosen_idx].item())]
        loss_map[y, x] = loss.item() if torch.is_tensor(loss) else float(loss)

        # mark as used
        if not reuse:
            active_mask[chosen_idx] = False

        # progress
        progress = f"{(n+1)/len(order)*100:.1f}% ({n+1}/{len(order)}) | reuse: {reuse_cnt}"
        print('\r' + progress, end='', flush=True)

    print(f"\nElapsed time: {(time()-st)/60:.1f}min")
    return mosaic_map, label_map, loss_map
