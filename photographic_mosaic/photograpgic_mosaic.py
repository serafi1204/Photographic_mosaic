import cv2
import numpy as np
import torch
from importlib.resources import files as fls

from .LPIPS import LPIPS

try:
    ENVIROMENT = 'colab'

    from google.colab import files
    download = lambda : list(files.upload().keys())[0]
    
except:

    ENVIROMENT = 'pc'
    download = lambda : input("Image file(.png, .jpg) : ")



class PhotograpgicMosaic():
    def __init__(self, mosaic_resolution=(60, 60)):
        self.enviroment = ENVIROMENT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameter
        self.mosaic_resolution = mosaic_resolution # (H, W)
        self.mosaic_size = (24, 48)
        self.source_size = (540, 960)
        self.output_size = None

        # Data
        self.dataset_path = fls('photograpgic_mosaic').joinpath('dataset.pt')
        self.dataset = torch.from_numpy(np.load("dataset.npz")).to(self.device)

        # utiles
        self.compare_func = LPIPS()
        self.download = download

        # Reset
        self.reset()

    def reset(self):
        self.output_size = (self.mosaic_size[0]*self.mosaic_resolution[0], self.mosaic_size[1]*self.mosaic_resolution[1])

    def resize(self, source:np.array, size:tuple, interpolation=cv2.INTER_LINEAR):
        h, w = size
        rate = w/h
        s_h, s_w = source.shape[:2]
        s_rate = s_w/s_h

        if (rate < s_rate):
            crop_h, crop_w = s_h, int(s_h*rate)
            offset_h, offset_w = 0, (s_w-crop_w)//2
        else:
            crop_h, crop_w = int(s_w/rate), s_w
            offset_h, offset_w = (s_h-crop_h)//2, 0
            
        
        cropped = source[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]
        resized = cv2.resize(cropped, size[::-1], interpolation=interpolation)
        
        return resized

    def makeMosicMap(self, target_image_path):
        # init
        source = cv2.imread(target_image_path)
        source = self.resize(source, self.output_size, interpolation=cv2.INTER_LANCZOS4)

        mosaic_map = np.zeros(self.mosaic_resolution)

        # make mosaic map
        H, W = self.mosaic_resolution
        for h in range(H):
            for w in range(W):
                pass

