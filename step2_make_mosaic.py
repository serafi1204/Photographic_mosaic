import numpy as np
import cv2
import os
from copy import deepcopy

import params 
import utile.resize as rs
import utile.snr as snr

label_color = [[255, 0, 0], [0, 255, 255], [0, 0, 255], [0, 255, 0]]

root = 'source'
path_resized = os.path.join(root, "resized")
path_sample = os.path.join(root, f"sampel_{params.SAMPLE_GRID_SIZE[0]}_{params.SAMPLE_GRID_SIZE[1]}")


samplelist = os.listdir(path_sample)
samples = [(cv2.cvtColor(cv2.imread(os.path.join(path_sample, filename)), params.COMPARE_FORMAT), filename) for filename in samplelist]
backup = deepcopy(samples)

target = cv2.cvtColor(cv2.imread("target_resized.png"), params.COMPARE_FORMAT)
if target is None:
    raise FileNotFoundError("target_resized.png 를 불러올 수 없습니다.")

mosaic = np.zeros((*params.TARGET_SIZE, 3))
label_map = np.zeros((*params.TARGET_GRID, 3))
for x in range(params.TARGET_GRID[0]):
    for y in range(params.TARGET_GRID[1]):
        x0, y0 = x*params.SAMPLE_GRID_SIZE[0], y*params.SAMPLE_GRID_SIZE[1]
        x1, y1 = x0+params.SAMPLE_GRID_SIZE[0], y0+params.SAMPLE_GRID_SIZE[1]
        piece = target[x0:x1, y0:y1]
        
        maxSampleFilename = '' 
        maxSNR = -np.inf
        maxIndex = -1
        sample_img = None
        for i, (sample, filename) in enumerate(samples):
            sn = params.COMPARE_FUNCTION(piece, sample)
            if (sn > maxSNR or maxSNR == -np.inf):
                maxSampleFilename = filename
                maxIndex = i
                maxSNR = sn
                sample_img = sample 

        
        x0, y0 = x*params.FRAME_SIZE[0], y*params.FRAME_SIZE[1]
        x1, y1 = x0+params.FRAME_SIZE[0], y0+params.FRAME_SIZE[1]

        label_map[x, y] = label_color[int(maxSampleFilename[0])] 
        mosaic[x0:x1, y0:y1] = cv2.imread(os.path.join(path_resized, maxSampleFilename))
        print(f"({x}, {y})    {maxSampleFilename} ({maxSNR:0.3f})")

        """
        combined_img = np.hstack((piece, sample_img))  # 가로로 붙이기
        cv2.imshow('sample', combined_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        """
        #del samples[maxIndex]
        if (len(samples) == 0): 
            samples = deepcopy(backup)
    #if (len(samples) == 0): break

cv2.imwrite("mosaic.png", mosaic)
cv2.imwrite("label_map.png", label_map)
print("\n---------------------------------------------")
target = rs.resize(target, params.TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
print(f"Processing compete...SNR: {snr.SNR(target, mosaic)}")