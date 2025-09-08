

import numpy as np
import cv2
import os 

import params 
import utile.resize as rs

root = 'source'
path = os.path.join(root, 'temp')
path_resized = os.path.join(root, "resized")
path_sample = os.path.join(root, f"sampel_{params.SAMPLE_GRID_SIZE[0]}_{params.SAMPLE_GRID_SIZE[1]}")

if (not os.path.exists(path_sample)):
    os.makedirs(path_sample)

folderList = os.listdir(root+r"\temp")
for j, label in enumerate(folderList):
    path_label = os.path.join(root, "temp", label)
    filelist = os.listdir(path_label)
    for i, filename in enumerate(filelist):

        print(f"{filename}...", end="")  

        pt = os.path.join(path_label, filename)
        pt_resized = os.path.join(path_resized, f"{j}_{i}.png")
        pt_sample = os.path.join(path_sample, f"{j}_{i}.png")
 
        img = cv2.imread(pt)
        resized = rs.resize(img, params.FRAME_SIZE)
        sample = cv2.resize(resized, params.SAMPLE_GRID_SIZE[::-1], interpolation=cv2.INTER_CUBIC )
        
        cv2.imwrite(pt_resized, resized)
        cv2.imwrite(pt_sample, sample)

        print("complete")