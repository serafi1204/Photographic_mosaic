import numpy as np
import cv2
import os 

import params 
import utile.resize as rs

root = 'source'
path = os.path.join(root, 'temp')
path_resized = os.path.join(root, "resized")
path_sample = os.path.join(root, f"sampel_{params.SAMPLE_GRID_SIZE[0]}_{params.SAMPLE_GRID_SIZE[1]}")

params.FRAME_SIZE = tuple(map(int, input("FRAME_SIZE(540, 960): ").strip().split()))

dataset = {"img": [], 'sample':[], 'label': []}

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
        
        img = cv2.imdecode(np.fromfile(pt, dtype=np.uint8), cv2.IMREAD_COLOR)
        #img = cv2.imread(pt)
        resized = rs.resize(img, params.FRAME_SIZE)
        sample = cv2.resize(resized, params.SAMPLE_GRID_SIZE[::-1], interpolation=cv2.INTER_CUBIC )
        
        dataset['img'].append(resized)
        dataset['sample'].append(sample)
        dataset['label'].append(j)

        print("complete")

np.savez_compressed(f"dataset_{params.FRAME_SIZE[0]}", img=np.asarray(dataset['img']), sample=np.asarray(dataset['sample']), label=np.asarray(dataset['label']))
