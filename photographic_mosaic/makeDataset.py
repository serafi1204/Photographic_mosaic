import numpy as np
import cv2
import os 

from .configuration import *
from .resize import resize

def makeDataset(path, dataset_filename = 'dataset.npz', source_filename = 'source.npz'):
    tag = []
    label = []
    source = []
    data = []

    for i, name in enumerate(os.listdir(path)):
        path_source = os.path.join(path, name)
        filelist = os.listdir(path_source)
        
        tag.append(name)

        print(f'Loading..."{name}"-{len(filelist)}...', end='')
        for filename in filelist:
            try:
                pt = os.path.join(path_source, filename)
        
                img = np.array(cv2.imread(pt))
                img_source = resize(img, SOURCE_SIZE)
                img_resized = resize(img, MOSAIC_SIZE)
                img_data = (img_resized - 128)/256

                label.append(i)
                source.append(img_source)
                data.append(img_data)
            except:
                print(f"Error file: {pt}")

        print("complete.")
    
    np_label = np.asarray(label).astype(np.int8)
    np_source = np.asarray(source).astype(np.int8)
    np_data = np.asarray(data).astype(np.float32)

    np.savez_compressed(dataset_filename, label=np_label, data=np_data)
    np.savez_compressed(source_filename, data=np_source)

    print('Complete to generate dataset.')