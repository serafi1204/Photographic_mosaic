import numpy as np
import cv2
import os 
import h5py
import numpy as np


from .configuration import *
from .resize import resize

def makeDataset(path, dataset_filename = 'dataset.npz', source_filename = 'source.npz', source_save= True):
    label = []
    source = []
    data = []

    for path_sub in path:
        for i, name in enumerate(os.listdir(path_sub)):
            path_source = os.path.join(path_sub, name)
            filelist = os.listdir(path_source)

            print(f'Loading..."{name}"-{len(filelist)}...', end='')
            for filename in filelist:
                try:
                    pt = os.path.join(path_source, filename)
            
                    img = np.array(cv2.imread(pt))
                    img_source = resize(img, SOURCE_SIZE)
                    img_resized = resize(img, MOSAIC_SIZE)
                    img_data = img_resized.astype(np.float32) / 255.0

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
    if (source_save):
        with h5py.File("source.h5", "w") as f:
            f.create_dataset("data", data=np_source, compression="gzip", chunks=True)

    print('Complete to generate dataset.')