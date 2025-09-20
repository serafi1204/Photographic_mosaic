import numpy as np
import cv2
import os 
import h5py
import numpy as np


from .configuration import *
from .resize import resize

def makeDataset(path, dataset_filename='dataset.npz', source_filename='source.h5', source_save=True):
    label = []
    data = []

    # 먼저 전체 이미지 개수를 구함
    total_files = sum(len(os.listdir(os.path.join(p, d))) 
                      for p in path for d in os.listdir(p))

    if source_save:
        # h5 파일 생성 (이미지 shape 미리 지정해야 효율적)
        f = h5py.File(source_filename, "w")
        dset = f.create_dataset(
            "data",
            shape=(total_files, *SOURCE_SIZE, 3),  # (N, H, W, C)
            dtype=np.uint8,
            compression="gzip",
            chunks=(1, *SOURCE_SIZE, 3)
        )

    idx = 0
    for path_sub in path:
        for i, name in enumerate(os.listdir(path_sub)):
            path_source = os.path.join(path_sub, name)
            filelist = os.listdir(path_source)

            print(f'Loading..."{name}"-{len(filelist)}...', end='')
            for filename in filelist:
                try:
                    pt = os.path.join(path_source, filename)
                    img = cv2.imread(pt)

                    if img is None:
                        print(f"Error file: {pt}")
                        continue

                    img_source = resize(img, SOURCE_SIZE).astype(np.uint8)
                    img_resized = resize(img, MOSAIC_SIZE).astype(np.float32) / 255.0

                    label.append(i)
                    data.append(img_resized)

                    if source_save:
                        dset[idx] = img_source  # 한 장씩 바로 저장
                    idx += 1

                except Exception as e:
                    print(f"Error file: {pt}, {e}")

            print("complete.")

    np_label = np.asarray(label, dtype=np.int8)
    np_data = np.asarray(data, dtype=np.float32)

    np.savez_compressed(dataset_filename, label=np_label, data=np_data)
    if source_save:
        f.close()
