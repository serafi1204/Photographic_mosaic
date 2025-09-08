import cv2
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity as ssim

def SNR(ref:np.array, source:np.array):
  if ref.shape != source.shape:
    raise ValueError(f"ref and source must have the same shape(ref:{ref.shape}/source:{source.shape})")

  ref = ref.astype(np.float32)
  source = source.astype(np.float32)
  noise =np.sum(np.pow(np.abs(ref - source), 2))
  snr = np.sum(np.pow(ref, 2))/noise

  return 10*np.log10(snr)

def color_difference(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Shape mismatch")

    lab1 = color.rgb2lab(img1 / 255.0)
    lab2 = color.rgb2lab(img2 / 255.0)

    delta_e = color.deltaE_ciede2000(lab1, lab2)

    return -np.mean(delta_e)

def SSIM(img1, img2):
    if img1.shape != img2.shape:
      raise ValueError(f"ref and source must have the same shape(ref:{img1.shape}/source:{img2.shape})")
    lab1 = color.rgb2lab(img1 / 255.0)
    lab2 = color.rgb2lab(img2 / 255.0)

    similarity = ssim(lab1, lab2, data_range=1.0, win_size=3)
    return similarity
