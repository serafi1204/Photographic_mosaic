import numpy as np
import cv2

def resize(source:np.array, size:tuple, interpolation=cv2.INTER_LINEAR):
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
