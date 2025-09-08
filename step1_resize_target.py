import numpy as np
import cv2
import os

import params 
import utile.resize as rs


target = cv2.imread('target.png')
resized = rs.resize(target, params.TARGET_SAMPLE_SIZE, interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('target_resized.png', resized)