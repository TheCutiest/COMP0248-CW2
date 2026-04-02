import cv2
import numpy as np
import glob

files = sorted(glob.glob("object01_depth/*.png"))
img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)

print("shape:", img.shape)
print("dtype:", img.dtype)
print("min:", np.min(img))
print("max:", np.max(img))
print("unique sample:", np.unique(img)[0:10])