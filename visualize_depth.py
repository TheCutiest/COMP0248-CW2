import cv2
import numpy as np
import glob
import os

in_dir = "object01_depth"
out_dir = "object01_depth_vis"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(f"{in_dir}/*.png"))

for f in files[:20]:
    depth = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32)

    valid = depth > 0
    if np.sum(valid) == 0:
        continue

    dmin = np.percentile(depth[valid], 2)
    dmax = np.percentile(depth[valid], 98)

    depth_clip = np.clip(depth, dmin, dmax)
    depth_norm = ((depth_clip - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    name = os.path.basename(f)
    cv2.imwrite(os.path.join(out_dir, name), depth_color)

print("saved to", out_dir)