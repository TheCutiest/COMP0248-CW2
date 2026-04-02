import cv2
import numpy as np
import torch

rgb_path = 'object01_rgb/0000.png'
depth_gt_path = 'object01_depth/0000.png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

img_bgr = cv2.imread(rgb_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

pred_depth = prediction.cpu().numpy()
gt_depth = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

mask = gt_depth > 0
pred_valid = pred_depth[mask]
gt_valid = gt_depth[mask]

scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
pred_aligned = pred_depth * scale
pred_valid_aligned = pred_aligned[mask]

rmse = np.sqrt(np.mean((pred_valid_aligned - gt_valid) ** 2))
mae = np.mean(np.abs(pred_valid_aligned - gt_valid))
abs_rel = np.mean(np.abs(pred_valid_aligned - gt_valid) / (gt_valid + 1e-8))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"AbsRel: {abs_rel:.4f}")
print(f"Scale used: {scale:.4f}")