import os
import glob
import cv2
import numpy as np
import torch
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiDaS depth evaluation on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. object01 or object02"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MiDaS_small",
        help="MiDaS model type, e.g. MiDaS_small, DPT_Large, DPT_Hybrid"
    )
    parser.add_argument(
        "--save_vis_count",
        type=int,
        default=5,
        help="How many prediction visualizations to save"
    )
    return parser.parse_args()


def get_transform(model_type):
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "small" in model_type.lower():
        return transforms.small_transform
    return transforms.dpt_transform


def safe_read_rgb(rgb_path):
    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def safe_read_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {depth_path}")
    return depth.astype(np.float32)


def make_depth_vis(depth_map):
    depth_vis = depth_map.copy()
    valid = depth_vis > 0
    if valid.sum() == 0:
        return None

    dmin = np.percentile(depth_vis[valid], 2)
    dmax = np.percentile(depth_vis[valid], 98)

    depth_vis = np.clip(depth_vis, dmin, dmax)
    depth_vis = ((depth_vis - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def main():
    args = parse_args()

    dataset_name = args.dataset
    model_type = args.model
    save_vis_count = args.save_vis_count

    rgb_dir = f"{dataset_name}_rgb"
    depth_dir = f"{dataset_name}_depth"
    vis_dir = f"{dataset_name}_midas_vis"

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    metrics_csv = os.path.join(output_dir, f"{dataset_name}_dpt_metrics.csv")
    summary_txt = os.path.join(output_dir, f"{dataset_name}_dpt_summary.txt")

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    if len(rgb_files) == 0:
        raise RuntimeError(f"No RGB files found in {rgb_dir}")
    if len(depth_files) == 0:
        raise RuntimeError(f"No depth files found in {depth_dir}")

    assert len(rgb_files) == len(depth_files), f"RGB {len(rgb_files)} != Depth {len(depth_files)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"RGB dir: {rgb_dir}")
    print(f"Depth dir: {depth_dir}")
    print(f"Vis dir: {vis_dir}")

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transform = get_transform(model_type)

    rmse_list = []
    mae_list = []
    absrel_list = []
    records = []

    for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        try:
            img_rgb = safe_read_rgb(rgb_path)
            gt_depth = safe_read_depth(depth_path)
        except Exception as e:
            print(f"[WARN] Skipping pair due to read error: {e}")
            continue

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

        mask = gt_depth > 0
        if mask.sum() == 0:
            print(f"[WARN] No valid GT depth pixels in {depth_path}, skipping.")
            continue

        pred_valid = pred_depth[mask]
        gt_valid = gt_depth[mask]

        pred_median = np.median(pred_valid)
        if abs(pred_median) < 1e-8:
            print(f"[WARN] Predicted median depth too small in {rgb_path}, skipping.")
            continue

        scale = np.median(gt_valid) / (pred_median + 1e-8)
        pred_aligned = pred_depth * scale
        pred_valid_aligned = pred_aligned[mask]

        rmse = np.sqrt(np.mean((pred_valid_aligned - gt_valid) ** 2))
        mae = np.mean(np.abs(pred_valid_aligned - gt_valid))
        abs_rel = np.mean(np.abs(pred_valid_aligned - gt_valid) / (gt_valid + 1e-8))

        rmse_list.append(rmse)
        mae_list.append(mae)
        absrel_list.append(abs_rel)

        records.append({
            "image": os.path.basename(rgb_path),
            "rmse": float(rmse),
            "mae": float(mae),
            "abs_rel": float(abs_rel),
            "scale": float(scale),
            "valid_pixels": int(mask.sum()),
        })

        if i < save_vis_count:
            pred_vis = make_depth_vis(pred_aligned)
            if pred_vis is not None:
                vis_path = os.path.join(vis_dir, f"{i:04d}.png")
                cv2.imwrite(vis_path, pred_vis)

        print(
            f"[{i+1}/{len(rgb_files)}] "
            f"{os.path.basename(rgb_path)} | "
            f"RMSE={rmse:.2f}, MAE={mae:.2f}, AbsRel={abs_rel:.4f}"
        )

    if len(rmse_list) == 0:
        raise RuntimeError("No valid images were evaluated.")

    mean_rmse = float(np.mean(rmse_list))
    std_rmse = float(np.std(rmse_list))
    mean_mae = float(np.mean(mae_list))
    std_mae = float(np.std(mae_list))
    mean_absrel = float(np.mean(absrel_list))
    std_absrel = float(np.std(absrel_list))

    print("\n==== Final Results ====")
    print(f"Mean RMSE:   {mean_rmse:.4f}")
    print(f"Std RMSE:    {std_rmse:.4f}")
    print(f"Mean MAE:    {mean_mae:.4f}")
    print(f"Std MAE:     {std_mae:.4f}")
    print(f"Mean AbsRel: {mean_absrel:.4f}")
    print(f"Std AbsRel:  {std_absrel:.4f}")
    print(f"Num images:  {len(rmse_list)}")

    df = pd.DataFrame(records)
    df.to_csv(metrics_csv, index=False)

    with open(summary_txt, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Mean RMSE: {mean_rmse:.4f}\n")
        f.write(f"Std RMSE: {std_rmse:.4f}\n")
        f.write(f"Mean MAE: {mean_mae:.4f}\n")
        f.write(f"Std MAE: {std_mae:.4f}\n")
        f.write(f"Mean AbsRel: {mean_absrel:.4f}\n")
        f.write(f"Std AbsRel: {std_absrel:.4f}\n")
        f.write(f"Num images: {len(rmse_list)}\n")

    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved summary to {summary_txt}")
    print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()