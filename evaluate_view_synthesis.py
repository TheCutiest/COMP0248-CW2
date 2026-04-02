import os
import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple view synthesis evaluation using depth + relative pose.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. object01")
    parser.add_argument("--img1", type=str, required=True, help="Source image filename, e.g. 0000.png")
    parser.add_argument("--img2", type=str, required=True, help="Target image filename, e.g. 0010.png")
    parser.add_argument("--max_features", type=int, default=2000, help="ORB max features")
    parser.add_argument("--good_match_ratio", type=float, default=0.8, help="Lowe ratio test threshold")
    return parser.parse_args()


def load_camera_intrinsics(dataset_name):
    intr_path = os.path.join("output", f"{dataset_name}_camera_intrinsics.npz")
    if os.path.exists(intr_path):
        data = np.load(intr_path)
        return data["K"]

    fx = 615.0
    fy = 615.0
    cx = 320.0
    cy = 240.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    print("[WARN] Using fallback intrinsics. Replace with true camera intrinsics if available.")
    return K


def estimate_pose(img1_gray, img2_gray, K, max_features=2000, ratio=0.8):
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("ORB failed to find descriptors.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        raise RuntimeError(f"Not enough good matches: {len(good_matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    E, mask_E = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed.")

    retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, kp1, kp2, good_matches, retval


def warp_image_with_depth(img1, depth1, R, t, K):
    h, w = depth1.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.indices((h, w))
    z = depth1.astype(np.float32)

    valid = z > 0
    xs_valid = xs[valid].astype(np.float32)
    ys_valid = ys[valid].astype(np.float32)
    z_valid = z[valid]

    X = (xs_valid - cx) * z_valid / fx
    Y = (ys_valid - cy) * z_valid / fy

    pts3d = np.stack([X, Y, z_valid], axis=0)  # 3 x N
    pts3d_2 = R @ pts3d + t.reshape(3, 1)

    X2 = pts3d_2[0]
    Y2 = pts3d_2[1]
    Z2 = pts3d_2[2]

    valid2 = Z2 > 1e-6
    X2 = X2[valid2]
    Y2 = Y2[valid2]
    Z2 = Z2[valid2]

    src_x = xs_valid[valid2]
    src_y = ys_valid[valid2]

    u2 = (fx * X2 / Z2 + cx).round().astype(np.int32)
    v2 = (fy * Y2 / Z2 + cy).round().astype(np.int32)

    inside = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)

    u2 = u2[inside]
    v2 = v2[inside]
    src_x = src_x[inside].astype(np.int32)
    src_y = src_y[inside].astype(np.int32)
    Z2 = Z2[inside]

    warped = np.zeros_like(img1)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)
    valid_mask = np.zeros((h, w), dtype=np.uint8)

    for uu, vv, sx, sy, zz in zip(u2, v2, src_x, src_y, Z2):
        if zz < z_buffer[vv, uu]:
            z_buffer[vv, uu] = zz
            warped[vv, uu] = img1[sy, sx]
            valid_mask[vv, uu] = 255

    return warped, valid_mask


def main():
    args = parse_args()

    dataset = args.dataset
    rgb_dir = f"{dataset}_rgb"
    depth_dir = f"{dataset}_depth"

    img1_path = os.path.join(rgb_dir, args.img1)
    img2_path = os.path.join(rgb_dir, args.img2)
    depth1_path = os.path.join(depth_dir, args.img1)

    if not os.path.exists(img1_path):
        raise FileNotFoundError(img1_path)
    if not os.path.exists(img2_path):
        raise FileNotFoundError(img2_path)
    if not os.path.exists(depth1_path):
        raise FileNotFoundError(depth1_path)

    out_dir = os.path.join("output", f"{dataset}_view_synthesis")
    os.makedirs(out_dir, exist_ok=True)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if img1 is None or img2 is None or depth1 is None:
        raise RuntimeError("Failed to read one of the inputs.")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    K = load_camera_intrinsics(dataset)

    R, t, kp1, kp2, good_matches, pose_inliers = estimate_pose(
        img1_gray, img2_gray, K,
        max_features=args.max_features,
        ratio=args.good_match_ratio
    )

    warped, valid_mask = warp_image_with_depth(img1, depth1, R, t, K)

    valid = valid_mask > 0
    if valid.sum() == 0:
        raise RuntimeError("Warp produced no valid pixels.")

    diff = np.abs(warped.astype(np.float32) - img2.astype(np.float32))
    mae = float(np.mean(diff[valid]))

    error_map = np.mean(diff, axis=2)
    error_vis = np.zeros_like(error_map, dtype=np.uint8)
    err_valid = error_map[valid]
    if err_valid.size > 0:
        emin = np.percentile(err_valid, 2)
        emax = np.percentile(err_valid, 98)
        error_vis = np.clip(error_map, emin, emax)
        error_vis = ((error_vis - emin) / (emax - emin + 1e-8) * 255).astype(np.uint8)
        error_vis = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

    pair_name = f"{args.img1[:-4]}_{args.img2[:-4]}"
    cv2.imwrite(os.path.join(out_dir, f"{pair_name}_warped.png"), warped)
    cv2.imwrite(os.path.join(out_dir, f"{pair_name}_target.png"), img2)
    cv2.imwrite(os.path.join(out_dir, f"{pair_name}_valid_mask.png"), valid_mask)
    cv2.imwrite(os.path.join(out_dir, f"{pair_name}_error.png"), error_vis)

    panel = np.hstack([
        img1,
        warped,
        img2,
        error_vis
    ])
    cv2.imwrite(os.path.join(out_dir, f"{pair_name}_panel.png"), panel)

    result_txt = os.path.join(out_dir, f"{pair_name}_metrics.txt")
    with open(result_txt, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Source image: {args.img1}\n")
        f.write(f"Target image: {args.img2}\n")
        f.write(f"Good matches: {len(good_matches)}\n")
        f.write(f"recoverPose inliers: {int(pose_inliers)}\n")
        f.write(f"Valid warped pixels: {int(valid.sum())}\n")
        f.write(f"Photometric MAE: {mae:.4f}\n")
        f.write("R:\n")
        f.write(np.array2string(R, precision=6, suppress_small=True))
        f.write("\n\n")
        f.write("t (unit direction, up to scale):\n")
        f.write(np.array2string(t, precision=6, suppress_small=True))
        f.write("\n")

    print("=== View Synthesis Evaluation ===")
    print(f"Dataset: {dataset}")
    print(f"Pair: {args.img1} -> {args.img2}")
    print(f"Good matches: {len(good_matches)}")
    print(f"recoverPose inliers: {int(pose_inliers)}")
    print(f"Valid warped pixels: {int(valid.sum())}")
    print(f"Photometric MAE: {mae:.4f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()