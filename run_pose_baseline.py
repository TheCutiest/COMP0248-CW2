import os
import cv2
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Relative pose baseline using ORB + Essential Matrix.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. object01")
    parser.add_argument("--img1", type=str, required=True, help="First image filename, e.g. 0000.png")
    parser.add_argument("--img2", type=str, required=True, help="Second image filename, e.g. 0010.png")
    parser.add_argument("--max_features", type=int, default=2000, help="ORB max features")
    parser.add_argument("--good_match_ratio", type=float, default=0.8, help="Lowe ratio test threshold")
    return parser.parse_args()


def load_camera_intrinsics(dataset_name):
    """
    Load camera intrinsics from a saved npz if available.
    Otherwise use a reasonable fallback.
    You can replace this with your exact intrinsics later.
    """
    intr_path = os.path.join("output", f"{dataset_name}_camera_intrinsics.npz")
    if os.path.exists(intr_path):
        data = np.load(intr_path)
        K = data["K"]
        return K

    # fallback values; replace if you later extract true intrinsics
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


def main():
    args = parse_args()

    dataset = args.dataset
    rgb_dir = f"{dataset}_rgb"
    img1_path = os.path.join(rgb_dir, args.img1)
    img2_path = os.path.join(rgb_dir, args.img2)

    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Image not found: {img2_path}")

    out_dir = os.path.join("output", f"{dataset}_pose_baseline")
    os.makedirs(out_dir, exist_ok=True)

    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)

    if img1_color is None or img2_color is None:
        raise RuntimeError("Failed to read one or both images.")

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    K = load_camera_intrinsics(dataset)

    orb = cv2.ORB_create(nfeatures=args.max_features)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("ORB failed to find descriptors in one of the images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < args.good_match_ratio * n.distance:
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

    inlier_count = int(mask_E.sum()) if mask_E is not None else 0

    retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    match_vis = cv2.drawMatches(
        img1_color, kp1, img2_color, kp2, good_matches[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    match_vis_path = os.path.join(out_dir, f"{args.img1[:-4]}_{args.img2[:-4]}_matches.png")
    cv2.imwrite(match_vis_path, match_vis)

    result_txt = os.path.join(out_dir, f"{args.img1[:-4]}_{args.img2[:-4]}_pose.txt")
    with open(result_txt, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Image 1: {args.img1}\n")
        f.write(f"Image 2: {args.img2}\n")
        f.write(f"Num keypoints img1: {len(kp1)}\n")
        f.write(f"Num keypoints img2: {len(kp2)}\n")
        f.write(f"Num good matches: {len(good_matches)}\n")
        f.write(f"Essential matrix inliers: {inlier_count}\n")
        f.write(f"recoverPose inliers: {int(retval)}\n")
        f.write("R:\n")
        f.write(np.array2string(R, precision=6, suppress_small=True))
        f.write("\n\n")
        f.write("t (unit direction, up to scale):\n")
        f.write(np.array2string(t, precision=6, suppress_small=True))
        f.write("\n")

    print("=== Pose Baseline Result ===")
    print(f"Dataset: {dataset}")
    print(f"Image pair: {args.img1}, {args.img2}")
    print(f"Keypoints: {len(kp1)} / {len(kp2)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"Essential matrix inliers: {inlier_count}")
    print(f"recoverPose inliers: {int(retval)}")
    print("R =")
    print(R)
    print("t =")
    print(t)
    print(f"Saved match visualization to: {match_vis_path}")
    print(f"Saved pose result to: {result_txt}")


if __name__ == "__main__":
    main()