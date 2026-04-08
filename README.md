# COMP0248 Coursework 2 - Task 2 Pipeline

This repository contains the Task 2 code for monocular depth estimation, relative pose estimation, and view synthesis evaluation.

## Environment

We used a Python virtual environment named `cw2`.

Activate it with:

```bash
source ~/venvs/cw2/bin/activate
```

Install required packages:

```bash
pip install torch torchvision opencv-python pandas numpy rosbags
```

---

# Directory Structure

```text
COMP0248-CW2/
├── object01_rgb/
├── object01_depth/
├── object02_rgb/
├── object02_depth/
├── output/
├── run_midas_batch.py
├── run_task2_depth_eval.py
├── run_pose_baseline.py
├── run_view_ablation.py
├── evaluate_view_synthesis.py
├── visualize_depth.py
├── check_depth.py
├── export_rgb_depth.py
├── Group_11_calibration_01.bag
├── Group_11_calibration_02.bag
├── Group_11_object_01.bag
└── Group_11_object_02.bag
```

---

# 1. Export RGB and Depth Images

If needed, first export RGB and aligned depth images from the ROS bag files.

```bash
python export_rgb_depth.py
```

This should generate folders such as:

```text
object01_rgb/
object01_depth/
object02_rgb/
object02_depth/
```

The exported depth images come from the aligned RealSense depth stream and are used as reference depth for evaluation.

---

# 2. Run Depth Evaluation

Run monocular depth evaluation for one dataset:

```bash
python run_midas_batch.py --dataset object01
python run_midas_batch.py --dataset object02
```

You can also specify another model:

```bash
python run_midas_batch.py --dataset object01 --model MiDaS_small
python run_midas_batch.py --dataset object01 --model DPT_Large

python run_midas_batch.py --dataset object02 --model MiDaS_small
python run_midas_batch.py --dataset object02 --model DPT_Large
```

To save more visualization examples:

```bash
python run_midas_batch.py --dataset object01 --save_vis_count 10
```

To save aligned predicted depth maps:

```bash
python run_midas_batch.py --dataset object01 --model MiDaS_small --save_aligned_depth
python run_midas_batch.py --dataset object02 --model DPT_Large --save_aligned_depth
```

Output files include:

```text
output/object01_MiDaS_small_metrics.csv
output/object01_MiDaS_small_summary.txt
output/object01_DPT_Large_metrics.csv
output/object01_DPT_Large_summary.txt

output/object02_MiDaS_small_metrics.csv
output/object02_MiDaS_small_summary.txt
output/object02_DPT_Large_metrics.csv
output/object02_DPT_Large_summary.txt

object01_MiDaS_small_vis/
object01_DPT_Large_vis/
object02_MiDaS_small_vis/
object02_DPT_Large_vis/

output/object01_MiDaS_small_depth_aligned_npy/
output/object01_MiDaS_small_depth_aligned_png/
output/object01_DPT_Large_depth_aligned_npy/
output/object01_DPT_Large_depth_aligned_png/
```

Each evaluation records:

* RMSE
* MAE
* AbsRel
* scale factor
* valid depth pixels

The predicted monocular depth is median-scaled to the aligned RealSense depth before evaluation.

---

# 3. Run Full Depth Pipeline

To automatically run depth evaluation for both object01 and object02:

```bash
python run_task2_depth_eval.py
```

This script:

* checks required folders
* runs `run_midas_batch.py`
* saves metrics csv files
* saves summary txt files
* saves visualization images

---

# 4. Run Relative Pose Baseline

We use ORB features + brute-force matching + Essential Matrix + `recoverPose`.

Example commands:

```bash
python run_pose_baseline.py --dataset object01 --img1 0000.png --img2 0010.png
python run_pose_baseline.py --dataset object01 --img1 0100.png --img2 0110.png
python run_pose_baseline.py --dataset object02 --img1 0200.png --img2 0210.png
```

Outputs:

```text
output/object01_pose_baseline/
output/object02_pose_baseline/
```

Each run saves:

* feature matching visualization
* estimated rotation matrix `R`
* estimated translation direction `t`
* number of detected keypoints
* number of good matches
* essential matrix inlier count
* recoverPose inlier count

---

# 5. Run View Synthesis Evaluation

This script combines:

* source RGB image
* source depth image
* estimated relative pose

and reprojects image 1 into image 2 view.

Example commands:

```bash
python evaluate_view_synthesis.py --dataset object01 --img1 0000.png --img2 0010.png
python evaluate_view_synthesis.py --dataset object01 --img1 0100.png --img2 0110.png
python evaluate_view_synthesis.py --dataset object02 --img1 0200.png --img2 0210.png
```

Outputs:

```text
output/object01_view_synthesis/
output/object02_view_synthesis/
```

Each run saves:

* warped image
* target image
* valid mask
* error map
* side-by-side visualization panel
* photometric MAE
* number of valid warped pixels

---

# 6. Run Depth-Source Ablation for View Synthesis

To compare different depth sources under the same estimated pose:

```bash
python run_view_ablation.py --dataset object01 --img1 0000.png --img2 0010.png --depth_source gt
python run_view_ablation.py --dataset object01 --img1 0000.png --img2 0010.png --depth_source MiDaS_small
python run_view_ablation.py --dataset object01 --img1 0000.png --img2 0010.png --depth_source DPT_Large
```

Additional examples:

```bash
python run_view_ablation.py --dataset object01 --img1 0100.png --img2 0110.png --depth_source gt
python run_view_ablation.py --dataset object01 --img1 0100.png --img2 0110.png --depth_source MiDaS_small
python run_view_ablation.py --dataset object01 --img1 0100.png --img2 0110.png --depth_source DPT_Large

python run_view_ablation.py --dataset object02 --img1 0200.png --img2 0210.png --depth_source gt
python run_view_ablation.py --dataset object02 --img1 0200.png --img2 0210.png --depth_source MiDaS_small
python run_view_ablation.py --dataset object02 --img1 0200.png --img2 0210.png --depth_source DPT_Large
```

Supported depth sources:

* `gt`
* `MiDaS_small`
* `DPT_Large`

Outputs:

```text
output/object01_view_synthesis_gt/
output/object01_view_synthesis_MiDaS_small/
output/object01_view_synthesis_DPT_Large/

output/object02_view_synthesis_gt/
output/object02_view_synthesis_MiDaS_small/
output/object02_view_synthesis_DPT_Large/
```

This experiment helps separate the effect of depth quality from pose quality.

---

# Notes / Limitations

* The bag files contain RGB images, aligned depth images, and camera info topics.
* The aligned RealSense depth images are used as reference depth for evaluation.
* No external tracking, motion-capture, or PhaseSpace pose topics were found in the bag files.
* Therefore, direct comparison between estimated pose and pose ground truth could not be performed.
* Pose quality is instead evaluated indirectly using:

  * feature match quality
  * essential matrix inliers
  * recoverPose inliers
  * reprojection quality
  * photometric MAE
* Current scripts use fallback camera intrinsics.
* These intrinsics can later be replaced with values extracted from `/camera/color/camera_info`.

---

# Main Dependencies

```text
Python 3.10+
Torch
OpenCV
NumPy
Pandas
rosbags
```

---

# Suggested Report Figures

Useful figures to include in the report:

1. RGB image / reference depth / predicted depth / error map
2. Best, median, and worst depth prediction examples
3. MiDaS_small vs DPT_Large comparison table
4. Feature matching visualizations for pose estimation
5. View synthesis panel:

   * source image
   * warped image
   * target image
   * valid mask
   * error map
6. Depth-source ablation table:

   * GT depth
   * MiDaS_small
   * DPT_Large
7. Example showing that lower photometric MAE usually corresponds to better depth quality
