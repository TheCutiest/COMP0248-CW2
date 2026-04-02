# COMP0248 Coursework 2 - Task 2 Pipeline

This repository contains the Task 2 code for depth estimation, pose estimation, and view synthesis evaluation.

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

If needed, first export RGB and depth images from the ROS bag files.

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

---

# 2. Run Depth Evaluation

Run MiDaS depth evaluation for one dataset:

```bash
python run_midas_batch.py --dataset object01
python run_midas_batch.py --dataset object02
```

You can also specify another model:

```bash
python run_midas_batch.py --dataset object01 --model DPT_Large
python run_midas_batch.py --dataset object02 --model DPT_Hybrid
```

To save more visualization examples:

```bash
python run_midas_batch.py --dataset object01 --save_vis_count 10
```

Output files:

```text
output/object01_midas_metrics.csv
output/object01_midas_summary.txt
object01_midas_vis/

output/object02_midas_metrics.csv
output/object02_midas_summary.txt
object02_midas_vis/
```

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

We use ORB features + Essential Matrix + recoverPose.

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
* estimated rotation matrix R
* estimated translation direction t
* inlier counts

---

# 5. Run View Synthesis Evaluation

This script combines:

* RGB image 1
* depth image 1
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
* side-by-side panel
* photometric MAE result

---

# Notes / Limitations

* The bag files only contain RGB images, aligned depth images, and camera info topics.
* No external tracking / PhaseSpace pose topics were found.
* Therefore, direct comparison between estimated pose and tracking ground truth could not be performed.
* Relative pose estimation was evaluated using image-based methods only.
* View synthesis and reprojection error were used as joint evaluation metrics.
* Current pose scripts use fallback camera intrinsics. These can later be replaced with intrinsics extracted from `/camera/color/camera_info`.

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

1. RGB / GT depth / Pred depth / Error map
2. Best, median, and worst depth prediction cases
3. Feature matching visualizations for pose estimation
4. View synthesis panel: source image / warped image / target image / error map
5. Table comparing MiDaS_small vs DPT_Large / DPT_Hybrid
