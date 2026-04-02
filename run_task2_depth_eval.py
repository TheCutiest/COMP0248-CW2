import os
import sys
import subprocess
from pathlib import Path


DATASETS = ["object01", "object02"]
MODEL = "MiDaS_small"
SAVE_VIS_COUNT = 5

SCRIPT_NAME = "run_midas_batch.py"


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def check_required_dirs(dataset_name):
    rgb_dir = Path(f"{dataset_name}_rgb")
    depth_dir = Path(f"{dataset_name}_depth")

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing directory: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing directory: {depth_dir}")

    rgb_files = sorted(rgb_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))

    if len(rgb_files) == 0:
        raise RuntimeError(f"No RGB images found in {rgb_dir}")
    if len(depth_files) == 0:
        raise RuntimeError(f"No depth images found in {depth_dir}")

    print(f"[{dataset_name}] RGB images:   {len(rgb_files)}")
    print(f"[{dataset_name}] Depth images: {len(depth_files)}")


def check_output_files(dataset_name):
    metrics_csv = Path("output") / f"{dataset_name}_midas_metrics.csv"
    summary_txt = Path("output") / f"{dataset_name}_midas_summary.txt"
    vis_dir = Path(f"{dataset_name}_midas_vis")

    ok = True

    if not metrics_csv.exists():
        print(f"[WARN] Missing metrics file: {metrics_csv}")
        ok = False
    if not summary_txt.exists():
        print(f"[WARN] Missing summary file: {summary_txt}")
        ok = False
    if not vis_dir.exists():
        print(f"[WARN] Missing visualization dir: {vis_dir}")
        ok = False

    if ok:
        print(f"[{dataset_name}] Outputs look good.")
        print(f"  - {metrics_csv}")
        print(f"  - {summary_txt}")
        print(f"  - {vis_dir}")


def main():
    if not Path(SCRIPT_NAME).exists():
        raise FileNotFoundError(f"Cannot find {SCRIPT_NAME} in current directory.")

    os.makedirs("output", exist_ok=True)

    print("Task 2 depth evaluation pipeline")
    print(f"Model: {MODEL}")
    print(f"Datasets: {DATASETS}")

    for dataset_name in DATASETS:
        check_required_dirs(dataset_name)

        cmd = [
            sys.executable,
            SCRIPT_NAME,
            "--dataset", dataset_name,
            "--model", MODEL,
            "--save_vis_count", str(SAVE_VIS_COUNT),
        ]
        run_command(cmd)
        check_output_files(dataset_name)

    print("\n" + "=" * 80)
    print("All depth evaluations completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()