import glob
import imageio.v2 as imageio
from PIL import Image
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='pos', type=str)
args = parser.parse_args()

expr_name = args.config

cont_frame_dir = f"experiments/expr3/assets/learn_dynamics/cont_{expr_name}"
cont_paths = sorted(glob.glob(f"{cont_frame_dir}/frame_*.png"))

soft_frame_dir = f"experiments/expr3/assets/learn_dynamics/soft_{expr_name}"
soft_paths = sorted(glob.glob(f"{soft_frame_dir}/frame_*.png"))

soft_lr5_frame_dir = f"experiments/expr3/assets/learn_dynamics/soft_{expr_name}_lr5"
soft_lr5_paths = sorted(glob.glob(f"{soft_lr5_frame_dir}/frame_*.png"))

last_img = None
fps = 10
hold_sec = 3
hold_frames = fps * hold_sec

img0 = cv2.imread(cont_paths[0])
h, w, _ = img0.shape

out = cv2.VideoWriter(
    f"experiments/expr3/assets/videos/{args.config}.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=fps,
    frameSize=(w, h * 3)
)

for idx, (p1, p2, p3) in enumerate(zip(cont_paths, soft_paths, soft_lr5_paths), start=1):
    # exec.py에서 bbox_inches='tight'로 인해 W, H가 달라지는 case가 존재
    img1 = cv2.resize(cv2.imread(p1), (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(cv2.imread(p2), (w, h), interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(cv2.imread(p3), (w, h), interpolation=cv2.INTER_AREA)

    merged = cv2.vconcat([img1, img2, img3])
    out.write(merged)
    last_img = merged

    if idx == 300:
        break

for _ in range(hold_frames):
    out.write(last_img)

out.release()