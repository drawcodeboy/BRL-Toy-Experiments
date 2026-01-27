import glob
import imageio.v2 as imageio
from PIL import Image
import numpy as np
import cv2
import argparse


cont_frame_dir = f"experiments/expr6/assets/cont"
cont_paths = sorted(glob.glob(f"{cont_frame_dir}/frame_*.png"))

soft_frame_dir = f"experiments/expr6/assets/soft_cont"
soft_paths = sorted(glob.glob(f"{soft_frame_dir}/frame_*.png"))

last_img = None
fps = 100
hold_sec = 3
hold_frames = fps * hold_sec

img0 = cv2.imread(cont_paths[0])
h, w, _ = img0.shape

out = cv2.VideoWriter(
    f"experiments/expr6/videos/merge.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=fps,
    frameSize=(w * 2, h)
)

for idx, (p1, p2) in enumerate(zip(cont_paths, soft_paths), start=1):
    # exec.py에서 bbox_inches='tight'로 인해 W, H가 달라지는 case가 존재
    img1 = cv2.resize(cv2.imread(p1), (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(cv2.imread(p2), (w, h), interpolation=cv2.INTER_AREA)

    merged = cv2.hconcat([img1, img2])
    out.write(merged)
    last_img = merged

for _ in range(hold_frames):
    out.write(last_img)

out.release()