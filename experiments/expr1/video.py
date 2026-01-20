import glob
import imageio.v2 as imageio
from PIL import Image
import numpy as np
import cv2

video_name = "both_case_neg_opposite"
frame_dir = f"experiments/expr1/{video_name}"
paths = sorted(glob.glob(f"{frame_dir}/frame_*.png"))

last_img = None
fps = 10
hold_sec = 3
hold_frames = fps * hold_sec

img0 = cv2.imread(paths[0])
h, w, _ = img0.shape

out = cv2.VideoWriter(
    f"{frame_dir}/../videos/{video_name}.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=fps,
    frameSize=(w, h)
)

for p in paths:
    img = cv2.imread(p)
    out.write(img)
    last_img = img

for _ in range(hold_frames):
    out.write(last_img)

out.release()