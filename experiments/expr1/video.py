import glob
import imageio.v2 as imageio
from PIL import Image
import numpy as np
import cv2

frame_dir = "experiments/expr1/both_open_case"
paths = sorted(glob.glob(f"{frame_dir}/frame_*.png"))

img0 = cv2.imread(paths[0])
h, w, _ = img0.shape

out = cv2.VideoWriter(
    f"{frame_dir}/_simulation.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=15,
    frameSize=(w, h)
)

for p in paths:
    img = cv2.imread(p)
    out.write(img)

out.release()