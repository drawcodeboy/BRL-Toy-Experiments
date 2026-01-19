import glob
import cv2
import imageio.v2 as imageio

frame_dir = "experiments/expr1/both_open_case"
paths = sorted(glob.glob(f"{frame_dir}/frame_*.png"))

frames = []
for p in paths:
    img = cv2.imread(p)              # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(img)

imageio.mimsave(
    f"{frame_dir}/_simulation.gif",
    frames,
    fps=15
)