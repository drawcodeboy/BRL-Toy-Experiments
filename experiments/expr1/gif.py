import glob
import cv2
import imageio.v2 as imageio

gif_name = "both_case_neg_opposite"
frame_dir = f"experiments/expr1/{gif_name}"
paths = sorted(glob.glob(f"{frame_dir}/frame_*.png"))

frames = []
for p in paths:
    img = cv2.imread(p)              # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(img)

imageio.mimsave(
    f"{frame_dir}/../gifs/{gif_name}.gif",
    frames,
    fps=10
)