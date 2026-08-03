# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

IMAGE_PATH = r"D:\HA\apple_gui\data\sessions\20260803_094736\raw_frames\ch1\frame_000001.bmp"


img = plt.imread(IMAGE_PATH)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)

plt.show()