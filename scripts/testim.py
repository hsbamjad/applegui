# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

IMAGE_PATH = r"S:\MSU_Research\apple_gui\frame_000492.jpg"


img = plt.imread(IMAGE_PATH)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)

plt.show()