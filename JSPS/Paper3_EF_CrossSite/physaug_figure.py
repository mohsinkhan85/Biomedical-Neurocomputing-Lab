r"""
Paper figure: one EchoNet-Dynamic training frame under each physics-informed augmentation effect on its own.
CPU only, about a minute. Run from D:\EchoNet (train_physaug.py must be in the same folder):
  !"{sys.executable}" physaug_figure.py
Writes D:\EchoNet\physaug_effects.png and .pdf (send both). Try --seed 1, 2, ... to pick a clearer video.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_physaug as tp

NAMES = [("dr", "Dynamic range"), ("gain", "Gain"), ("tgc", "Depth gain (TGC)"), ("near", "Near field"),
         ("blur", "Point-spread blur"), ("speckle", "Speckle"), ("noise", "Noise floor")]

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"D:\EchoNet\EchoNet-Dynamic")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=r"D:\EchoNet\physaug_effects")
a = ap.parse_args()

df = tp.read_filelist(Path(a.data), None)
path = df[df.Split == "TRAIN"].path.sample(1, random_state=a.seed).iloc[0]
v = tp.load_video(path)[0][:8].astype(np.float32)           # (T, H, W) grey, first channel
tiles = [("Original", v[0])]
for i, (key, name) in enumerate(NAMES):
    aug = tp.PhysAug([key], 1.0, seed=a.seed * 100 + i)
    tiles.append((name, aug(v.copy())[0]))
tiles.append(("All effects", tp.PhysAug([k for k, _ in NAMES], 1.0, seed=a.seed + 7)(v.copy())[0]))

fig, axs = plt.subplots(1, len(tiles), figsize=(7.16, 1.25), constrained_layout=True)
for ax, (name, img) in zip(axs, tiles):
    ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(name, fontsize=6.5)
    ax.axis("off")
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=300)
print(f"Wrote {a.out}.png / .pdf  (video {Path(path).name})")
