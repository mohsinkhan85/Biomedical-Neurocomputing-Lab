r"""
EchoNet-Dynamic EF training with GENERIC (non-physics) intensity augmentation: the comparison baseline
for physics-informed augmentation (JSPS Paper 3, package C).

Same model, schedule, data, crop/pad and evaluation as train_physaug.py; only the augmentation differs.
Effects are standard image-intensity augmentations with no ultrasound model, applied to the displayed
grey-level clip (whole frame), each independently with probability --p (default 0.5, same as physaug),
parameters drawn once per clip. Ranges follow the nnU-Net default intensity augmentations
(Isensee et al., Nature Methods 2021), fixed a priori, NOT fitted to CAMUS.

Effects (--aug, comma-separated, or "all"), applied in this order (as in nnU-Net):
  noise        additive Gaussian noise, sd = U(0, sqrt(0.1)) x clip sd   (nnU-Net: variance 0-0.1 on unit-sd image)
  gblur        isotropic Gaussian blur, sigma 0.5-1.0 px
  brightness   multiply by 0.75-1.25
  contrast     (x - mean) x 0.75-1.25 + mean, clipped to the clip's original range
  gamma        255 (x / 255)^g, g 0.7-1.5

Usage (notebook 1, from D:\EchoNet):
  Look at augmented clips:   python train_genaug.py --preview_aug --out D:\EchoNet\runs\genaug_seed0
  Smoke test:                python train_genaug.py --out D:\EchoNet\runs\smoke_gen --epochs 1 --limit 200 --workers 2
  Full run:                  python train_genaug.py --out D:\EchoNet\runs\genaug_seed0 --seed 0 --workers 2
--out is required (so a run can never resume into a physaug folder).
"""
import sys
import time

import cv2
import numpy as np
import torch

import train_physaug as tp

GEN_AUGS = ["noise", "gblur", "brightness", "contrast", "gamma"]
GEN_P = dict(
    NOISE_SD_FRAC=(0.0, float(np.sqrt(0.1))),
    BLUR_SIGMA=(0.5, 1.0),
    BRIGHTNESS=(0.75, 1.25),
    CONTRAST=(0.75, 1.25),
    GAMMA=(0.7, 1.5),
    reference="nnU-Net default intensity augmentation (Isensee et al., Nat Methods 2021); p per effect = --p",
)


class GenericAug:
    def __init__(self, enabled, p, seed=None):
        self.on = set(enabled)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def use(self, name):
        return name in self.on and self.rng.random() < self.p

    def __call__(self, clip):
        """clip: float (T, H, W) grey 0-255 -> augmented float (T, H, W) 0-255."""
        rng = self.rng
        x = clip.astype(np.float32)
        if self.use("noise"):
            sd = rng.uniform(*GEN_P["NOISE_SD_FRAC"]) * x.std()
            x = x + rng.standard_normal(x.shape).astype(np.float32) * sd
        if self.use("gblur"):
            s = rng.uniform(*GEN_P["BLUR_SIGMA"])
            x = np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=s, sigmaY=s) for f in x])
        if self.use("brightness"):
            x = x * rng.uniform(*GEN_P["BRIGHTNESS"])
        if self.use("contrast"):
            lo, hi, m = x.min(), x.max(), x.mean()
            x = np.clip((x - m) * rng.uniform(*GEN_P["CONTRAST"]) + m, lo, hi)
        x = np.clip(x, 0, 255)
        if self.use("gamma"):
            x = 255.0 * (x / 255.0) ** rng.uniform(*GEN_P["GAMMA"])
        return x.astype(np.float32)


class GenericTrainDataset(tp.PhysAugTrainDataset):
    def __getitem__(self, i):
        if self.aug is None:  # one generator per worker process (same seeding as physaug)
            wi = torch.utils.data.get_worker_info()
            self.aug = GenericAug(self.augs, self.p, seed=self.seed * 1000 + (wi.id if wi else 0) + int(time.time()))
        return super().__getitem__(i)


def main():
    if not any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:]):
        raise SystemExit("Please give --out, e.g. --out D:\\EchoNet\\runs\\genaug_seed0")
    # Reuse train_physaug.main unchanged, with the generic augmentations swapped in.
    tp.ALL_AUGS = GEN_AUGS
    tp.P = GEN_P                         # written to config.json
    tp.PhysAug = GenericAug              # used by --preview_aug
    tp.PhysAugTrainDataset = GenericTrainDataset
    _preview = tp.preview
    tp.preview = lambda df, augs, p, out_png, **kw: _preview(df, augs, p, out_png.replace("physaug", "genaug"), **kw)
    tp.main()


if __name__ == "__main__":
    main()
