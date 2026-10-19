# Bladder Segmentation in Ultrasound Video

`bladder_segmentation.py` finds the bladder in every frame of a B-mode ultrasound video and tracks it over time. It needs no training data. It relies on the fact that urine is anechoic, so the bladder shows up as a large dark region.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Basic
python bladder_segmentation.py my_scan.mp4 --out results --save-masks

# Crop away on-screen menus, click-point inside the bladder, and report cm^2
python bladder_segmentation.py my_scan.mp4 --roi 120,60,500,420 --seed 360,250 --pixel-spacing 0.3
```

Outputs in `--out`:

| File | Content |
|------|---------|
| `overlay.mp4` | Original video with the bladder outline and area drawn on it |
| `metrics.csv` | Per-frame area (px, and mm² if `--pixel-spacing` is given), equivalent diameter, centroid, width/height |
| `masks/frame_XXXXX.png` | Binary mask per frame (with `--save-masks`) |

## How it works

1. **Field of view:** detects the ultrasound fan so the black screen border is ignored.
2. **Denoising:** a median and a bilateral filter reduce speckle.
3. **Thresholding:** keeps dark pixels inside the fan (Otsu by default, or `--threshold`).
4. **Morphology:** an opening removes speckle, a closing bridges gaps, and holes are filled.
5. **Component selection:** on the first frame it takes the region under `--seed`, or else the largest region near the center. On later frames it takes the region that overlaps most with the previous frame's mask.
6. **Temporal smoothing:** a moving average over frames reduces flicker (`--smoothing`).

## Tuning tips

| Problem | Try |
|---------|-----|
| Wrong dark region picked (shadow, vessel, cyst) | `--seed x,y` on the bladder in the first frame |
| Text/menus/scale bar interfere | `--roi x,y,w,h` around the image area |
| Mask leaks into surrounding tissue | lower `--threshold` (e.g. 40) or smaller `--close-size` |
| Mask has holes / ragged edge | larger `--close-size`, or `--convex` |
| Mask flickers between frames | higher `--smoothing` (e.g. 0.8) |
| Small bladder not detected | lower `--min-area` |

## Moving to deep learning later

If you want higher accuracy on hard cases, save the masks with `--save-masks`, correct a few hundred of them (e.g. in CVAT or Label Studio), and use them as labels to train a U-Net. The image/mask pairs are already in the right format.
