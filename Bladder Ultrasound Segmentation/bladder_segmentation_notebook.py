# ===== Bladder segmentation in an ultrasound video (Jupyter version) =====
# pip install opencv-python numpy matplotlib
import os, csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Settings: change these for your video ----------
VIDEO_PATH    = "20251109_110615.MP4"           # path to your video
OUT_DIR       = "bladder_results"        # output folder
ROI           = (400, 225, 1360, 1040)   # x, y, w, h of the scan area (None = whole frame)
SEED          = None                     # x, y of a point inside the bladder (None = automatic)
PIXEL_SPACING = "auto"                   # mm per pixel, "auto" = read the depth ruler, None = report in pixels
RULER_X       = (1940, 1990)             # x range of the depth ruler's 1 cm tick marks (Handy Echo, 2048 px wide)
THRESHOLD     = 45                       # darkness threshold after depth compensation
MIN_AREA      = 1500                     # smallest bladder area in pixels
OPEN_SIZE, CLOSE_SIZE = 7, 21            # morphology kernel sizes
TAIL_SIZE     = None                     # cuts thin dark strips off the bladder (None = 6% of image size)
NECK_RATIO    = 0.65                     # cuts off dark areas joined to the bladder by a narrow neck (0 = off)
SMOOTHING     = 0.6                      # temporal smoothing 0-1 (higher = steadier)
BLANK_LEVEL   = 5.0                      # frames darker than this are treated as blank
SAVE_MASKS    = True
SHAPE         = "smooth"        # how to draw the bladder: "box", "rotated_box", "ellipse", "smooth" or "pixels"


def read_ruler_spacing(path, ruler_x=RULER_X, min_brightness=150):
    """Measure mm per pixel from the 1 cm tick marks of the on-screen depth ruler."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    band = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, ruler_x[0]:ruler_x[1]] > min_brightness
    rows = np.where(band.sum(axis=1) >= 0.4 * (ruler_x[1] - ruler_x[0]))[0]
    if len(rows) == 0:
        return None
    ticks = [g.mean() for g in np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)]
    gaps = np.diff(ticks)
    if len(gaps) < 3:
        return None
    step = np.median(gaps)
    regular = gaps[np.abs(gaps - step) < 0.1 * step]        # ignore the logo/markers that are not ticks
    if len(regular) < 3:
        return None
    return 10.0 / regular.mean()                             # 10 mm per tick gap


def crop(img, roi):
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 1, cv2.FILLED)
    return out


def estimate_fan_mask(path, roi, n_samples=30, min_intensity=12):
    """Find the ultrasound field of view so the black border is ignored."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_samples
    max_img = None
    for idx in np.linspace(0, total - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.medianBlur(crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), roi), 5)
        max_img = g if max_img is None else np.maximum(max_img, g)
    cap.release()
    fan = (max_img > min_intensity).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fan)
    if n > 1:
        fan = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    fan = fill_holes(fan)
    return cv2.erode(fan, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def compensate_depth(gray, fan, percentile=60, target=100.0):
    """Scale each row so tissue has similar brightness at every depth."""
    img = gray.astype(np.float32)
    inside = np.where(fan > 0, img, np.nan)
    valid = (fan > 0).sum(axis=1) > 0
    ref = np.full(gray.shape[0], np.nan, dtype=np.float32)
    ref[valid] = np.nanpercentile(inside[valid], percentile, axis=1)
    rows = np.arange(len(ref))
    ref = np.interp(rows, rows[valid], ref[valid]).astype(np.float32)
    ref = cv2.GaussianBlur(ref.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=gray.shape[0] / 40).ravel()
    return np.clip(img * (target / np.maximum(ref, 5.0))[:, None], 0, 255).astype(np.uint8)


_edge_cache = {}


def compensate_edges(img, fan, percentile=60, target=100.0, band=10):
    """Brighten the sides of the fan, which ultrasound images darker than the middle."""
    key = id(fan)
    if key not in _edge_cache:                         # distance-to-edge bands depend only on the fan: compute once
        bins = (cv2.distanceTransform(fan, cv2.DIST_L2, 5) // band).astype(np.int32)
        ys, xs = np.nonzero(fan[::4, ::4])               # a 1-in-16 pixel sample is enough for the statistics
        sample = (ys * 4, xs * 4)
        _edge_cache[key] = (bins, sample, bins[sample])
    bins, sample, sample_bins = _edge_cache[key]
    values = img[sample]
    gain = np.ones(bins.max() + 1, np.float32)
    for b in np.unique(sample_bins):
        v = values[sample_bins == b]
        if v.size > 20:
            gain[b] = target / max(np.percentile(v, percentile), 5.0)
    gain = np.minimum(gain, 3.0)                       # never brighten more than 3x
    return np.clip(img.astype(np.float32) * gain[bins], 0, 255).astype(np.uint8)


def dark_region_mask(gray, fan):
    smooth = compensate_edges(compensate_depth(cv2.medianBlur(gray, 7), fan), fan)
    smooth = cv2.bilateralFilter(smooth, 9, 50, 9)
    mask = ((smooth < THRESHOLD) & (fan > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE)))
    mask = fill_holes(mask)
    # A large opening removes thin strips (e.g. the darker edge of the scan) but keeps the round bladder body.
    tail = TAIL_SIZE or int(0.06 * min(gray.shape)) | 1
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail, tail)))


def select_component(mask, previous, seed):
    """Keep the bladder: overlaps the previous frame, else under the seed, else large and central."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    if not cands:
        return None
    if previous is not None and previous.any():
        overlap, best = max((np.logical_and(labels == i, previous).sum(), i) for i in cands)
        if overlap > 0.1 * previous.sum():
            return (labels == best).astype(np.uint8)
    if seed is not None and labels[seed[1], seed[0]] in cands:
        return (labels == labels[seed[1], seed[0]]).astype(np.uint8)
    h, w = mask.shape
    center, diag = np.array([w / 2, h / 2]), np.hypot(w, h)
    best = max(cands, key=lambda i: stats[i, cv2.CC_STAT_AREA] * (1 - np.linalg.norm(cents[i] - center) / diag) ** 2)
    return (labels == best).astype(np.uint8)


def cut_necks(mask):
    """Separate the bladder from dark areas attached to it through a narrow neck.

    The cut size scales with the bladder itself (radius of the largest circle that fits inside),
    so it works for small and large bladders alike.
    """
    if NECK_RATIO <= 0 or not mask.any():
        return mask
    radius = cv2.distanceTransform(mask, cv2.DIST_L2, 5).max()
    k = int(NECK_RATIO * radius) * 2 + 1
    if k < 5:
        return mask
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n < 2:
        return mask
    body = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # Grow the body back inside the original mask to restore its edges and tips, but not the cut-off part.
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k // 2 | 1, k // 2 | 1))
    return cv2.dilate(body, grow) & mask


def fit_shape(mask):
    """Turn the pixel mask into a simple shape (polygon points) and its size (length, width in px)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, (0.0, 0.0)
    c = max(contours, key=cv2.contourArea)
    if SHAPE == "box":                                   # upright bounding box
        x, y, w, h = cv2.boundingRect(c)
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), tuple(sorted((w, h), reverse=True))
    if SHAPE == "rotated_box":                           # tightest box, can be tilted
        rect = cv2.minAreaRect(c)
        return cv2.boxPoints(rect).astype(np.int32), tuple(sorted(rect[1], reverse=True))
    if SHAPE == "ellipse" and len(c) >= 5:               # best-fitting ellipse
        (cx, cy), (a, b), ang = cv2.fitEllipse(c)
        pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(a / 2), int(b / 2)), int(ang), 0, 360, 5)
        return pts, tuple(sorted((a, b), reverse=True))
    rect = cv2.minAreaRect(c)
    size = tuple(sorted(rect[1], reverse=True))
    if SHAPE == "smooth":                                # pixel outline with the jagged edges smoothed away
        pts = c[:, 0, :].astype(np.float32)
        k = max(len(pts) // 15, 3) | 1
        pad = np.vstack([pts[-k:], pts, pts[:k]])
        smooth = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="same")[k:-k] for i in (0, 1)], axis=1)
        return smooth.astype(np.int32), size
    return c[:, 0, :], size                              # "pixels": raw outline


def draw_overlay(frame, shape_pts, label):
    ox, oy = (ROI[0], ROI[1]) if ROI else (0, 0)
    out = frame.copy()
    if shape_pts is not None:
        pts = (shape_pts + np.array([ox, oy])).astype(np.int32)
        tint = frame.copy()
        cv2.fillPoly(tint, [pts], (0, 200, 255))
        out = cv2.addWeighted(tint, 0.25, frame, 0.75, 0)
        cv2.polylines(out, [pts], True, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(out, label, (ox + 10, oy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    return out


# ---------- Run ----------
os.makedirs(OUT_DIR, exist_ok=True)
if SAVE_MASKS:
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

if PIXEL_SPACING == "auto":
    PIXEL_SPACING = read_ruler_spacing(VIDEO_PATH)
    print("Pixel spacing from depth ruler:", f"{PIXEL_SPACING:.4f} mm/px" if PIXEL_SPACING else "not found, using pixels")

fan = estimate_fan_mask(VIDEO_PATH, ROI)
seed = (SEED[0] - ROI[0], SEED[1] - ROI[1]) if (SEED and ROI) else SEED
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
writer, previous, smoothed = None, None, None
results, previews = [], {}
preview_frames = {60, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2}

idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ROI)

    if gray[fan > 0].mean() < BLANK_LEVEL:          # blank frame -> no bladder, reset tracking
        mask, previous, smoothed = np.zeros_like(gray), None, None
    else:
        mask = select_component(dark_region_mask(gray, fan), previous, seed)
        if mask is not None:
            mask = cut_necks(mask)
        else:
            mask = previous if previous is not None else np.zeros_like(gray)

    smoothed = mask.astype(np.float32) if smoothed is None else SMOOTHING * smoothed + (1 - SMOOTHING) * mask
    final = (smoothed >= 0.5).astype(np.uint8)
    previous = final

    shape_pts, (length_px, width_px) = fit_shape(final)
    area_px = int(final.sum())
    to_cm = PIXEL_SPACING / 10 if PIXEL_SPACING else None
    row = {"frame": idx, "time_s": round(idx / fps, 3), "area_px": area_px,
           "length_px": round(length_px, 1), "width_px": round(width_px, 1)}
    if to_cm:
        row.update(area_cm2=round(area_px * to_cm ** 2, 2),
                   length_cm=round(length_px * to_cm, 2), width_cm=round(width_px * to_cm, 2))
        label = f"Bladder {row['length_cm']:.1f} x {row['width_cm']:.1f} cm, area {row['area_cm2']:.1f} cm^2"
    else:
        label = f"Bladder {length_px:.0f} x {width_px:.0f} px, area {area_px} px"
    results.append(row)

    overlay = draw_overlay(frame, shape_pts, label)
    if writer is None:
        h, w = overlay.shape[:2]
        writer = cv2.VideoWriter(os.path.join(OUT_DIR, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(overlay)

    if SAVE_MASKS:
        full = np.zeros(frame.shape[:2], np.uint8)
        x, y, w, h = ROI if ROI else (0, 0, frame.shape[1], frame.shape[0])
        full[y:y + h, x:x + w] = final * 255
        cv2.imwrite(os.path.join(OUT_DIR, "masks", f"frame_{idx:05d}.png"), full)
    if idx in preview_frames:
        previews[idx] = overlay
    idx += 1

cap.release()
writer.release()

with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"Processed {idx} frames -> {OUT_DIR}/overlay.mp4 and {OUT_DIR}/metrics.csv")

# ---------- Show results in the notebook ----------
fig, axes = plt.subplots(1, len(previews) + 1, figsize=(18, 5))
for ax, (i, img) in zip(axes, sorted(previews.items())):
    ax.imshow(cv2.cvtColor(crop(img, ROI), cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {i}")
    ax.axis("off")
t = [r["time_s"] for r in results]
a = [r.get("area_cm2", r["area_px"]) for r in results]
axes[-1].plot(t, a)
axes[-1].set_xlabel("Time (s)")
axes[-1].set_ylabel("Bladder area (cm²)" if PIXEL_SPACING else "Bladder area (px)")
axes[-1].set_title("Bladder area over time")
plt.tight_layout()
plt.show()
