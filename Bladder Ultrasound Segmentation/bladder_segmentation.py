"""
Bladder segmentation in ultrasound videos (no training data needed).

In B-mode ultrasound the urine-filled bladder is anechoic, so it appears as a
large dark region surrounded by brighter tissue. This script finds that region
in every frame and tracks it through the video:

    1. Detect the ultrasound field of view (the "fan"), so the black screen
       border is not mistaken for the bladder.
    2. Reduce speckle noise (median + bilateral filter) and even out the
       brightness loss with depth, so deep tissue does not look like fluid.
    3. Threshold dark pixels inside the fan.
    4. Clean the mask with morphology and fill holes.
    5. Pick the bladder component: the one under --seed, or the best large,
       central component in the first frame, then the component that overlaps
       most with the previous frame's mask.
    6. Smooth the mask over time to reduce flicker.

Outputs (in --out):
    overlay.mp4   video with the bladder outline and area drawn on each frame
    metrics.csv   per-frame area, centroid, width/height and diameter
    masks/        binary mask PNG per frame (with --save-masks); these can be
                  corrected and reused as labels to train a U-Net later

Example:
    python bladder_segmentation.py scan.mp4 --out results --save-masks
    python bladder_segmentation.py scan.mp4 --roi 120,60,500,420 --seed 360,250 --pixel-spacing 0.3
"""

import argparse
import csv
import os

import cv2
import numpy as np


def parse_pair(text, n):
    values = [int(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated integers, got '{text}'")
    return tuple(values)


def to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def crop(image, roi):
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y:y + h, x:x + w]


def estimate_fan_mask(video_path, roi, n_samples=30, min_intensity=12):
    """Pixels that are bright in at least one sampled frame belong to the scan area."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_samples
    indices = np.linspace(0, max(total - 1, 0), n_samples).astype(int)
    max_image = None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.medianBlur(crop(to_gray(frame), roi), 5)
        max_image = gray if max_image is None else np.maximum(max_image, gray)
    cap.release()
    if max_image is None:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    fan = (max_image > min_intensity).astype(np.uint8)
    # Keep the largest bright region (the fan) and drop text/markers around it.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fan)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fan = (labels == largest).astype(np.uint8)
    fan = fill_holes(fan)
    # Shrink slightly so the dark fan edge is not segmented.
    return cv2.erode(fan, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def compensate_depth(gray, fan, percentile=60, target=100.0):
    """Even out brightness loss with depth, so deep tissue is not mistaken for fluid.

    Each row is scaled so its typical in-fan brightness becomes `target`. The
    bladder is darker than the surrounding tissue at the same depth, so it stays dark.
    """
    image = gray.astype(np.float32)
    inside = np.where(fan > 0, image, np.nan)
    valid = (fan > 0).sum(axis=1) > 0
    reference = np.full(gray.shape[0], np.nan, dtype=np.float32)
    reference[valid] = np.nanpercentile(inside[valid], percentile, axis=1)
    rows = np.arange(len(reference))
    reference = np.interp(rows, rows[valid], reference[valid]) if valid.any() else np.full_like(reference, target)
    reference = cv2.GaussianBlur(reference.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=gray.shape[0] / 40).ravel()
    scaled = image * (target / np.maximum(reference, 5.0))[:, None]
    return np.clip(scaled, 0, 255).astype(np.uint8)


_edge_cache = {}


def compensate_edges(img, fan, percentile=60, target=100.0, band=10):
    """Brighten the sides of the fan, which ultrasound images darker than the middle."""
    key = id(fan)
    if key not in _edge_cache:  # distance-to-edge bands depend only on the fan: compute once
        bins = (cv2.distanceTransform(fan, cv2.DIST_L2, 5) // band).astype(np.int32)
        ys, xs = np.nonzero(fan[::4, ::4])  # a 1-in-16 pixel sample is enough for the statistics
        sample = (ys * 4, xs * 4)
        _edge_cache[key] = (bins, sample, bins[sample])
    bins, sample, sample_bins = _edge_cache[key]
    values = img[sample]
    gain = np.ones(bins.max() + 1, np.float32)
    for b in np.unique(sample_bins):
        v = values[sample_bins == b]
        if v.size > 20:
            gain[b] = target / max(np.percentile(v, percentile), 5.0)
    gain = np.minimum(gain, 3.0)  # never brighten more than 3x
    return np.clip(img.astype(np.float32) * gain[bins], 0, 255).astype(np.uint8)


def dark_region_mask(gray, fan, threshold, open_size, close_size, depth_compensation, tail_size=None):
    smooth = cv2.medianBlur(gray, 7)
    if depth_compensation:
        smooth = compensate_edges(compensate_depth(smooth, fan), fan)
    smooth = cv2.bilateralFilter(smooth, 9, 50, 9)

    if threshold is None and depth_compensation:
        # After compensation typical tissue sits at ~100; fluid is well below half of that.
        threshold = 45
    elif threshold is None:
        inside = smooth[fan > 0]
        threshold, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((smooth < threshold) & (fan > 0)).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)))
    mask = fill_holes(mask)
    # A large opening removes thin strips (e.g. the darker edge of the scan) but keeps the round bladder body.
    tail = tail_size or int(0.06 * min(gray.shape)) | 1
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail, tail)))


def select_component(mask, previous, seed, min_area):
    """Return a mask with only the bladder component, or None if nothing fits."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        return None

    if previous is not None and previous.any():
        overlaps = [(np.logical_and(labels == i, previous).sum(), i) for i in candidates]
        best_overlap, best = max(overlaps)
        if best_overlap > 0.1 * previous.sum():
            return (labels == best).astype(np.uint8)

    if seed is not None:
        label = labels[seed[1], seed[0]]
        if label in candidates:
            return (labels == label).astype(np.uint8)

    # No tracking info: prefer large components close to the image center.
    h, w = mask.shape
    center = np.array([w / 2, h / 2])
    diag = np.hypot(w, h)

    def score(i):
        distance = np.linalg.norm(centroids[i] - center) / diag
        return stats[i, cv2.CC_STAT_AREA] * (1.0 - distance) ** 2

    return (labels == max(candidates, key=score)).astype(np.uint8)


def measure(mask, pixel_spacing):
    area_px = int(mask.sum())
    row = {"area_px": area_px, "centroid_x": "", "centroid_y": "", "width_px": "", "height_px": "",
           "equiv_diameter_px": "", "area_mm2": "", "equiv_diameter_mm": ""}
    if area_px == 0:
        return row
    m = cv2.moments(mask, binaryImage=True)
    x, y, w, h = cv2.boundingRect(mask)
    diameter = np.sqrt(4 * area_px / np.pi)
    row.update(centroid_x=round(m["m10"] / m["m00"], 1), centroid_y=round(m["m01"] / m["m00"], 1),
               width_px=w, height_px=h, equiv_diameter_px=round(diameter, 1))
    if pixel_spacing:
        row.update(area_mm2=round(area_px * pixel_spacing ** 2, 1),
                   equiv_diameter_mm=round(diameter * pixel_spacing, 1))
    return row


def draw_overlay(frame, mask, roi, row, pixel_spacing):
    overlay = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c + np.array([ox, oy]) for c in contours]

    tint = overlay.copy()
    cv2.drawContours(tint, contours, -1, (0, 200, 255), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(tint, 0.25, overlay, 0.75, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    if pixel_spacing and row["area_px"]:
        label = f"Bladder area: {row['area_mm2'] / 100:.2f} cm^2"
    else:
        label = f"Bladder area: {row['area_px']} px"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


def segment_video(args):
    os.makedirs(args.out, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    fan = estimate_fan_mask(args.video, args.roi)
    seed = None
    if args.seed:
        seed = (args.seed[0] - args.roi[0], args.seed[1] - args.roi[1]) if args.roi else args.seed

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    previous = None
    smoothed = None

    csv_path = os.path.join(args.out, "metrics.csv")
    fields = ["frame", "time_s", "area_px", "area_mm2", "equiv_diameter_px", "equiv_diameter_mm",
              "centroid_x", "centroid_y", "width_px", "height_px"]
    with open(csv_path, "w", newline="") as f:
        table = csv.DictWriter(f, fieldnames=fields)
        table.writeheader()

        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = crop(to_gray(frame), args.roi)

            if gray[fan > 0].mean() < args.blank_level:
                # Blank frame (e.g. before imaging starts): no bladder, restart tracking.
                mask = np.zeros_like(gray, dtype=np.uint8)
                previous = smoothed = None
            else:
                candidates = dark_region_mask(gray, fan, args.threshold, args.open_size, args.close_size,
                                              not args.no_depth_compensation, args.tail_size)
                mask = select_component(candidates, previous, seed, args.min_area)
            if mask is None:
                # Nothing found (e.g. probe lifted): keep the last mask rather than jumping.
                mask = previous if previous is not None else np.zeros_like(gray, dtype=np.uint8)
            if args.convex and mask.any():
                hull = cv2.convexHull(cv2.findNonZero(mask))
                mask = np.zeros_like(mask)
                cv2.fillPoly(mask, [hull], 1)

            # Exponential moving average over frames removes flicker.
            smoothed = mask.astype(np.float32) if smoothed is None else \
                args.smoothing * smoothed + (1 - args.smoothing) * mask
            final = (smoothed >= 0.5).astype(np.uint8)
            previous = final

            row = measure(final, args.pixel_spacing)
            table.writerow({"frame": index, "time_s": round(index / fps, 3), **row})

            if args.save_masks:
                full = np.zeros(frame.shape[:2], dtype=np.uint8)
                if args.roi:
                    x, y, w, h = args.roi
                    full[y:y + h, x:x + w] = final * 255
                else:
                    full = final * 255
                cv2.imwrite(os.path.join(args.out, "masks", f"frame_{index:05d}.png"), full)

            overlay = draw_overlay(frame, final, args.roi, row, args.pixel_spacing)
            if writer is None:
                h, w = overlay.shape[:2]
                writer = cv2.VideoWriter(os.path.join(args.out, "overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(overlay)
            if args.show:
                cv2.imshow("Bladder segmentation (q to quit)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Processed {index} frames -> {args.out}/overlay.mp4, {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment the bladder in an ultrasound video.")
    parser.add_argument("video", help="input video file (mp4, avi, ...)")
    parser.add_argument("--out", default="bladder_results", help="output folder")
    parser.add_argument("--roi", type=lambda s: parse_pair(s, 4),
                        help="x,y,w,h crop of the scan area (skip on-screen text/menus)")
    parser.add_argument("--seed", type=lambda s: parse_pair(s, 2),
                        help="x,y pixel inside the bladder on the first frame (full-frame coordinates)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="darkness threshold 0-255 (default: 45 with depth compensation, else automatic Otsu)")
    parser.add_argument("--no-depth-compensation", action="store_true",
                        help="turn off the depth and fan-edge brightness normalisation")
    parser.add_argument("--blank-level", type=float, default=5.0,
                        help="frames whose mean scan brightness is below this are treated as blank")
    parser.add_argument("--min-area", type=int, default=1500, help="minimum bladder area in pixels")
    parser.add_argument("--open-size", type=int, default=7, help="opening kernel size (removes speckle)")
    parser.add_argument("--tail-size", type=int, default=None,
                        help="opening size that cuts thin dark strips off the bladder (default: 6%% of image size)")
    parser.add_argument("--close-size", type=int, default=21, help="closing kernel size (bridges gaps)")
    parser.add_argument("--smoothing", type=float, default=0.6,
                        help="temporal smoothing 0-1 (0 = off, higher = steadier but slower to react)")
    parser.add_argument("--convex", action="store_true", help="replace the mask with its convex hull")
    parser.add_argument("--pixel-spacing", type=float, default=None,
                        help="mm per pixel, to report area in mm^2 / cm^2")
    parser.add_argument("--save-masks", action="store_true", help="write one mask PNG per frame")
    parser.add_argument("--show", action="store_true", help="show the result live (needs a display)")
    segment_video(parser.parse_args())


if __name__ == "__main__":
    main()
