r"""
EchoNet-Dynamic EF training with physics-informed ultrasound augmentation (JSPS Paper 3).
Same model, schedule and evaluation as echonet_baseline.py; only the training clips are augmented.

Each training clip is mapped from displayed grey levels back to an approximate echo amplitude
(assuming EchoNet was displayed with DR_SRC dB of dynamic range), the selected physics effects are
applied in the amplitude domain, and the clip is re-displayed with a (possibly new) dynamic range
and gain, keeping the clip's bright end (99th percentile) at its original grey level. Every effect is applied independently with probability --p, and parameters are drawn
once per clip (one "scanner setting" per clip). Ranges were fixed a priori from typical cardiac
scanner settings, NOT fitted to CAMUS.

Effects (--aug, comma-separated, or "all" / "none"):
  dr       display dynamic range           50-80 dB (source assumed 60 dB)
  gain     overall gain                    -6 to +6 dB
  tgc      depth gain curve (TGC)          4 control points over depth, each -6 to +6 dB
  near     near-field brightening          0 to +12 dB, decaying with depth (scale 8% of depth)
  blur     point-spread function           lateral sigma 0-1.2 px, axial 0-0.6 px (at 112 px)
  speckle  speckle re-synthesis            Rayleigh speckle, mixing weight 0-0.5
  noise    thermal noise / clutter floor   -70 to -45 dB below peak, added incoherently
  half     half-heartbeat clip             ED->ES only, mirrored (as CAMUS); needs VolumeTracings

Usage (notebook, from D:\EchoNet):
  Look at augmented clips first:
    python train_physaug.py --preview_aug --aug all
  Smoke test:
    python train_physaug.py --aug all --out D:\EchoNet\runs\smoke_phys --epochs 1 --limit 200
  Full run:
    python train_physaug.py --aug all --out D:\EchoNet\runs\physaug_all_seed0 --seed 0
  Ablation, e.g. without speckle:
    python train_physaug.py --aug dr,gain,tgc,near,blur,noise,half --out D:\EchoNet\runs\physaug_nospeckle_seed0
Imbalance-aware (label-distribution smoothing, Yang et al. ICML 2021), baseline or physics-augmented:
    python train_physaug.py --aug none --lds --out D:\EchoNet\runs\lds_baseline_seed0 --seed 0
    python train_physaug.py --aug all  --lds --out D:\EchoNet\runs\lds_physaug_seed0 --seed 0
  (--aug none trains exactly like echonet_baseline.py, with the weighted loss.)
Then test on CAMUS:
    python camus_eval.py --ckpt D:\EchoNet\runs\physaug_all_seed0\best.pt --out D:\EchoNet\runs\physaug_all_seed0\camus
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from echonet_baseline import (EchoDataset, build_model, compute_mean_std, evaluate, load_video,
                              metrics, read_filelist)
from camus_eval import make_cycle, resample_time
from halfcycle_diag import ed_es_frames

# Power saving: one CPU thread per process. Without this, OpenCV/NumPy/PyTorch in every data-loader
# worker each start one thread per CPU core, which drives the whole CPU to full load (and power).
cv2.setNumThreads(1)


def _limit_threads(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

ALL_AUGS = ["dr", "gain", "tgc", "near", "blur", "speckle", "noise", "half"]

# Parameter ranges, fixed a priori from the literature (not fitted to CAMUS). Report these in the paper.
# Display dynamic range: B-mode is typically log-compressed to a 40-60 dB display range
#   (Harput et al., IEEE IUS 2014); scanner compression maps of 50-85 dB with 60 dB as a common
#   default are described in US patent 5,993,392. ASE guidance: dynamic range is lab/patient adjusted.
# Speckle: fully developed speckle has Rayleigh amplitude statistics (Burckhardt 1978; Wagner et al. 1983).
# PSF: phased-array cardiac probes (~1.5-4 MHz) have coarser lateral than axial resolution.
P = dict(
    DR_SRC=60.0,                  # assumed display dynamic range of EchoNet (dB): common default
    DR_RANGE=(50.0, 80.0),        # new display dynamic range (dB), within reported 40-85 dB settings
    GAIN_RANGE=(-6.0, 6.0),       # overall gain (dB)
    TGC_POINTS=4, TGC_RANGE=(-6.0, 6.0),
    NEAR_RANGE=(0.0, 12.0), NEAR_SCALE=0.08,
    BLUR_LAT=(0.0, 1.2), BLUR_AX=(0.0, 0.6),
    SPECKLE_ALPHA=(0.0, 0.5), SPECKLE_GRAIN=0.8, SPECKLE_FIXED=0.8,
    NOISE_DB=(-70.0, -45.0),
)


# ---------------------------------------------------------------- physics augmentation
def _speckle_field(rng, shape, grain):
    """Fully developed (Rayleigh) speckle amplitude field with grain size `grain` px, mean 1."""
    re = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), grain)
    im = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), grain)
    return re, im


def scan_sector(clip, thr=10):
    """Smooth mask of the ultrasound fan: largest bright component, holes filled, edges cleaned,
    so augmentation does not follow the noisy, compressed edge pixels of the original."""
    m = (clip.max(0) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == keep).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(m)
    cv2.drawContours(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, -1)
    hull = cv2.erode(hull, np.ones((3, 3), np.uint8))
    return hull.astype(bool)


class PhysAug:
    def __init__(self, enabled, p, seed=None):
        self.on = set(enabled)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def use(self, name):
        return name in self.on and self.rng.random() < self.p

    def __call__(self, clip):
        """clip: float (T, H, W) grey 0-255 -> augmented float (T, H, W) 0-255."""
        rng = self.rng
        T, H, W = clip.shape
        sector = scan_sector(clip)
        if not sector.any():
            return clip
        dr_src = P["DR_SRC"]
        level0 = clip / 255.0 * dr_src - dr_src                         # dB, 0 = display peak
        ref = np.percentile(level0[:, sector], 99)                      # bright-end anchor (pre-augmentation)
        amp = 10 ** (level0 / 20.0)                                      # linear amplitude
        z = np.linspace(0, 1, H, dtype=np.float32)[None, :, None]      # depth, 0 = apex (top)

        if self.use("tgc"):
            ctrl = rng.uniform(*P["TGC_RANGE"], size=P["TGC_POINTS"])
            g = np.interp(z.ravel(), np.linspace(0, 1, P["TGC_POINTS"]), ctrl)[None, :, None]
            amp = amp * 10 ** (g / 20.0)
        if self.use("near"):
            n = rng.uniform(*P["NEAR_RANGE"])
            amp = amp * 10 ** (n * np.exp(-z / P["NEAR_SCALE"]) / 20.0)
        if self.use("speckle"):
            a = rng.uniform(*P["SPECKLE_ALPHA"])
            fr, fi = _speckle_field(rng, (H, W), P["SPECKLE_GRAIN"])
            w = np.sqrt(P["SPECKLE_FIXED"])
            s = np.empty((T, H, W), np.float32)
            for t in range(T):                                       # mostly fixed pattern + per-frame decorrelation
                tr, ti = _speckle_field(rng, (H, W), P["SPECKLE_GRAIN"])
                s[t] = np.hypot(w * fr + np.sqrt(1 - w ** 2) * tr, w * fi + np.sqrt(1 - w ** 2) * ti)
            s /= s[:, sector].mean()
            amp = amp * ((1 - a) + a * s)
        if self.use("noise"):
            level = 10 ** (rng.uniform(*P["NOISE_DB"]) / 20.0)
            nr = rng.standard_normal((T, H, W)).astype(np.float32)
            ni = rng.standard_normal((T, H, W)).astype(np.float32)
            noise = level * np.hypot(nr, ni) / np.sqrt(2)
            amp = np.sqrt(amp ** 2 + noise ** 2)                       # incoherent sum
        if self.use("blur"):
            sl, sa = rng.uniform(*P["BLUR_LAT"]), rng.uniform(*P["BLUR_AX"])
            if sl > 0.05 or sa > 0.05:
                amp = np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=max(sl, 1e-3), sigmaY=max(sa, 1e-3))
                                for f in amp.astype(np.float32)])

        level = 20 * np.log10(amp + 1e-12)
        gain = rng.uniform(*P["GAIN_RANGE"]) if self.use("gain") else 0.0
        dr = rng.uniform(*P["DR_RANGE"]) if self.use("dr") else dr_src
        # Re-display: the clip's own 99th-percentile level keeps its grey value (bright end anchored,
        # as when only the dynamic range setting changes); a wider DR lifts the dark floor,
        # a narrower one deepens it. With dr = DR_SRC and no other effect this is the identity.
        grey_ref = 255.0 * (ref + dr_src) / dr_src
        out = np.clip(grey_ref + 255.0 * (level - ref + gain) / dr, 0, 255)
        out = np.where(sector[None], out, clip)                       # outside the fan: unchanged
        return out.astype(np.float32)


# ---------------------------------------------------------------- imbalance-aware loss (LDS)
def lds_weights(ef_train, sigma=2.0, power=0.5, max_w=10.0):
    """Label-distribution smoothing (Yang et al., ICML 2021, 'Delving into Deep Imbalanced Regression').
    Histogram of training EF in 1-point bins, smoothed with a Gaussian kernel (sigma, in EF points);
    weight = (smoothed density)^-power (power 0.5 = square-root inverse, 1.0 = inverse), normalized so the
    average training weight is 1, capped at max_w. Returns a lookup table indexed by round(EF)."""
    bins = np.arange(0, 101)
    hist = np.bincount(np.clip(np.round(ef_train).astype(int), 0, 100), minlength=101).astype(np.float64)
    k = np.exp(-0.5 * ((np.arange(-int(4 * sigma), int(4 * sigma) + 1)) / sigma) ** 2)
    dens = np.convolve(hist, k / k.sum(), mode="same")
    w = np.where(dens > 0, dens, dens[dens > 0].min()) ** (-power)
    idx = np.clip(np.round(ef_train).astype(int), 0, 100)
    w = w / w[idx].mean()
    return np.minimum(w, max_w).astype(np.float32), bins


# ---------------------------------------------------------------- dataset
class PhysAugTrainDataset(Dataset):
    def __init__(self, df, mean, std, augs, p, edes, frames=32, period=2, pad=12, seed=0):
        self.paths = df.path.tolist()
        self.names = df.FileName.tolist()
        self.fps = df.FPS.values.astype(np.float32) if "FPS" in df else np.full(len(df), 50.0, np.float32)
        self.ef = df.EF.values.astype(np.float32)
        self.mean, self.std = mean[:, None, None, None], std[:, None, None, None]
        self.frames, self.period, self.pad = frames, period, pad
        self.augs, self.p, self.edes, self.seed = set(augs), p, edes, seed
        self.aug = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.aug is None:  # one generator per worker process
            wi = torch.utils.data.get_worker_info()
            self.aug = PhysAug(self.augs, self.p, seed=self.seed * 1000 + (wi.id if wi else 0) + int(time.time()))
        rng = self.aug.rng
        v = load_video(self.paths[i])[0]                                # (T, H, W) grey 0-255
        need = self.frames * self.period
        key = self.names[i] if self.names[i].endswith(".avi") else self.names[i] + ".avi"
        if "half" in self.augs and key in self.edes and rng.random() < self.p:
            ed, es = self.edes[key]
            lo, hi = sorted((min(ed, len(v) - 1), min(es, len(v) - 1)))
            if hi - lo >= 2:
                seg = v[lo:hi + 1]
                if ed > es:
                    seg = seg[::-1]
                v = make_cycle(resample_time(seg, float(self.fps[i])), need, "mirror")
        if len(v) < need:
            v = np.concatenate([v, np.zeros((need - len(v), *v.shape[1:]), np.float32)])
        start = rng.integers(0, len(v) - need + 1)
        clip = np.ascontiguousarray(v[start + self.period * np.arange(self.frames)])
        if self.augs - {"half"}:
            clip = self.aug(clip)
        clip = (np.repeat(clip[None], 3, 0) - self.mean) / self.std     # (3, T, H, W)
        if self.pad:
            c, t, h, w = clip.shape
            p = np.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), np.float32)
            p[:, :, self.pad:self.pad + h, self.pad:self.pad + w] = clip
            y, x = rng.integers(0, 2 * self.pad + 1, size=2)
            clip = p[:, :, y:y + h, x:x + w]
        return torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)), self.ef[i]


def load_edes(root):
    vt = pd.read_csv(root / "VolumeTracings.csv")
    out = {}
    for name, g in vt.groupby("FileName", sort=False):
        r = ed_es_frames(g)
        if r is not None:
            out[name] = r
    return out


def preview(df, augs, p, out_png, n=4, seed=0):
    """Grid: each row = one video; first column original, then 5 random augmentations (first frame)."""
    rng = np.random.default_rng(seed)
    aug = PhysAug(augs, p, seed=seed)
    rows = []
    for path in df.path.sample(n, random_state=seed):
        v = load_video(path)[0][:8]
        tiles = [v[0]] + [aug(v.copy())[0] for _ in range(5)]
        rows.append(np.concatenate(tiles, 1))
    img = np.concatenate(rows, 0).astype(np.uint8)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, img)
    print(f"Wrote {out_png}: each row one video; column 1 original, columns 2-6 augmented.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\physaug_all_seed0")
    ap.add_argument("--aug", default="all", help="comma list of " + ",".join(ALL_AUGS) + ", or all / none")
    ap.add_argument("--p", type=float, default=0.5, help="probability of applying each effect per clip")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4, help="data-loader processes (fewer = less CPU power)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val_stride", type=int, default=32)
    ap.add_argument("--test_stride", type=int, default=4)
    ap.add_argument("--lds", action="store_true",
                    help="imbalance-aware training: label-distribution-smoothed loss re-weighting")
    ap.add_argument("--lds_sigma", type=float, default=2.0, help="LDS kernel width in EF points")
    ap.add_argument("--lds_power", type=float, default=0.5, help="0.5 = sqrt-inverse density, 1 = inverse")
    ap.add_argument("--preview_aug", action="store_true")
    a = ap.parse_args()

    augs = ALL_AUGS if a.aug == "all" else [] if a.aug == "none" else a.aug.split(",")
    bad = set(augs) - set(ALL_AUGS)
    if bad:
        raise SystemExit(f"Unknown augmentation(s): {bad}. Choose from {ALL_AUGS}")

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    root = Path(a.data)
    df = read_filelist(root, a.limit)
    tr, va, te = (df[df.Split == s] for s in ("TRAIN", "VAL", "TEST"))

    if a.preview_aug:
        preview(tr, augs, 1.0 if a.p < 1 else a.p, f"physaug_preview_{a.aug.replace(',', '-')}.png", seed=a.seed)
        return

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"args": vars(a), "augs": augs, "params": P}, indent=2))
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | augs {augs} | p {a.p}")

    ms_file = root.parent / "mean_std.json"
    if ms_file.exists():
        d = json.loads(ms_file.read_text()); mean, std = np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
    else:
        mean, std = compute_mean_std(tr.path, seed=a.seed)
        ms_file.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))

    edes = load_edes(root) if "half" in augs else {}
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
              worker_init_fn=_limit_threads)
    dl_tr = DataLoader(PhysAugTrainDataset(tr, mean, std, augs, a.p, edes, seed=a.seed),
                       batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(EchoDataset(va, mean, std, test_stride=a.val_stride), batch_size=1, **kw)

    model = build_model(True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    w_table = None
    if a.lds:
        w_np, _ = lds_weights(tr.EF.values, a.lds_sigma, a.lds_power)
        w_table = torch.from_numpy(w_np).to(device)
        for lo_, hi_ in ((0, 40), (40, 50), (50, 101)):
            sel = (tr.EF.values >= lo_) & (tr.EF.values < hi_)
            print(f"LDS mean weight for EF {lo_}-{hi_ - 1}: {w_np[np.round(tr.EF.values[sel]).astype(int)].mean():.2f}")

    start_ep, best = 0, float("inf")
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"Resumed from epoch {start_ep}")
    log = out / "log.csv"
    if not log.exists():
        log.write_text("epoch,train_loss,val_MAE,val_RMSE,val_R2,minutes\n")

    for ep in range(start_ep, a.epochs):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                err2 = (model(x).float().view(-1) - y) ** 2
                if w_table is not None:
                    err2 = err2 * w_table[y.round().long().clamp(0, 100)]
                loss = err2.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
            if b % 100 == 0:
                print(f"  epoch {ep} batch {b}/{len(dl_tr)} loss {loss.item():.1f}", flush=True)
        sched.step()
        ys, ps, _ = evaluate(model, dl_va, device)
        m = metrics(ys, ps)
        mins = (time.time() - t0) / 60
        print(f"Epoch {ep}: train MSE {tot / n:.2f} | val MAE {m['MAE']:.2f} RMSE {m['RMSE']:.2f} R2 {m['R2']:.3f} | {mins:.1f} min")
        with open(log, "a") as f:
            f.write(f"{ep},{tot / n:.4f},{m['MAE']:.4f},{m['RMSE']:.4f},{m['R2']:.4f},{mins:.1f}\n")
        if m["MAE"] < best:
            best = m["MAE"]
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, out / "last.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    results = {}
    for name, part, stride in (("val", va, a.val_stride), ("test", te, a.test_stride)):
        dl = DataLoader(EchoDataset(part, mean, std, test_stride=stride), batch_size=1, **kw)
        ys, ps, idx = evaluate(model, dl, device)
        results[name] = metrics(ys, ps)
        pd.DataFrame({"FileName": part.FileName.values[idx], "EF_true": ys, "EF_pred": ps}).to_csv(
            out / f"{name}_predictions.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("FINAL", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
