r"""
Quantify the image-physics gap between EchoNet-Dynamic (training domain) and CAMUS (external domain).
CPU only: safe to run while the GPU trains.

Usage (from D:\EchoNet, .venv active):
    python physics_gap.py                       # 500 EchoNet training videos vs all 500 CAMUS patients
    python physics_gap.py --n_echonet 100 --limit_camus 100   # quick test

Both datasets are first brought to the same 112x112 format (CAMUS through the same
harmonization as camus_eval.py, no zoom). All measures use the end-diastolic (ED) frame and
two regions derived the same way in both datasets:
  - blood pool  = expert LV cavity mask (EchoNet: VolumeTracings; CAMUS: label 1 of the ED mask)
  - myocardium  = a ring just outside the cavity (dilation band), restricted to the scan sector

Measures per video:
  intensity      mean, std, 5th/50th/95th percentile inside the sector
  contrast       myocardium-blood contrast-to-noise ratio (CNR)
  speckle        Nakagami m and Omega, and speckle SNR (mean/std) in the myocardium, computed on
                 an approximately decompressed envelope (log compression undone assuming --dr dB)
  resolution     axial (vertical) and lateral (horizontal) speckle autocorrelation FWHM, pixels
  sharpness      mean gradient magnitude inside the sector
  depth profile  mean intensity in 10 depth bins (gain / attenuation / TGC)

Outputs (in --out):
  physics_gap_per_video.csv   one row per video
  physics_gap_summary.csv     per measure: EchoNet mean+-sd, CAMUS mean+-sd, Cohen's d, Wasserstein-1
  physics_gap.png             histograms + depth profiles
"""
import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from echonet_baseline import load_video

N_DEPTH = 10


# ---------------------------------------------------------------- loading + harmonization
def echonet_ed(root, fname, tracings):
    """Return ED frame (112x112 float) and its LV cavity mask from VolumeTracings."""
    key = fname if fname.endswith(".avi") else fname + ".avi"
    t = tracings.get(key)
    if t is None:
        return None
    best = None
    for frame, g in t.groupby("Frame"):
        g = g[["X1", "Y1", "X2", "Y2"]].values  # first row = long axis
        if len(g) < 3:
            continue
        x = np.concatenate([g[1:, 0], g[1:, 2][::-1]])
        y = np.concatenate([g[1:, 1], g[1:, 3][::-1]])
        m = np.zeros((112, 112), np.uint8)
        cv2.fillPoly(m, [np.round(np.stack([x, y], 1)).astype(np.int32)], 1)
        if best is None or m.sum() > best[1].sum():
            best = (int(frame), m.astype(bool))
    if best is None:
        return None
    v = load_video(root / "Videos" / key)[0]          # (T, H, W), grey
    fr = min(best[0], v.shape[0] - 1)
    return v[fr], best[1]


def load_nifti_hw(path):
    img = nib.load(str(path))
    x = np.asarray(img.dataobj)
    if x.ndim == 2:
        x = x[..., None]
    sx, sy = (float(s) for s in img.header.get_zooms()[:2])
    return x.transpose(1, 0, 2), sy, sx               # (H, W, T), row spacing, col spacing


def camus_ed(pdir):
    """CAMUS ED frame + LV cavity mask, harmonized exactly like camus_eval.to_echonet_frames (zoom 1)."""
    pid = pdir.name
    seq, rs, cs = load_nifti_hw(pdir / f"{pid}_4CH_half_sequence.nii.gz")
    gt, _, _ = load_nifti_hw(pdir / f"{pid}_4CH_ED_gt.nii.gz")
    s = min(rs, cs)
    h, w = seq.shape[:2]
    nh, nw = int(round(h * rs / s)), int(round(w * cs / s))
    frames = np.stack([cv2.resize(seq[..., t].astype(np.float32), (nw, nh)) for t in range(seq.shape[2])])
    mask = cv2.resize((gt[..., 0] == 1).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    sector = frames.max(0) > 0
    ys, xs = np.where(sector)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    img, mask = frames[0][sl], mask[sl]
    hh, ww = img.shape
    m = max(hh, ww)
    y0, x0 = (m - hh) // 2, (m - ww) // 2
    I = np.zeros((m, m), np.float32); I[y0:y0 + hh, x0:x0 + ww] = img
    M = np.zeros((m, m), np.uint8); M[y0:y0 + hh, x0:x0 + ww] = mask
    I = np.clip(cv2.resize(I, (112, 112), interpolation=cv2.INTER_AREA), 0, 255)
    M = cv2.resize(M, (112, 112), interpolation=cv2.INTER_NEAREST).astype(bool)
    return I, M


# ---------------------------------------------------------------- measures
def sector_mask(img, thr=3):
    m = (img > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def ring(mask, sector, inner=1, outer=5):
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8)
    band = cv2.dilate(m8, k(outer)).astype(bool) & ~cv2.dilate(m8, k(inner)).astype(bool)
    return band & sector


def nakagami(env):
    e2 = env.astype(np.float64) ** 2
    omega = e2.mean()
    var = e2.var()
    return (omega ** 2 / var if var > 0 else np.nan), omega


def fwhm_autocorr(img, region):
    """Axial (rows) and lateral (cols) FWHM of the normalized speckle autocorrelation in a region."""
    ys, xs = np.where(region)
    if len(ys) < 50:
        return np.nan, np.nan
    patch = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float64)
    pm = region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    patch = np.where(pm, patch - patch[pm].mean(), 0.0)
    F = np.fft.fft2(patch, s=(2 * patch.shape[0], 2 * patch.shape[1]))
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    ac /= ac.max()
    cy, cx = np.unravel_index(np.argmax(ac), ac.shape)

    def width(profile, c):
        """Full width at half maximum with linear interpolation between samples (sub-pixel)."""
        l = c
        while l > 0 and profile[l] > 0.5:
            l -= 1
        r = c
        while r < len(profile) - 1 and profile[r] > 0.5:
            r += 1
        xl = l + (0.5 - profile[l]) / (profile[l + 1] - profile[l]) if profile[l + 1] != profile[l] else l
        xr = r - 1 + (profile[r - 1] - 0.5) / (profile[r - 1] - profile[r]) if profile[r - 1] != profile[r] else r
        return float(xr - xl)
    return width(ac[:, cx], cy), width(ac[cy, :], cx)


def measure(img, cavity, dr):
    sec = sector_mask(img)
    cavity = cavity & sec
    myo = ring(cavity, sec)
    v = img[sec]
    r = dict(mean=v.mean(), std=v.std(),
             p05=np.percentile(v, 5), p50=np.percentile(v, 50), p95=np.percentile(v, 95))
    mb, mm = img[cavity], img[myo]
    r["blood_mean"], r["myo_mean"] = mb.mean(), mm.mean()
    r["CNR"] = (mm.mean() - mb.mean()) / np.sqrt(mm.var() + mb.var() + 1e-6)
    env = 10 ** ((img / 255.0) * dr / 20.0)          # approximate inverse log compression
    r["nakagami_m_myo"], r["nakagami_omega_myo"] = nakagami(env[myo])
    r["nakagami_m_blood"], _ = nakagami(env[cavity])
    e = env[myo]
    r["speckle_SNR_myo"] = e.mean() / (e.std() + 1e-9)
    r["fwhm_axial"], r["fwhm_lateral"] = fwhm_autocorr(img, myo)
    gy, gx = np.gradient(img.astype(np.float64))
    r["sharpness"] = np.hypot(gy, gx)[sec].mean()
    rows = np.where(sec.any(1))[0]
    top, bot = rows.min(), rows.max() + 1
    edges = np.linspace(top, bot, N_DEPTH + 1).astype(int)
    for i in range(N_DEPTH):
        band = sec.copy()
        band[:edges[i]] = False
        band[edges[i + 1]:] = False
        r[f"depth_{i}"] = img[band].mean() if band.any() else np.nan
    return r


# ---------------------------------------------------------------- comparison
def wasserstein(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--echonet", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--camus", default=r"D:\CAMUS\database_nifti")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\physics_gap")
    ap.add_argument("--n_echonet", type=int, default=500, help="random TRAIN videos")
    ap.add_argument("--limit_camus", type=int, default=None)
    ap.add_argument("--dr", type=float, default=60.0, help="assumed dynamic range (dB) for decompression")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    eroot = Path(a.echonet)
    fl = pd.read_csv(eroot / "FileList.csv")
    fl = fl[(fl.Split == "TRAIN") & (fl.FrameHeight == 112) & (fl.FrameWidth == 112)]
    fl = fl.sample(min(a.n_echonet, len(fl)), random_state=a.seed)
    vt = pd.read_csv(eroot / "VolumeTracings.csv")
    vt = dict(tuple(vt.groupby("FileName", sort=False)))
    print(f"EchoNet: measuring {len(fl)} training videos ...")
    for i, f in enumerate(fl.FileName.astype(str)):
        res = echonet_ed(eroot, f, vt)
        if res is None:
            continue
        rows.append(dict(dataset="EchoNet", id=f, **measure(*res, a.dr)))
        if i % 100 == 0:
            print(f"  {i}/{len(fl)}", flush=True)

    pdirs = sorted(p for p in Path(a.camus).iterdir() if p.is_dir() and p.name.startswith("patient"))
    if a.limit_camus:
        pdirs = pdirs[:a.limit_camus]
    print(f"CAMUS: measuring {len(pdirs)} patients ...")
    for i, p in enumerate(pdirs):
        q = "NA"
        cfg = p / "Info_4CH.cfg"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith("ImageQuality"):
                    q = line.split(":", 1)[1].strip()
        rows.append(dict(dataset="CAMUS", id=p.name, quality=q, **measure(*camus_ed(p), a.dr)))
        if i % 100 == 0:
            print(f"  {i}/{len(pdirs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "physics_gap_per_video.csv", index=False)

    measures = [c for c in df.columns if c not in ("dataset", "id", "quality")]
    summ = []
    for c in measures:
        e = df.loc[df.dataset == "EchoNet", c].dropna().values
        k = df.loc[df.dataset == "CAMUS", c].dropna().values
        if len(e) < 2 or len(k) < 2:
            continue
        sd = np.sqrt((e.var(ddof=1) + k.var(ddof=1)) / 2)
        summ.append(dict(measure=c, EchoNet_mean=e.mean(), EchoNet_sd=e.std(ddof=1),
                         CAMUS_mean=k.mean(), CAMUS_sd=k.std(ddof=1),
                         cohens_d=(k.mean() - e.mean()) / sd if sd > 0 else np.nan,
                         wasserstein1=wasserstein(e, k)))
    summ = pd.DataFrame(summ)
    summ["abs_d"] = summ.cohens_d.abs()
    summ = summ.sort_values("abs_d", ascending=False).drop(columns="abs_d")
    summ.to_csv(out / "physics_gap_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = ["mean", "p95", "CNR", "nakagami_m_myo", "speckle_SNR_myo",
                "fwhm_axial", "fwhm_lateral", "sharpness"]
        fig, ax = plt.subplots(3, 3, figsize=(12, 10))
        for axi, c in zip(ax.flat, show):
            for name, col in (("EchoNet", "#2a6fdb"), ("CAMUS", "#d9534f")):
                x = df.loc[df.dataset == name, c].dropna()
                axi.hist(x, bins=40, alpha=0.5, density=True, color=col, label=name)
            axi.set_title(c)
        axd = ax.flat[-1]
        for name, col in (("EchoNet", "#2a6fdb"), ("CAMUS", "#d9534f")):
            d = df.loc[df.dataset == name, [f"depth_{i}" for i in range(N_DEPTH)]]
            m, s = d.mean().values, d.std().values
            xx = np.arange(N_DEPTH) / (N_DEPTH - 1)
            axd.plot(xx, m, color=col, label=name)
            axd.fill_between(xx, m - s, m + s, color=col, alpha=0.15)
        axd.set_title("Mean intensity vs depth"); axd.set_xlabel("relative depth (0 = apex)")
        ax.flat[0].legend()
        plt.tight_layout()
        plt.savefig(out / "physics_gap.png", dpi=150)
        print("Saved", out / "physics_gap.png")
    except ImportError:
        print("matplotlib not installed: skipped the figure")


if __name__ == "__main__":
    main()
