r"""
External test of the EchoNet-trained EF model on CAMUS (no retraining, no tuning on CAMUS).

Usage (from D:\EchoNet, with the .venv active):
  1) Check that CAMUS frames look like EchoNet frames (orientation, cropping):
       python camus_eval.py --preview
     -> writes camus_preview.png (left: EchoNet frame, right: CAMUS frames). Check it before step 2.
  2) Evaluate:
       python camus_eval.py --ckpt D:\EchoNet\runs\baseline_seed0\best.pt --out D:\EchoNet\runs\baseline_seed0\camus
  Field of view: --zoom 1.3 crops CAMUS to EchoNet's tighter view (apex kept at top).
  Diagnostic: --histmatch maps each CAMUS video's in-sector grey-level distribution onto EchoNet's
  (reference built from training videos only). A test of whether brightness/gain drives the gap,
  not part of the proposed method.
  Orientation fixes if the preview looks wrong: add --no_transpose, --flip_ud and/or --flip_lr to BOTH commands.

Harmonization (report these in the paper):
  - 4-chamber half sequence (ED -> ES) only.
  - Resampled to isotropic pixels using the NIfTI spacing, cropped to the scan sector,
    padded to square, resized to 112x112, grey replicated to 3 channels.
  - Resampled in time to 50 fps (EchoNet's typical rate). The half cycle is mirrored
    (ED->ES->ED) and repeated to form full-cycle-like videos; --temporal pad uses zero padding instead.
  - Normalized with the EchoNet training mean/std (mean_std.json), never with CAMUS statistics.
  - Targets: CAMUS reference EF (biplane), and an apical 4-chamber monoplane EF computed
    from the expert LV masks with the area-length method.
"""
import argparse
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from echonet_baseline import build_model, load_video, metrics

TARGET_FPS = 50.0


# ---------------------------------------------------------------- reading
def read_cfg(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def get_float(cfg, *keys):
    for k in keys:
        if k in cfg:
            try:
                return float(cfg[k])
            except ValueError:
                pass
    return np.nan


def load_nifti(path, a):
    img = nib.load(str(path))
    x = np.asarray(img.dataobj)
    sp = img.header.get_zooms()
    if x.ndim == 2:
        x = x[..., None]
    sx, sy = float(sp[0]), float(sp[1])
    if not a.no_transpose:  # NIfTI stores (width, height, frames) -> (height, width, frames)
        x = x.transpose(1, 0, 2)
        row_sp, col_sp = sy, sx
    else:
        row_sp, col_sp = sx, sy
    if a.flip_ud:
        x = x[::-1]
    if a.flip_lr:
        x = x[:, ::-1]
    return np.ascontiguousarray(x), row_sp, col_sp


# ---------------------------------------------------------------- harmonization
def to_echonet_frames(seq, row_sp, col_sp, size=112, zoom=1.0):
    """seq (H, W, T) -> uint8 (T, 112, 112): isotropic, sector-cropped, square, resized."""
    s = min(row_sp, col_sp)
    h, w = seq.shape[:2]
    nh, nw = int(round(h * row_sp / s)), int(round(w * col_sp / s))
    frames = [cv2.resize(seq[..., t].astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
              for t in range(seq.shape[2])]
    v = np.stack(frames)
    mask = v.max(0) > 0
    ys, xs = np.where(mask)
    if len(ys):
        v = v[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    t, h, w = v.shape
    m = max(h, w)
    sq = np.zeros((t, m, m), np.float32)
    y0, x0 = (m - h) // 2, (m - w) // 2
    sq[:, y0:y0 + h, x0:x0 + w] = v
    if zoom > 1:  # match EchoNet's tighter field of view: keep the apex (top), crop sides and bottom
        c = int(round(m / zoom))
        xs0 = (m - c) // 2
        sq = sq[:, :c, xs0:xs0 + c]
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in sq])
    return np.clip(out, 0, 255)


def resample_time(v, fps_in, fps_out=TARGET_FPS):
    t = v.shape[0]
    if t < 2 or not np.isfinite(fps_in) or fps_in <= 0:
        return v
    n = max(2, int(round((t - 1) * fps_out / fps_in)) + 1)
    pos = np.linspace(0, t - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (pos - i0)[:, None, None]
    return v[i0] * (1 - w) + v[i1] * w


def make_cycle(v, need, mode):
    if mode == "pad" or v.shape[0] >= need:
        return v
    cyc = np.concatenate([v, v[-2:0:-1]]) if v.shape[0] > 2 else v
    reps = int(np.ceil(need / len(cyc))) + 1
    return np.concatenate([cyc] * reps)[:max(need, len(v))]


# ---------------------------------------------------------------- histogram matching (diagnostic)
def echonet_reference_quantiles(echonet_root, n=200, seed=0, nq=1001):
    fl = pd.read_csv(Path(echonet_root) / "FileList.csv")
    fl = fl[(fl.Split == "TRAIN") & (fl.FrameHeight == 112) & (fl.FrameWidth == 112)]
    fl = fl.sample(min(n, len(fl)), random_state=seed)
    vals = []
    for f in fl.FileName.astype(str):
        v = load_video(Path(echonet_root) / "Videos" / (f if f.endswith(".avi") else f + ".avi"))[0]
        sec = v.max(0) > 3
        vals.append(v[:, sec].ravel()[::7])
    return np.quantile(np.concatenate(vals), np.linspace(0, 1, nq))


def histmatch(v, ref_q):
    """Map in-sector intensities of video v (T,H,W) so their distribution matches ref_q."""
    sec = v.max(0) > 0
    x = v[:, sec]
    q = np.linspace(0, 1, len(ref_q))
    src_q = np.quantile(x, q)
    src_q = np.maximum.accumulate(src_q + np.arange(len(q)) * 1e-6)
    out = v.copy()
    out[:, sec] = np.interp(x, src_q, ref_q)
    return out


# ---------------------------------------------------------------- monoplane EF from masks
def lv_volume_area_length(mask, row_sp, col_sp):
    """Single-plane area-length: V = 0.85 * A^2 / L (mm -> mL)."""
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return np.nan
    area = len(ys) * row_sp * col_sp
    pts = np.stack([ys * row_sp, xs * col_sp], 1)
    pts = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[0]
    length = proj.max() - proj.min()
    return 0.85 * area ** 2 / length / 1000.0


def monoplane_ef(pdir, pid, a):
    vols = []
    for ph in ("ED", "ES"):
        f = pdir / f"{pid}_4CH_{ph}_gt.nii.gz"
        if not f.exists():
            return np.nan
        m, rs, cs = load_nifti(f, a)
        vols.append(lv_volume_area_length(m[..., 0] == 1, rs, cs))  # label 1 = LV endocardium
    edv, esv = vols
    return 100.0 * (edv - esv) / edv if edv and np.isfinite(edv) else np.nan


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, v_uint8, mean, std, device, frames=32, period=2, stride=4, chunk=16):
    v = np.repeat(v_uint8[None], 3, 0).astype(np.float32)          # (3, T, H, W)
    v = (v - mean[:, None, None, None]) / std[:, None, None, None]
    need = frames * period
    if v.shape[1] < need:
        v = np.concatenate([v, np.zeros((3, need - v.shape[1], *v.shape[2:]), np.float32)], 1)
    starts = np.arange(0, v.shape[1] - need + 1, stride)
    clips = np.stack([v[:, s + period * np.arange(frames)] for s in starts])
    clips = torch.from_numpy(clips)
    out = []
    for k in range(0, len(clips), chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(clips[k:k + chunk].to(device)).float().view(-1))
    return torch.cat(out).mean().item()


def summarize(df, ycol, pcol="EF_pred"):
    d = df[[ycol, pcol]].dropna()
    y, p = d[ycol].values, d[pcol].values
    r = metrics(y, p)
    e = p - y
    r.update(n=int(len(d)), bias=float(e.mean()),
             LoA_low=float(e.mean() - 1.96 * e.std()), LoA_high=float(e.mean() + 1.96 * e.std()))
    try:
        from sklearn.metrics import roc_auc_score
        for thr in (40, 45):
            lab = (y < thr).astype(int)
            if 0 < lab.sum() < len(lab):
                r[f"AUC_EF<{thr}"] = float(roc_auc_score(lab, -p))
    except ImportError:
        pass
    return r


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus", default=r"D:\CAMUS\database_nifti")
    ap.add_argument("--echonet", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--meanstd", default=r"D:\EchoNet\mean_std.json")
    ap.add_argument("--ckpt", default=r"D:\EchoNet\runs\baseline_seed0\best.pt")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\baseline_seed0\camus")
    ap.add_argument("--temporal", choices=["mirror", "pad"], default="mirror")
    ap.add_argument("--default_fps", type=float, default=np.nan,
                    help="frame rate to assume if Info_4CH.cfg has none")
    ap.add_argument("--no_transpose", action="store_true")
    ap.add_argument("--flip_ud", action="store_true")
    ap.add_argument("--flip_lr", action="store_true")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help=">1 crops CAMUS to a tighter field of view like EchoNet (e.g. 1.3)")
    ap.add_argument("--histmatch", action="store_true",
                    help="diagnostic: match CAMUS grey levels to EchoNet's distribution")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    root = Path(a.camus)
    pdirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("patient"))
    if a.limit:
        pdirs = pdirs[:a.limit]
    print(f"{len(pdirs)} CAMUS patients found")
    cfg0 = read_cfg(pdirs[0] / "Info_4CH.cfg")
    print("Info_4CH.cfg keys:", cfg0)

    tag = f"{a.temporal}_zoom{a.zoom:g}" + ("_histmatch" if a.histmatch else "")
    ref_q = None
    if a.histmatch:
        print("Building EchoNet reference grey-level distribution (200 training videos) ...")
        ref_q = echonet_reference_quantiles(a.echonet)

    if a.preview:
        ef = pd.read_csv(Path(a.echonet) / "FileList.csv").iloc[0].FileName
        en = load_video(Path(a.echonet) / "Videos" / (ef if ef.endswith(".avi") else ef + ".avi"))[0, 0]
        tiles = [en.astype(np.uint8)]
        for p in pdirs[:3]:
            seq, rs, cs = load_nifti(p / f"{p.name}_4CH_half_sequence.nii.gz", a)
            v = to_echonet_frames(seq, rs, cs, zoom=a.zoom)
            if ref_q is not None:
                v = histmatch(v, ref_q)
            tiles.append(v[0].astype(np.uint8))
        img = cv2.resize(np.concatenate(tiles, 1), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"camus_preview_{tag}.png", img)
        print(f"Wrote camus_preview_{tag}.png  (left: EchoNet; right: CAMUS patients 1-3). "
              "Apex should be at the top and the views should look alike.")
        return

    device = torch.device("cuda")
    ms = json.loads(Path(a.meanstd).read_text())
    mean, std = np.array(ms["mean"], np.float32), np.array(ms["std"], np.float32)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(pdirs):
        cfg = read_cfg(p / "Info_4CH.cfg")
        fps = get_float(cfg, "FrameRate", "FPS", "frame_rate")
        if not np.isfinite(fps):
            fps = a.default_fps
        seq, rs, cs = load_nifti(p / f"{p.name}_4CH_half_sequence.nii.gz", a)
        v = to_echonet_frames(seq, rs, cs, zoom=a.zoom)
        if ref_q is not None:
            v = histmatch(v, ref_q)
        n_raw = v.shape[0]
        v = resample_time(v, fps)
        v = make_cycle(v, 64, a.temporal)
        rows.append(dict(
            patient=p.name,
            EF_ref=get_float(cfg, "EF", "LVef"),
            EF_mono_A4C=monoplane_ef(p, p.name, a),
            quality=cfg.get("ImageQuality", "NA"),
            fps=fps, frames=n_raw,
            EF_pred=predict(model, v, mean, std, device),
        ))
        if i % 50 == 0:
            print(f"  {i}/{len(pdirs)}  {rows[-1]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"camus_predictions_{tag}.csv", index=False)
    res = {"settings": vars(a), "overall_vs_reference_EF": summarize(df, "EF_ref"),
           "overall_vs_A4C_monoplane_EF": summarize(df, "EF_mono_A4C"), "by_quality": {}}
    for q, g in df.groupby("quality"):
        res["by_quality"][q] = summarize(g, "EF_ref")
    (out / f"camus_results_{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
