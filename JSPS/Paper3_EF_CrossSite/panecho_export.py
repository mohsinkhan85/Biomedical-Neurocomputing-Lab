r"""
PanEcho foundation model (Holste et al., JAMA 2025; github.com/CarDS-Yale/PanEcho) on the same videos as the
EchoNet-trained models (JSPS Paper 3, package D). No training: the published EF head is used as is, and the
768-d video embedding is saved for the few-label adaptation analysis (cheap_analysis.py).

Preprocessing follows the PanEcho paper: 16 consecutive frames, frames resized to 256 x 256 (bicubic) and
centre-cropped to 224 x 224, grey -> 3 channels, ImageNet normalization; 4 evenly spaced clips per video,
EF = mean of the 4 clip predictions, spread = SD of the 4. CAMUS is harmonized exactly as for the other models
(isotropic, sector crop, 50 fps, ED->ES->ED cycle) but rendered at 256 px instead of 112.

Outputs
  <runs_dir>\panecho\val_predictions.csv, test_predictions.csv, results.json   (EchoNet-Dynamic VAL/TEST;
        val_predictions.csv gives the EchoNet error profile used by the norm/asym intervals)
  <export>\camus_panecho.npz, pediatric_panecho.npz                          (format of adapt_benchmark --export)
  In the npz, F_source = [standardized embedding (768), PanEcho EF], w0 = [0,...,0, 1], b0 = 0, so that
  'last_layer' in cheap_analysis.py is a ridge correction of PanEcho's own EF prediction from its embedding
  (penalized towards "no change"), exactly like for the EchoNet-trained models. The embedding is standardized
  with EchoNet TEST statistics (source domain, no target labels used).

Usage (notebook 1, from D:\EchoNet; first run downloads the code and weights from GitHub):
  !"{sys.executable}" panecho_export.py --limit 8          # quick check (prints shapes, EF units, timing)
  !"{sys.executable}" panecho_export.py                    # full run (EchoNet val+test, CAMUS, pediatric)
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch

from echonet_baseline import load_video, read_filelist

# Use the Windows certificate store for HTTPS downloads (GitHub, PyTorch weights). Python's own certificate
# list fails on networks that inspect HTTPS (CERTIFICATE_VERIFY_FAILED); pip already uses the system store.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("Note: 'pip install truststore' if downloads fail with CERTIFICATE_VERIFY_FAILED")

cv2.setNumThreads(1)
CLIP, NCLIPS = 16, 4
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None, None] * 255
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None, None] * 255


# ---------------------------------------------------------------- video -> clips
def to_224(v):
    """grey (T, H, W) -> (T, 224, 224): resize to 256 (bicubic), centre crop 224 (PanEcho test preprocessing)."""
    v = np.stack([cv2.resize(f.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC) for f in v])
    return np.clip(v[:, 16:240, 16:240], 0, 255)


def clips(v, n=NCLIPS, length=CLIP):
    """n evenly spaced clips of `length` consecutive frames -> float (n, 3, length, 224, 224), normalized."""
    if len(v) < length:
        v = np.concatenate([v] * int(np.ceil(length / len(v))))
    starts = np.linspace(0, len(v) - length, n).round().astype(int)
    out = []
    for s in starts:
        c = np.repeat(v[s:s + length][None], 3, 0)                      # (3, T, H, W)
        out.append((c - IMNET_MEAN) / IMNET_STD)
    return np.stack(out).astype(np.float32)


def echonet_videos(root, split, limit=None):
    df = read_filelist(Path(root))
    df = df[df.Split == split]
    if limit:
        df = df.head(limit)
    for r in df.itertuples():
        yield str(r.FileName), float(r.EF), "NA", np.nan, to_224(load_video(r.path)[0])


def pediatric_videos(root, limit=None):
    root = Path(root)
    df = pd.read_csv(root / "FileList.csv")
    df = df[df.EF.notna()]
    n = 0
    for r in df.itertuples():
        f = str(r.FileName)
        path = root / "Videos" / (f if f.lower().endswith(".avi") else f + ".avi")
        if not path.exists():
            continue
        yield f, float(r.EF), str(getattr(r, "Age", "NA")), np.nan, to_224(load_video(path)[0])
        n += 1
        if limit and n >= limit:
            break


def camus_videos(root, limit=None):
    from camus_eval import get_float, load_nifti, monoplane_ef, read_cfg, resample_time, to_echonet_frames
    a = SimpleNamespace(no_transpose=False, flip_ud=False, flip_lr=False)
    pdirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("patient"))[:limit]
    for p in pdirs:
        cfg = read_cfg(p / "Info_4CH.cfg")
        seq, rs, cs = load_nifti(p / f"{p.name}_4CH_half_sequence.nii.gz", a)
        v = resample_time(to_echonet_frames(seq, rs, cs, size=256), get_float(cfg, "FrameRate", "FPS"))
        cyc = np.concatenate([v, v[-2:0:-1]]) if len(v) > 2 else v      # ED->ES->ED, as for the other models
        long = np.concatenate([cyc] * (int(np.ceil(3 * CLIP / len(cyc))) + 1))
        long = np.clip(long[:, 16:240, 16:240], 0, 255)                  # already 256 px: centre crop only
        yield p.name, get_float(cfg, "EF", "LVef"), cfg.get("ImageQuality", "NA"), monoplane_ef(p, p.name, a), long


# ---------------------------------------------------------------- model
def load_model(device):
    """PanEcho with only the EF head. model.encoder(x) -> 768-d video embedding; model.EF_head(emb) -> EF (%)."""
    m = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", tasks=["EF"], clip_len=CLIP, trust_repo=True)
    return m.to(device).eval()


@torch.no_grad()
def run(model, video_iter, device, ef_scale):
    rows, feats = [], []
    t0 = time.time()
    for i, (vid, y, q, y2, v) in enumerate(video_iter):
        x = torch.from_numpy(clips(v)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            emb = model.encoder(x)
            ef = model.EF_head(emb)
        ef = ef.float().view(len(x), -1)[:, 0].cpu().numpy()
        if ef_scale is None:                                             # decide units from the first video
            ef_scale = 100.0 if np.median(ef) < 1.5 else 1.0
            print(f"  EF head output {ef.round(3)} -> scale x{ef_scale:g} (percent)", flush=True)
        ef = ef * ef_scale
        rows.append(dict(id=vid, y=y, quality=q, y2=y2, p=float(ef.mean()), s=float(ef.std())))
        feats.append(emb.float().mean(0).cpu().numpy())
        if i % 200 == 0:
            print(f"  {i}: {vid} EF {y:.1f} pred {ef.mean():.1f}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows), np.stack(feats), ef_scale


def summary(y, p):
    from sklearn.metrics import roc_auc_score
    e = p - y
    r = dict(n=int(len(y)), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
             R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), slope=float(np.polyfit(y, p, 1)[0]))
    red = y < 40
    if 0 < red.sum() < len(y):
        r["AUC_EF<40"] = float(roc_auc_score(red.astype(int), -p))
        r["reduced_called_preserved"] = float((p[red] >= 50).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--echonet", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--camus", default=r"D:\CAMUS\database_nifti")
    ap.add_argument("--pediatric", default=r"D:\EchoNet - Pediatric\pediatric_echo_avi\A4C")
    ap.add_argument("--runs_dir", default=r"D:\EchoNet\runs")
    ap.add_argument("--export", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--tag", default="panecho")
    ap.add_argument("--limit", type=int, default=None, help="videos per dataset, for a quick check")
    ap.add_argument("--cpu_threads", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    run_dir = Path(a.runs_dir) / (a.tag if not a.limit else a.tag + "_check")
    run_dir.mkdir(parents=True, exist_ok=True)
    exp = Path(a.export) if not a.limit else run_dir
    exp.mkdir(parents=True, exist_ok=True)
    results, scale = {}, None

    # EchoNet-Dynamic (source domain for the other models): VAL gives the error profile, TEST the reference result
    feats_src = None
    for split in ("VAL", "TEST"):
        print(f"EchoNet {split} ...", flush=True)
        df, F, scale = run(model, echonet_videos(a.echonet, split, a.limit), device, scale)
        pd.DataFrame({"FileName": df.id, "EF_true": df.y, "EF_pred": df.p}).to_csv(
            run_dir / f"{split.lower()}_predictions.csv", index=False)
        results[f"echonet_{split.lower()}"] = summary(df.y.values, df.p.values)
        if split == "TEST":
            feats_src = F
    mu, sd = feats_src.mean(0), feats_src.std(0) + 1e-6

    for name, it in (("camus", camus_videos(a.camus, a.limit)), ("pediatric", pediatric_videos(a.pediatric, a.limit))):
        print(f"{name} ...", flush=True)
        df, F, scale = run(model, it, device, scale)
        results[name] = summary(df.y.values, df.p.values)
        Fz = (F - mu) / sd
        np.savez_compressed(
            exp / f"{name}_{a.tag}.npz", ids=df.id.values, y=df.y.values, y2=df.y2.values.astype(np.float64),
            quality=df.quality.values, F_source=np.hstack([Fz, df.p.values[:, None]]).astype(np.float32),
            w0=np.r_[np.zeros(Fz.shape[1]), 1.0], b0=0.0, p_source=df.p.values, s_source=df.s.values)
        print(f"  saved {exp / f'{name}_{a.tag}.npz'}", flush=True)

    results["settings"] = dict(clip_len=CLIP, clips_per_video=NCLIPS, resize="256 bicubic, centre crop 224",
                               ef_scale=scale, limit=a.limit)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
