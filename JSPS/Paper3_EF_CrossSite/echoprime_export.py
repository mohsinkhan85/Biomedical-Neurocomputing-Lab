r"""
EchoPrime foundation model (Vukadinovic et al., arXiv 2410.09704; github.com/echonet/EchoPrime, MIT licence) on
the same videos as the EchoNet-trained models (JSPS Paper 3, package D). No training.

EchoPrime has no EF regression head: it predicts measurements by retrieval. A video is embedded (MViT-v2-S video
encoder, 512-d; video embedding = mean of its 4 clip embeddings, as EchoPrime averages embeddings into a study
embedding), compared with the report embeddings of its training studies, and EF = mean EF of the 50 most similar
studies, as in EchoPrime.predict_metrics(). Only the ~607,000 candidate studies that have an EF label are searched
(with all ~1.2 million, some videos have no labelled neighbour among the top 50 and get no EF).

Preprocessing follows EchoPrime's own code (process_mp4s / utils.crop_and_scale): 10% border crop, resize to
224 x 224 (bicubic), grey -> 3 channels, EchoPrime mean/std; clips of 32 frames taken with stride 2 (16 frames);
4 evenly spaced clips per video, EF = mean over clips, spread = SD over clips, embedding = mean over clips.
CAMUS is harmonized as for the other models (isotropic, sector crop, 50 fps, ED->ES->ED cycle), rendered at
280 px so that the 10% crop leaves 224 px.

Outputs (same format as panecho_export.py)
  <runs_dir>\echoprime\val_predictions.csv, test_predictions.csv, results.json
  <export>\camus_echoprime.npz, pediatric_echoprime.npz
  F_source = [standardized embedding (512), EchoPrime EF], w0 = [0,...,0, 1], b0 = 0.

Model files (about 3.9 GB, downloaded once into --model_dir from the EchoPrime v1.0.0 GitHub release):
  model_data.zip (only the encoder weights and candidate studies/labels are extracted),
  candidate_embeddings_p1.pt, candidate_embeddings_p2.pt

Usage (notebook 1, from D:\EchoNet):
  !"{sys.executable}" echoprime_export.py --limit 8      # quick check (downloads the model files first)
  !"{sys.executable}" echoprime_export.py                # full run
"""
import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

from echonet_baseline import load_video, read_filelist

try:  # Windows certificate store for HTTPS (university network), as in panecho_export.py
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("Note: 'pip install truststore' if downloads fail with CERTIFICATE_VERIFY_FAILED")

cv2.setNumThreads(1)
RELEASE = "https://github.com/echonet/EchoPrime/releases/download/v1.0.0/"
FRAMES, STRIDE, NCLIPS, TOPK = 32, 2, 4, 50
EP_MEAN = np.array([29.110628, 28.076836, 29.096405], np.float32)[:, None, None, None]
EP_STD = np.array([47.989223, 46.456997, 47.20083], np.float32)[:, None, None, None]
ZIP_MEMBERS = ["model_data/weights/echo_prime_encoder.pt",
               "model_data/candidates_data/candidate_studies.csv",
               "model_data/candidates_data/candidate_labels.pkl"]


# ---------------------------------------------------------------- model files
def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, t0, last = 0, time.time(), 0
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 15:
                last = time.time()
                print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({done / 1e6 / (last - t0):.0f} MB/s)", flush=True)
    tmp.rename(dest)


def get_model_files(model_dir):
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    if not all((d / m).exists() for m in ZIP_MEMBERS):
        z = d / "model_data.zip"
        download(RELEASE + "model_data.zip", z)
        with zipfile.ZipFile(z) as zf:
            for m in ZIP_MEMBERS:
                zf.extract(m, d)
        z.unlink()                                                     # 1.3 GB, the rest is not needed
    for p in ("candidate_embeddings_p1.pt", "candidate_embeddings_p2.pt"):
        download(RELEASE + p, d / p)
    return d


def load_echoprime(model_dir, device):
    d = get_model_files(model_dir)
    enc = torchvision.models.video.mvit_v2_s()
    enc.head[-1] = torch.nn.Linear(enc.head[-1].in_features, 512)
    enc.load_state_dict(torch.load(d / ZIP_MEMBERS[0], map_location="cpu"))
    enc = enc.to(device).eval()
    cand = torch.cat([torch.load(d / f"candidate_embeddings_p{i}.pt", map_location="cpu") for i in (1, 2)])
    studies = pd.read_csv(d / ZIP_MEMBERS[1])["Study"].values
    ef_lab = pd.read_pickle(d / ZIP_MEMBERS[2])["ejection_fraction"]
    # EF label of each candidate (NaN if that study has no EF), aligned with the candidate embeddings
    cand_ef = torch.tensor([ef_lab.get(s, np.nan) for s in studies], dtype=torch.float32)
    keep = torch.isfinite(cand_ef)                   # retrieve only among studies that have an EF label
    print(f"EchoPrime: {len(cand)} candidate embeddings, {int(keep.sum())} with EF (used for retrieval)", flush=True)
    return enc, cand[keep].to(device), cand_ef[keep].to(device)


# ---------------------------------------------------------------- video -> clips
def crop_scale(v):
    """EchoPrime crop_and_scale for square frames: cut 10% of each border, resize to 224 (bicubic)."""
    h, w = v.shape[1:]
    py, px = int(h * 0.1), int(w * 0.1)
    v = v[:, py:h - py, px:w - px]
    return np.clip(np.stack([cv2.resize(f.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
                             for f in v]), 0, 255)


def clips(v):
    """4 evenly spaced clips of 32 frames, stride 2 -> float (4, 3, 16, 224, 224), EchoPrime-normalized."""
    if len(v) < FRAMES:                                               # EchoPrime pads short videos with zeros
        v = np.concatenate([v, np.zeros((FRAMES - len(v), *v.shape[1:]), v.dtype)])
    starts = np.linspace(0, len(v) - FRAMES, NCLIPS).round().astype(int)
    out = [(np.repeat(v[s:s + FRAMES:STRIDE][None], 3, 0) - EP_MEAN) / EP_STD for s in starts]
    return np.stack(out).astype(np.float32)


def echonet_videos(root, split, limit=None):
    df = read_filelist(Path(root))
    df = df[df.Split == split]
    if limit:
        df = df.head(limit)
    for r in df.itertuples():
        yield str(r.FileName), float(r.EF), "NA", np.nan, crop_scale(load_video(r.path)[0])


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
        yield f, float(r.EF), str(getattr(r, "Age", "NA")), np.nan, crop_scale(load_video(path)[0])
        n += 1
        if limit and n >= limit:
            break


def camus_videos(root, limit=None):
    from camus_eval import get_float, load_nifti, monoplane_ef, read_cfg, resample_time, to_echonet_frames
    a = SimpleNamespace(no_transpose=False, flip_ud=False, flip_lr=False)
    pdirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("patient"))[:limit]
    for p in pdirs:
      try:
        cfg = read_cfg(p / "Info_4CH.cfg")
        seq, rs, cs = load_nifti(p / f"{p.name}_4CH_half_sequence.nii.gz", a)
        v = resample_time(to_echonet_frames(seq, rs, cs, size=280), get_float(cfg, "FrameRate", "FPS"))
        cyc = np.concatenate([v, v[-2:0:-1]]) if len(v) > 2 else v      # ED->ES->ED, as for the other models
        long = np.concatenate([cyc] * (int(np.ceil(2 * FRAMES / len(cyc))) + 1))
        item = (p.name, get_float(cfg, "EF", "LVef"), cfg.get("ImageQuality", "NA"), monoplane_ef(p, p.name, a),
                crop_scale(long))
      except Exception as err:                                        # report and skip one bad patient
        print(f"  SKIPPED {p.name}: {type(err).__name__}: {err}", flush=True)
        continue
      yield item


# ---------------------------------------------------------------- inference
@torch.no_grad()
def ef_by_retrieval(emb, cand, cand_ef):
    """EchoPrime predict_metrics for a single video: cosine-normalized video embedding x candidate embeddings,
    EF = nanmean of the EF labels of the TOPK most similar candidate studies."""
    sims = torch.nn.functional.normalize(emb, dim=1) @ cand.T
    idx = torch.topk(sims, k=TOPK, dim=1).indices
    return torch.nanmean(cand_ef[idx], dim=1)


@torch.no_grad()
def run(enc, cand, cand_ef, video_iter, device):
    rows, feats, t0 = [], [], time.time()
    for i, (vid, y, q, y2, v) in enumerate(video_iter):
        try:
            x = torch.from_numpy(clips(v)).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                emb = enc(x)
            emb = emb.float()
            ef_clips = ef_by_retrieval(emb, cand, cand_ef).cpu().numpy()          # per clip (for the spread)
            ef_video = float(ef_by_retrieval(emb.mean(0, keepdim=True), cand, cand_ef)[0])  # video embedding
        except Exception as err:                                      # report and skip one bad video
            print(f"  SKIPPED {vid}: {type(err).__name__}: {err}", flush=True)
            continue
        rows.append(dict(id=vid, y=y, quality=q, y2=y2, p=ef_video, s=float(np.std(ef_clips))))
        feats.append(emb.mean(0).cpu().numpy())
        if i % 200 == 0:
            print(f"  {i}: {vid} EF {y:.1f} pred {ef_video:.1f}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows), np.stack(feats)


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
    ap.add_argument("--model_dir", default=r"D:\EchoNet\EchoPrime_model")
    ap.add_argument("--runs_dir", default=r"D:\EchoNet\runs")
    ap.add_argument("--export", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--tag", default="echoprime")
    ap.add_argument("--limit", type=int, default=None, help="videos per dataset, for a quick check")
    ap.add_argument("--cpu_threads", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, cand, cand_ef = load_echoprime(a.model_dir, device)
    run_dir = Path(a.runs_dir) / (a.tag if not a.limit else a.tag + "_check")
    run_dir.mkdir(parents=True, exist_ok=True)
    exp = Path(a.export) if not a.limit else run_dir
    exp.mkdir(parents=True, exist_ok=True)
    results, feats_src = {}, None

    def cached_run(name, it):
        """Run one dataset, or reload it if an earlier run already finished it (resume after a crash)."""
        c = run_dir / f"{name}_cache.npz"
        if c.exists():
            z = np.load(c, allow_pickle=True)
            print(f"{name}: reloaded {len(z['p'])} videos from {c.name}", flush=True)
            df = pd.DataFrame({k: z[k] for k in ("id", "y", "quality", "y2", "p", "s")})
            with torch.no_grad():   # recompute EF from the saved video embeddings (labelled candidates only)
                df["p"] = ef_by_retrieval(torch.from_numpy(z["F"]).float().to(device), cand, cand_ef).cpu().numpy()
            df["s"] = np.nan_to_num(df["s"].values.astype(float))
            return df, z["F"]
        print(f"{name} ...", flush=True)
        df, F = run(enc, cand, cand_ef, it, device)
        np.savez_compressed(c, F=F, **{k: df[k].values for k in df.columns})
        return df, F

    val_csv = run_dir / "val_predictions.csv"
    if False:                                                         # VAL is recomputed with the current method
        d = pd.read_csv(val_csv)
        print(f"EchoNet VAL: reloaded {len(d)} predictions from {val_csv.name}", flush=True)
        results["echonet_val"] = summary(d.EF_true.values, d.EF_pred.values)
    else:
        df, _ = cached_run("echonet_val", echonet_videos(a.echonet, "VAL", a.limit))
        pd.DataFrame({"FileName": df.id, "EF_true": df.y, "EF_pred": df.p}).to_csv(val_csv, index=False)
        results["echonet_val"] = summary(df.y.values, df.p.values)
    df, feats_src = cached_run("echonet_test", echonet_videos(a.echonet, "TEST", a.limit))   # embeddings needed
    pd.DataFrame({"FileName": df.id, "EF_true": df.y, "EF_pred": df.p}).to_csv(
        run_dir / "test_predictions.csv", index=False)
    results["echonet_test"] = summary(df.y.values, df.p.values)
    mu, sd = feats_src.mean(0), feats_src.std(0) + 1e-6

    for name, it in (("camus", camus_videos(a.camus, a.limit)), ("pediatric", pediatric_videos(a.pediatric, a.limit))):
        df, F = cached_run(name, it)
        results[name] = summary(df.y.values, df.p.values)
        Fz = (F - mu) / sd
        np.savez_compressed(
            exp / f"{name}_{a.tag}.npz", ids=df.id.values, y=df.y.values, y2=df.y2.values.astype(np.float64),
            quality=df.quality.values, F_source=np.hstack([Fz, df.p.values[:, None]]).astype(np.float32),
            w0=np.r_[np.zeros(Fz.shape[1]), 1.0], b0=0.0, p_source=df.p.values, s_source=df.s.values)
        print(f"  saved {exp / f'{name}_{a.tag}.npz'}", flush=True)

    results["settings"] = dict(frames=FRAMES, stride=STRIDE, clips_per_video=NCLIPS, topk=TOPK,
                               preprocessing="10% border crop, resize 224 bicubic, EchoPrime mean/std",
                               ef="retrieval over candidate studies (EchoPrime predict_metrics)", limit=a.limit)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
