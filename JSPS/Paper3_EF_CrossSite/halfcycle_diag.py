r"""
Diagnostic: does CAMUS's half-heartbeat clip format alone raise the error?
Uses EchoNet's own TEST videos (same scanner, same patients' labels), so only the clip structure changes.

For each test video (ED and ES frames from VolumeTracings; ED = larger LV area):
  full     : the whole video, as in normal testing
  mirror   : only ED -> ES, resampled to 50 fps, played forward then backward and repeated
             (exactly the CAMUS pipeline in camus_eval.py)
  pad      : only ED -> ES, resampled to 50 fps, then zero-padded

Usage (notebook 1, from D:\EchoNet):
    python halfcycle_diag.py
    python halfcycle_diag.py --limit 100      # quick test
Outputs in --out: halfcycle_predictions.csv, halfcycle_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from echonet_baseline import build_model, load_video, metrics
from camus_eval import make_cycle, predict, resample_time


def ed_es_frames(tr):
    """Return (ED frame, ES frame) from one video's tracings: ED = larger traced area."""
    areas = {}
    for frame, g in tr.groupby("Frame"):
        g = g[["X1", "Y1", "X2", "Y2"]].values
        if len(g) < 3:
            continue
        x = np.concatenate([g[1:, 0], g[1:, 2][::-1]])
        y = np.concatenate([g[1:, 1], g[1:, 3][::-1]])
        areas[int(frame)] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if len(areas) < 2:
        return None
    ed = max(areas, key=areas.get)
    es = min(areas, key=areas.get)
    return ed, es


def summarize(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = metrics(y, p)
    e = p - y
    r.update(n=int(len(y)), bias=float(e.mean()),
             LoA_low=float(e.mean() - 1.96 * e.std()), LoA_high=float(e.mean() + 1.96 * e.std()))
    try:
        from sklearn.metrics import roc_auc_score
        r["AUC_EF<40"] = float(roc_auc_score((y < 40).astype(int), -p))
    except Exception:
        pass
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--echonet", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--meanstd", default=r"D:\EchoNet\mean_std.json")
    ap.add_argument("--ckpt", default=r"D:\EchoNet\runs\baseline_seed0\best.pt")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\baseline_seed0\halfcycle")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    root = Path(a.echonet)
    fl = pd.read_csv(root / "FileList.csv")
    fl = fl[(fl.Split == "TEST") & (fl.FrameHeight == 112) & (fl.FrameWidth == 112)]
    if a.limit:
        fl = fl.head(a.limit)
    vt = dict(tuple(pd.read_csv(root / "VolumeTracings.csv").groupby("FileName", sort=False)))

    device = torch.device("cuda")
    ms = json.loads(Path(a.meanstd).read_text())
    mean, std = np.array(ms["mean"], np.float32), np.array(ms["std"], np.float32)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], 0
    for i, r in enumerate(fl.itertuples()):
        name = str(r.FileName)
        key = name if name.endswith(".avi") else name + ".avi"
        if key not in vt or (fr := ed_es_frames(vt[key])) is None:
            skipped += 1
            continue
        ed, es = fr
        v = load_video(root / "Videos" / key)[0]                     # (T, H, W) grey
        lo, hi = sorted((min(ed, v.shape[0] - 1), min(es, v.shape[0] - 1)))
        half = v[lo:hi + 1]
        if ed > es:                                                   # make it run ED -> ES
            half = half[::-1]
        half = resample_time(half, float(r.FPS))
        rows.append(dict(
            FileName=name, EF=float(r.EF), fps=float(r.FPS), half_frames=hi - lo + 1,
            pred_full=predict(model, v, mean, std, device),
            pred_mirror=predict(model, make_cycle(half, 64, "mirror"), mean, std, device),
            pred_pad=predict(model, make_cycle(half, 64, "pad"), mean, std, device),
        ))
        if i % 100 == 0:
            print(f"  {i}/{len(fl)}  {rows[-1]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "halfcycle_predictions.csv", index=False)
    res = {k: summarize(df.EF, df[f"pred_{k}"]) for k in ("full", "mirror", "pad")}
    res["skipped_no_tracings"] = skipped
    d = (df.pred_mirror - df.EF).abs() - (df.pred_full - df.EF).abs()
    res["mirror_minus_full_abs_error"] = dict(mean=float(d.mean()), median=float(d.median()))
    try:
        from scipy.stats import wilcoxon
        res["mirror_minus_full_abs_error"]["wilcoxon_p"] = float(wilcoxon(d).pvalue)
    except Exception:
        pass
    (out / "halfcycle_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
