r"""
External test of an EchoNet-Dynamic-trained EF model on EchoNet-Pediatric (apical 4-chamber view).
No retraining. Same 112x112 AVI format as EchoNet-Dynamic, so no conversion is needed.

Usage (notebook, from D:\EchoNet, .venv active):
    python pediatric_eval.py --check                    # print the label file's columns and a summary
    python pediatric_eval.py --ckpt "D:\EchoNet\runs\baseline_seed0\best.pt" --out "D:\EchoNet\runs\baseline_seed0\pediatric"
    python pediatric_eval.py --limit 100 ...            # quick test

Outputs in --out: pediatric_predictions.csv, pediatric_results.json
(overall, by EF range, by age group if an Age column exists, and heart-failure category agreement).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from echonet_baseline import EchoDataset, build_model, evaluate, metrics

DEFAULT_ROOT = r"D:\EchoNet - Pediatric\pediatric_echo_avi\A4C"


def load_filelist(root, limit=None):
    df = pd.read_csv(root / "FileList.csv")
    df.columns = [c.strip() for c in df.columns]
    name_col = next(c for c in df.columns if c.lower() in ("filename", "file_name", "file"))
    ef_col = next(c for c in df.columns if c.lower() in ("ef", "lvef"))
    df = df.rename(columns={name_col: "FileName", ef_col: "EF"})
    df["FileName"] = df.FileName.astype(str)
    df["path"] = [root / "Videos" / (f if f.lower().endswith(".avi") else f + ".avi") for f in df.FileName]
    n0 = len(df)
    df = df[df.path.map(lambda p: p.exists()) & df.EF.notna()].copy()
    if len(df) < n0:
        print(f"Skipped {n0 - len(df)} rows without a video file or EF label")
    if limit:
        df = df.head(limit)
    return df


def hf_category(ef):
    return np.where(ef < 40, "reduced <40", np.where(ef < 50, "mildly reduced 40-49", "preserved >=50"))


def summarize(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = metrics(y, p)
    e = p - y
    r.update(n=int(len(y)), bias=float(e.mean()),
             LoA_low=float(e.mean() - 1.96 * e.std()), LoA_high=float(e.mean() + 1.96 * e.std()),
             slope=float(np.polyfit(y, p, 1)[0]) if len(y) > 2 else float("nan"))
    try:
        from sklearn.metrics import roc_auc_score
        lab = (y < 40).astype(int)
        if 0 < lab.sum() < len(lab):
            r["AUC_EF<40"] = float(roc_auc_score(lab, -p))
    except ImportError:
        pass
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--meanstd", default=r"D:\EchoNet\mean_std.json")
    ap.add_argument("--ckpt", default=r"D:\EchoNet\runs\baseline_seed0\best.pt")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\baseline_seed0\pediatric")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    df = load_filelist(root, a.limit)

    if a.check:
        raw = pd.read_csv(root / "FileList.csv")
        print("Columns:", list(raw.columns))
        print(raw.head(3).to_string())
        print(f"\nUsable videos: {len(df)}")
        print("EF:", df.EF.describe().round(1).to_dict())
        print("Heart-failure categories:", pd.Series(hf_category(df.EF.values)).value_counts().to_dict())
        cap = cv2.VideoCapture(str(df.path.iloc[0])); ok, fr = cap.read(); cap.release()
        print("First video frame shape:", None if not ok else fr.shape)
        return

    device = torch.device("cuda")
    ms = json.loads(Path(a.meanstd).read_text())
    mean, std = np.array(ms["mean"], np.float32), np.array(ms["std"], np.float32)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    dl = DataLoader(EchoDataset(df, mean, std, test_stride=a.stride), batch_size=1, num_workers=a.workers)
    ys, ps, idx = evaluate(model, dl, device)
    pred = df.iloc[idx].drop(columns="path").copy()
    pred["EF_pred"] = ps
    pred.to_csv(out / "pediatric_predictions.csv", index=False)

    y, p = pred.EF.values, pred.EF_pred.values
    res = {"settings": {k: str(v) for k, v in vars(a).items()}, "overall": summarize(y, p), "by_EF_range": {}}
    rng = pd.cut(pred.EF, [-1e-9, 30, 40, 50, 60, 101], right=False, labels=["<30", "30-39", "40-49", "50-59", ">=60"])
    for g, sub in pred.groupby(rng, observed=True):
        e = sub.EF_pred - sub.EF
        res["by_EF_range"][str(g)] = dict(n=int(len(sub)), bias=float(e.mean()), MAE=float(e.abs().mean()))
    if "Age" in pred.columns:
        res["by_age"] = {}
        ages = pd.cut(pred.Age, [-1e-9, 1, 5, 12, 200], right=False, labels=["<1 y", "1-4 y", "5-11 y", ">=12 y"])
        for g, sub in pred.groupby(ages, observed=True):
            res["by_age"][str(g)] = summarize(sub.EF, sub.EF_pred) if len(sub) > 2 else {"n": int(len(sub))}
    t, q = hf_category(y), hf_category(p)
    res["hf_category"] = {
        "agreement": float((t == q).mean()),
        "reduced_called_preserved": int(((t == "reduced <40") & (q == "preserved >=50")).sum()),
        "n_reduced": int((t == "reduced <40").sum()),
        "confusion": pd.crosstab(pd.Series(t, name="true"), pd.Series(q, name="pred")).to_dict(),
    }
    (out / "pediatric_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=2, default=str))


if __name__ == "__main__":
    main()
