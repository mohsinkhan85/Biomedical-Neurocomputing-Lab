r"""
EchoNet-Dynamic EF baseline: R(2+1)D-18 video regression, following Ouyang et al. (Nature 2020)
settings (32 frames, sampling period 2, Kinetics-400 pretraining, SGD lr 1e-4, 45 epochs,
LR step every 15 epochs, random 12-px pad-and-crop). Adapted for an 11 GB GPU (mixed precision,
smaller batch) and Windows.

Smoke test (few minutes):
    python echonet_baseline.py --data D:\EchoNet\EchoNet-Dynamic --out D:\EchoNet\runs\smoke --epochs 1 --limit 200
Full run:
    python echonet_baseline.py --data D:\EchoNet\EchoNet-Dynamic --out D:\EchoNet\runs\baseline_seed0 --seed 0

Re-running the same command resumes from the last finished epoch.
Outputs in --out: last.pt, best.pt, log.csv, val_predictions.csv, test_predictions.csv, results.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

# Power saving: one CPU thread per process (OpenCV otherwise uses every core in every data worker).
# Does not change results.
cv2.setNumThreads(1)


def _limit_threads(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


# ---------------------------------------------------------------- data
def load_video(path):
    """Return float32 array (C, T, H, W), RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read video: {path}")
    return np.stack(frames).astype(np.float32).transpose(3, 0, 1, 2)


def read_filelist(root, limit=None):
    df = pd.read_csv(root / "FileList.csv")
    n0 = len(df)
    # Exclude the 6 non-standard-resolution videos (768x1024 exports)
    df = df[(df.FrameHeight == 112) & (df.FrameWidth == 112)].copy()
    print(f"Excluded {n0 - len(df)} non-112x112 videos; {len(df)} remain")
    df["FileName"] = df.FileName.astype(str)
    df["path"] = [root / "Videos" / (f if f.endswith(".avi") else f + ".avi") for f in df.FileName]
    if limit:  # small subset per split for smoke tests
        df = df.groupby("Split", group_keys=False).head(limit)
    return df


def compute_mean_std(paths, n=200, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(list(paths), min(n, len(paths)))
    s, s2, cnt = np.zeros(3), np.zeros(3), 0
    for p in sample:
        v = load_video(p).reshape(3, -1)
        s += v.sum(1)
        s2 += (v ** 2).sum(1)
        cnt += v.shape[1]
    mean = s / cnt
    std = np.sqrt(s2 / cnt - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


class EchoDataset(Dataset):
    def __init__(self, df, mean, std, frames=32, period=2, train=False, pad=12, test_stride=16):
        self.paths = df.path.tolist()
        self.names = df.FileName.tolist()
        self.ef = df.EF.values.astype(np.float32)
        self.mean = mean[:, None, None, None]
        self.std = std[:, None, None, None]
        self.frames, self.period, self.train, self.pad = frames, period, train, pad
        self.test_stride = test_stride

    def __len__(self):
        return len(self.paths)

    def _prepare(self, i):
        v = (load_video(self.paths[i]) - self.mean) / self.std
        need = self.frames * self.period
        if v.shape[1] < need:  # pad short videos with zeros
            z = np.zeros((3, need - v.shape[1], *v.shape[2:]), np.float32)
            v = np.concatenate([v, z], 1)
        return v, need

    def _clip(self, v, start):
        return v[:, start + self.period * np.arange(self.frames)]

    def __getitem__(self, i):
        v, need = self._prepare(i)
        T = v.shape[1]
        if self.train:
            clip = self._clip(v, np.random.randint(0, T - need + 1))
            if self.pad:
                c, t, h, w = clip.shape
                p = np.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), np.float32)
                p[:, :, self.pad:self.pad + h, self.pad:self.pad + w] = clip
                y, x = np.random.randint(0, 2 * self.pad + 1, size=2)
                clip = p[:, :, y:y + h, x:x + w]
            return torch.from_numpy(np.ascontiguousarray(clip)), self.ef[i]
        # evaluation: all clips at a fixed stride, predictions averaged per video
        starts = np.arange(0, T - need + 1, self.test_stride)
        clips = np.stack([self._clip(v, s) for s in starts])
        return torch.from_numpy(np.ascontiguousarray(clips)), self.ef[i], i


# ---------------------------------------------------------------- model / eval
def build_model(pretrained=True):
    w = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    m = torchvision.models.video.r2plus1d_18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m.fc.bias.data[0] = 55.6  # approx. mean EF, as in the original code
    return m


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


@torch.no_grad()
def evaluate(model, loader, device, chunk=16):
    model.eval()
    ys, ps, idx = [], [], []
    for clips, ef, i in loader:  # batch_size=1: clips (1, N, C, T, H, W)
        clips = clips[0]
        preds = []
        for k in range(0, len(clips), chunk):
            with torch.autocast("cuda", dtype=torch.float16):
                preds.append(model(clips[k:k + chunk].to(device, non_blocking=True)).float().view(-1))
        ps.append(torch.cat(preds).mean().item())
        ys.append(float(ef))
        idx.append(int(i))
    return ys, ps, idx


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="videos per split, for smoke tests")
    ap.add_argument("--val_stride", type=int, default=32)
    ap.add_argument("--test_stride", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = read_filelist(root, a.limit)
    tr, va, te = (df[df.Split == s] for s in ("TRAIN", "VAL", "TEST"))
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    ms_file = root.parent / "mean_std.json"  # training-set statistics only
    if ms_file.exists() and not a.limit:
        d = json.loads(ms_file.read_text()); mean, std = np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
    else:
        print("Computing mean/std from training videos ...")
        mean, std = compute_mean_std(tr.path, seed=a.seed)
        if not a.limit:
            ms_file.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    print("mean", mean.round(2), "std", std.round(2))

    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
              worker_init_fn=_limit_threads)
    dl_tr = DataLoader(EchoDataset(tr, mean, std, train=True), batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(EchoDataset(va, mean, std, test_stride=a.val_stride), batch_size=1, **kw)

    model = build_model(not a.no_pretrained).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.MSELoss()

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
                loss = loss_fn(model(x).float().view(-1), y)
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

    # final evaluation with the best checkpoint
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
