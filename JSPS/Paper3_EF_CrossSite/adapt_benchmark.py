r"""
Site-adaptation benchmark (JSPS Paper 3): how much does each adaptation method fix cross-site EF error,
and what does it cost in labels, trainable parameters and time?

For every checkpoint (model trained on EchoNet-Dynamic) and every random split of the target dataset:
  test set  = a fixed half of the patients (never used for adaptation in that split)
  pool      = the other half; the first k pool patients are the "labeled" adaptation patients
Methods (labels needed):
  source               0   model as is
  bn_adapt             0   re-estimate BatchNorm statistics on unlabeled target scans
  ssa                  0   SSA test-time adaptation (Adachi et al., ICLR 2025): align target features with the
                           source feature subspace by updating BatchNorm affine parameters, unlabeled target scans
  offset               k   a single offset (mean error on the k patients) added to the source model's predictions
  linear               k   slope + offset on the source model's predictions
  bn_adapt+linear      k   bn_adapt, then linear
  ssa+linear           k   ssa, then linear
  last_layer           k   ridge regression of the final layer on frozen features, pulled towards the original layer
  adapter              k   small residual 1x1x1 bottleneck adapters after every residual block + final layer
  full_ft              k   all weights
Unlabeled 0-label methods adapt on all target scans (labels never used). Expensive methods (adapter, full_ft)
use fewer splits than the cheap ones.

Uncertainty: each video gets 4 phase-shifted clips; sigma = spread of the 4 clip predictions.
Normalized split conformal 90% intervals use the k labeled patients as calibration (leave-one-out residuals for
methods fitted on them); not computed for adapter / full_ft. Referral: error after removing the 20% of test scans
with the largest sigma.

Usage (notebook 1, from D:\EchoNet):
  python adapt_benchmark.py --dataset camus
  python adapt_benchmark.py --dataset pediatric --methods source,bn_adapt,ssa,linear,bn_adapt+linear,ssa+linear,last_layer
Results are appended to --out (one row per checkpoint x method x k x split) and the script resumes: re-running
skips rows already done.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from echonet_baseline import build_model, load_video

CHEAP = ["source", "bn_adapt", "ssa", "offset", "linear", "bn_adapt+linear", "ssa+linear", "last_layer"]
EXPENSIVE = ["adapter", "full_ft"]
DEFAULT_CKPTS = [rf"D:\EchoNet\runs\{n}\best.pt" for n in
                 ("baseline_seed0", "baseline_seed1", "baseline_seed2",
                  "physaug_all_seed0", "physaug_all_seed1", "physaug_all_seed2")]
FRAMES, PERIOD, NCLIP = 32, 2, 4


# ======================================================================= data -> cached clips
NORM = {}


def as_u8(v):
    """grey clip (T, H, W) 0-255 -> uint8 (stored compactly; normalized on the GPU in to_input)."""
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def to_input(batch_u8, device):
    """uint8 (B, T, H, W) -> normalized float (B, 3, T, H, W) on device, EchoNet training mean/std."""
    x = batch_u8.to(device).float()[:, None].expand(-1, 3, -1, -1, -1)
    return (x - NORM["mean"]) / NORM["std"]


def clips_from_video(v, n=NCLIP):
    """n evenly spaced 32-frame clips (period 2) from a long-enough video (T, H, W)."""
    need = FRAMES * PERIOD
    if len(v) < need:
        v = np.concatenate([v, np.zeros((need - len(v), *v.shape[1:]), v.dtype)])
    starts = np.linspace(0, len(v) - need, n).round().astype(int)
    return [v[s + PERIOD * np.arange(FRAMES)] for s in starts]


def camus_items(root, limit=None):
    from camus_eval import get_float, load_nifti, monoplane_ef, read_cfg, resample_time, to_echonet_frames
    a = SimpleNamespace(no_transpose=False, flip_ud=False, flip_lr=False)
    pdirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("patient"))[:limit]
    items = []
    for p in pdirs:
        cfg = read_cfg(p / "Info_4CH.cfg")
        seq, rs, cs = load_nifti(p / f"{p.name}_4CH_half_sequence.nii.gz", a)
        v = resample_time(to_echonet_frames(seq, rs, cs), get_float(cfg, "FrameRate", "FPS"))
        cyc = np.concatenate([v, v[-2:0:-1]]) if len(v) > 2 else v          # ED->ES->ED, as camus_eval
        reps = int(np.ceil((FRAMES * PERIOD + len(cyc)) / len(cyc))) + 1
        long = np.concatenate([cyc] * reps)[:FRAMES * PERIOD + len(cyc)]     # room for phase-shifted clips
        items.append(dict(id=p.name, y=get_float(cfg, "EF", "LVef"), quality=cfg.get("ImageQuality", "NA"),
                          y2=monoplane_ef(p, p.name, a), frames=clips_from_video(long)))
    return items


def pediatric_items(root, limit=None):
    root = Path(root)
    df = pd.read_csv(root / "FileList.csv")
    df = df[df.EF.notna()]
    items = []
    for r in df.itertuples():
        f = str(r.FileName)
        path = root / "Videos" / (f if f.lower().endswith(".avi") else f + ".avi")
        if not path.exists():
            continue
        v = load_video(path)[0]
        items.append(dict(id=f, y=float(r.EF), quality=str(getattr(r, "Age", "NA")), y2=np.nan,
                          frames=clips_from_video(v)))
        if limit and len(items) >= limit:
            break
    return items


def echonet_source_clips(root, n=500, seed=0):
    fl = pd.read_csv(Path(root) / "FileList.csv")
    fl = fl[(fl.Split == "TRAIN") & (fl.FrameHeight == 112) & (fl.FrameWidth == 112)].sample(
        min(n, (fl.Split == "TRAIN").sum()), random_state=seed)
    out = []
    for f in fl.FileName.astype(str):
        v = load_video(Path(root) / "Videos" / (f if f.endswith(".avi") else f + ".avi"))[0]
        out.append(as_u8(clips_from_video(v, 1)[0]))
    return np.stack(out)


# ======================================================================= model helpers
def features(model, x):
    x = model.stem(x)
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return model.avgpool(x).flatten(1)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_all(model, X, device, bs=16):
    """X (N videos, NCLIP, T, H, W) uint8 -> per-video mean prediction, sigma, and features (mean)."""
    model.eval()
    flat = torch.from_numpy(X.reshape(-1, *X.shape[2:]))
    preds, feats = [], []
    for i in range(0, len(flat), bs):
        x = to_input(flat[i:i + bs], device)
        with autocast(device):
            f = features(model, x).float()
            preds.append(model.fc(f).float().view(-1).cpu())
            feats.append(f.cpu())
    p = torch.cat(preds).view(len(X), -1).numpy()
    f = torch.cat(feats).view(len(X), X.shape[1], -1).mean(1).numpy()
    return p.mean(1), p.std(1), f


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]


@torch.no_grad()
def bn_adapt(model, X, device, bs=16):
    for m in bn_layers(model):
        m.reset_running_stats(); m.momentum = None                    # cumulative average over target batches
    model.train()
    flat = torch.from_numpy(X.reshape(-1, *X.shape[2:]))
    perm = torch.randperm(len(flat))
    for i in range(0, len(flat), bs):
        with autocast(device):
            features(model, to_input(flat[perm[i:i + bs]], device))
    for m in bn_layers(model):
        m.momentum = 0.1
    model.eval()
    return model


def source_stats(model, S, device, K=64, bs=16):
    _, _, F = predict_all(model, S[:, None], device, bs)
    mu = F.mean(0)
    cov = np.cov(F - mu, rowvar=False)
    lam, vec = np.linalg.eigh(cov)
    idx = np.argsort(lam)[::-1][:K]
    lam, V = lam[idx], vec[:, idx]
    w = model.fc.weight.detach().float().cpu().numpy().ravel()
    alpha = 1 + np.abs(w @ V)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(mu=t(mu), V=t(V), lam=t(np.maximum(lam, 1e-6)), alpha=t(alpha))


def ssa_adapt(model, X, stats, device, epochs=2, bs=10, lr=1e-4):
    """SSA (Adachi et al., ICLR 2025): symmetric KL between source and target Gaussians in the weighted
    source PCA subspace; only BatchNorm affine parameters are updated; BatchNorm uses target batch statistics."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in bn_layers(model):
        m.weight.requires_grad_(True); m.bias.requires_grad_(True)
        params += [m.weight, m.bias]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    mu, V, lam, alpha = (stats[k].to(device) for k in ("mu", "V", "lam", "alpha"))
    flat = torch.from_numpy(X.reshape(-1, *X.shape[2:]))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(flat))
        for i in range(0, len(flat) - 1, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            with autocast(device):
                f = features(model, to_input(flat[idx], device)).float()
            p = (f - mu) @ V
            m, v = p.mean(0), p.var(0) + 1e-6
            kl1 = 0.5 * (torch.log(lam / v) + (v + m ** 2) / lam - 1)
            kl2 = 0.5 * (torch.log(v / lam) + (lam + m ** 2) / v - 1)
            loss = (alpha * (kl1 + kl2)).sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


class Adapter3d(nn.Module):
    def __init__(self, c, r):
        super().__init__()
        self.down = nn.Conv3d(c, r, 1)
        self.up = nn.Conv3d(r, c, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


def add_adapters(model):
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for i, blk in enumerate(layer):
            c = blk.conv2[1].num_features
            layer[i] = nn.Sequential(blk, Adapter3d(c, max(8, c // 16)))
    return model


def finetune(model, Xk, yk, device, mode, steps=150, bs=8):
    """Train on the k labeled patients' clips. BatchNorm stays in eval mode (small batches)."""
    if mode == "adapter":
        model = add_adapters(model).to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        train = [p for n, p in model.named_parameters() if ".1.down" in n or ".1.up" in n or n.startswith("fc.")]
        for p in train:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(train, lr=1e-3, weight_decay=1e-4)
    else:
        train = list(model.parameters())
        opt = torch.optim.AdamW(train, lr=1e-5, weight_decay=1e-4)
    n_train = sum(p.numel() for p in train)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    flat = torch.from_numpy(Xk.reshape(-1, *Xk.shape[2:]))
    yflat = torch.from_numpy(np.repeat(yk, Xk.shape[1]).astype(np.float32))
    model.eval()
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, len(flat), (min(bs, len(flat)),), generator=g)
        with autocast(device):
            loss = ((model(to_input(flat[idx], device)).float().view(-1) - yflat[idx].to(device)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    return model, n_train


# ======================================================================= fitting + metrics
def fit_linear(p, y):
    if len(p) < 2 or np.std(p) < 1e-6:
        return 1.0, float(np.mean(y) - np.mean(p))
    s, i = np.polyfit(p, y, 1)
    return float(s), float(i)


def loo_linear(p, y):
    out = np.empty(len(p))
    for j in range(len(p)):
        m = np.arange(len(p)) != j
        s, i = fit_linear(p[m], y[m])
        out[j] = s * p[j] + i
    return out


def fit_ridge(F, y, w0, b0, lam=10.0):
    """Last layer refit: min ||F w + b - y||^2 + lam ||w - w0||^2 (shrinks towards the original layer)."""
    r = y - (F @ w0 + b0)                         # what the original layer gets wrong
    Fc, rc = F - F.mean(0), r - r.mean()
    dw = np.linalg.solve(Fc.T @ Fc + lam * np.eye(F.shape[1]), Fc.T @ rc)
    w = w0 + dw
    return w, float(np.mean(y - F @ w))


def loo_ridge(F, y, w0, b0, lam):
    out = np.empty(len(y))
    for j in range(len(y)):
        m = np.arange(len(y)) != j
        w, b = fit_ridge(F[m], y[m], w0, b0, lam)
        out[j] = F[j] @ w + b
    return out


def conformal(cal_pred, cal_y, cal_sig, test_pred, test_sig, alpha=0.1):
    k = len(cal_y)
    if k == 0:
        return np.nan, np.nan, np.nan
    tau = np.median(cal_sig) + 1e-3
    r = np.abs(cal_y - cal_pred) / (cal_sig + tau)
    lvl = math.ceil((k + 1) * (1 - alpha)) / k
    if lvl > 1:
        return np.inf, np.inf, np.nan
    q = np.quantile(r, lvl, method="higher")
    half = q * (test_sig + tau)
    return q, half, None


def score(y, p, sig, half=None, refer=0.2):
    e = p - y
    r = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
             R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
             slope=float(np.polyfit(y, p, 1)[0]) if np.std(p) > 1e-9 else 0.0)
    try:
        from sklearn.metrics import roc_auc_score
        lab = (y < 40).astype(int)
        r["AUC_EF<40"] = float(roc_auc_score(lab, -p)) if 0 < lab.sum() < len(lab) else np.nan
    except ImportError:
        r["AUC_EF<40"] = np.nan
    keep = sig <= np.quantile(sig, 1 - refer)
    r["MAE_after_20pct_referral"] = float(np.abs(e[keep]).mean())
    cat = lambda x: np.digitize(x, [40, 50])
    r["HF_category_agreement"] = float((cat(y) == cat(p)).mean())
    red = y < 40
    r["reduced_called_preserved"] = float(((p >= 50) & red).sum() / max(red.sum(), 1))
    if half is not None and np.all(np.isfinite(half)):
        r["coverage90"] = float((np.abs(e) <= half).mean())
        r["width90"] = float(2 * np.mean(half))
    else:
        r["coverage90"], r["width90"] = np.nan, np.nan
    return r


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["camus", "pediatric"], default="camus")
    ap.add_argument("--camus", default=r"D:\CAMUS\database_nifti")
    ap.add_argument("--pediatric", default=r"D:\EchoNet - Pediatric\pediatric_echo_avi\A4C")
    ap.add_argument("--echonet", default=r"D:\EchoNet\EchoNet-Dynamic")
    ap.add_argument("--meanstd", default=r"D:\EchoNet\mean_std.json")
    ap.add_argument("--ckpts", default=",".join(DEFAULT_CKPTS))
    ap.add_argument("--methods", default=",".join(CHEAP + EXPENSIVE))
    ap.add_argument("--ks", default="5,10,20,50,100")
    ap.add_argument("--splits_cheap", type=int, default=20)
    ap.add_argument("--splits_ft", type=int, default=5)
    ap.add_argument("--ft_steps", type=int, default=150)
    ap.add_argument("--ssa_epochs", type=int, default=2)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--n_source", type=int, default=500)
    ap.add_argument("--cpu_threads", type=int, default=2, help="PyTorch CPU threads (power saving)")
    ap.add_argument("--bs", type=int, default=16, help="clips per GPU batch for prediction / statistics")
    ap.add_argument("--export", default=None,
                    help="folder: only save per-patient predictions / features of the 0-label models as .npz "
                         "(for cheap_analysis.py) and skip the per-split methods")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="patients/videos, for quick tests")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(a.cpu_threads)          # power saving: limit CPU threads (OpenCV already 1)
    out = Path(a.out or rf"D:\EchoNet\runs\adapt_benchmark_{a.dataset}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        d0 = pd.read_csv(out)
        done = set(zip(d0.ckpt, d0.method, d0.k, d0.split))
    methods = a.methods.split(",")
    ks = [int(k) for k in a.ks.split(",")]
    ms = json.loads(Path(a.meanstd).read_text())
    mean, std = np.array(ms["mean"], np.float32), np.array(ms["std"], np.float32)

    t0 = time.time()
    print(f"Loading {a.dataset} ...", flush=True)
    items = camus_items(a.camus, a.limit) if a.dataset == "camus" else pediatric_items(a.pediatric, a.limit)
    X = np.stack([np.stack([as_u8(c) for c in it["frames"]]) for it in items])   # N, NCLIP, T, H, W uint8
    for it in items:
        it.pop("frames")
    y = np.array([it["y"] for it in items], np.float64)
    N = len(y)
    print(f"{N} videos cached ({X.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f} s", flush=True)
    NORM["mean"] = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    NORM["std"] = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    S = echonet_source_clips(a.echonet, a.n_source) if ("ssa" in a.methods) else None

    n_splits = max(a.splits_cheap, a.splits_ft)
    splits = []
    for s in range(n_splits):
        perm = np.random.default_rng(1000 + s).permutation(N)
        n_test = N // 2
        splits.append((perm[:n_test], perm[n_test:]))                  # test, adaptation pool
    ks = [k for k in ks if k <= len(splits[0][1])]

    rows_buffer = []

    def emit(row):
        rows_buffer.append(row)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)

    def needed(tag):
        for s_ in range(n_splits):
            for m_ in methods:
                if s_ >= (a.splits_ft if m_ in EXPENSIVE else a.splits_cheap):
                    continue
                for k_ in ([0] if m_ in ("source", "bn_adapt", "ssa") else ks):
                    if (tag, m_, k_, s_) not in done:
                        return True
        return False

    exp_dir = Path(a.export) if a.export else None
    if exp_dir:
        exp_dir.mkdir(parents=True, exist_ok=True)
    for ck in a.ckpts.split(","):
        tag = Path(ck).parent.name
        if exp_dir and (exp_dir / f"{a.dataset}_{tag}.npz").exists():
            print(f"=== {tag}: export exists, skipped ===", flush=True)
            continue
        if not exp_dir and not needed(tag):
            print(f"=== {tag}: all results already done, skipped ===", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        base = build_model(pretrained=False).to(device)
        base.load_state_dict(torch.load(ck, map_location=device))
        base.eval()
        w0 = base.fc.weight.detach().float().cpu().numpy().ravel()
        b0 = float(base.fc.bias.detach().float().cpu())

        # ---- 0-label models, adapted once on all unlabeled target scans
        preds = {}
        t = time.time(); preds["source"] = (*predict_all(base, X, device, a.bs), time.time() - t, 0)
        if any(m.startswith("bn_adapt") for m in methods):
            t = time.time(); mdl = bn_adapt(copy.deepcopy(base), X, device, a.bs)
            preds["bn_adapt"] = (*predict_all(mdl, X, device, a.bs), time.time() - t, 0); del mdl
        if any(m.startswith("ssa") for m in methods):
            t = time.time(); stats = source_stats(base, S, device, bs=a.bs)
            mdl = ssa_adapt(copy.deepcopy(base), X, stats, device, a.ssa_epochs, bs=min(a.bs, 10))
            n_bn = sum(m.weight.numel() + m.bias.numel() for m in bn_layers(mdl))
            preds["ssa"] = (*predict_all(mdl, X, device, a.bs), time.time() - t, n_bn); del mdl
        print(f"  0-label adaptations done ({time.time() - t0:.0f} s elapsed)", flush=True)
        if exp_dir:
            save = dict(ids=np.array([it["id"] for it in items]), y=y,
                        y2=np.array([it["y2"] for it in items], np.float64),
                        quality=np.array([it["quality"] for it in items]), w0=w0, b0=b0,
                        p_source=preds["source"][0], s_source=preds["source"][1], F_source=preds["source"][2])
            for m in ("bn_adapt", "ssa"):
                if m in preds:
                    save[f"p_{m}"], save[f"s_{m}"] = preds[m][0], preds[m][1]
                    save[f"secs_{m}"] = preds[m][3]
            np.savez_compressed(exp_dir / f"{a.dataset}_{tag}.npz", **save)
            print(f"  saved {exp_dir / f'{a.dataset}_{tag}.npz'}", flush=True)
            continue

        for s, (te, pool) in enumerate(splits):
            for m in methods:
                expensive = m in EXPENSIVE
                if s >= (a.splits_ft if expensive else a.splits_cheap):
                    continue
                for k in ([0] if m in ("source", "bn_adapt", "ssa") else ks):
                    if (tag, m, k, s) in done:
                        continue
                    cal = pool[:k]
                    half = None
                    t = time.time()
                    if m in ("source", "bn_adapt", "ssa"):
                        p, sig, _, secs, npar = preds[m]
                        cal_p = p[pool[:20]]                              # conformal: 20 labeled patients
                        _, half, _ = conformal(cal_p, y[pool[:20]], sig[pool[:20]], p[te], sig[te])
                        pt, st = p[te], sig[te]
                    elif m == "offset":
                        p, sig, _, secs, npar = preds["source"]
                        off = float(np.mean(y[cal] - p[cal]))
                        pt, st = p[te] + off, sig[te]
                        loo = np.array([p[cal][j] + np.mean(np.delete(y[cal] - p[cal], j)) for j in range(k)])
                        _, half, _ = conformal(loo, y[cal], sig[cal], pt, st)
                        secs += time.time() - t; npar = 1
                    elif m in ("linear", "bn_adapt+linear", "ssa+linear"):
                        src = {"linear": "source", "bn_adapt+linear": "bn_adapt", "ssa+linear": "ssa"}[m]
                        p, sig, _, secs, npar = preds[src]
                        sl, ic = fit_linear(p[cal], y[cal])
                        pt, st = sl * p[te] + ic, abs(sl) * sig[te]
                        _, half, _ = conformal(loo_linear(p[cal], y[cal]), y[cal], abs(sl) * sig[cal], pt, st)
                        secs += time.time() - t; npar += 2
                    elif m == "last_layer":
                        p, sig, F, secs, _ = preds["source"]
                        w, b = fit_ridge(F[cal], y[cal], w0, b0, a.ridge_lambda)
                        pt, st = F[te] @ w + b, sig[te]
                        _, half, _ = conformal(loo_ridge(F[cal], y[cal], w0, b0, a.ridge_lambda), y[cal],
                                               sig[cal], pt, st)
                        secs += time.time() - t; npar = len(w0) + 1
                    else:
                        mdl, npar = finetune(copy.deepcopy(base), X[cal], y[cal], device, m, a.ft_steps, bs=min(a.bs, 8))
                        pt, st, _ = predict_all(mdl, X[te], device, a.bs)
                        del mdl
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        secs = time.time() - t
                    row = dict(dataset=a.dataset, ckpt=tag, method=m, k=k, split=s, n_test=len(te),
                               trainable_params=int(npar), adapt_seconds=round(float(secs), 2))
                    row.update(score(y[te], pt, st, half))
                    emit(row)
            if s % 5 == 0:
                print(f"  split {s} done ({time.time() - t0:.0f} s elapsed)", flush=True)

    if out.exists():
        d = pd.read_csv(out)
        summ = (d.groupby(["method", "k"])[["MAE", "bias", "slope", "AUC_EF<40", "coverage90", "width90",
                                            "MAE_after_20pct_referral"]].mean().round(2))
        summ.to_csv(out.with_name(out.stem + "_summary.csv"))
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(summ)


if __name__ == "__main__":
    main()
