r"""
CPU analysis of site adaptation from exported predictions (no GPU, seconds to minutes).
Input: .npz files written by  adapt_benchmark.py --dataset <camus|pediatric> --export <folder>
       (one per model: per-patient EF labels, source predictions + clip spread + 512-d features,
        and the BatchNorm-re-estimated and SSA-adapted predictions).

For each model x split x number of labels k x patient-selection strategy x method:
  splits     identical to adapt_benchmark.py (seed 1000+s; first half = test, second half = adaptation pool)
  selection  random            k random pool patients (as in adapt_benchmark.py)
             pred_stratified   k pool patients spread evenly over the model's predicted EF (no labels needed
                               to choose them: realistic)
             label_stratified  k pool patients balanced over true EF <40 / 40-49 / >=50 (needs records of
                               EF category; reference)
  methods    source, bn_adapt, ssa (0 labels); offset, linear, last_layer, bn_adapt+linear, ssa+linear
  uncertainty  split conformal 90%: 'plain' (one interval width from leave-one-out residuals) and
             'mondrian' (separate widths for predicted EF <50 and >=50 when >= 9 calibration patients in
             each group, else plain); coverage overall and among reduced-EF (<40) test patients
             'norm'  normalized conformal: residuals divided by the model's EchoNet-validation error profile
                     g(predicted EF) (from <runs_dir>/<model>/val_predictions.csv), interval half-width q x g(p)
             'asym'  as 'norm' but with separate upper and lower quantiles (5% each side) of the signed errors,
                     because target errors are directional (low EF over-estimated)
  referral   MAE after removing the 20% of test scans with the largest clip-to-clip spread; AUROC of the
             spread for detecting errors above 10 EF points

Usage:
  python cheap_analysis.py --npz_dir D:\EchoNet\runs\adapt_preds --dataset camus
  python cheap_analysis.py --npz_dir D:\EchoNet\runs\adapt_preds --dataset pediatric
Outputs (in --npz_dir): cheap_<dataset>.csv (all rows) and cheap_<dataset>_summary.csv
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

METHODS0 = ["source", "bn_adapt", "ssa"]
METHODSK = ["offset", "linear", "last_layer", "bn_adapt+linear", "ssa+linear"]
SELECTIONS = ["random", "pred_stratified", "label_stratified"]


# ---------------------------------------------------------------- fitting
def fit_linear(p, y):
    if len(p) < 2 or np.std(p) < 1e-6:
        return 1.0, float(np.mean(y - p))
    s, i = np.polyfit(p, y, 1)
    return float(s), float(i)


def fit_ridge_loo(F, y, w0, b0, lam):
    """Last-layer refit  min ||F w + b - y||^2 + lam ||w - w0||^2 (intercept not penalized).
    Exact leave-one-out predictions via the hat matrix of the penalized least-squares fit."""
    r = y - (F @ w0 + b0)                                   # correction the original layer needs
    X = np.hstack([np.ones((len(y), 1)), F])
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0.0
    G = np.linalg.solve(X.T @ X + P, X.T)                   # (d+1) x k
    beta = G @ r
    H = X @ G
    loo_res = (r - H @ r) / np.clip(1 - np.diag(H), 1e-9, None)
    return w0 + beta[1:], b0 + beta[0], y - loo_res


def fit_predict(method, cal, test, D, lam):
    """Return (test predictions, leave-one-out predictions for the calibration patients, slope scale)."""
    y = D["y"]
    if method == "offset":
        p = D["p_source"]
        off = np.mean(y[cal] - p[cal])
        loo = np.array([p[c] + np.mean(np.delete(y[cal] - p[cal], j)) for j, c in enumerate(cal)])
        return p[test] + off, loo, 1.0
    if method in ("linear", "bn_adapt+linear", "ssa+linear"):
        p = D[{"linear": "p_source", "bn_adapt+linear": "p_bn_adapt", "ssa+linear": "p_ssa"}[method]]
        s, i = fit_linear(p[cal], y[cal])
        loo = np.empty(len(cal))
        for j in range(len(cal)):
            m = np.arange(len(cal)) != j
            sj, ij = fit_linear(p[cal][m], y[cal][m])
            loo[j] = sj * p[cal][j] + ij
        return s * p[test] + i, loo, abs(s)
    if method == "last_layer":
        F, w0, b0 = D["F_source"], D["w0"], float(D["b0"])
        w, b, loo = fit_ridge_loo(F[cal], y[cal], w0, b0, lam)
        return F[test] @ w + b, loo, 1.0
    raise ValueError(method)


# ---------------------------------------------------------------- selection
def select(strategy, pool, k, D, rng):
    if strategy == "random":
        return pool[:k]
    if strategy == "pred_stratified":
        order = pool[np.argsort(D["p_source"][pool], kind="stable")]
        idx = np.unique(np.linspace(0, len(order) - 1, k).round().astype(int))
        return order[idx]
    if strategy == "label_stratified":
        y = D["y"][pool]
        groups = [pool[y < 40], pool[(y >= 40) & (y < 50)], pool[y >= 50]]
        take, chosen = [k // 3 + (1 if i < k % 3 else 0) for i in range(3)], []
        spare = []
        for g, t in zip(groups, take):
            g = rng.permutation(g)
            chosen += list(g[:t]); spare += list(g[t:])
        if len(chosen) < k:
            chosen += list(rng.permutation(np.array(spare, int))[:k - len(chosen)])
        return np.array(chosen, int)
    raise ValueError(strategy)


# ---------------------------------------------------------------- conformal + metrics
def conf_q(res, alpha=0.1):
    k = len(res)
    if k == 0:
        return np.inf
    lvl = math.ceil((k + 1) * (1 - alpha)) / k
    return np.inf if lvl > 1 else float(np.quantile(res, lvl, method="higher"))


def conf_q_one_sided(s, alpha=0.1):
    """Upper (1 - alpha/2) split-conformal quantile of signed scores s (one side of a two-sided interval)."""
    return conf_q(s, alpha / 2) if len(s) else np.inf


def difficulty_fn(runs_dir, tag, nbins=10):
    """Error-vs-predicted-EF profile from the model's own EchoNet validation predictions:
    mean |error| in 10 equal-count bins of predicted EF, linearly interpolated (flat beyond the ends),
    scaled to mean 1. Returns None if <runs_dir>/<tag>/val_predictions.csv is missing."""
    f = Path(runs_dir) / tag / "val_predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    p, e = d.EF_pred.values, np.abs(d.EF_pred.values - d.EF_true.values)
    order = np.argsort(p)
    chunks = np.array_split(order, nbins)
    xc = np.array([np.median(p[c]) for c in chunks])
    yc = np.array([e[c].mean() for c in chunks])
    yc = yc / e.mean()
    return lambda q: np.interp(q, xc, yc)


def metrics(y, p, sig, half=None, half_m=None, extra=None):
    e = p - y
    r = dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(),
             R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
             slope=np.polyfit(y, p, 1)[0] if np.std(p) > 1e-9 else 0.0)
    red = y < 40
    try:
        from sklearn.metrics import roc_auc_score
        r["AUC_EF<40"] = roc_auc_score(red.astype(int), -p) if 0 < red.sum() < len(y) else np.nan
        big = (np.abs(e) > 10).astype(int)
        r["sigma_AUROC_err>10"] = roc_auc_score(big, sig) if 0 < big.sum() < len(y) else np.nan
    except ImportError:
        r["AUC_EF<40"] = r["sigma_AUROC_err>10"] = np.nan
    r["MAE_reduced"] = np.abs(e[red]).mean() if red.any() else np.nan
    r["sens_reduced"] = (p[red] < 40).mean() if red.any() else np.nan
    r["reduced_called_preserved"] = (p[red] >= 50).mean() if red.any() else np.nan
    cat = lambda x: np.digitize(x, [40, 50])
    r["HF_category_agreement"] = (cat(y) == cat(p)).mean()
    keep = sig <= np.quantile(sig, 0.8)
    r["MAE_after_20pct_referral"] = np.abs(e[keep]).mean()
    for name, h in (("plain", half), ("mondrian", half_m)):
        if h is not None and np.all(np.isfinite(h)):
            cov = np.abs(e) <= h
            r[f"coverage90_{name}"], r[f"width90_{name}"] = cov.mean(), 2 * np.mean(h)
            r[f"coverage90_{name}_reduced"] = cov[red].mean() if red.any() else np.nan
        else:
            r[f"coverage90_{name}"] = r[f"width90_{name}"] = r[f"coverage90_{name}_reduced"] = np.nan
    # extra intervals: name -> (lower width, upper width), interval = [p - lo, p + hi]
    for name, (lo, hi) in (extra or {}).items():
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            cov = (y >= p - lo) & (y <= p + hi)
            r[f"coverage90_{name}"], r[f"width90_{name}"] = cov.mean(), np.mean(lo + hi)
            r[f"coverage90_{name}_reduced"] = cov[red].mean() if red.any() else np.nan
        else:
            r[f"coverage90_{name}"] = r[f"width90_{name}"] = r[f"coverage90_{name}_reduced"] = np.nan
    return {k: float(v) for k, v in r.items()}


def family(tag):
    return re.sub(r"_seed\d+$", "", tag)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--dataset", choices=["camus", "pediatric"], default="camus")
    ap.add_argument("--ks", default="5,10,20,50,100")
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--ridge_lambda", type=float, default=10.0)
    ap.add_argument("--selections", default=",".join(SELECTIONS))
    ap.add_argument("--runs_dir", default=r"D:\EchoNet\runs",
                    help="folder with <model>\\val_predictions.csv (EchoNet error profile for norm/asym intervals)")
    a = ap.parse_args()
    d = Path(a.npz_dir)
    files = sorted(d.glob(f"{a.dataset}_*.npz"))
    if not files:
        raise SystemExit(f"No {a.dataset}_*.npz files in {d}")
    ks = [int(k) for k in a.ks.split(",")]
    sels = a.selections.split(",")
    rows = []
    for f in files:
        tag = f.stem[len(a.dataset) + 1:]
        D = dict(np.load(f, allow_pickle=True))
        y = D["y"]; N = len(y)
        has = {m: f"p_{m}" in D for m in ("bn_adapt", "ssa")}
        g = difficulty_fn(a.runs_dir, tag)
        print(f"{tag}: {N} patients | EchoNet error profile: {'found' if g else 'MISSING (norm/asym skipped)'}",
              flush=True)
        for s in range(a.splits):
            perm = np.random.default_rng(1000 + s).permutation(N)
            te, pool = perm[:N // 2], perm[N // 2:]
            rng = np.random.default_rng(2000 + s)
            for m in METHODS0:
                if m != "source" and not has[m]:
                    continue
                p, sig = D[f"p_{m}"], D[f"s_{m}"]
                rows.append(dict(ckpt=tag, family=family(tag), split=s, k=0, selection="none", method=m,
                                 **metrics(y[te], p[te], sig[te])))
            for k in ks:
                if k > len(pool):
                    continue
                for sel in sels:
                    cal = select(sel, pool, k, D, rng)
                    for m in METHODSK:
                        if (m.startswith("bn_adapt") and not has["bn_adapt"]) or (m.startswith("ssa") and not has["ssa"]):
                            continue
                        pt, loo, scale = fit_predict(m, cal, te, D, a.ridge_lambda)
                        res = np.abs(y[cal] - loo)
                        half = np.full(len(te), conf_q(res))
                        # Mondrian by predicted EF group (<50 vs >=50)
                        g_cal, g_te = loo < 50, pt < 50
                        if g_cal.sum() >= 9 and (~g_cal).sum() >= 9:
                            half_m = np.where(g_te, conf_q(res[g_cal]), conf_q(res[~g_cal]))
                        else:
                            half_m = half
                        extra = {}
                        if g is not None:
                            gc, gt = g(loo), g(pt)
                            # 'norm': normalized conformal, width scaled by the EchoNet error profile at the prediction
                            hn = conf_q(res / gc) * gt
                            extra["norm"] = (hn, hn)
                            # 'asym': normalized and asymmetric (separate upper / lower quantiles of signed errors)
                            sres = (y[cal] - loo) / gc
                            extra["asym"] = (conf_q_one_sided(-sres) * gt, conf_q_one_sided(sres) * gt)
                        sig_src = D["s_bn_adapt"] if m.startswith("bn_adapt") else D["s_ssa"] if m.startswith("ssa") else D["s_source"]
                        rows.append(dict(ckpt=tag, family=family(tag), split=s, k=k, selection=sel, method=m,
                                         **metrics(y[te], pt, scale * sig_src[te], half, half_m, extra)))
    df = pd.DataFrame(rows)
    out = d / f"cheap_{a.dataset}.csv"
    df.to_csv(out, index=False)
    cols = ["MAE", "bias", "slope", "AUC_EF<40", "MAE_reduced", "sens_reduced", "reduced_called_preserved",
            "HF_category_agreement", "coverage90_plain", "width90_plain", "coverage90_plain_reduced",
            "coverage90_mondrian", "width90_mondrian", "coverage90_mondrian_reduced",
            "coverage90_norm", "width90_norm", "coverage90_norm_reduced",
            "coverage90_asym", "width90_asym", "coverage90_asym_reduced",
            "MAE_after_20pct_referral", "sigma_AUROC_err>10"]
    cols = [c for c in cols if c in df.columns]
    summ = df.groupby(["family", "method", "selection", "k"])[cols].mean().round(3)
    summ.to_csv(d / f"cheap_{a.dataset}_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_columns", 30):
        show = ["MAE", "reduced_called_preserved", "coverage90_plain", "width90_plain", "coverage90_plain_reduced",
                "coverage90_norm", "width90_norm", "coverage90_norm_reduced",
                "coverage90_asym", "width90_asym", "coverage90_asym_reduced"]
        print(summ[[c for c in show if c in summ.columns]])
    print(f"\nSaved {out} and {d / f'cheap_{a.dataset}_summary.csv'}")


if __name__ == "__main__":
    main()
