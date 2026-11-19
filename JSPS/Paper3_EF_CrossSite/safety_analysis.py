r"""
Package I (JSPS Paper 3): safety-aware recalibration. CPU only; reads the same .npz files as fusion_analysis.py
(which must be in the same folder) and writes only new files: safety_<dataset>.csv and safety_<dataset>_summary.csv.

Question: can the most accurate configuration (3 physics-augmented seeds + PanEcho + EchoPrime, 'combo') keep its
accuracy while protecting reduced-EF patients as well as the safest one (physics ensemble + offset, 'phys3_offset')?

Methods (k labelled patients; splits identical to cheap_analysis.py / fusion_analysis.py)
  combo_ridge        reference: ridge correction of the combination (as in fusion_analysis.py)
  phys3_offset       reference: physics ensemble + offset (safest so far)
  combo_wridge       weighted ridge: labelled patients with EF < 45 count --w times in the fit (w chosen by
                     leave-one-out among 1, 3, 10 using the weighted error, so it adapts to how many low-EF
                     patients are in the k)
  blend              lambda * combo_ridge + (1 - lambda) * phys3_offset, lambda in {0, .25, .5, .75, 1} chosen by
                     leave-one-out MAE on the k patients only
Safety flag (decision rule on top of any method): flag "possible reduced EF" when prediction - q < 40, where q is the
split-conformal 90% one-sided bound of over-estimation from the leave-one-out errors on the k patients. Reported:
sensitivity of the flag for true EF < 40, and the false-flag rate among true EF >= 50.

Usage (from D:\EchoNet, CPU):
  !"{sys.executable}" safety_analysis.py --dataset camus
  !"{sys.executable}" safety_analysis.py --dataset pediatric
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fusion_analysis as fa

LAMS = fa.LAMS
WEIGHTS = [1.0, 3.0, 10.0]
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]


def fit_wridge_loo(XL, r, wts, lams):
    """Weighted ridge with unpenalized intercept; exact weighted leave-one-out; returns best by weighted LOO error."""
    k, d = XL.shape
    X1 = np.hstack([np.ones((k, 1)), XL])
    W = wts[:, None]
    best = None
    for lam in lams:
        A = X1.T @ (W * X1)
        A[1:, 1:] += lam * np.eye(d)
        G = np.linalg.solve(A, (W * X1).T)                       # (d+1) x k
        beta = G @ r
        h = np.einsum("ij,ji->i", X1, G)
        loo_corr = r - (r - X1 @ beta) / np.clip(1 - h, 1e-9, None)
        err = np.sum(wts * np.abs(r - loo_corr)) / wts.sum()
        if best is None or err < best[0]:
            best = (err, beta, loo_corr)
    return best


def upper_q(x, alpha=0.1):
    """split-conformal one-sided (1 - alpha) quantile."""
    k = len(x)
    lvl = math.ceil((k + 1) * (1 - alpha)) / k
    return np.inf if lvl > 1 else float(np.quantile(x, lvl, method="higher"))


def metrics(y, pt, q_over):
    red, pres = y < 40, y >= 50
    flag = (pt - q_over) < 40 if np.isfinite(q_over) else np.zeros_like(pt, bool)
    e = pt - y
    return dict(MAE=float(np.abs(e).mean()), bias=float(e.mean()),
                MAE_reduced=float(np.abs(e[red]).mean()) if red.any() else np.nan,
                reduced_called_preserved=float((pt[red] >= 50).mean()) if red.any() else np.nan,
                flag_sensitivity_reduced=float(flag[red].mean()) if red.any() and np.isfinite(q_over) else np.nan,
                flag_rate_preserved=float(flag[pres].mean()) if pres.any() and np.isfinite(q_over) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--dataset", choices=["camus", "pediatric"], default="camus")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--pcs", type=int, default=32)
    a = ap.parse_args()
    ks = [int(k) for k in (a.ks or ("5,10,20,50,100,250" if a.dataset == "camus" else
                                    "5,10,20,50,100,250,500,1000")).split(",")]

    M = fa.load_models(a.npz_dir, a.dataset)
    A = fa.align(M, "baseline_seed0" if "baseline_seed0" in M else sorted(M)[0])
    phys = [f"physaug_all_seed{s}" for s in range(3)]
    fm = [t for t in ("panecho", "echoprime") if t in A]
    missing = [t for t in phys if t not in A]
    if missing:
        raise SystemExit(f"missing models: {missing}")
    combo = phys + fm
    print(f"{a.dataset}: {len(A[phys[0]]['y'])} patients; combination = {combo}", flush=True)

    X = np.hstack([fa.model_block(A[m]["F"], A[m]["p"], a.pcs) for m in combo])
    base_c = np.mean([A[m]["p"] for m in combo], axis=0)
    base_p = np.mean([A[m]["p"] for m in phys], axis=0)
    y = A[phys[0]]["y"]
    N = len(y)
    rows, t0 = [], time.time()
    for s in range(a.splits):
        perm = np.random.default_rng(1000 + s).permutation(N)
        te, pool = perm[:N // 2], perm[N // 2:]
        for k in ks:
            if k > len(pool):
                continue
            for sel in ("random", "pred_stratified"):
                cal = fa.select(sel, pool, k, base_c)
                yc = y[cal]
                # phys3 + offset
                rp = yc - base_p[cal]
                off = rp.mean()
                loo_p = base_p[cal] + np.array([np.delete(rp, j).mean() for j in range(k)])
                pt_p = base_p[te] + off
                # combo ridge
                rc = yc - base_c[cal]
                beta, loo_corr, _, _ = fa.fit_loo(X[cal], rc, np.zeros((X.shape[1], X.shape[1])), LAMS, [0.0])
                loo_c = base_c[cal] + loo_corr
                pt_c = base_c[te] + beta[0] + X[te] @ beta[1:]
                # weighted ridge (EF < 45 up-weighted), weight chosen by weighted LOO
                bw = None
                for w in WEIGHTS:
                    wts = np.where(yc < 45, w, 1.0)
                    err, b, lc = fit_wridge_loo(X[cal], rc, wts, LAMS)
                    crit = np.abs(rc - lc).mean() + np.abs(rc - lc)[yc < 45].mean() if (yc < 45).any() else np.abs(rc - lc).mean()
                    if bw is None or crit < bw[0]:
                        bw = (crit, b, lc, w)
                _, b, lc, w_sel = bw
                loo_w = base_c[cal] + lc
                pt_w = base_c[te] + b[0] + X[te] @ b[1:]
                # blend chosen by LOO MAE
                best_l = min(BLENDS, key=lambda L: np.abs(yc - (L * loo_c + (1 - L) * loo_p)).mean())
                loo_b = best_l * loo_c + (1 - best_l) * loo_p
                pt_b = best_l * pt_c + (1 - best_l) * pt_p
                for meth, pt, loo, extra in (("phys3_offset", pt_p, loo_p, {}), ("combo_ridge", pt_c, loo_c, {}),
                                             ("combo_wridge", pt_w, loo_w, {"weight": w_sel}),
                                             ("blend", pt_b, loo_b, {"blend_lambda": best_l})):
                    q = upper_q(loo - yc)
                    rows.append(dict(split=s, k=k, selection=sel, method=meth, **extra, **metrics(y[te], pt, q)))
        if s % 5 == 4:
            print(f"  split {s + 1}/{a.splits} ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.npz_dir) / f"safety_{a.dataset}.csv"
    df.to_csv(out, index=False)
    cols = ["MAE", "bias", "MAE_reduced", "reduced_called_preserved", "flag_sensitivity_reduced", "flag_rate_preserved",
            "blend_lambda", "weight"]
    summ = df.groupby(["method", "selection", "k"])[[c for c in cols if c in df.columns]].mean().round(3)
    summ.to_csv(Path(a.npz_dir) / f"safety_{a.dataset}_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 20):
        print(summ.xs("random", level="selection"))
    print(f"\nSaved {out} and safety_{a.dataset}_summary.csv")


if __name__ == "__main__":
    main()
