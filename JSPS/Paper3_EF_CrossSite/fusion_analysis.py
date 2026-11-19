r"""
Package F (JSPS Paper 3): can combining models and using the unlabelled scans of the new site give a large
gain with few labels?  CPU only, reads the .npz files written by adapt_benchmark.py --export,
panecho_export.py and echoprime_export.py (folder --npz_dir). Writes new files only:
fusion_<dataset>.csv (all rows) and fusion_<dataset>_summary.csv.

Configurations (sets of models whose features are combined)
  single models      baseline_seed{0,1,2}, physaug_all_seed{0,1,2}, panecho, echoprime
  FM2                panecho + echoprime
  phys_s+FM2         physaug_all_seed{s} + panecho + echoprime        (s = 0, 1, 2)
  base_s+FM2         baseline_seed{s} + panecho + echoprime           (control for phys_s+FM2)
  phys3, phys3+FM2   the 3 physics-augmented seeds (ensemble), and with both foundation models
For each configuration: base prediction = mean EF prediction of its models; features = for each model, its
embedding z-scored and reduced to --pcs principal components over all scans of the new site (unlabelled, so no
target labels are used), plus its own EF prediction.

Methods (k labelled patients, the rest of the split as in cheap_analysis.py)
  source     base prediction, no labels
  offset     base prediction + mean error on the k patients
  ridge      ridge correction of the base prediction from the features; penalty chosen by exact
             leave-one-out error on the k patients only
  lapridge   semi-supervised: ridge plus a graph-Laplacian penalty built from ALL scans of the new site
             (labelled and unlabelled), so patients with similar features get similar corrections
             (Laplacian-regularized least squares, Belkin et al. JMLR 2006); both penalties chosen by
             leave-one-out on the k labelled patients (the Laplacian weight 0 = plain ridge is one choice)
Targets: CAMUS is analysed against its biplane reference EF (target 'reference') and against the matching
apical 4-chamber EF computed from its masks (target 'a4c'); pediatric against its reference only.
Uncertainty: split conformal 90% intervals from leave-one-out residuals (as 'plain' in cheap_analysis.py).

Usage (notebook, from D:\EchoNet, after D3):
  !"{sys.executable}" fusion_analysis.py --dataset camus
  !"{sys.executable}" fusion_analysis.py --dataset pediatric
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

LAMS = [0.3, 3.0, 30.0, 300.0, 3000.0]
GAMMAS = [0.0, 0.1, 1.0, 10.0]


# ---------------------------------------------------------------- data
def load_models(npz_dir, dataset):
    files = sorted(Path(npz_dir).glob(f"{dataset}_*.npz"))
    if not files:
        raise SystemExit(f"No {dataset}_*.npz files in {npz_dir}")
    M = {}
    for f in files:
        D = dict(np.load(f, allow_pickle=True))
        M[f.stem[len(dataset) + 1:]] = D
    return M


def align(M, ref_tag):
    """Reorder every model's arrays to the patient order of ref_tag (= the order cheap_analysis.py splits on)."""
    ref = [str(i) for i in M[ref_tag]["ids"]]
    out = {}
    for tag, D in M.items():
        pos = {str(i): j for j, i in enumerate(D["ids"])}
        if not all(r in pos for r in ref):
            print(f"  {tag}: {sum(r not in pos for r in ref)} patients missing -> skipped", flush=True)
            continue
        idx = np.array([pos[r] for r in ref])
        F = D["F_source"][idx].astype(np.float64)
        if F.shape[1] in (769, 513) and np.allclose(D["w0"][:-1], 0):   # foundation-model files: last col = EF
            F = F[:, :-1]
        out[tag] = dict(p=D["p_source"][idx].astype(np.float64), F=F,
                        y=M[ref_tag]["y"].astype(np.float64), y2=M[ref_tag]["y2"].astype(np.float64))
    return out


def model_block(F, p, pcs):
    """Unlabelled-site PCA of the z-scored embedding (pcs components, unit variance) + the centred EF prediction."""
    Z = (F - F.mean(0)) / (F.std(0) + 1e-8)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    q = min(pcs, len(S))
    P = U[:, :q] * S[:q]
    P = P / (P.std(0) + 1e-8)
    return np.hstack([P, ((p - p.mean()) / (p.std() + 1e-8))[:, None]])


def laplacian_quadratic(X, knn=10):
    """X' L X for a symmetric kNN graph (Gaussian weights, median-distance bandwidth) over all scans,
    scaled so that its trace equals trace(X'X) (makes the Laplacian weight dimensionless)."""
    n = len(X)
    sq = (X ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :knn]
    rows = np.repeat(np.arange(n), knn)
    dist = d2[rows, nn.ravel()]
    sig2 = np.median(dist)
    W = np.zeros((n, n))
    W[rows, nn.ravel()] = np.exp(-dist / sig2)
    W = np.maximum(W, W.T)
    L = np.diag(W.sum(1)) - W
    Q = X.T @ L @ X
    return Q * (np.trace(X.T @ X) / max(np.trace(Q), 1e-12))


# ---------------------------------------------------------------- fitting
def fit_loo(XL, r, Q, lams, gammas):
    """min ||[1 XL] b - r||^2 + lam ||b_x||^2 + gamma b_x' Q b_x, intercept unpenalized.
    Returns the coefficient vector of the (lam, gamma) with the lowest exact leave-one-out MAE, and the
    leave-one-out fitted residual corrections for that choice."""
    k, d = XL.shape
    X1 = np.hstack([np.ones((k, 1)), XL])
    best = None
    G0 = X1.T @ X1
    for g in gammas:
        Pq = np.zeros((d + 1, d + 1))
        Pq[1:, 1:] = g * Q
        for lam in lams:
            A = G0 + Pq
            A[1:, 1:] += lam * np.eye(d)
            Ainv_Xt = np.linalg.solve(A, X1.T)
            beta = Ainv_Xt @ r
            h = np.einsum("ij,ji->i", X1, Ainv_Xt)
            loo_corr = r - (r - X1 @ beta) / np.clip(1 - h, 1e-9, None)
            err = np.abs(r - loo_corr).mean()
            if best is None or err < best[0]:
                best = (err, beta, loo_corr, lam, g)
    return best[1], best[2], best[3], best[4]


def conf_q(res, alpha=0.1):
    k = len(res)
    lvl = math.ceil((k + 1) * (1 - alpha)) / k
    return np.inf if lvl > 1 else float(np.quantile(res, lvl, method="higher"))


def select(strategy, pool, k, p):
    if strategy == "random":
        return pool[:k]
    order = pool[np.argsort(p[pool], kind="stable")]
    return order[np.unique(np.linspace(0, len(order) - 1, k).round().astype(int))]


def metrics(y, pt, half=None):
    e = pt - y
    red = y < 40
    r = dict(MAE=np.abs(e).mean(), bias=e.mean(),
             MAE_reduced=np.abs(e[red]).mean() if red.any() else np.nan,
             reduced_called_preserved=(pt[red] >= 50).mean() if red.any() else np.nan,
             HF_category_agreement=(np.digitize(y, [40, 50]) == np.digitize(pt, [40, 50])).mean())
    if half is not None and np.isfinite(half):
        cov = np.abs(e) <= half
        r.update(coverage90=cov.mean(), width90=2 * half, coverage90_reduced=cov[red].mean() if red.any() else np.nan)
    else:
        r.update(coverage90=np.nan, width90=np.nan, coverage90_reduced=np.nan)
    return {k_: float(v) for k_, v in r.items()}


# ---------------------------------------------------------------- main
def configurations(tags):
    fm = [t for t in ("panecho", "echoprime") if t in tags]
    C = {}
    for t in tags:
        if t.startswith(("baseline_seed", "physaug_all_seed")) or t in fm:
            C[t] = (t.rsplit("_seed", 1)[0] if "_seed" in t else t, [t])
    if len(fm) == 2:
        C["FM2"] = ("FM2", fm)
    for s in range(3):
        for fam, short in (("physaug_all", "phys"), ("baseline", "base")):
            t = f"{fam}_seed{s}"
            if t in tags and fm:
                C[f"{short}{s}+FM"] = (f"{short}+" + "+".join(fm), [t] + fm)
    phys = [f"physaug_all_seed{s}" for s in range(3) if f"physaug_all_seed{s}" in tags]
    if len(phys) == 3:
        C["phys3"] = ("phys3", phys)
        if fm:
            C["phys3+FM"] = ("phys3+" + "+".join(fm), phys + fm)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--dataset", choices=["camus", "pediatric"], default="camus")
    ap.add_argument("--ks", default=None, help="default camus 5,10,20,50,100,250; pediatric adds 500,1000")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--pcs", type=int, default=32, help="principal components per model")
    ap.add_argument("--knn", type=int, default=10)
    a = ap.parse_args()
    ks = [int(k) for k in (a.ks or ("5,10,20,50,100,250" if a.dataset == "camus" else
                                    "5,10,20,50,100,250,500,1000")).split(",")]

    M = load_models(a.npz_dir, a.dataset)
    ref = next((t for t in ("baseline_seed0",) if t in M), sorted(M)[0])
    A = align(M, ref)
    C = configurations(list(A))
    print(f"{a.dataset}: {len(A[ref]['y'])} patients, models {sorted(A)}", flush=True)
    print("configurations:", {c: v[1] for c, v in C.items()}, flush=True)

    targets = [("reference", A[ref]["y"])]
    if a.dataset == "camus" and np.isfinite(A[ref]["y2"]).sum() > 100:
        targets.append(("a4c", A[ref]["y2"]))

    rows, t0 = [], time.time()
    for cname, (family, members) in C.items():
        X = np.hstack([model_block(A[m]["F"], A[m]["p"], a.pcs) for m in members])
        base = np.mean([A[m]["p"] for m in members], axis=0)
        Q = laplacian_quadratic(X, a.knn)
        for tname, yall in targets:
            keep = np.where(np.isfinite(yall))[0]
            N = len(keep)
            for s in range(a.splits):
                perm = keep[np.random.default_rng(1000 + s).permutation(N)]
                te, pool = perm[:N // 2], perm[N // 2:]
                y = yall
                rows.append(dict(config=cname, family=family, target=tname, split=s, k=0, selection="none",
                                 method="source", lam=np.nan, gamma=np.nan, **metrics(y[te], base[te])))
                for k in ks:
                    if k > len(pool):
                        continue
                    for sel in ("random", "pred_stratified"):
                        cal = select(sel, pool, k, base)
                        r = y[cal] - base[cal]
                        # offset
                        off = r.mean()
                        loo = np.array([np.delete(r, j).mean() for j in range(k)])
                        rows.append(dict(config=cname, family=family, target=tname, split=s, k=k, selection=sel,
                                         method="offset", lam=np.nan, gamma=np.nan,
                                         **metrics(y[te], base[te] + off, conf_q(np.abs(r - loo)))))
                        for meth, gams in (("ridge", [0.0]), ("lapridge", GAMMAS)):
                            beta, loo_c, lam, g = fit_loo(X[cal], r, Q, LAMS, gams)
                            pt = base[te] + beta[0] + X[te] @ beta[1:]
                            rows.append(dict(config=cname, family=family, target=tname, split=s, k=k,
                                             selection=sel, method=meth, lam=lam, gamma=g,
                                             **metrics(y[te], pt, conf_q(np.abs(r - loo_c)))))
        print(f"  {cname} done ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.npz_dir) / f"fusion_{a.dataset}.csv"
    df.to_csv(out, index=False)
    cols = ["MAE", "bias", "MAE_reduced", "reduced_called_preserved", "HF_category_agreement",
            "coverage90", "width90", "coverage90_reduced", "gamma"]
    summ = df.groupby(["target", "family", "method", "selection", "k"])[cols].mean().round(3)
    summ.to_csv(Path(a.npz_dir) / f"fusion_{a.dataset}_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        show = summ.reset_index()
        show = show[(show.selection.isin(["none", "random"]))]
        print(show.pivot_table(index=["target", "family", "method"], columns="k", values="MAE").round(2))
    print(f"\nSaved {out} and fusion_{a.dataset}_summary.csv")


if __name__ == "__main__":
    main()
