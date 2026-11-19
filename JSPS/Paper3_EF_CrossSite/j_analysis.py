r"""
Package J (JSPS Paper 3): reviewer-proofing analyses for the combined recalibration. CPU only.
Reads the same .npz exports as fusion_analysis.py (D:\EchoNet\runs\adapt_preds) and the EchoNet validation
predictions (D:\EchoNet\runs\<model>\val_predictions.csv). Uses the SAME 20 splits as fusion_analysis.py.
Writes only new files, into --out (default D:\EchoNet\runs\package_J):

  J_<dataset>_main.csv / _summary.csv
      J1  ensemble control: 3 baseline seeds (base3, base3+FM) vs 3 physics seeds (phys3, phys3+FM)
      J2  clinical agreement: MAE, bias, Bland-Altman limits of agreement, Pearson r, AUROC for EF<40,
          sensitivity / specificity for EF<40, heart-failure category agreement and Cohen's kappa (3 classes)
      J3  conformal intervals FOR THE COMBINATION: plain, Mondrian, normalized (EchoNet error profile, mean over
          the member models), asymmetric; coverage overall and in reduced EF (<40), width
      J6  fit time of the combination (seconds, CPU)
  J_<dataset>_sensitivity.csv / _sensitivity_summary.csv
      J5  phys3+FM with 8 / 16 / 32 / 64 principal components, without the EF-prediction column,
          and with a wider penalty grid
  J_<dataset>_patients.csv     per-patient test predictions (all splits) for Bland-Altman plots and subgroups
  J_<dataset>_subgroups.csv
      J4  MAE / bias / reduced-EF called preserved by CAMUS image quality, or pediatric age group and sex

Usage (notebook 2, from D:\EchoNet, fusion_analysis.py must be in the same folder):
  %env OMP_NUM_THREADS=2
  !"{sys.executable}" j_analysis.py --dataset camus
  !"{sys.executable}" j_analysis.py --dataset pediatric
Check first (2 splits, 2 k values, about a minute):
  !"{sys.executable}" j_analysis.py --dataset camus --splits 2 --ks 20,100 --out D:\EchoNet\runs\package_J_check
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fusion_analysis as fa

FM = ["panecho", "echoprime"]
LAMS_WIDE = [0.03, 0.3, 3.0, 30.0, 300.0, 3000.0, 30000.0]


# ---------------------------------------------------------------- helpers
def auroc(score, pos):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().values
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa(a, b, n=3):
    C = np.zeros((n, n))
    np.add.at(C, (a, b), 1)
    C /= C.sum()
    po, pe = np.trace(C), (C.sum(0) * C.sum(1)).sum()
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def clinical(y, p):
    e = p - y
    red, pres = y < 40, y >= 50
    cy, cp = np.digitize(y, [40, 50]), np.digitize(p, [40, 50])
    return dict(MAE=np.abs(e).mean(), bias=e.mean(), LoA_low=e.mean() - 1.96 * e.std(ddof=1),
                LoA_high=e.mean() + 1.96 * e.std(ddof=1), r=np.corrcoef(y, p)[0, 1],
                AUROC_EF40=auroc(-p, red),
                sens_EF40=(p[red] < 40).mean() if red.any() else np.nan,
                spec_EF40=(p[~red] >= 40).mean() if (~red).any() else np.nan,
                reduced_called_preserved=(p[red] >= 50).mean() if red.any() else np.nan,
                HF_agreement=(cy == cp).mean(), HF_kappa=kappa(cy, cp))


def qlevel(k, alpha):
    lvl = math.ceil((k + 1) * (1 - alpha)) / k
    return None if lvl > 1 else lvl


def intervals(y, pt, loo_pred, r_loo_res, g):
    """r_loo_res: LOO residuals y - loo_pred on the labelled patients. Returns coverage/width per variant."""
    k = len(r_loo_res)
    out = {}
    red = y < 40

    def rec(name, lo, hi):
        cov = (y >= lo) & (y <= hi)
        out[f"cov_{name}"] = cov.mean()
        out[f"cov_{name}_reduced"] = cov[red].mean() if red.any() else np.nan
        out[f"width_{name}"] = np.mean(hi - lo)

    a = np.abs(r_loo_res)
    lv = qlevel(k, 0.1)
    if lv is None:
        return out
    q = np.quantile(a, lv, method="higher")
    rec("plain", pt - q, pt + q)
    lo_c, hi_c = loo_pred < 50, loo_pred >= 50                               # Mondrian by predicted EF group
    if lo_c.sum() >= 9 and hi_c.sum() >= 9:
        qa = np.quantile(a[lo_c], qlevel(lo_c.sum(), 0.1), method="higher")
        qb = np.quantile(a[hi_c], qlevel(hi_c.sum(), 0.1), method="higher")
        h = np.where(pt < 50, qa, qb)
    else:
        h = np.full_like(pt, q)
    rec("mondrian", pt - h, pt + h)
    if g is not None:
        gs, gt = g(loo_pred), g(pt)
        qn = np.quantile(a / gs, lv, method="higher")
        rec("norm", pt - qn * gt, pt + qn * gt)
        s = r_loo_res / gs                                                   # signed: y - pred
        lv5 = math.ceil((k + 1) * 0.95) / k
        if lv5 <= 1:
            up = np.quantile(s, lv5, method="higher")
            dn = -np.quantile(-s, lv5, method="higher")
            rec("asym", pt + dn * gt, pt + up * gt)
    return out


def load_profile(runs_dir, tag, nbins=10):
    f = Path(runs_dir) / tag / "val_predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    cols = {c.lower(): c for c in d.columns}
    pc = next((cols[c] for c in ("ef_pred", "p", "pred", "prediction") if c in cols), None)
    yc = next((cols[c] for c in ("ef_true", "y", "ef", "true") if c in cols), None)
    if pc is None or yc is None:
        return None
    p, e = d[pc].values.astype(float), np.abs(d[pc].values - d[yc].values).astype(float)
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    ch = np.array_split(np.argsort(p), nbins)
    xc, yc_ = np.array([np.median(p[c]) for c in ch]), np.array([e[c].mean() for c in ch]) / e.mean()
    return lambda q: np.interp(q, xc, yc_)


def fit(X, base, cal, y, lams):
    r = y[cal] - base[cal]
    beta, loo_c, lam, _ = fa.fit_loo(X[cal], r, np.zeros((X.shape[1], X.shape[1])), lams, [0.0])
    return beta, loo_c, lam, r


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default=r"D:\EchoNet\runs\adapt_preds")
    ap.add_argument("--runs_dir", default=r"D:\EchoNet\runs")
    ap.add_argument("--out", default=r"D:\EchoNet\runs\package_J")
    ap.add_argument("--dataset", choices=["camus", "pediatric"], default="camus")
    ap.add_argument("--ped_filelist", default=r"D:\EchoNet - Pediatric\pediatric_echo_avi\A4C\FileList.csv")
    ap.add_argument("--ks", default=None, help="default camus 5,10,20,50,100,250; pediatric adds 500,1000")
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    ks = [int(k) for k in (a.ks or ("5,10,20,50,100,250" if a.dataset == "camus" else
                                    "5,10,20,50,100,250,500,1000")).split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    M = fa.load_models(a.npz_dir, a.dataset)
    ref = "baseline_seed0" if "baseline_seed0" in M else sorted(M)[0]
    A = fa.align(M, ref)
    ids = np.array([str(i) for i in M[ref]["ids"]])
    y_all = A[ref]["y"]
    grp = np.array([str(x) for x in M[ref].get("quality", np.array(["NA"] * len(ids)))])
    print(f"{a.dataset}: {len(ids)} patients, models {sorted(A)}", flush=True)

    base_s = [f"baseline_seed{s}" for s in range(3) if f"baseline_seed{s}" in A]
    phys_s = [f"physaug_all_seed{s}" for s in range(3) if f"physaug_all_seed{s}" in A]
    fm = [t for t in FM if t in A]
    C = {"baseline_seed0": [ref]}
    if len(base_s) == 3:
        C["base3"] = base_s
        if fm:
            C["base3+FM"] = base_s + fm
    if len(phys_s) == 3:
        C["phys3"] = phys_s
        if fm:
            C["phys3+FM"] = phys_s + fm
    if fm:
        C["FM2"] = fm
    print("configurations:", C, flush=True)

    prof = {t: load_profile(a.runs_dir, t) for t in A}
    print("EchoNet error profiles:", {t: ("found" if g else "MISSING") for t, g in prof.items()}, flush=True)

    def combo_profile(members):
        gs = [prof[m] for m in members if prof[m] is not None]
        return (lambda q: np.mean([g(q) for g in gs], axis=0)) if gs else None

    keep = np.where(np.isfinite(y_all))[0]
    N = len(keep)
    splits = []
    for s in range(a.splits):
        perm = keep[np.random.default_rng(1000 + s).permutation(N)]
        splits.append((perm[:N // 2], perm[N // 2:]))

    rows, prow = [], []
    save_pat = {("baseline_seed0", 0), ("baseline_seed0", 20), ("phys3+FM", 0), ("phys3+FM", 20), ("phys3+FM", 100)}
    t0 = time.time()
    for cname, members in C.items():
        X = np.hstack([fa.model_block(A[m]["F"], A[m]["p"], 32) for m in members])
        base = np.mean([A[m]["p"] for m in members], axis=0)
        g = combo_profile(members)
        for s, (te, pool) in enumerate(splits):
            y = y_all
            rows.append(dict(config=cname, split=s, k=0, selection="none", lam=np.nan, fit_seconds=0.0,
                             **clinical(y[te], base[te])))
            if (cname, 0) in save_pat:
                prow.append(pd.DataFrame(dict(config=cname, k=0, split=s, id=ids[te], group=grp[te], y=y[te],
                                              pred=base[te])))
            for k in ks:
                if k > len(pool):
                    continue
                for sel in ("random", "pred_stratified"):
                    cal = fa.select(sel, pool, k, base)
                    t1 = time.perf_counter()
                    beta, loo_c, lam, r = fit(X, base, cal, y, fa.LAMS)
                    secs = time.perf_counter() - t1
                    pt = base[te] + beta[0] + X[te] @ beta[1:]
                    loo_pred = base[cal] + loo_c
                    row = dict(config=cname, split=s, k=k, selection=sel, lam=lam, fit_seconds=secs,
                               **clinical(y[te], pt))
                    row.update(intervals(y[te], pt, loo_pred, y[cal] - loo_pred, g))
                    rows.append(row)
                    if sel == "random" and (cname, k) in save_pat:
                        prow.append(pd.DataFrame(dict(config=cname, k=k, split=s, id=ids[te], group=grp[te],
                                                      y=y[te], pred=pt)))
        print(f"  {cname} done ({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"J_{a.dataset}_main.csv", index=False)
    num = [c for c in df.columns if c not in ("config", "split", "k", "selection")]
    summ = df.groupby(["config", "selection", "k"])[num].mean().round(4)
    summ.to_csv(out / f"J_{a.dataset}_summary.csv")

    # J1 paired test: phys3(+FM) vs base3(+FM), same splits
    from scipy.stats import wilcoxon
    tests = []
    for p_, b_ in (("phys3+FM", "base3+FM"), ("phys3", "base3"), ("phys3+FM", "phys3"), ("base3+FM", "base3")):
        if p_ not in C or b_ not in C:
            continue
        for k in ks:
            P = df[(df.config == p_) & (df.k == k) & (df.selection == "random")].set_index("split").MAE
            B = df[(df.config == b_) & (df.k == k) & (df.selection == "random")].set_index("split").MAE
            P, B = P.align(B, join="inner")
            if len(P) >= 5:
                tests.append(dict(a=p_, b=b_, k=k, MAE_a=P.mean(), MAE_b=B.mean(), diff=(P - B).mean(),
                                  p_wilcoxon=wilcoxon(P, B).pvalue if (P - B).abs().sum() > 0 else 1.0))
    pd.DataFrame(tests).round(4).to_csv(out / f"J_{a.dataset}_ensemble_control.csv", index=False)

    # J4 subgroups from per-patient predictions
    pats = pd.concat(prow, ignore_index=True)
    if a.dataset == "pediatric":
        age = pd.to_numeric(pats.group, errors="coerce")
        pats["group"] = pd.cut(age, [-1e-9, 1, 5, 12, 200], right=False,
                               labels=["<1 y", "1-4 y", "5-11 y", ">=12 y"]).astype(str)
        fl = Path(a.ped_filelist)
        if fl.exists():
            f = pd.read_csv(fl)
            f["id"] = f.FileName.astype(str)
            if "Sex" in f.columns:
                sex = dict(zip(f.id, f.Sex.astype(str)))
                sex.update({k_.replace(".avi", ""): v for k_, v in sex.items()})
                pats["sex"] = pats.id.map(sex).fillna("NA")
        else:
            print(f"  (pediatric FileList not found at {fl}: sex subgroups skipped)", flush=True)
    pats.to_csv(out / f"J_{a.dataset}_patients.csv", index=False)
    sub = []
    for col in [c for c in ("group", "sex") if c in pats.columns]:
        for (cfg, k, gv), d in pats.groupby(["config", "k", col]):
            e = d.pred - d.y
            red = d.y < 40
            sub.append(dict(variable=col, config=cfg, k=k, subgroup=gv, n_patients=d.id.nunique(),
                            MAE=np.abs(e).mean(), bias=e.mean(),
                            reduced_called_preserved=(d.pred[red] >= 50).mean() if red.any() else np.nan))
    pd.DataFrame(sub).round(3).to_csv(out / f"J_{a.dataset}_subgroups.csv", index=False)

    # J5 sensitivity (phys3+FM, random selection)
    sens = []
    if "phys3+FM" in C:
        mem = C["phys3+FM"]
        base = np.mean([A[m]["p"] for m in mem], axis=0)
        variants = {f"pcs{q}": (np.hstack([fa.model_block(A[m]["F"], A[m]["p"], q) for m in mem]), fa.LAMS)
                    for q in (8, 16, 32, 64)}
        X32 = variants["pcs32"][0]
        noef = np.hstack([fa.model_block(A[m]["F"], A[m]["p"], 32)[:, :-1] for m in mem])
        variants["no_EF_column"] = (noef, fa.LAMS)
        variants["wide_lambda_grid"] = (X32, LAMS_WIDE)
        for vname, (X, lams) in variants.items():
            for s, (te, pool) in enumerate(splits):
                for k in ks:
                    if k > len(pool):
                        continue
                    cal = pool[:k]
                    beta, loo_c, lam, r = fit(X, base, cal, y_all, lams)
                    pt = base[te] + beta[0] + X[te] @ beta[1:]
                    sens.append(dict(variant=vname, split=s, k=k, lam=lam, MAE=np.abs(pt - y_all[te]).mean()))
        sd = pd.DataFrame(sens)
        sd.to_csv(out / f"J_{a.dataset}_sensitivity.csv", index=False)
        sd.pivot_table(index="variant", columns="k", values="MAE").round(3).to_csv(
            out / f"J_{a.dataset}_sensitivity_summary.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        s = summ.reset_index()
        s = s[s.selection.isin(["none", "random"])]
        print("\nMAE by configuration and k:")
        print(s.pivot_table(index="config", columns="k", values="MAE").round(2))
        print("\nEnsemble control (phys vs base):")
        print(pd.DataFrame(tests).round(3).to_string(index=False))
        cols = [c for c in s.columns if c.startswith("cov_") or c.startswith("width_")]
        if cols:
            print("\nConformal intervals, phys3+FM:")
            print(s[s.config == "phys3+FM"].set_index("k")[cols].round(3))
        if sens:
            print("\nSensitivity, phys3+FM:")
            print(pd.read_csv(out / f"J_{a.dataset}_sensitivity_summary.csv").to_string(index=False))
    print(f"\nSaved files in {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
