"""Explainable-AI analysis for RASAD (Wav2Vec2 + BERT + TF-IDF -> PCA -> soft-voting ensemble).

Produces every number, table and figure for the XAI section of the paper from the
trained model and the extracted features. See README.md in this folder for the
expected input files.

Method
------
1. SHAP values are computed for the soft-voting ensemble's P(AD) in PCA space with a
   model-agnostic permutation explainer (training set as background).
2. Because PCA (and an optional StandardScaler) is linear, each component's SHAP value
   is split exactly over the original features in proportion to that feature's
   contribution to the component's deviation from the background mean. The original-
   feature attributions still sum to f(x) - E[f(x)], so nothing is lost.
3. Attributions are then summed per modality (Wav2Vec2, BERT, TF-IDF) and reported per
   TF-IDF term, per test speaker, and across 5 cross-validation refits (stability).

Usage
-----
    python xai_analysis.py --data-dir path/to/features --model path/to/model.joblib --out results
    python xai_analysis.py --selftest --out selftest   # synthetic data, checks the install only
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODALITIES = ("Wav2Vec2 (acoustic)", "BERT (linguistic)", "TF-IDF (linguistic)")
SEED = 42


# ---------------------------------------------------------------- loading
def _load_matrix(path_stem):
    """Load <stem>.npy or <stem>.npz (sparse) as a dense float array."""
    npy, npz = Path(f"{path_stem}.npy"), Path(f"{path_stem}.npz")
    if npy.exists():
        return np.asarray(np.load(npy), dtype=float)
    if npz.exists():
        return sparse.load_npz(npz).toarray().astype(float)
    raise FileNotFoundError(f"Neither {npy} nor {npz} exists")


def load_split(data_dir, split):
    d = Path(data_dir)
    blocks = [_load_matrix(d / f"{split}_{name}") for name in ("acoustic", "bert", "tfidf")]
    y = np.load(d / f"{split}_labels.npy").astype(int)
    ids_file = d / f"{split}_ids.txt"
    ids = ids_file.read_text().split() if ids_file.exists() else [f"{split}{i:03d}" for i in range(len(y))]
    return blocks, y, ids


def to_pipeline(obj):
    """Accept a fitted Pipeline or a dict {'scaler' (optional), 'pca', 'ensemble'}."""
    if isinstance(obj, Pipeline):
        return obj
    steps = [("scaler", obj["scaler"])] if obj.get("scaler") is not None else []
    steps += [("pca", obj["pca"]), ("ensemble", obj["ensemble"])]
    return Pipeline(steps)


def split_pipeline(pipe):
    """Return (scaler or None, pca, classifier) from a fitted pipeline."""
    steps = [s for _, s in pipe.steps]
    pca_idx = next(i for i, s in enumerate(steps) if isinstance(s, PCA))
    pre = steps[:pca_idx]
    if len(pre) > 1 or (pre and not isinstance(pre[0], StandardScaler)):
        raise ValueError("Only an optional StandardScaler may precede PCA (linear steps are required).")
    clf = steps[pca_idx + 1:]
    if len(clf) != 1:
        raise ValueError("Expected exactly one classifier step after PCA.")
    return (pre[0] if pre else None), steps[pca_idx], clf[0]


# ---------------------------------------------------------------- SHAP
def raw_feature_shap(pipe, X_bg, X_explain, max_background=100):
    """SHAP values for P(AD) on the original (pre-PCA) features. Returns (phi, base_value)."""
    scaler, pca, clf = split_pipeline(pipe)
    scale = (lambda X: scaler.transform(X)) if scaler is not None else (lambda X: X)
    U_bg, U_ex = scale(X_bg), scale(X_explain)
    Z_bg, Z_ex = pca.transform(U_bg), pca.transform(U_ex)

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(Z_bg), size=min(max_background, len(Z_bg)), replace=False)
    Z_ref, U_ref = Z_bg[bg_idx], U_bg[bg_idx]

    f = lambda Z: clf.predict_proba(Z)[:, 1]  # noqa: E731
    k = Z_ex.shape[1]
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(Z_ref, max_samples=len(Z_ref)), seed=SEED)
    sv = explainer(Z_ex, max_evals=max(2 * k + 1, 10 * k))
    phi_z = sv.values                                   # (n, k)

    # Exact linear redistribution: z_k - zbar_k = sum_j W'_kj (u_j - ubar_j)
    W = pca.components_.copy()
    if getattr(pca, "whiten", False):
        W /= np.sqrt(pca.explained_variance_)[:, None]
    dU = U_ex - U_ref.mean(axis=0)
    dZ = dU @ W.T
    ratio = np.divide(phi_z, dZ, out=np.zeros_like(phi_z), where=np.abs(dZ) > 1e-12)
    phi_raw = dU * (ratio @ W)                          # (n, d), rows sum to phi_z.sum(1)
    lost = np.abs(phi_z[np.abs(dZ) <= 1e-12]).sum()
    if lost > 1e-6:
        print(f"note: {lost:.2e} of attribution on components at their background mean was not redistributed")
    return phi_raw, float(np.mean(sv.base_values))


def modality_sums(phi, dims):
    edges = np.cumsum([0, *dims])
    return np.stack([phi[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)


# ---------------------------------------------------------------- analyses
def modality_table(phi, dims, y):
    mod = modality_sums(phi, dims)
    absmod = np.abs(mod)
    rows = []
    for m, name in enumerate(MODALITIES):
        share = lambda mask: 100 * absmod[mask, m].sum() / absmod[mask].sum()  # noqa: E731
        rows.append({
            "Modality": name,
            "Share of |SHAP| all (%)": share(np.ones_like(y, bool)),
            "Share of |SHAP| AD (%)": share(y == 1),
            "Share of |SHAP| CN (%)": share(y == 0),
            "Mean SHAP AD": mod[y == 1, m].mean(),
            "Mean SHAP CN": mod[y == 0, m].mean(),
        })
    return pd.DataFrame(rows)


def word_table(phi, dims, vocab, X_tfidf, y, top=15):
    start = dims[0] + dims[1]
    tf_phi = phi[:, start:start + dims[2]]
    df = pd.DataFrame({
        "term": vocab,
        "mean_shap": tf_phi.mean(axis=0),
        "mean_abs_shap": np.abs(tf_phi).mean(axis=0),
        "docs_AD": (X_tfidf[y == 1] > 0).sum(axis=0),
        "docs_CN": (X_tfidf[y == 0] > 0).sum(axis=0),
    })
    toward_ad = df.sort_values("mean_shap", ascending=False).head(top)
    toward_cn = df.sort_values("mean_shap").head(top)
    return df.sort_values("mean_abs_shap", ascending=False), toward_ad, toward_cn


def case_table(phi, base, dims, vocab, y, p, ids, n_words=5):
    """Explain the most confident correct AD case, the most confident correct CN case and every error."""
    pred = (p >= 0.5).astype(int)
    picks = []
    for label in (1, 0):
        ok = np.where((y == label) & (pred == label))[0]
        if len(ok):
            picks.append(ok[np.argmax(np.abs(p[ok] - 0.5))])
    picks += list(np.where(pred != y)[0])
    mod = modality_sums(phi, dims)
    start = dims[0] + dims[1]
    rows = []
    for i in picks:
        w = phi[i, start:start + dims[2]]
        top_ad = [vocab[j] for j in np.argsort(-w)[:n_words] if w[j] > 0]
        top_cn = [vocab[j] for j in np.argsort(w)[:n_words] if w[j] < 0]
        rows.append({
            "speaker": ids[i], "true": "AD" if y[i] else "CN", "pred": "AD" if pred[i] else "CN",
            "P(AD)": p[i], "base": base,
            **{f"SHAP {m.split(' ')[0]}": mod[i, k] for k, m in enumerate(MODALITIES)},
            "words toward AD": ", ".join(top_ad), "words toward CN": ", ".join(top_cn),
        })
    return pd.DataFrame(rows)


def stability(pipe, X_train, y_train, dims, vocab, n_splits=5, top=20):
    """Refit the pipeline (same hyperparameters) on CV folds; compare explanations across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    shares, word_imp = [], []
    start = dims[0] + dims[1]
    for tr, va in skf.split(X_train, y_train):
        m = clone(pipe).fit(X_train[tr], y_train[tr])
        phi, _ = raw_feature_shap(m, X_train[tr], X_train[va])
        a = np.abs(modality_sums(phi, dims))
        shares.append(100 * a.sum(0) / a.sum())
        word_imp.append(np.abs(phi[:, start:start + dims[2]]).mean(0))
    shares = np.array(shares)
    tops = [set(np.argsort(-w)[:top]) for w in word_imp]
    jacc = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    rho = [spearmanr(a, b).correlation for a, b in combinations(word_imp, 2)]
    fold_df = pd.DataFrame(shares, columns=MODALITIES)
    fold_df.index = [f"fold {i + 1}" for i in range(n_splits)]
    summary = {
        "modality_share_mean": dict(zip(MODALITIES, shares.mean(0).round(1).tolist())),
        "modality_share_sd": dict(zip(MODALITIES, shares.std(0, ddof=1).round(1).tolist())),
        f"top{top}_terms_jaccard_mean": float(np.mean(jacc)),
        f"top{top}_terms_jaccard_min": float(np.min(jacc)),
        "term_importance_spearman_mean": float(np.nanmean(rho)),
        "terms_in_top_of_every_fold": [vocab[j] for j in sorted(set.intersection(*tops))],
    }
    return fold_df, summary


# ---------------------------------------------------------------- outputs
def plot_modality(mod_df, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(mod_df))
    for off, col, lab in ((-0.2, "Share of |SHAP| AD (%)", "AD"), (0.2, "Share of |SHAP| CN (%)", "CN")):
        ax.bar(x + off, mod_df[col], width=0.4, label=lab)
    ax.set_xticks(x, [m.split(" ")[0] for m in mod_df["Modality"]])
    ax.set_ylabel("Share of total |SHAP| (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_xai_modality.pdf")
    fig.savefig(out / "fig_xai_modality.png", dpi=300)
    plt.close(fig)


def plot_words(toward_ad, toward_cn, out):
    d = pd.concat([toward_cn.iloc[::-1], toward_ad.iloc[::-1]])
    fig, ax = plt.subplots(figsize=(4.2, 0.18 * len(d) + 0.8))
    ax.barh(d["term"], d["mean_shap"], color=np.where(d["mean_shap"] > 0, "#c0392b", "#2c6fbb"))
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Mean SHAP value on P(AD)  (< 0: toward CN,  > 0: toward AD)")
    fig.tight_layout()
    fig.savefig(out / "fig_xai_terms.pdf")
    fig.savefig(out / "fig_xai_terms.png", dpi=300)
    plt.close(fig)


def latex_tables(mod_df, toward_ad, toward_cn, stab, out, synthetic):
    warn = "% !!! SYNTHETIC SELF-TEST DATA. NOT RESULTS. DO NOT USE IN THE PAPER !!!\n" if synthetic else ""
    sm, ss = stab["modality_share_mean"], stab["modality_share_sd"]
    lines = [warn + r"\begin{table}[!t]", r"\centering",
             r"\caption{Contribution of each modality to the ensemble's predictions on the test set "
             r"(share of total $|$SHAP$|$; cross-validation mean $\pm$ SD in the last column)}",
             r"\label{tab:xai_modality}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Modality & All (\%) & AD (\%) & CN (\%) & CV (\%) \\", r"\midrule"]
    for _, r in mod_df.iterrows():
        m = r["Modality"]
        lines.append(f"{m} & {r['Share of |SHAP| all (%)']:.1f} & {r['Share of |SHAP| AD (%)']:.1f} & "
                     f"{r['Share of |SHAP| CN (%)']:.1f} & {sm[m]:.1f} $\\pm$ {ss[m]:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "", ""]

    n = min(len(toward_ad), len(toward_cn), 10)
    lines += [r"\begin{table}[!t]", r"\centering",
              r"\caption{TF--IDF terms with the largest mean SHAP contribution toward AD and toward CN (test set)}",
              r"\label{tab:xai_terms}", r"\begin{tabular}{lclc}", r"\toprule",
              r"\multicolumn{2}{c}{Toward AD} & \multicolumn{2}{c}{Toward CN} \\",
              r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}",
              r"Term & SHAP ($\times10^{-3}$) & Term & SHAP ($\times10^{-3}$) \\", r"\midrule"]
    for (_, a), (_, c) in zip(toward_ad.head(n).iterrows(), toward_cn.head(n).iterrows()):
        lines.append(f"{a['term']} & {1e3 * a['mean_shap']:+.1f} & {c['term']} & {1e3 * c['mean_shap']:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "xai_tables.tex").write_text("\n".join(lines) + "\n")


def run(blocks_tr, y_tr, blocks_te, y_te, ids_te, vocab, pipe, out, synthetic=False):
    out.mkdir(parents=True, exist_ok=True)
    dims = [b.shape[1] for b in blocks_te]
    if len(vocab) != dims[2]:
        raise ValueError(f"Vocabulary has {len(vocab)} terms but TF-IDF matrix has {dims[2]} columns")
    X_tr, X_te = np.hstack(blocks_tr), np.hstack(blocks_te)
    p = pipe.predict_proba(X_te)[:, 1]
    print(f"Test accuracy of loaded model: {np.mean((p >= 0.5) == y_te):.4f}  (paper: 0.8451)")

    phi, base = raw_feature_shap(pipe, X_tr, X_te)
    gap = np.max(np.abs(base + phi.sum(1) - p))
    print(f"Additivity check, max |base + sum(SHAP) - P(AD)| = {gap:.2e}")

    mod_df = modality_table(phi, dims, y_te)
    all_terms, toward_ad, toward_cn = word_table(phi, dims, vocab, blocks_te[2], y_te)
    cases = case_table(phi, base, dims, vocab, y_te, p, ids_te)
    fold_df, stab = stability(pipe, X_tr, y_tr, dims, vocab)

    mod_df.round(3).to_csv(out / "modality_attribution.csv", index=False)
    all_terms.round(5).to_csv(out / "term_attribution.csv", index=False)
    cases.round(3).to_csv(out / "case_explanations.csv", index=False)
    fold_df.round(1).to_csv(out / "stability_folds.csv")
    (out / "stability_summary.json").write_text(json.dumps(stab, indent=2))
    np.save(out / "shap_values_test.npy", phi)
    plot_modality(mod_df, out)
    plot_words(toward_ad, toward_cn, out)
    latex_tables(mod_df, toward_ad, toward_cn, stab, out, synthetic)

    print("\nModality attribution:\n", mod_df.round(1).to_string(index=False))
    print("\nStability:", json.dumps(stab, indent=2))
    print(f"\nAll outputs written to {out.resolve()}")


def selftest(out):
    """Synthetic data only: checks that the installation and the code run end to end."""
    rng = np.random.default_rng(SEED)
    n_tr, n_te, da, db, dt = 166, 71, 64, 64, 40
    y_tr, y_te = rng.permutation([1] * 87 + [0] * 79), rng.permutation([1] * 35 + [0] * 36)

    def make(y):
        a = rng.normal(size=(len(y), da)) + 0.5 * y[:, None] * (np.arange(da) < 4)
        b = rng.normal(size=(len(y), db)) + 0.8 * y[:, None] * (np.arange(db) < 4)
        t = np.abs(rng.normal(size=(len(y), dt))) * (rng.random((len(y), dt)) < 0.3)
        t[:, 0] += 0.6 * y
        t[:, 1] += 0.6 * (1 - y)
        return [a, b, t]

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    btr, bte = make(y_tr), make(y_te)
    vocab = ["uh", "boy"] + [f"term{i}" for i in range(2, dt)]
    ens = VotingClassifier([("xgb", XGBClassifier(n_estimators=50, max_depth=3, random_state=SEED)),
                            ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
                            ("lr", LogisticRegression(max_iter=1000))], voting="soft")
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=SEED)),
                     ("ensemble", ens)]).fit(np.hstack(btr), y_tr)
    run(btr, y_tr, bte, y_te, [f"S{i:03d}" for i in range(n_te)], vocab, pipe, out, synthetic=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder with {train,test}_{acoustic,bert,tfidf,labels} files and tfidf_vocab.txt")
    ap.add_argument("--model", help="joblib file: fitted Pipeline, or dict with 'scaler' (optional), 'pca', 'ensemble'")
    ap.add_argument("--out", default="xai_results")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data to check the installation")
    args = ap.parse_args()
    out = Path(args.out)
    if args.selftest:
        return selftest(out)
    if not (args.data_dir and args.model):
        ap.error("--data-dir and --model are required (or use --selftest)")
    btr, ytr, _ = load_split(args.data_dir, "train")
    bte, yte, ids = load_split(args.data_dir, "test")
    vocab = (Path(args.data_dir) / "tfidf_vocab.txt").read_text(encoding="utf-8").splitlines()
    run(btr, ytr, bte, yte, ids, vocab, to_pipeline(joblib.load(args.model)), out)


if __name__ == "__main__":
    main()
