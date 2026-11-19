"""Package H: final statistics tables and figures for JSPS Paper 3 (IEEE JBHI, single-column figure width 3.5 in,
double-column 7.16 in). Reads the exported result CSVs; writes PDFs/PNGs and tables into OUT."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind

A = Path("/mnt/user-data/attachments/3d9fb2e0-5b4d-51bc-a91b-b22783524106")
U = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/package_h/out"); OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

F = dict(
    cheap_camus=U / "cheap_camus_3.csv", cheap_ped=U / "cheap_pediatric_3.csv",
    fus_camus=A / "9910d91c-fusion_camus.csv", fus_ped=A / "7499a992-fusion_pediatric.csv",
    bench_camus=A / "44435642-adapt_benchmark_camus.csv", bench_ped=A / "0f93ce23-adapt_benchmark_pediatric.csv",
    pred_camus=A / "f9fdc96f-camus_predictions_mirror_zoom1.csv", pred_ped=A / "744f8bab-pediatric_predictions.csv",
)

# ---------------------------------------------------------------- style (validated categorical slots 1-6)
C = dict(offset="#2a78d6", linear="#eb6834", last_layer="#1baf7a", adapter="#eda100", full_ft="#e87ba4",
         combo="#008300")
MK = dict(offset="o", linear="s", last_layer="^", adapter="D", full_ft="v", combo="*")
LAB = dict(offset="Offset", linear="Linear", last_layer="Last layer", adapter="Adapter",
           full_ft="Full fine-tuning", combo="Combination (ours)")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.4,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})


def boot_ci(x, n=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def curve(df, cond, metric="MAE"):
    """mean and 95% bootstrap CI over splits, per k (seed models averaged within each split first)."""
    s = df[cond].groupby(["k", "split"])[metric].mean().reset_index()
    rows = []
    for k, g in s.groupby("k"):
        lo, hi = boot_ci(g[metric].values)
        rows.append((k, g[metric].mean(), lo, hi, len(g)))
    return pd.DataFrame(rows, columns=["k", "mean", "lo", "hi", "n"])


def load():
    D = {}
    for ds in ("camus", "ped"):
        ch = pd.read_csv(F[f"cheap_{ds}"]); fu = pd.read_csv(F[f"fus_{ds}"]); be = pd.read_csv(F[f"bench_{ds}"])
        be["family"] = be.ckpt.astype(str).str.extract(r"(baseline|physaug_all)")[0]
        fu = fu[fu.target == "reference"]
        D[ds] = dict(cheap=ch, fus=fu, bench=be)
    return D


def series(D, ds, metric="MAE"):
    ch, fu, be = D[ds]["cheap"], D[ds]["fus"], D[ds]["bench"]
    base = (ch.family == "baseline") & (ch.selection == "random")
    S = {m: curve(ch, base & (ch.method == m), metric) for m in ("offset", "linear", "last_layer")}
    for m in ("adapter", "full_ft"):
        S[m] = curve(be, (be.method == m) & (be.family == "baseline"), metric)
    S["combo"] = curve(fu, (fu.family == "phys3+panecho+echoprime") & (fu.method == "ridge") & (fu.selection == "random"), metric)
    ref = {}
    for m in ("source", "bn_adapt", "ssa"):
        ref[m] = ch[(ch.family == "baseline") & (ch.method == m)][metric].mean()
    return S, ref


# ---------------------------------------------------------------- Figure 1: label curves
def fig_labels(D):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)
    for ax, ds, title in zip(axs, ("camus", "ped"), ("(a) CAMUS (adult, biplane EF)", "(b) EchoNet-Pediatric")):
        S, ref = series(D, ds)
        for m in ("offset", "linear", "last_layer", "adapter", "full_ft", "combo"):
            c = S[m]; c = c[c.k > 0]
            lw = 2.0 if m == "combo" else 1.3
            ax.fill_between(c.k, c.lo, c.hi, color=C[m], alpha=0.13, lw=0)
            ax.plot(c.k, c["mean"], color=C[m], lw=lw, marker=MK[m], ms=5.5 if m == "combo" else 4,
                    mec="white", mew=0.6, label=LAB[m], zorder=3 if m == "combo" else 2)
        for m, ls, txt in (("source", (0, (4, 2)), "No adaptation"), ("bn_adapt", (0, (1, 1.5)), "BatchNorm re-est."),
                           ("ssa", (0, (3, 1, 1, 1)), "SSA (test-time)")):
            ax.axhline(ref[m], color=INK2, lw=0.9, ls=ls, zorder=1)
            ax.text(ax.get_xlim()[1] if False else (1000 if ds == "ped" else 250), ref[m], f" {txt} {ref[m]:.1f}",
                    va="bottom" if m != "ssa" else "top", ha="right", fontsize=6.2, color=INK2)
        ax.set_xscale("log"); ax.set_title(title, loc="left", color=INK)
        ks = [5, 10, 20, 50, 100, 250] + ([500, 1000] if ds == "ped" else [])
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Labelled patients at the new site (k)")
        ax.set_ylabel("MAE (EF points)")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        lo = min(S["combo"]["mean"].min(), S["last_layer"]["mean"].min()) - 0.4
        ax.set_ylim(lo, max(ref.values()) + 0.8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_label_curves.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: failure explanation (error vs EF)
def fig_failure():
    cam = pd.read_csv(F["pred_camus"]); ped = pd.read_csv(F["pred_ped"])
    bins = [0, 30, 40, 50, 60, 100]; labs = ["<30", "30-39", "40-49", "50-59", "≥60"]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), constrained_layout=True)
    rows = [("CAMUS vs biplane reference", cam.EF_ref, cam.EF_pred, "#2a78d6", "o"),
            ("CAMUS vs matched 4-chamber EF", cam.EF_mono_A4C, cam.EF_pred, "#eb6834", "s"),
            ("EchoNet-Pediatric", ped.EF, ped.EF_pred, "#1baf7a", "^")]
    for name, y, p, col, mk in rows:
        b = pd.cut(y, bins, right=False, labels=labs)
        g = (p - y).groupby(b, observed=False).agg(["mean", "count"])
        axs[0].plot(range(len(labs)), g["mean"], color=col, marker=mk, ms=4, mec="white", mew=0.6, label=name)
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_xticks(range(len(labs))); axs[0].set_xticklabels(labs)
    axs[0].set_xlabel("Reference EF (%)"); axs[0].set_ylabel("Mean error, predicted − reference")
    axs[0].set_title("(a) Bias grows as EF falls (no adaptation)", loc="left")
    axs[0].grid(axis="y", color=GRID, lw=0.6); axs[0].legend(frameon=False, loc="upper right")
    # (b) scatter CAMUS predicted vs reference, both references
    axs[1].plot([10, 85], [10, 85], color=INK2, lw=0.8)
    axs[1].scatter(cam.EF_ref, cam.EF_pred, s=6, color="#2a78d6", alpha=0.55, lw=0, label="vs biplane reference")
    axs[1].scatter(cam.EF_mono_A4C, cam.EF_pred, s=6, color="#eb6834", alpha=0.55, lw=0, marker="s",
                   label="vs matched 4-chamber EF")
    mae1 = (cam.EF_pred - cam.EF_ref).abs().mean(); mae2 = (cam.EF_pred - cam.EF_mono_A4C).abs().mean()
    axs[1].text(12, 80, f"MAE {mae1:.1f} (biplane)\nMAE {mae2:.1f} (4-chamber)", fontsize=6.5, va="top", color=INK)
    axs[1].set_xlabel("Reference EF (%)"); axs[1].set_ylabel("Predicted EF (%)")
    axs[1].set_title("(b) CAMUS: EF definition explains part of the gap", loc="left")
    axs[1].set_xlim(10, 85); axs[1].set_ylim(10, 85); axs[1].legend(frameon=False, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_failure.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: safety (reduced-EF called preserved)
def fig_safety(D):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4), constrained_layout=True)
    for ax, ds, title in zip(axs, ("camus", "ped"), ("(a) CAMUS", "(b) EchoNet-Pediatric")):
        S, ref = series(D, ds, "reduced_called_preserved")
        fu = D[ds]["fus"]
        phys_off = curve(fu, (fu.family == "phys3") & (fu.method == "offset") & (fu.selection == "random"),
                         "reduced_called_preserved")
        for m in ("offset", "last_layer", "full_ft", "combo"):
            c = S[m]; c = c[c.k > 0]
            ax.plot(c.k, 100 * c["mean"], color=C[m], marker=MK[m], ms=5 if m == "combo" else 4, mec="white",
                    mew=0.6, lw=2.0 if m == "combo" else 1.3, label=LAB[m])
        c = phys_off[phys_off.k > 0]
        ax.plot(c.k, 100 * c["mean"], color="#6250d6", marker="P", ms=4.5, mec="white", mew=0.6, lw=1.3,
                label="Physics ensemble, offset")
        ax.axhline(100 * ref["source"], color=INK2, lw=0.9, ls=(0, (4, 2)))
        ax.text(1000 if ds == "ped" else 250, 100 * ref["source"], f" No adaptation {100 * ref['source']:.0f}%",
                ha="right", va="bottom", fontsize=6.2, color=INK2)
        ax.set_xscale("log"); ks = [5, 10, 20, 50, 100, 250] + ([500, 1000] if ds == "ped" else [])
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Labelled patients at the new site (k)")
        ax.set_ylabel("Reduced EF (<40%) called\npreserved (≥50%), %")
        ax.set_title(title, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6)
        if ds == "camus":
            ax.set_ylim(0, 14)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_safety.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4: uncertainty
def fig_uncertainty(D):
    ch = D["ped"]["cheap"]
    sel = (ch.family == "baseline") & (ch.selection == "random") & (ch.method == "last_layer") & (ch.k >= 20)
    g = ch[sel].groupby("k").mean(numeric_only=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), constrained_layout=True)
    spec = [("plain", "Split conformal", "#2a78d6", "o"), ("mondrian", "Mondrian (EF <50 / ≥50)", "#eb6834", "s"),
            ("norm", "Error-profile scaled (ours)", "#1baf7a", "^"), ("asym", "Scaled + asymmetric (ours)", "#eda100", "D")]
    for key, name, col, mk in spec:
        axs[0].plot(g.index, g[f"coverage90_{key}_reduced"], color=col, marker=mk, ms=4, mec="white", mew=0.6, label=name)
        axs[1].plot(g.index, g[f"width90_{key}"], color=col, marker=mk, ms=4, mec="white", mew=0.6, label=name)
    axs[0].axhline(0.9, color=INK2, lw=0.9, ls=(0, (4, 2))); axs[0].text(1000, 0.9, "target 0.90 ", ha="right",
                                                                         va="bottom", fontsize=6.2, color=INK2)
    axs[0].set_ylabel("Coverage, reduced-EF children"); axs[1].set_ylabel("Mean interval width (EF points)")
    axs[0].set_title("(a) Coverage where it matters", loc="left"); axs[1].set_title("(b) Interval width", loc="left")
    for ax in axs:
        ax.set_xscale("log"); ks = [20, 50, 100, 250, 500, 1000]; ax.set_xticks(ks); ax.set_xticklabels(map(str, ks))
        ax.set_xlabel("Labelled patients (k), last-layer correction"); ax.grid(axis="y", color=GRID, lw=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_uncertainty.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- Tables
def fmt(m, lo, hi):
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def table_combo_vs_ft(D):
    """Combination vs full fine-tuning (and baseline last layer) on the SAME splits; paired difference with bootstrap CI."""
    out = []
    for ds, ks in (("camus", [5, 20, 100, 250]), ("ped", [5, 20, 100, 1000])):
        ch, fu, be = D[ds]["cheap"], D[ds]["fus"], D[ds]["bench"]
        for k in ks:
            cb = fu[(fu.family == "phys3+panecho+echoprime") & (fu.method == "ridge") & (fu.selection == "random")
                    & (fu.k == k)].groupby("split").MAE.mean()
            ft = be[(be.method == "full_ft") & (be.k == k)].groupby("split").MAE.mean()          # both families
            ll = ch[(ch.family == "baseline") & (ch.method == "last_layer") & (ch.selection == "random")
                    & (ch.k == k)].groupby("split").MAE.mean()
            sp = cb.index.intersection(ft.index).intersection(ll.index)
            d1, d2 = (cb[sp] - ft[sp]).values, (cb[sp] - ll[sp]).values
            sp2 = cb.index.intersection(ll.index)
            d3 = (cb[sp2] - ll[sp2]).values
            out.append(dict(dataset="CAMUS" if ds == "camus" else "Pediatric", k=k, n_matched_splits=len(sp),
                            combination=fmt(cb[sp].mean(), *boot_ci(cb[sp])), full_ft=fmt(ft[sp].mean(), *boot_ci(ft[sp])),
                            diff_vs_full_ft=fmt(d1.mean(), *boot_ci(d1)),
                            diff_vs_last_layer_all_splits=fmt(d3.mean(), *boot_ci(d3)),
                            p_vs_last_layer=f"{wilcoxon(d3).pvalue:.1e}", n_splits_last_layer=len(sp2)))
    return pd.DataFrame(out)


def table_augmentation(D):
    """Seed-level comparison of training augmentations after adaptation (last layer, random patients)."""
    out = []
    for ds, ks in (("camus", [20, 100]), ("ped", [20, 100])):
        ch = D[ds]["cheap"]
        for k in ks:
            s = ch[(ch.method == "last_layer") & (ch.selection == "random") & (ch.k == k)]
            per = s.groupby(["family", "ckpt"]).MAE.mean()
            b = per["baseline"]
            for fam in ("baseline", "genaug", "only_appearance", "physaug_all"):
                v = per[fam]
                p = ttest_ind(v, b, equal_var=False).pvalue if fam != "baseline" else np.nan
                out.append(dict(dataset="CAMUS" if ds == "camus" else "Pediatric", k=k, family=fam,
                                MAE=f"{v.mean():.2f} ± {v.std(ddof=1):.2f}", n_seeds=len(v),
                                p_vs_baseline_seed_level=f"{p:.3f}" if fam != "baseline" else ""))
    return pd.DataFrame(out)


def table_foundation(D):
    out = []
    for ds, ks in (("camus", [20, 100]), ("ped", [20, 100])):
        fu = D[ds]["fus"]
        for fam in ("baseline", "physaug_all", "panecho", "echoprime", "FM2", "phys3", "phys3+panecho+echoprime"):
            r = dict(dataset="CAMUS" if ds == "camus" else "Pediatric", configuration=fam)
            s = fu[(fu.family == fam)]
            r["no_labels"] = f"{s[s.method == 'source'].MAE.mean():.2f}"
            for k in ks:
                v = s[(s.method == "ridge") & (s.selection == "random") & (s.k == k)].groupby("split").MAE.mean()
                r[f"ridge_k{k}"] = fmt(v.mean(), *boot_ci(v))
                rc = s[(s.method == "ridge") & (s.selection == "random") & (s.k == k)].reduced_called_preserved.mean()
                r[f"reducedEF_called_preserved_k{k}"] = f"{100 * rc:.1f}%"
            out.append(r)
    return pd.DataFrame(out)


def main():
    D = load()
    fig_labels(D); fig_failure(); fig_safety(D); fig_uncertainty(D)
    T = {"table_combination_vs_finetuning": table_combo_vs_ft(D), "table_augmentation_seed_level": table_augmentation(D),
         "table_foundation_models": table_foundation(D)}
    with pd.ExcelWriter(OUT / "package_H_tables.xlsx") as xw:
        for name, t in T.items():
            t.to_csv(OUT / f"{name}.csv", index=False)
            t.to_excel(xw, sheet_name=name[:31], index=False)
            print(f"\n=== {name}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
