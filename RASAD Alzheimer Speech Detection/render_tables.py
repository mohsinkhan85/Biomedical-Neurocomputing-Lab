"""Render the four tables in tables.tex as IEEE-style PNG images (booktabs look).

Run: python render_tables.py   ->  table_png/table_{2,3,4,5}.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
OUT = Path(__file__).parent / "table_png"

TABLES = [
    dict(num="II", caption="ADReSSo21 Data Used in This Study",
         header=["Split", "AD", "CN", "Total"], align="lccc",
         rows=[["Training (5-fold CV)", "87", "79", "166"],
               ["Test (held out)", "35", "36", "71"],
               None,
               ["Total", "122", "115", "237"]]),
    dict(num="III", caption="Test-Set Performance of the Proposed Ensemble ($n=71$; AD Is the Positive Class)",
         header=["Metric", "Value", "95% CI"], align="lcc",
         rows=[["Accuracy", "84.5%", "74.3–91.1%"],
               ["Sensitivity (AD recall)", "82.9%", "67.3–91.9%"],
               ["Specificity (CN recall)", "86.1%", "71.3–93.9%"],
               ["Precision (PPV)", "85.3%", "69.9–93.6%"],
               ["NPV", "83.8%", "68.9–92.3%"],
               None,
               ["F1-score (AD)", "0.841", "–"],
               ["Macro F1-score", "0.845", "–"],
               ["Matthews corr. coef.", "0.690", "–"],
               ["ROC AUC", "0.89", "–"],
               ["Average precision", "0.89", "–"]]),
    dict(num="IV", caption="Comparison With Published Systems on the ADReSSo21 Test Set",
         header=["System", "Modalities", "Inference", "Acc. (%)"], align="lllc",
         rows=[["Luz et al. [35] (baseline)", "A + L", "Offline", "78.87"],
               ["Zhu et al. [20] (WavBERT)", "A + L", "Offline", "83.10"],
               ["Rohanian et al. [15]", "A + L", "Offline", "84.00"],
               ["Pan et al. [8]", "L", "Offline", "84.51"],
               ["Syed et al. [18]", "L", "Offline", "84.51"],
               None,
               ["**RASAD (ours)", "A + L", "Real-time", "**84.51"]],
         note="A: acoustic; L: linguistic."),
    dict(num="V", caption="Projected Screening Performance of RASAD at Different AD Prevalences (per 1,000 People Screened)",
         header=["Prevalence", "PPV (%)", "NPV (%)", "TP", "FP", "FN"], align="cccccc",
         rows=[["5%", "23.9", "99.0", "41", "132", "9"],
               ["10%", "39.9", "97.8", "83", "125", "17"],
               ["20%", "59.9", "95.3", "166", "111", "34"],
               ["30%", "71.9", "92.1", "249", "97", "51"],
               ["50%", "85.6", "83.4", "414", "69", "86"]],
         note="Positive likelihood ratio 5.97; negative likelihood ratio 0.20."),
]

FS, ROW_H, PAD_X = 9.5, 0.215, 0.14  # font size (pt), row height (in), column padding (in)


def text_width(fig, s, bold=False):
    t = fig.text(0, 0, s, fontsize=FS, weight="bold" if bold else "normal")
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def render(t):
    fig = plt.figure(dpi=300)
    cells = [t["header"]] + [r for r in t["rows"] if r]
    ncol = len(t["header"])
    colw = [max(text_width(fig, c.lstrip("*"), c.startswith("**")) for c in (r[i] for r in cells)) + 2 * PAD_X
            for i in range(ncol)]
    width = max(sum(colw), 3.2)
    colw = [w * width / sum(colw) for w in colw]
    nrows = 1 + len(t["rows"])
    n_mid = sum(r is None for r in t["rows"])
    cap_lines = 2 if text_width(fig, t["caption"]) > width else 1
    height = 0.30 + 0.18 * cap_lines + 0.12 + (nrows - n_mid) * ROW_H + 0.12 + (0.22 if t.get("note") else 0.05)
    fig.set_size_inches(width + 0.2, height)
    W, H = width + 0.2, height
    fig.patch.set_facecolor("white")

    y = H - 0.08
    fig.text(0.5, (y - 0.08) / H, f"TABLE {t['num']}", ha="center", va="center", fontsize=FS - 0.5)
    y -= 0.25
    cap = t["caption"] if cap_lines == 1 else _wrap(fig, t["caption"], width)
    fig.text(0.5, (y - 0.09 * cap_lines) / H, cap, ha="center", va="center", fontsize=FS - 0.5,
             fontvariant="small-caps", linespacing=1.3)
    y -= 0.18 * cap_lines + 0.08
    x0 = 0.1

    def rule(yy, lw):
        fig.add_artist(plt.Line2D([x0 / W, (x0 + width) / W], [yy / H, yy / H], lw=lw, color="black"))

    def row(cells_, yy):
        x = x0
        for c, w, a in zip(cells_, colw, t["align"]):
            bold = c.startswith("**")
            c = c.lstrip("*")
            xx = {"l": x + PAD_X, "c": x + w / 2, "r": x + w - PAD_X}[a]
            fig.text(xx / W, (yy - ROW_H / 2) / H, c, ha={"l": "left", "c": "center", "r": "right"}[a],
                     va="center", fontsize=FS, weight="bold" if bold else "normal")
            x += w

    rule(y, 1.0)
    y -= 0.03
    row(t["header"], y)
    y -= ROW_H + 0.02
    rule(y, 0.6)
    y -= 0.03
    for r in t["rows"]:
        if r is None:
            y -= 0.015
            rule(y, 0.6)
            y -= 0.03
            continue
        row(r, y)
        y -= ROW_H
    y -= 0.02
    rule(y, 1.0)
    if t.get("note"):
        fig.text((x0 + 0.02) / W, (y - 0.12) / H, t["note"], ha="left", va="center", fontsize=FS - 1.5)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"table_{ {'II': 2, 'III': 3, 'IV': 4, 'V': 5}[t['num']] }.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def _wrap(fig, s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if text_width(fig, trial) > width - 0.1 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return "\n".join(lines + [cur])


if __name__ == "__main__":
    for t in TABLES:
        print(render(t))
