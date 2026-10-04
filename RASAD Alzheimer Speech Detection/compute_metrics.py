"""Derive the RASAD test-set metrics from the confusion matrix reported in the paper (Fig. 6).

Every number in tables.tex that is marked as derived comes from this script, so the
tables stay consistent with the confusion matrix. Run: python compute_metrics.py
"""
from math import sqrt

# Fig. 6, ADReSSo21 test set (AD = positive class)
TP, FN = 29, 6   # 35 AD speakers
TN, FP = 31, 5   # 36 CN speakers


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def main():
    n = TP + TN + FP + FN
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    f1_ad = 2 * ppv * sens / (ppv + sens)
    f1_cn = 2 * npv * spec / (npv + spec)
    acc = (TP + TN) / n
    bal_acc = (sens + spec) / 2
    mcc = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    po, pe = acc, ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / n**2
    kappa = (po - pe) / (1 - pe)

    rows = [
        ("Accuracy", acc, wilson(TP + TN, n)),
        ("Balanced accuracy", bal_acc, None),
        ("Sensitivity / Recall (AD)", sens, wilson(TP, TP + FN)),
        ("Specificity / Recall (CN)", spec, wilson(TN, TN + FP)),
        ("Precision / PPV (AD)", ppv, wilson(TP, TP + FP)),
        ("NPV / Precision (CN)", npv, wilson(TN, TN + FN)),
        ("F1 (AD)", f1_ad, None),
        ("F1 (CN)", f1_cn, None),
        ("Macro F1", (f1_ad + f1_cn) / 2, None),
        ("MCC", mcc, None),
        ("Cohen's kappa", kappa, None),
        ("LR+", sens / (1 - spec), None),
        ("LR-", (1 - sens) / spec, None),
    ]
    print(f"n = {n}  (AD = {TP + FN}, CN = {TN + FP})")
    for name, v, ci in rows:
        ci_s = f"  95% CI [{ci[0]*100:.1f}, {ci[1]*100:.1f}] %" if ci else ""
        print(f"{name:28s} {v:.4f}{ci_s}")


if __name__ == "__main__":
    main()
