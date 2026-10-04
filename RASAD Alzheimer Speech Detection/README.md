# RASAD: tabular results

LaTeX tables for *RASAD: Real-Time Alzheimer's Speech Analysis & Diagnosis*.

- `tables.tex`: seven IEEEtran tables (dataset, configuration, modality ablation, classifier AUCs, confusion matrix, detailed metrics with 95% CIs, ADReSSo21 comparison). They contain only values reported in the paper or derived from its confusion matrix.
- `compute_metrics.py`: computes every derived metric from the Fig. 6 confusion matrix (TP 29, FN 6, TN 31, FP 5).

The inference time (36.98 ms per batch) stays in the text of Sec. IV-D.

## Inconsistencies in the current draft to fix before submission

| # | Item | Where | Conflict | Suggested fix |
|---|------|-------|----------|---------------|
| 1 | Unimodal accuracies | Abstract / Sec. IV-A / Fig. 2 caption | Acoustic 73.2 / 76.1 / 81.7 %; linguistic 81.7 / 78.9 / 73.2 % | Use the abstract values (73.2 acoustic, 81.7 linguistic). Change Sec. IV-A (76.1/78.9 → 73.2/81.7) and the Fig. 2 caption (swapped), and regenerate the Fig. 2 bar chart if its bars show other values. |
| 2 | F1-score | Sec. IV-B (CM paragraph) | Says ≈ 0.89; the confusion matrix gives **0.841** (AD) / 0.845 (macro) | Use 0.84 |
| 3 | Average precision | Sec. IV-B text vs. Fig. 5 caption | 0.85 vs. 0.89 | Use 0.89: change the Sec. IV-B text (0.85 → 0.89) |
| 4 | Inference time | Sec. IV-D text vs. Fig. 10 caption | 45.2 ms vs. 36.98 ms | Use 36.98 ms: change the Sec. IV-D text (45.2 → 36.98) |
| 5 | Dataset size | Sec. III | "87 AD and 79 CN" is the training split only; test set is 35 AD / 36 CN (n = 237 total) | State both splits (Table `tab:dataset`) |
| 6 | Fig. 3 | Sec. IV-B | Computed on *training* data but quotes the 84.5 % *test* accuracy | Report test-set numbers per classifier (Table `tab:classifiers`) |
| 7 | Table I corpora | Pan et al. [8], Rohanian et al. [15] | Listed as ADReSS. Both are ADReSSo 2021 challenge papers | Change to ADReSSo21. Rohanian et al. report 84 % overall. |
| 8 | Pan et al. acoustic | Sec. II vs. Table I | 74.6 % vs. 74.7 % | Use one value |
| 9 | Nasreen et al. | Sec. II-D vs. Table I | "roughly 90 %" vs. 87 % | Use one value |
| 10 | Missing words | Abstract, Contributions | "recordings from ___ speakers", "combined by ___", "per batch on ___", "retaining ___ components", "under ___ on consumer hardware" | Fill in |
| 11 | Ref [6] | Throughout | Cited as "Qi et al." but [6] is Mobtahej et al. (2026) | Fix the citation or the author name |
| 12 | Incomplete sentences | Sec. II-A ("Pan et al. (2021) reported this gap directly within a single setup:"), Sec. II-F | Sentences cut off | Complete them |

Note: 84.51 % equals 60/71, the same test accuracy reported by Pan et al. and Syed et al. on ADReSSo21. The Wilson 95 % CI for accuracy is 74.3–91.1 %, so differences of a few points between systems on this 71-speaker test set are not statistically meaningful. That supports the limitation stated in the Conclusion.
