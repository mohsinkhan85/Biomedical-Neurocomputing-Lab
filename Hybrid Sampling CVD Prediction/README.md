# A Leakage-Free ML Pipeline for CVD Prediction

Code for the paper **"A Leakage-Free Machine Learning Pipeline for
Cardiovascular Disease Prediction from Imbalanced Survey Data"** (Daniyal, Jadoon, Miakhil, Iqbal,
Qureshi, Ullah, Rehman). The manuscript is in [`paper/`](paper/) and every
table and figure in it can be regenerated from this code.

The experiments use the BRFSS 2015 Heart Disease Health Indicators dataset
(`data/heart_disease_health_indicators_BRFSS2015.csv`, 253,680 records,
229,781 after removing duplicates) and, for the external comparison, the UCI
Cleveland heart disease data (`data/cleveland/`).

| Paper section | What the code does | Where |
|---|---|---|
| 3.1–3.3 Data, preprocessing, feature selection | Load CSV, drop duplicates, class counts, Pearson correlation ranking | `hscvd/data.py` |
| 3.4 Hybrid sampling | SMOTEENN followed by Random Oversampling of the training data; Figs. 2–3 and the statistics in the text | `hscvd/sampling.py`, `sampling_analysis.py` |
| 3.5 Evaluation protocol | Real test set (3,000 per class) and validation set held out before any sampling | `hscvd/sampling.py` |
| 3.6 Improved pipeline | All 21 features, class weights, regularised models, threshold tuned on validation | `hscvd/models.py`, `hscvd/evaluation.py` |
| 3.7 External evaluation | Improved pipeline on Cleveland, 5 × 5-fold cross-validation | `run_cleveland.py` |
| 4 Models and metrics | DT, SVM, XGBoost, LR; autoencoder + CNN / NN / MLP; accuracy, precision, recall, F1, AUC | `hscvd/models.py`, `hscvd/evaluation.py` |
| 4.4 Statistical analysis | Bootstrap CIs, odds ratios, calibration and prior correction | `run_robustness.py` |
| 5 Results | Tables 3–9, Figs. 4–7 | `run_improved.py`, `run_ablation.py`, `run_robustness.py`, `run_cleveland.py`, `make_paper_assets.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
# Figs. 2-3 and the hybrid sampling statistics of Section 3.4
python sampling_analysis.py

# Table 3 (left): baseline HS pipeline on the real test set
python run_improved.py

# Table 4: ablation study
python run_ablation.py

# Table 3 (right), Table 6, Figs. 4-6: improved pipeline with 5-fold CV
# (DL results depend on the training order, so the improved column of the
# paper is taken from this run, not from run_improved.py above)
python run_improved.py --no-baseline --cv --out results/improved_cv

# Tables 5, 7, 8 and Fig. 7: bootstrap CIs, odds ratios, calibration
python run_robustness.py

# Table 9, Cleveland rows
python run_cleveland.py

# Build the manuscript's figures and tables, then the PDF
python make_paper_assets.py
cd paper && latexmk -pdf main.tex
```

All runs use random seed 42, and TensorFlow runs deterministically, so
repeated runs give identical numbers.

## Resampling before splitting (`run_experiments.py`)

`run_experiments.py` also supports the resample-before-split setting that
is common in published pipelines (`--protocol paper`): the *entire* dataset
is resampled with SMOTEENN + ROS before the test set is drawn. Duplicated
and cleaned records then reach the test set, and the HS models appear to
reach 94.7–99.8% accuracy; the paper cites this figure once (Section 5.5) to
explain why published results can be higher. `--protocol strict` resamples
only the training data. A quick smoke test runs in about two minutes:

```bash
python run_experiments.py --sample-frac 0.1 --epochs 1 --skip-cv --out results/smoke
```

Other options: `--sampling hs`, `--epochs N`, `--batch-size N`,
`--no-autoencoder`, `--skip-cv`, `--test-per-class N`, `--verbose 1`.

## Results on real, unseen patients

6,000 real test records (3,000 per class); training 179,024 and validation
44,757 real records. Accuracy / recall / F1 / AUC, in %:

| Model | Baseline HS pipeline | Improved pipeline |
|---|---|---|
| DT | 57.40 / 18.53 / 30.32 / 57.40 | **74.90 / 78.90 / 75.87 / 82.50** |
| SVM | 70.78 / 60.17 / 67.31 / 81.06 | **75.98 / 80.37 / 76.99 / 83.67** |
| XGBoost | 61.58 / 30.47 / 44.23 / 74.31 | **76.00 / 82.77 / 77.52 / 83.99** |
| LR | — | **76.00 / 80.33 / 77.00 / 83.70** |
| CNN | 66.85 / 48.70 / 59.50 / 77.58 | **75.55 / 80.07 / 76.61 / 83.41** |
| NN | 64.48 / 40.07 / 53.01 / 74.67 | **75.88 / 78.80 / 76.57 / 83.85** |
| MLP | 63.05 / 34.17 / 48.04 / 72.53 | **75.47 / 82.40 / 77.06 / 83.79** |

The improved pipeline makes four changes, each quantified in the paper's
ablation study (Table 4):

| Change | Why |
|---|---|
| All 21 features instead of the 9 correlation-selected ones | Pearson correlation misses useful signal (e.g. sex, OR 2.13); XGBoost AUC 0.823 → 0.835 |
| Class weights instead of resampling | SMOTEENN and ENN cleaning distort the data; XGBoost AUC 0.743 with HS vs. 0.823 without |
| Decision threshold tuned on the validation set (Youden's J) | With imbalanced data the right cut-off is not 0.5 |
| Regularised models (depth-limited DT, shallow early-stopped XGBoost, early-stopped DL models) | A fully grown tree memorises records (57.4% on real patients) |

All gains are significant (95% bootstrap CIs). The raw probabilities of the
class-weighted models overestimate risk three- to fourfold; a prior
correction for the class weights restores calibration (Brier score
0.070–0.072 vs. 0.084 for the prevalence alone) without changing AUC or
classifications.

Applied unchanged to the Cleveland data, the improved pipeline reaches
80.1–81.3% accuracy and an AUC of 0.893–0.900 (5 × 5-fold cross-validation).

## Implementation notes

- **Dataset copy.** The copy of the BRFSS CSV in
  `../Early Detection of Cardiovascular Disease using DL/` is truncated
  (244,680 rows, 221,542 after removing duplicates); the paper uses the full
  file in `data/`. Use `--data` to switch.
- **Autoencoder features.** The DL models use the 64-dimensional encoder
  output as features; `--no-autoencoder` feeds the scaled features directly.
- **SVM.** `SVC(kernel="linear")` cannot handle about 300k rows in practical
  time, so the code uses `LinearSVC`, which fits the same linear-kernel model.
- **AUC** is computed from continuous scores (probabilities or decision
  values), not from hard labels.
