# Hybrid Sampling + ML/DL for Cardiovascular Disease Prediction

Code for the paper **"Integrating Hybrid Sampling with Machine Learning for
Enhanced Cardiovascular Disease Prediction"** (Daniyal, Jadoon, Miakhil, Iqbal,
Qureshi, Ullah, Rehman).

It implements the pipeline of Fig. 1 on the BRFSS 2015 heart disease
health indicators dataset (`data/heart_disease_health_indicators_BRFSS2015.csv`,
253,680 records, 22 columns):

| Paper section | What the code does | Where |
|---|---|---|
| 3.1–3.2 Data collection & preprocessing | Load CSV, drop duplicates, check missing values, class counts, Figs. 2–3 | `hscvd/data.py`, `hscvd/plots.py` |
| 3.3 Feature selection | Pearson correlation with `HeartDiseaseorAttack`; keeps GenHlth, Age, DiffWalk, HighBP, Stroke, PhysHlth, Diabetes, HighChol, Smoker | `hscvd/data.py`, `hscvd/config.py` |
| 3.4 Hybrid sampling | SMOTEENN followed by Random Oversampling; 3,000 rows per class held out as the test set; remaining data split 80/20 into train/validation; Figs. 4–5 | `hscvd/sampling.py` |
| 3.5 ML models | Decision tree (C4.5-style, entropy), linear SVM, XGBoost | `hscvd/models.py` |
| 4 DL models | MinMax scaling → autoencoder (64-unit encoder) feature extraction → CNN / NN / MLP | `hscvd/models.py` |
| 4 Metrics | Accuracy, precision, recall, F1, AUC (Eqs. 1–5), ROC, confusion matrix | `hscvd/evaluation.py` |
| 5 Results | Tables 3, 4, 5, 7 and Figs. 6–16 | `run_experiments.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Full reproduction, exactly as described in the paper
python run_experiments.py

# Leakage-free evaluation (see "Evaluation protocols" below)
python run_experiments.py --protocol strict --out results/strict

# Quick smoke test (~2 minutes)
python run_experiments.py --sample-frac 0.1 --epochs 1 --skip-cv --out results/smoke
```

Useful options: `--sampling hs` (run only some sampling methods),
`--epochs N`, `--batch-size N`, `--no-autoencoder`, `--skip-cv`,
`--test-per-class N`, `--verbose 1` (Keras progress bars).

Outputs go to `results/` (or `--out`):

```
eda/       correlation_with_target.csv, fig2-fig5 (class / age distributions)
splits/    <method>_{train,val,test}.csv for smoteenn, ros, hs
tables/    table3_validation.csv, table4_test.csv,
           table5_dl_cross_validation.csv, table7_ml_cross_validation.csv
figures/   roc_<method>_{dl,ml}.png (Figs. 6-9), confusion_matrix_HS_*.png
           (Figs. 10-12), learning_curve_HS_*.png (Figs. 13-16)
```

## Model details

| Model | Architecture / settings |
|---|---|
| Autoencoder | Dense(64, ReLU) encoder → Dense(9, sigmoid) decoder, MSE, Adam, 10 epochs; the 64-d encoding is the input to CNN/NN/MLP |
| CNN | Conv1D(8, k=2, ReLU) → MaxPool(2) → Flatten → Dense(16, ReLU) → Dense(1, sigmoid); 20 epochs |
| NN | Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, sigmoid); 10 epochs |
| MLP | Dense(128) → Dense(64) → Dense(32) (ReLU) → Dense(1, sigmoid); 20 epochs |
| DT | `DecisionTreeClassifier(criterion="entropy")` |
| SVM | Linear kernel (`LinearSVC` on MinMax-scaled features) |
| XGBoost | `XGBClassifier` defaults |

All DL models use binary cross-entropy, Adam and batch size 32. Random seed 42.

## Evaluation protocols

**`--protocol paper` (default)** follows Section 3.4 exactly: the *entire*
dataset is resampled with SMOTEENN + ROS, then 3,000 rows per class are drawn
from the resampled data as the test set and the rest is split 80/20.
Cross-validation (Tables 5 and 7) runs on that resampled pool.

Because ROS duplicates minority rows and SMOTE creates synthetic points that
sit between real ones, copies of the same patient end up in both training
and test sets. Memorising models such as a fully grown decision tree or
XGBoost score close to 100% in this setting. That is why the paper reports
HS_DT at 99.7%.

**`--protocol strict`** keeps the test set (3,000 real records per class)
and the validation set apart *before* any sampling, and resamples only the
training data (and, in cross-validation, only the training folds). It shows
how the models do on real patients they have never seen, and we recommend
reporting it next to the paper numbers.

## Improved pipeline (`run_improved.py`)

Tested under the strict protocol, the paper's design choices hurt the models
on real, unseen patients. `run_improved.py` trains both setups on the same
leakage-free split and compares them on the same 6,000 real test records:

| Change | Why |
|---|---|
| Use all 21 features instead of the 9 correlation-selected ones | Pearson correlation misses useful signal (Sex, Income, BMI, Education, ...); XGBoost AUC 0.823 → 0.835 |
| No resampling; class weights instead (`class_weight="balanced"`, `scale_pos_weight`) | SMOTEENN's synthetic points and ENN cleaning distort the data; XGBoost AUC with HS 0.743 vs. 0.823 without |
| Tune the decision cut-off on the validation set (Youden's J) | With imbalanced data the right cut-off is not 0.5; this choice drives accuracy and F1 much more than resampling does |
| Regularise the models: depth-limited DT (`max_depth=8`, `min_samples_leaf=50`), XGBoost with shallow trees, subsampling and early stopping on validation AUC, DL models with early stopping | A fully grown tree memorises records (57.4% on real patients) |
| Add logistic regression | A strong, interpretable baseline on these mostly binary features |

```bash
python run_improved.py                          # all models, ~45 min on 4 CPUs
python run_improved.py --models XGBoost LogReg DT --no-baseline   # ~2 min
```

Writes `results/improved/comparison.csv`, `roc_improved.png` and confusion
matrices.

Results of `python run_improved.py` on the full dataset (6,000 real test
records, 3,000 per class; training 179,024 and validation 44,757 real records):

| Model | Paper setup: Acc / Recall / F1 / AUC (%) | Improved: Acc / Recall / F1 / AUC (%) |
|---|---|---|
| DT | 57.40 / 18.53 / 30.32 / 57.40 | **74.90 / 78.90 / 75.87 / 82.50** |
| SVM | 70.78 / 60.17 / 67.31 / 81.06 | **75.98 / 80.37 / 76.99 / 83.67** |
| XGBoost | 61.58 / 30.47 / 44.23 / 74.31 | **76.00 / 82.77 / 77.52 / 83.99** |
| LogReg | — | 76.00 / 80.33 / 77.00 / 83.70 |
| CNN | 66.85 / 48.70 / 59.50 / 77.58 | **75.80 / 81.73 / 77.16 / 83.74** |
| NN | 64.48 / 40.07 / 53.01 / 74.67 | **76.00 / 81.17 / 77.18 / 83.80** |
| MLP | 63.05 / 34.17 / 48.04 / 72.53 | **75.70 / 81.90 / 77.12 / 83.89** |

Every model improves; the largest gains are in recall (heart disease cases
found). All improved models converge to about 76% accuracy and 0.84 AUC,
which is about the best this dataset supports: BRFSS answers are
self-reported survey data. The 93–99% scores reported in the paper come
from leakage between training and test data (see "Evaluation protocols"),
not from better models.

## Results

`python run_experiments.py` writes Tables 3–5 and 7 to `results/paper/tables/`
(and to `results/strict/tables/` with `--protocol strict`). With the paper
protocol the code reproduces the paper's pattern, e.g. on a 10% sample
HS_DT ≈ 99% and HS_XGBoost ≈ 98% test accuracy. On real, unseen patients
the same models score far lower; see the table in "Improved pipeline".

## Revised manuscript

`paper/` contains the revised manuscript (LaTeX source and compiled
`main.pdf`) with all results above, and `paper/CHANGES.md` lists every
change from the original. To rebuild it after re-running the experiments:

```bash
python run_improved.py                                   # paper vs improved
python run_improved.py --no-baseline --cv --out results/improved_cv
python make_paper_assets.py                              # figures + tables
cd paper && latexmk -pdf main.tex
```

## Notes and differences from the paper

- **Dataset size.** The paper reports 221,542 rows after removing
  duplicates. The Kaggle CSV in `data/` has 253,680 rows, which leaves
  229,781 after removing duplicates; the copy in
  `../Early Detection of Cardiovascular Disease using DL/` has only 244,680
  rows, which leaves 221,542. Use `--data` to switch between them.
- **Autoencoder features.** The original notebook fed the autoencoder's
  *reconstruction* into the classifiers. This code uses the 64-d encoder
  output, which is the usual way to extract features. Use `--no-autoencoder`
  to skip this step.
- **SVM.** `SVC(kernel="linear")` cannot handle about 300k rows in a
  practical time, so the code uses `LinearSVC`, which fits the same
  linear-kernel model.
- **AUC.** AUC is computed from continuous scores (probabilities or decision
  values) rather than from hard labels.
- **MLP.** The paper does not give the MLP's architecture; the 128-64-32 network above is our choice.
