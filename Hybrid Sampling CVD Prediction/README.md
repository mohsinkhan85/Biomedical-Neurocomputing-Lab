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

## Results

See `results/paper/tables/` and `results/strict/tables/`. Summary of the test
set (Table 4) from our full run:

RESULTS_PLACEHOLDER

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
