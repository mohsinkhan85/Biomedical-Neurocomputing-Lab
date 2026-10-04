# RASAD explainability (XAI) analysis

`xai_analysis.py` produces every number, table and figure for the XAI section (`xai_section.tex`) from the trained RASAD model. Run it once on the real features. Nothing in the section text should be filled in by hand.

## 1. Install

```bash
pip install -r requirements.txt
python xai_analysis.py --selftest --out selftest   # about 2 minutes; synthetic data, checks the install only
```

The self-test output is marked `SYNTHETIC`. Do not use it in the paper.

## 2. Prepare the inputs

Put these files in one folder (e.g. `features/`). Rows must be in the same speaker order across all files of a split.

| File | Content | Shape |
|---|---|---|
| `train_acoustic.npy`, `test_acoustic.npy` | Wav2Vec2 embeddings exactly as fed to the model | (n, d_a) |
| `train_bert.npy`, `test_bert.npy` | Fine-tuned BERT [CLS] embeddings | (n, 768) |
| `train_tfidf.npz` (or `.npy`), `test_tfidf.npz` | TF-IDF matrix (`scipy.sparse.save_npz`) | (n, V) |
| `train_labels.npy`, `test_labels.npy` | 1 = AD, 0 = CN | (n,) |
| `tfidf_vocab.txt` | One term per line, in column order: `"\n".join(vectorizer.get_feature_names_out())` | V lines |
| `test_ids.txt` (optional) | Speaker IDs, one per line | 71 lines |

The fused vector must be `[acoustic | bert | tfidf]` in that order, the same order used for training.

Save the trained model with joblib, as either:

```python
joblib.dump(pipeline, "model.joblib")                      # sklearn Pipeline: [StandardScaler] -> PCA -> VotingClassifier
# or
joblib.dump({"scaler": scaler_or_None, "pca": pca, "ensemble": voting_clf}, "model.joblib")
```

## 3. Run

```bash
python xai_analysis.py --data-dir features --model model.joblib --out xai_results
```

The first printed line is the test accuracy of the loaded model. **It must be 0.8451.** If it isn't, the wrong model or feature order was loaded, so stop and fix that first.

Runtime is roughly 10–60 minutes on a laptop CPU, depending on how many PCA components were kept.

## 4. Outputs (`xai_results/`)

| File | Use in paper |
|---|---|
| `xai_tables.tex` | Table: modality contribution, Table: top terms. Paste directly. |
| `fig_xai_modality.pdf` | Figure: modality contribution by class |
| `fig_xai_terms.pdf` | Figure: terms pushing toward AD / CN (replaces the PCA-component Fig. 8) |
| `case_explanations.csv` | Numbers for the case-study paragraph (confident AD, confident CN, each error) |
| `stability_summary.json`, `stability_folds.csv` | Numbers for the stability paragraph |
| `modality_attribution.csv`, `term_attribution.csv`, `shap_values_test.npy` | Full results / supplementary material |

Send the whole `xai_results/` folder back, and the section text in `xai_section.tex` can then be completed.

## Method in one paragraph

SHAP values for the ensemble's P(AD) are computed in PCA space with a permutation explainer (100 training speakers as background). PCA and standardisation are linear, so each component's SHAP value is split exactly over the original Wav2Vec2, BERT and TF-IDF features. The script checks that the attributions still sum to the model output (additivity error is printed). Attributions are then summed per modality, ranked per TF-IDF term, shown for individual speakers, and recomputed after refitting the pipeline on 5 cross-validation folds to measure stability.
