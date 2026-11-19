# Beyond Fine-Tuning: Label-Efficient and Safe Cross-Site Adaptation of Echocardiographic Ejection Fraction Models

Code for the paper by Muhammad Mohsin Khan, Takuro Ishii and Yoshifumi Saijo (Tohoku University / PAF-IAST),
submitted to the *IEEE Journal of Biomedical and Health Informatics*.
Supported in part by JSPS (Invitational Fellowship for Research in Japan, Grant JSPS/OF361, Fellowship ID L26517).

An EF model trained at one site (EchoNet-Dynamic) loses accuracy at new sites (CAMUS, EchoNet-Pediatric).
We explain why, and show that frozen models trained with scanner-simulating augmentation, combined with
PanEcho and EchoPrime and recalibrated by a small ridge correction from 20–50 labelled local patients,
beat full fine-tuning at every label budget on a CPU, with a safety mode and conformal intervals that
protect reduced-EF patients.

## Data (not included)

All three datasets are public but require registration; download them from their owners and keep their licences:

| Dataset | Use | Source |
|---|---|---|
| EchoNet-Dynamic | training, source test | https://echonet.github.io/dynamic/ |
| CAMUS | external test (adults) | https://www.creatis.insa-lyon.fr/Challenge/camus/ |
| EchoNet-Pediatric | external test (children) | https://echonet.github.io/pediatric/ |

Paths default to a Windows layout (`D:\EchoNet\...`); change them with each script's command-line options.
No patient data, model weights or result files are stored in this repository.

## Pipeline

All scripts sit in one folder because they import each other. Run them from this folder, in this order.
Each script's header explains its options.

| Step | Script | What it does | Hardware |
|---|---|---|---|
| 1 | `echonet_baseline.py` | Baseline R(2+1)D-18 EF model on EchoNet-Dynamic (3 seeds) | GPU |
| 1 | `train_physaug.py` | Training with scanner-simulating (physics-informed) augmentation | GPU |
| 1 | `train_genaug.py` | Control: generic nnU-Net-style intensity augmentation | GPU |
| 2 | `camus_eval.py` | Test on CAMUS (half-cycle → mirrored clip, biplane and A4C references) | GPU |
| 2 | `pediatric_eval.py` | Test on EchoNet-Pediatric | GPU |
| 2 | `physics_gap.py`, `halfcycle_diag.py` | Analyses of why the models fail (image physics, clip format) | CPU / GPU |
| 3 | `panecho_export.py` | PanEcho embeddings and EF on all datasets | GPU |
| 3 | `echoprime_export.py` | EchoPrime embeddings and retrieval EF on all datasets | GPU |
| 4 | `adapt_benchmark.py` | 8 adaptation methods, 0–1000 labels, 20 splits; `--export` writes features | GPU |
| 5 | `cheap_analysis.py` | Fast CPU adaptation methods and conformal intervals | CPU |
| 5 | `fusion_analysis.py` | **Combined recalibration (proposed method)** | CPU |
| 5 | `safety_analysis.py` | Safety mode (weighted ridge) and review flag | CPU |
| 5 | `j_analysis.py` | Ensemble control, clinical agreement, subgroups, sensitivity, intervals | CPU |
| 6 | `make_h.py`, `make_j.py`, `physaug_figure.py` | Figures and tables of the paper | CPU |

## Requirements

Python 3.10+, see `requirements.txt`. Experiments ran on one NVIDIA RTX 2080 Ti (11 GB);
steps 5–6 need only a CPU.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## Contact

Muhammad Mohsin Khan — mohsin.khan@spcai.paf-iast.edu.pk
