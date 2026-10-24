# Cleveland heart disease data

`heart_disease_cleveland.tsv.gz` is the UCI Cleveland heart disease dataset
(303 patients, 13 clinical features) as curated by the Penn Machine Learning
Benchmarks (PMLB):
https://github.com/EpistasisLab/pmlb/tree/master/datasets/heart_disease_cleveland

SHA-256: f901a05ccbd6bd1701b83ea8baa1c454be72cb8c41b00e2cec1bb09d6154e691
(matches the Git LFS pointer in the PMLB repository).

PMLB re-codes the categorical variables to start at 0 and binarises the
target (0 = no disease, 1 = UCI `num` > 0; 164 / 139 patients). Six values of
`fluoroscopy_vessels` and `thallium_stress_test` are missing.

Original source: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R.
(1989). Heart Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
