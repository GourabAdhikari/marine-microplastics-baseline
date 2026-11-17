# LightGBM grouped five-fold reproduction

## Result

**Pooled out-of-fold R² = 0.4726308398591973**, which rounds to the reported **0.473**. Difference from the rounded headline: **−0.0003691601408027**. All five fold R² values also match the original saved notebook to its displayed three decimals. This is successful reproduction at the reported precision, not proof of bitwise identity with historical predictions (not supplied).

R², MAE and RMSE below are on `log10(concentration + 0.001)`, not raw concentration. R² is not prediction accuracy.

| Fold | Train rows | Test rows | Test studies | R² | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10,853 | 6,135 | 1 | −0.555996 | 0.727919 | 0.909012 |
| 2 | 14,274 | 2,714 | 11 | 0.003502 | 0.812758 | 1.003800 |
| 3 | 14,276 | 2,712 | 20 | 0.301706 | 1.602672 | 2.273102 |
| 4 | 14,274 | 2,714 | 20 | −0.437683 | 1.275470 | 1.739840 |
| 5 | 14,275 | 2,713 | 20 | 0.332296 | 0.901872 | 1.234621 |

Pooled MAE: **0.996377**; pooled RMSE: **1.418198**; study-balanced MAE: **1.060942**. **63.94%** of predictions are within one log10 unit of the offset target. Arithmetic mean fold R² is **−0.071235**, not 0.473: each fold uses a different target-variance denominator, whereas the headline pools all held-out predictions.

## Data and exact procedure

- Input `data/noaa_microplastics.csv`: **29,076 rows × 35 columns**, 96 accessions and 68 organisations.
- Original ocean-water/pieces-m3 filter, preserving row order: **16,988 rows**, **72 linked studies**, 73 accessions, 73 publications, 52 organisations. No further row removal.
- Study groups are connected components through shared DOI or accession, using the original `linked_groups` implementation. An independent union-find check confirms these groups.
- Historical assumptions retained exactly: Barrows DOI `10.1016/j.envpol.2018.02.062` (**1,389 rows**) and Dehm DOI `10.1016/j.marpolbul.2020.111239` (**81 rows**) multiplied by **10^6** in memory. These assumptions were not independently revalidated. The 33 bongo-net records remain unchanged.
- Original features: latitude, longitude, year, month sine/cosine, log10 mesh, log1p depth and sampling-method family. Training-fold median imputation, StandardScaler, dense OneHotEncoder with unknown categories ignored; test numeric values clipped to training ranges as in the original wrapper.
- Exact `GroupKFold(5)` with no shuffle; original row ordering and group labels. Test sizes: **6,135 / 2,714 / 2,712 / 2,714 / 2,713**.
- Original LightGBM: **500 estimators, learning_rate=0.05, random_state=0, n_jobs=4, verbose=-1**, remaining parameters at library defaults. Full parameters saved in `model_parameters.json`.
- Only five LightGBM fits; no model searches, alternative models or architecture changes. The copied split-definition cell constructs other split indices but no models are evaluated on them.

## Environment and caveats

Original notebook metadata specifies Python **3.13.5**, without a full historical runtime manifest. Reproduction uses Python **3.12.7**, macOS ARM64, in `reproduction_outputs/venv/`. The starting host environment lacked LightGBM and had different NumPy/pandas/sklearn versions; it was not modified.

Core versions match `uv.lock`: **LightGBM 4.7.0, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1**. Despite the Python/platform caveat, headline and fold-level R² reproduce to the original displayed precision. See `environment.json`, `requirements-frozen.txt`, and `install.log`.

## Execution and verification

`lightgbm_reproduction.ipynb` is fully executed, with outputs. Preprocessing cells were copied verbatim from the original notebook and checked against the paired Python source; relevant helper definitions were extracted verbatim using AST. Unrelated plotting, R, other model training and search cells were excluded.

**31 focused checks passed**: original counts and fold sizes; historical scaling and target transform; independent grouping; no row/study/DOI/accession leakage; one test assignment per row; training-only imputation; finite predictions; R² agreement with sklearn; SHA-256 equality of every original tracked file before and after execution. `git diff --exit-code` also passed. No execution errors occurred.

Artifacts, all under `reproduction_outputs/`:
- `lightgbm_reproduction.ipynb`: executed notebook.
- `run_reproduction.py`: notebook builder/executor; rerun from repository root with `PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python reproduction_outputs/run_reproduction.py`.
- `summary.json`, `fold_metrics.csv`: full-precision metrics.
- `out_of_fold_predictions.csv`: source-row mapping (zero-based data row), study/DOI/accession, fold, target and prediction.
- `unit_adjustments.json`, `model_parameters.json`, `input_hashes.json`, `tests.json`: audit records.

Reproduction complete. Original source and dataset unchanged. No additional experiments performed.
