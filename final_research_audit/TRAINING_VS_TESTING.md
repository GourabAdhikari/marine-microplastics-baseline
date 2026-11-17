# Stage 4 S2: training versus study-held-out testing

Scenario: `historical_hypothetical_unit_scenario`. Target: `log10(concentration + 0.001)`. Historical Barrows/Dehm ×10^6 assumptions are unverified, not physical ground truth.

## Exact models and replication

No fitted prior LightGBM pipeline/booster artifacts were available: the old NPZ checkpoints contain held-out predictions and row IDs, not estimators. Exactly **five new fits** reconstructed S2's locked outer-fold models; no original model, inner grid or alternate parameter was trained. Folds 1–2 use C5_row_and_feature_subsample; folds 3–5 use C4_feature_subsample. Full parameters and original train-only preprocessing are preserved in checkpoint metadata. Seed 0, four threads, same core library versions.

**Every reconstructed held-out prediction matches the frozen Stage 4 S2 prediction bit-for-bit (maximum difference 0)**. Training scores therefore refer to the reconstructed locked S2 fold models, not a newly selected variant. Complete fitted pipelines with training clipping ranges and train/test predictions are saved locally for reuse. No cache in any prior output directory was changed.

## Counts and per-fold R²

| fold | candidate | train_rows | test_rows | train_studies | test_studies | train_R2 | test_R2 | R2_train_minus_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C5_row_and_feature_subsample | 10853 | 6135 | 71 | 1 | 0.770035279 | -0.431151475 | 1.201186754 |
| 2 | C5_row_and_feature_subsample | 14274 | 2714 | 61 | 11 | 0.779624624 | -0.424101081 | 1.203725704 |
| 3 | C4_feature_subsample | 14276 | 2712 | 52 | 20 | 0.786501009 | 0.408333968 | 0.378167041 |
| 4 | C4_feature_subsample | 14274 | 2714 | 52 | 20 | 0.795688661 | -0.007517498 | 0.803206159 |
| 5 | C4_feature_subsample | 14275 | 2713 | 52 | 20 | 0.782515582 | 0.268116985 | 0.514398597 |

## MAE and RMSE (log10 units)

| fold | train_MAE | test_MAE | train_RMSE | test_RMSE |
| --- | --- | --- | --- | --- |
| 1 | 0.627142379 | 0.717709557 | 1.036610248 | 0.871782243 |
| 2 | 0.596943598 | 1.018206116 | 0.943549167 | 1.199994825 |
| 3 | 0.465979712 | 1.535859501 | 0.655312640 | 2.092368278 |
| 4 | 0.570463984 | 1.066745978 | 0.898070764 | 1.456478757 |
| 5 | 0.580824715 | 0.964776948 | 0.940247229 | 1.292594435 |

- **Mean training R² across five folds: 0.782873030949.**
- **Mean testing R² across five folds: -0.037263820043.**
- **Pooled study-held-out R² across all 16,988 rows: 0.5256280075581679.**
- Mean of the five training-minus-testing R² gaps: **0.820136850992**.

The training average is unweighted over the five training fits. Training sets overlap heavily; there is no unique once-per-row training prediction and no pooled training R² is reported. The held-out pool contains each eligible row exactly once across the five disjoint test partitions. There are 72 DOI/accession-linked studies; test sizes remain 6,135 / 2,714 / 2,712 / 2,714 / 2,713.

## Generalization gap and interpretation

Training scores are resubstitution on samples already used to fit the estimator and preprocessing, not validation. The large positive fold-wise gaps and negative testing R² in folds 1, 2 and 4 show much weaker performance when whole studies are withheld. They are consistent with overfitting/study-specific pattern learning and limited cross-study transfer. They cannot attribute all loss to model complexity: train/test populations differ in protocols, geography, target range and variance. Each R² uses its own set's mean/variance, so a gap is descriptive, not a calibrated estimate of pure algorithmic overfitting.

Positive pooled R² alongside negative mean fold R² is not a contradiction: pooled R² includes between-study target variation, whereas each fold's R² uses that fold's own denominator. The largest 6,135-row study alone occupies fold 1. Do not subtract mean training R² from pooled held-out R² and call that the mean generalization gap.

## Original notebook: was approximately 0.7 a training score?

**Original LightGBM training/resubstitution R²: NOT AVAILABLE / not verified.** No such saved metric was identified, and the original model was not retrained.

The identifiable saved LightGBM value is **0.764 (rounded to three decimals)** in original notebook cell 27 (zero-based), model comparison. Its exact definition is **pooled out-of-fold R² under shuffled row-wise KFold(5, random_state=0)**, on the same log10 target; the scoring cell calls `out_of_fold` and scores withheld rows, not the training rows. Samples from a study can appear on both sides of this random split, so it is easier than whole-study-held-out testing. The displayed 0.473 beside it is grouped held-out R². Thus **0.764 must not be called original training R² or compared as if it were the same split as S2's grouped result**. The unrounded random-CV value is not available in the saved output. Source/output evidence: `original_notebook_metric_evidence.json`; Python counterparts are `microplastics.py:302`, `:468-486`.

## Verification and rerun

104 focused checks passed; 27377 pre-existing paths remain byte-identical. Tests cover immutable artifacts, exact rows/groups/targets/folds, no study overlap, exact held-out predictions, train-only imputer/scaler, saved-model prediction equivalence, independent R² calculation, all four reference metrics and the hard five-fit bound. Actual new fits in the full audit: **5**. `--tests-only` cannot fit.

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python final_research_audit/analyze.py
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python final_research_audit/analyze.py --tests-only
```

Default reruns reuse verified local caches; the completed marker prohibits additional fits. All new files are under `final_research_audit/`. No downloads, searches, target edits, chronological tests or independent external evaluations were run.
