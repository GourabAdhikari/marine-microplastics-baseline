You are an experienced machine-learning research engineer working inside the **existing cloned `marine-microplastics-baseline` repository**. Your task is to **attempt to improve the correctly reproduced LightGBM grouped-study benchmark while preserving the original experiment and scientific integrity**.

**Work autonomously**: inspect the project, write necessary code, run all requested experiments, create and execute the notebook cells, debug failures, run tests, and deliver a complete, reproducible report. Do not ask me to manually write/run notebook cells or copy files between project folders. Stop at the end of the bounded experiment; do not automatically begin another phase.

## 0. Non-negotiable baseline and objective

The original LightGBM reproduction has **already succeeded**. Read `reproduction_outputs/REPORT.md`, `summary.json`, `fold_metrics.csv`, `out_of_fold_predictions.csv`, `model_parameters.json`, `unit_adjustments.json`, and the executed reproduction notebook before modifying anything. If an artifact is named differently, discover the actual path and record it; do **not** regenerate the baseline needlessly.

Reference values for the reproduction, scored on `log10(concentration + 0.001)`:

- **Pooled grouped five-fold OOF R²:** `0.4726308398591973` (reported `0.473`).
- **Pooled MAE:** `0.996377` log10 units; **pooled RMSE:** `1.418198`.
- **Study-macro MAE:** `1.060942` log10 units; **within ±1 log10 unit:** `63.94%`.
- **Mean individual fold R²:** approximately `-0.071235`.
- **Rows and groups:** `16,988` eligible ocean-water samples; `72` DOI/accession-linked studies, from `29,076` original rows.
- **Fixed outer GroupKFold(5) test sizes:** `6,135 / 2,714 / 2,712 / 2,714 / 2,713` in the _same order and row IDs_ as the reproduction.
- **Original LightGBM:** `LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=0, n_jobs=4, verbose=-1)` plus original package defaults.

**Goal:** seek **R² > 0.4726308398591973 on the same reported benchmark** _and_ avoid unacceptable deterioration in per-study errors. Treat the desired higher number as an aspiration, **never a promised result**. The existing outer-fold outcomes and the 0.473 score are already known; improvements on these reused folds remain **exploratory**, even if all tuning is correctly nested. Genuine independent performance requires previously unseen studies.

The original report used the historical `×10^6` adjustments on Barrows (1,389 rows) and Dehm (81 rows). These adjustments remain **scientifically unresolved**. **For exact benchmark comparison, freeze the reproduction scenario exactly as originally implemented**—but label every result `historical_hypothetical_unit_scenario`, not a certified measurement correction or physical prediction of true synthetic microplastic concentration. Do not derive or change any unit multipliers based on scores. Do not compare raw-label R² and adjusted-label R² as if they measured the same outcome.

## 1. Folder isolation and preservation

1. Confirm you are in the cloned baseline repository, **not** the older `microplastic-2` folder.
2. Never overwrite `microplastics.ipynb`, `microplastics.py`, `gamm.R`, `data/`, `uv.lock`, `.git/`, or anything in `reproduction_outputs/`.
3. Create only **`optimization_outputs/`** and a new source subfolder there. If this directory already holds earlier experiments, create a versioned `optimization_outputs/run_02/` rather than overwriting them.
4. At start, record Git status, SHA-256 hashes of protected files, environment/package versions, CPU thread budget, seeds, row IDs, model configurations, and the exact path of every input artifact. Verify hashes again at completion. The raw dataset must remain byte-identical.
5. Reuse reproduction predictions and fold definitions, not a handwritten recreation. **Do not rerun the old entire notebook**. Only regenerate a baseline fold fit when needed to make a fair paired comparison; log any such fit.
6. Use fixed seeds and deterministic settings where possible. Limit unnecessary downloads, hyperparameter searches and background processes.

## 2. Strict data and evaluation invariants

Read and reuse the original preprocessing and study grouping, including:

- Ocean water + `pieces/m3` selection and original row order.
- Exactly the reproduction's 72 DOI/accession linked-group IDs, with a separate sanity-check that grouping does not join missing identifiers.
- The frozen historical unit scenario for the **reproduction benchmark only**.
- `y = log10(concentration + 0.001)`.
- Features: latitude, longitude, year, month sine/cosine, `log10(mesh size)`, `log1p(sample depth)`, and sampling-method family as used in the reproduction.
- Train-fold median imputation, scaling and categorical encoding, including unseen-category behavior and test-numeric clipping from original wrappers.

The original feature-processing pipeline must be fitted **inside each training fold**. No study IDs, DOIs, accessions, target-derived summaries, target bins, held-out sample measurements or fold identity may become predictors. Do not drop rows, silently exclude hard studies, reassign folds, or relabel units.

For direct comparisons, evaluate all candidates against the **same held-out target values, row ordering, folds, and study IDs**. Every row receives exactly one outer OOF prediction. Calculate pooled R² from the concatenated OOF predictions—not by averaging per-fold R². Report both.

## 3. Budgeted optimization design — make it rigorous, not huge

Implement one **predeclared, nested grouped tuning experiment**. Use the _already fixed_ five outer GroupKFold folds. On each outer training subset, perform **GroupKFold(3)** over its training studies only to rank candidate configurations. Confirm each inner training partition contains multiple independent study groups and record group counts. Do not use any outer-test labels to choose a parameter, fit preprocessing, stop training or choose among candidates.

**Freeze the following candidate registry before looking at tuned outer-fold outcomes.** Test exactly eight configurations (including baseline). Keep the package's LightGBM parameter names and check they are valid in the installed version.

| ID                             | Main differences relative to reproduced model                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `C0_original`                  | Exact reproduced default model (500 trees; learning rate 0.05)                                                     |
| `C1_shallow`                   | `max_depth=4`, `num_leaves=15`, `min_child_samples=40`                                                             |
| `C2_conservative`              | `max_depth=5`, `num_leaves=20`, `min_child_samples=70`, `reg_lambda=5.0`                                           |
| `C3_strong_regularization`     | `max_depth=5`, `num_leaves=20`, `min_child_samples=50`, `reg_alpha=0.5`, `reg_lambda=15.0`                         |
| `C4_feature_subsample`         | `max_depth=5`, `num_leaves=20`, `min_child_samples=50`, `colsample_bytree=0.8`, `reg_lambda=5.0`                   |
| `C5_row_and_feature_subsample` | C4 plus `subsample=0.8`, `subsample_freq=1`                                                                        |
| `C6_slower`                    | `learning_rate=0.03`, `n_estimators=800`, `max_depth=5`, `num_leaves=20`, `min_child_samples=50`, `reg_lambda=5.0` |
| `C7_fewer_trees`               | `learning_rate=0.05`, `n_estimators=250`, `max_depth=5`, `num_leaves=20`, `min_child_samples=50`, `reg_lambda=5.0` |

These are **hypotheses, not guaranteed optimal settings**. Preserve all other baseline settings unless required by a named candidate. If a candidate causes a compatibility error, document the issue; do not silently replace it with an entirely new model.

**Bound:** at most `5 outer × 3 inner × 8 candidates = 120` inner LightGBM fits, plus up to five outer refits of the selected configuration and a small number of essential verification fits. Do not invoke Optuna, evolutionary search, 1000-candidate grids, GPU infrastructure, or feature-download pipelines. If more than 120 inner fits would be needed, stop and report why **before** running them. Set learners to a moderate thread cap (e.g. 2–4 threads), depending on actual hardware. Use checkpointing keyed by hashes of code, data, splits, configs and versions so interrupted work resumes without repeating valid fits.

**No early stopping on the same inner fold used to judge a candidate.** Use the fixed estimator counts above for the primary comparison. If additional early-stopping research is proposed, leave it as an unexecuted recommendation, not an unplanned extension of this run.

### Predeclared selection rule

For each outer fold, obtain **inner out-of-fold predictions** for every candidate using identical inner study splits and compute:

- Primary: pooled inner OOF R² (on the common inner evaluation observations).
- Secondary: study-macro MAE and the 90th percentile of study MAE; report worst-study MAE.

Select **the candidate with the highest inner pooled R²**, with a prespecified safety constraint: it must not worsen inner study-macro MAE by more than **5%** relative to `C0_original`. If no nonbaseline candidate meets both criteria and improves inner R², choose `C0_original`. If inner pooled R² ties within `0.005`, prefer the candidate with lower study-macro MAE; if still tied, prefer the original/simple configuration according to ID order.

Do not retroactively change this rule after inspecting outer-fold results. Record each candidate's inner metrics and the selection reason for each outer fold. Report a _separate, descriptive_ comparison of raw candidate outer scores only if you evaluate them; never call the best post-hoc candidate an unbiased winner.

## 4. Fit and score the outer procedure

For each outer fold:

1. Read the original frozen train/test row indices.
2. Run the above inner study-grouped selection on **only outer training rows**.
3. Refit the selected candidate on the full outer training rows, fitting preprocessing only there.
4. Predict the outer test rows once, save exact row IDs and study IDs, and calculate all metrics.
5. Read the saved original reproduction's OOF prediction for the **same rows** and compute paired improvements/degradations.

At the end, compare these two distinct objects on the same 16,988 rows:

- `original_reproduced_fixed`: frozen, previously reproduced LightGBM benchmark.
- `nested_optimized_procedure`: outer OOF predictions from inner-selected LightGBM configurations.

Report exact pooled R² difference (`optimized - 0.4726308398591973`), pooled MAE, RMSE, study-macro MAE and MSE, median/90th-percentile/worst study MAE, within ±1 log10 fraction, arithmetic mean fold R², individual fold R², and performance by study size and method family. Highlight folds and studies that deteriorate materially. Quantify the influence of the largest 6,135-row study. Keep every negative fold metric and every poor result visible; no cherry-picking.

Do not automatically promote a configuration because one pooled R² score is higher. Explicitly distinguish **nested, predeclared-procedure performance** from exploratory candidate ranking, and report whether the selection rule actually produced improvement.

## 5. Audits and tests that must pass

Write automated tests for at least:

- Protected-file SHA-256 equality before/after.
- Exact 16,988 rows, 72 groups, original fold memberships and five fold sizes.
- Every test row appears once and no train/test study overlap in outer or inner folds.
- No target or study-derived predictor leakage.
- Training-only fit of imputer/scaler/encoder and any clipping thresholds.
- Correct target units/offset and frozen historical scenario labeling.
- Inner selection independent of outer labels (e.g. perturb outer-test labels in a test and verify selections do not change; do not use perturbed labels for final reporting).
- Exact alignment of original and optimized OOF predictions on row IDs; no duplicate or missing IDs.
- Deterministic repeat on a small fixed slice where practical; all predictions finite.
- Reported metrics recomputed independently from saved predictions.

Run tests and repair genuine implementation faults; **never change fold definitions or targets simply to make tests pass**.

## 6. Autonomous notebook creation and execution

Create an executed, self-contained research notebook at:

`optimization_outputs/LightGBM_Optimization_EXECUTED.ipynb`

Include clearly labeled executable cells for: (1) setup and read-only verification, (2) baseline artifacts and hashes, (3) dataset/group/fold assertions, (4) frozen parameter registry, (5) inner grouped selection, (6) outer refits and OOF predictions, (7) paired metrics and per-study analysis, (8) diagnostic tables and plots, (9) report generation, and (10) tests + final integrity verification.

Programmatically create **and execute all notebook cells**, in order, with a fresh kernel; capture cell execution statuses and outputs. Do not submit unexecuted code cells as completed work. Keep substantive work in importable modules/scripts and make the notebook call them, so both notebook and CLI can reproduce the same experiment. If any cell fails, fix the root cause and rerun as required. Avoid side-effect execution on import.

Use unambiguous output filenames, e.g.:

- `optimization_outputs/OPTIMIZATION_REPORT.md`
- `optimization_outputs/LightGBM_Optimization_EXECUTED.ipynb`
- `optimization_outputs/candidate_registry.json`
- `optimization_outputs/inner_metrics_by_outer_fold.csv`
- `optimization_outputs/outer_selections.csv`
- `optimization_outputs/optimized_oof_predictions.csv`
- `optimization_outputs/paired_per_study_metrics.csv`
- `optimization_outputs/paired_per_fold_metrics.csv`
- `optimization_outputs/optimization_summary.json`
- `optimization_outputs/reproduction_and_integrity_checks.json`
- `optimization_outputs/tests.txt`
- `optimization_outputs/figures/` (only plots that help interpret results)

Provide one clear, **actually tested** command for running the experiment again from the repository root, and another for the tests. Include package/environment metadata and SHA-256 checksums. Save a deployable full-development-data model **only if** you can select its parameters using _all-data grouped CV without looking at outer scores_, and clearly label it exploratory; otherwise omit saved model. Do not save or advertise a model trained on outer-test data as an independently validated one.

## 7. Stop rule and final response

At the end, stop after the bounded experiment. Produce a short plain-language summary including:

1. Original reproduced pooled R²: `0.4726308398591973`.
2. Actual nested optimized pooled R² (or explicit reason it could not be computed).
3. Exact difference; whether higher R² came with worse study-level errors.
4. Which parameters were selected in each outer fold.
5. Number of actual fits, execution status, tests passed, and original-file preservation results.
6. Exact paths to report, executed notebook and OOF predictions.
7. The crucial caveat that this is a previously inspected, historically adjusted grouped-study benchmark and not a scientifically verified or independent prospective result.

**Never manufacture metric improvements, never silently modify source measurements, never overwrite the 0.473 reproduction, and never start another phase automatically. If tuning does not help, say so and preserve the better reproducible reference.**
