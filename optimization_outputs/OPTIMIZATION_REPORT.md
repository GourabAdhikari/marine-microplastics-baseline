# Bounded LightGBM optimization report

**Scenario for every table and result: `historical_hypothetical_unit_scenario`.** Historical ×10^6 unit multipliers are frozen hypotheses, scientifically unresolved—not certified corrections or physical predictions of true synthetic microplastic concentration.

## Outcome

Original reproduced pooled R²: **0.4726308398591973**. Nested optimized pooled R²: **0.5033812847921568**. Exact difference (optimized − reference): **0.03075044493295953**.

Pooled R² improved. Study-macro MAE changed by **-0.059530833** log10 units (**-5.611%**). The inner 5% safety rule is not a guarantee of outer safety. 30 of 72 studies have worse MAE; 14 worsen by more than the predeclared descriptive threshold of 0.10 log10 units. No configuration is automatically promoted.

This is a **mixed improvement, not a uniformly better model**: pooled MAE increases from 0.996377 to 1.009343; within-one-log coverage falls from 63.94% to 61.01%; p90 study MAE rises from 1.813391 to 1.901218, and worst-study MAE from 4.943935 to 5.295836. Mean fold R² is also worse. Folds 2 and 5 exceed the descriptive 0.10-log MAE deterioration flag. The desired pooled-R² and average-study-error goals improved, but tail safety did not. No outer-tail acceptability threshold was specified in advance, so these harms cannot be declared acceptable after the fact. Retain both results and the unchanged reproducible reference; do not deploy or promote this procedure on this evidence alone.

R² is explained variance on `log10(concentration + 0.001)`, not accuracy. Within ±1 log measures a factor-ten tolerance on concentration **plus the offset**, not an unqualified percentage correct. All negative fold results remain visible.

## Pooled and study-balanced metrics

| metric | original_reproduced_fixed | nested_optimized_procedure |
| --- | --- | --- |
| R2 | 0.472631 | 0.503381 |
| MAE | 0.996377 | 1.009343 |
| RMSE | 1.418198 | 1.376230 |
| study_macro_MAE | 1.060942 | 1.001411 |
| study_macro_MSE | 2.171751 | 2.030075 |
| median_study_MAE | 0.866895 | 0.831673 |
| p90_study_MAE | 1.813391 | 1.901218 |
| worst_study_MAE | 4.943935 | 5.295836 |
| within_1_log | 0.639393 | 0.610078 |
| mean_fold_R2 | -0.071235 | -0.080676 |

## Selection and paired outer folds

| fold | candidate | differences from original |
| --- | --- | --- |
| 1 | C7_fewer_trees | {"learning_rate": 0.05, "max_depth": 5, "min_child_samples": 50, "n_estimators": 250, "num_leaves": 20, "reg_lambda": 5.0} |
| 2 | C1_shallow | {"max_depth": 4, "min_child_samples": 40, "num_leaves": 15} |
| 3 | C2_conservative | {"max_depth": 5, "min_child_samples": 70, "num_leaves": 20, "reg_lambda": 5.0} |
| 4 | C4_feature_subsample | {"colsample_bytree": 0.8, "max_depth": 5, "min_child_samples": 50, "num_leaves": 20, "reg_lambda": 5.0} |
| 5 | C6_slower | {"learning_rate": 0.03, "max_depth": 5, "min_child_samples": 50, "n_estimators": 800, "num_leaves": 20, "reg_lambda": 5.0} |

| fold | rows | studies | original_reproduced_fixed__R2 | nested_optimized_procedure__R2 | delta_R2 | original_reproduced_fixed__MAE | nested_optimized_procedure__MAE | delta_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6135 | 1 | -0.555996 | -0.498304 | 0.057692 | 0.727919 | 0.719173 | -0.008745 |
| 2 | 2714 | 11 | 0.003502 | -0.468474 | -0.471975 | 0.812758 | 1.002473 | 0.189714 |
| 3 | 2712 | 20 | 0.301706 | 0.376104 | 0.074398 | 1.602672 | 1.596711 | -0.005960 |
| 4 | 2714 | 20 | -0.437683 | -0.007517 | 0.430165 | 1.275470 | 1.066746 | -0.208724 |
| 5 | 2713 | 20 | 0.332296 | 0.194810 | -0.137486 | 0.901872 | 1.027807 | 0.125935 |

Full fold metrics including RMSE, all study-balanced metrics and tolerances: `paired_per_fold_metrics.csv`.

## Predeclared design

Exactly eight configurations from `optimization.md`, recorded before any tuned outer outcomes in `candidate_registry.json`. Baseline defaults are loaded from `reproduction_outputs/model_parameters.json`. No changed features or units, no early stopping, no model search beyond the registry. All other LightGBM parameters are unchanged; seed 0 and four learner threads on 8 logical CPUs. We preserve the baseline settings rather than adding different LightGBM deterministic/column-wise flags. A small repeated fit tests practical determinism on this environment, not cross-platform bitwise equivalence.

Fixed outer row memberships come directly from the baseline OOF CSV, never from a replacement split. Each outer training set has GroupKFold(3); `inner_split_counts.csv` records all 15 partition counts. The selector receives only outer-training targets. Candidates are ranked by pooled inner OOF R², not fold-mean R². Nonbaseline eligibility requires strictly better R² than C0 and study-macro MAE <=105% of C0. C0 remains admissible. Among candidates within 0.005 of the best admissible R², lower macro MAE wins; exact remaining ties use ID order. This precise tie interpretation was frozen before fits. `inner_metrics_by_outer_fold.csv` contains all 40 candidate summaries, including p90 and worst-study MAE; `outer_selections.csv` records reasons. All five selections were completed before any outer refit was scored.

Only the selected procedure was evaluated on outer rows, not eight candidate outer scores. This is nested predeclared-procedure performance, not a post-hoc best candidate. The outer folds and 0.473 headline were already inspected, so even correctly nested improvement is **exploratory**. Independent prospective performance requires new studies.

## Study-level deterioration

Ten largest MAE increases (all studies remain in the full CSV):

| study | rows | fold | original_reproduced_fixed__MAE | nested_optimized_procedure__MAE | delta_MAE |
| --- | --- | --- | --- | --- | --- |
| https://doi.org/10.1016/j.scitotenv.2018.12.192 | 23 | 4 | 1.246063 | 2.162956 | 0.916893 |
| https://doi.org/10.1016/j.scitotenv.2022.156259 | 57 | 5 | 1.094196 | 1.800683 | 0.706486 |
| https://doi.org/10.1016/j.chemosphere.2022.133980 | 69 | 4 | 0.169873 | 0.700701 | 0.530828 |
| https://doi.org/10.1038/srep14947 | 21 | 5 | 1.452892 | 1.912144 | 0.459252 |
| https://doi.org/10.1371/journal.pone.0080466 | 171 | 5 | 0.610816 | 1.049821 | 0.439005 |
| https://doi.org/10.1016/j.marpolbul.2022.114452 | 2 | 3 | 1.610055 | 1.991850 | 0.381794 |
| https://doi.org/10.1016/j.marpolbul.2024.117288 | 33 | 5 | 4.943935 | 5.295836 | 0.351901 |
| https://doi.org/10.3389/fenvs.2021.719274 | 5 | 2 | 1.085333 | 1.435189 | 0.349856 |
| https://doi.org/10.1126/sciadv.1600582 | 42 | 5 | 0.993794 | 1.309196 | 0.315401 |
| https://doi.org/10.1021/es4053076 | 2530 | 2 | 0.812465 | 1.025263 | 0.212797 |

All studies flagged as materially worse (>0.10 log10 MAE):

| study | rows | fold | delta_MAE |
| --- | --- | --- | --- |
| https://doi.org/10.1016/j.chemosphere.2022.133980 | 69 | 4 | 0.530828 |
| https://doi.org/10.1016/j.marpolbul.2022.114452 | 2 | 3 | 0.381794 |
| https://doi.org/10.1016/j.marpolbul.2024.117288 | 33 | 5 | 0.351901 |
| https://doi.org/10.1016/j.scitotenv.2018.12.192 | 23 | 4 | 0.916893 |
| https://doi.org/10.1016/j.scitotenv.2022.155958 | 124 | 3 | 0.121045 |
| https://doi.org/10.1016/j.scitotenv.2022.156259 | 57 | 5 | 0.706486 |
| https://doi.org/10.1021/es4053076 | 2530 | 2 | 0.212797 |
| https://doi.org/10.1038/srep14947 | 21 | 5 | 0.459252 |
| https://doi.org/10.1088/1748-9326/abbb4f | 663 | 5 | 0.134151 |
| https://doi.org/10.1126/sciadv.1600582 | 42 | 5 | 0.315401 |
| https://doi.org/10.1371/journal.pone.0080466 | 171 | 5 | 0.439005 |
| https://doi.org/10.1371/journal.pone.0111913 | 680 | 5 | 0.146304 |
| https://doi.org/10.3389/fenvs.2021.719274 | 5 | 2 | 0.349856 |
| https://doi.org/10.3389/fmars.2021.626026 | 54 | 5 | 0.212321 |

The full `paired_per_study_metrics.csv` preserves every study, including any undefined within-study R² for single-observation or constant-target studies (blank, not dropped). Macro MAE and MSE include all studies equally. Quantiles use pandas' default linear interpolation. Method-family groups are row-based; a study may contribute to multiple families. Study-size bins are fixed at 1–50, 51–200, 201–1000 and 1001+ eligible observations, never target-derived.

## Performance by method family and study size

| grouping | group | procedure | rows | studies | R2 | MAE | RMSE | study_macro_MAE | p90_study_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| method_family | bulk water | original_reproduced_fixed | 1903 | 12 | -0.041714 | 2.053115 | 2.727677 | 1.383764 | 2.199929 |
| method_family | bulk water | nested_optimized_procedure | 1903 | 12 | 0.002652 | 2.161275 | 2.668960 | 1.513184 | 2.826229 |
| method_family | pump | original_reproduced_fixed | 852 | 6 | -1.331179 | 1.299307 | 1.552286 | 1.228086 | 1.527223 |
| method_family | pump | nested_optimized_procedure | 852 | 6 | -0.642243 | 1.034807 | 1.302874 | 1.177425 | 1.443239 |
| method_family | surface net | original_reproduced_fixed | 14233 | 58 | 0.093210 | 0.836954 | 1.123198 | 1.010207 | 1.575414 |
| method_family | surface net | nested_optimized_procedure | 14233 | 58 | 0.132730 | 0.853801 | 1.098450 | 0.910459 | 1.511692 |
| study_size_bin | 1-50 | original_reproduced_fixed | 893 | 42 | 0.045598 | 1.046320 | 1.463811 | 1.025916 | 1.594339 |
| study_size_bin | 1-50 | nested_optimized_procedure | 893 | 42 | 0.070826 | 0.963189 | 1.444335 | 0.914501 | 1.900163 |
| study_size_bin | 1001+ | original_reproduced_fixed | 11199 | 4 | 0.500490 | 0.976627 | 1.423617 | 1.236195 | 1.870315 |
| study_size_bin | 1001+ | nested_optimized_procedure | 11199 | 4 | 0.539888 | 1.000913 | 1.366321 | 1.235661 | 1.860289 |
| study_size_bin | 201-1000 | original_reproduced_fixed | 3140 | 7 | 0.153245 | 1.062557 | 1.381007 | 1.234032 | 1.941764 |
| study_size_bin | 201-1000 | nested_optimized_procedure | 3140 | 7 | 0.162146 | 1.049045 | 1.373730 | 1.201645 | 1.785038 |
| study_size_bin | 51-200 | original_reproduced_fixed | 1756 | 19 | 0.294117 | 0.978602 | 1.425614 | 1.037704 | 1.838502 |
| study_size_bin | 51-200 | nested_optimized_procedure | 1756 | 19 | 0.311638 | 1.015581 | 1.407810 | 1.070444 | 1.801122 |

## Influence of the largest study

`https://doi.org/10.1126/science.1192321` contains **6135 rows (36.11%)**, comprising the entire first outer fold.

- Its fraction of total squared error: original **14.84%**, optimized **15.17%**.
- Its contribution to global ΔR²: **+0.002901070**; remaining studies contribute **+0.027849375**. These add to the overall change.
- Descriptively excluding this study from scoring the existing predictions: original R² **0.426215692**, optimized R² **0.461794956**. This changes the evaluation population and denominator, is not the benchmark, and was not used for selection; no refits or row exclusions occurred.

## Integrity, environment, execution and fit accounting

- Original eligible data: 16,988 rows, 72 study groups, 73 accessions and 73 publications; 29,076 original rows. The independent grouping audit rejects missing/placeholder identifiers; no missing-identifier bridge occurs in this dataset.
- Original preprocessing cells and helper functions are extracted verbatim read-only, with original row order, median imputation, scaling, one-hot unknown handling and test clipping. Preprocessing fits inside every training fold. No identifiers or targets enter predictors.
- Barrows 1,389 rows and Dehm 81 rows retain ×10^6; other measurements are unchanged. Exact target and row equality with the frozen reproduction are tested.
- Core packages: {"ipykernel": "7.4.0", "lightgbm": "4.7.0", "nbclient": "0.11.0", "nbformat": "5.11.1", "numpy": "2.5.3", "pandas": "3.0.6", "scikit-learn": "1.9.1", "scipy": "1.18.1"}. Python 3.12.7, macOS-15.6.1-arm64-arm-64bit. Same environment as reproduction; original historical notebook metadata had Python 3.13.5. No downloads were needed.
- Actual completed LightGBM fits: **120 inner + 5 selected outer + 2 essential deterministic verification = 127**. Started fits: 127. C0 inner fits are required candidate comparisons, not regenerated baseline OOF. Selected C0 outer fits, if present, are explicitly logged in `fit_log.jsonl`. No full baseline rerun, no early stopping, no deployable full-data model, no extra phase.
- Tests passed: **653**. `tests.txt` details every assertion. `reproduction_and_integrity_checks.json` records protected-file preservation, including the original dataset, original notebooks, lockfile, `.git/` and the complete reproduction directory. SHA-256 manifest: `protected_hashes_before.json`; symlinks are recorded as link targets without traversing external interpreters. No protected path was added or removed.
- Cache identities include core experiment code, data, original source, input artifacts, folds, all configurations, scenario and package versions, plus exact per-fit training labels and row IDs. Checkpoints are atomic and fit events logged. Changing the fit identity does not bypass the hard inner-fit cap. Checkpoint-only reruns do not repeat valid fits.
- The executed notebook invokes importable modules; imports have no fit side effects. `notebook_execution_status.json` records completed cells and outputs. The runner always uses a fresh kernel; runtime and IPython files are isolated under this run directory.

## Reproduce and test (from repository root)

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python optimization_outputs/source/run.py
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python optimization_outputs/source/run.py --tests
```

The first command resumes this frozen experiment from hash-verified checkpoints and executes the notebook; it does not start another experiment. Both commands are tested. For a different experiment, use a separate versioned directory rather than replacing this registry.

## Files

- `LightGBM_Optimization_EXECUTED.ipynb`: executed cells and outputs.
- `optimized_oof_predictions.csv`: exact source/water row IDs, studies, folds, common targets, both predictions and scenario label.
- `optimization_summary.json`, `paired_per_fold_metrics.csv`, `paired_per_study_metrics.csv`, `performance_by_method_and_study_size.csv`.
- `candidate_registry.json`, `inner_metrics_by_outer_fold.csv`, `inner_split_counts.csv`, `outer_selections.csv`, `frozen_splits.json`.
- `figures/fold_r2.svg`, `figures/all_study_mae.svg` (all 72 studies; order key saved alongside).
- `input_artifacts.json`, `environment.json`, `checkpoint_identity.json`, `fit_log.jsonl`, `tests.txt`, `reproduction_and_integrity_checks.json`.

**Stop:** this bounded experiment is complete. No independent validation claim and no scientifically verified unit correction is made. No automatic next phase.
