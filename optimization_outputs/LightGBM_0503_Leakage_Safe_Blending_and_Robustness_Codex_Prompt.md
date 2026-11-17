# Stage 3 — Controlled LightGBM Blending and Study-Robustness Experiment

> **For Codex.** Execute autonomously inside the cloned `marine-microplastics-baseline` repository. The aim is to test a specific, low-cost hypothesis: a conservative combination of the reproduced original LightGBM and the nested-optimized LightGBM procedure may reduce the optimized model's reliability penalties without forfeiting too much pooled R². **A better score is an objective, not a guaranteed outcome.**

## 0. The frozen evidence and the scientific caveat

The original fixed LightGBM baseline was reproduced on 16,988 ocean-water rows across 72 linked studies, with five fixed GroupKFold partitions:

| Metric                           | Original fixed LightGBM | Nested optimized LightGBM procedure |
| -------------------------------- | ----------------------: | ----------------------------------: |
| Pooled out-of-fold R²            |      0.4726308398591973 |                  0.5033812847921568 |
| Pooled MAE (log10)               |                0.996377 |                            1.009343 |
| Pooled RMSE (log10)              |                1.418198 |                            1.376230 |
| Study-macro MAE (log10)          |                1.060942 |                            1.001411 |
| Study-macro MSE                  |                2.171751 |                            2.030075 |
| 90th-percentile study MAE        |                1.813391 |                            1.901218 |
| Worst-study MAE                  |                4.943935 |                            5.295836 |
| Within ±1 log10 of offset target |                  63.94% |                              61.01% |
| Arithmetic mean of fold R²       |               -0.071235 |                           -0.080676 |

The original historical **×10^6 adjustments for Barrows (1,389 rows) and Dehm (81 rows) are unverified hypotheses**. This experiment must preserve that exact computational reproduction _as a labeled scenario_, never represent its predictions as certified estimates of true synthetic-only marine microplastic concentration. No unit/target edits or exclusions are authorized.

**The five outer validation folds and their outcomes have already been inspected.** Any retrospective improvement on those folds remains exploratory. Do not call it independent validation, a new holdout, production readiness, or proof of new-study generalization.

## 1. Mandatory preservation and working directory

1. Work in the existing cloned repository where `reproduction_outputs/` and `optimization_outputs/` already exist.
2. Inspect existing manifests, model configurations, source-row maps, fold memberships, OOF predictions and validation reports before writing new code.
3. Treat the repository's tracked files, original dataset, `reproduction_outputs/`, and `optimization_outputs/` as **read-only protected**. Do not modify, relocate or replace them. Record SHA-256 hashes before and after and fail loudly on unexpected changes.
4. Create all new files only under `blending_outputs/` with a separate `source/`, `figures/`, `tests/`, `checkpoints/` and executed notebook.
5. Preserve exactly the **same 16,988 rows, 72 study IDs, target `log10(concentration + 0.001)`, five original outer fold assignments, original unit-adjustment scenario and original preprocessing**.
6. Reuse recorded OOF predictions where they are valid; do not rerun the 127 optimization fits or any prior phase.
7. Compute no metrics by silently dropping NaNs, mismatched rows, extreme studies, or unsupported observations. Fail with an informative error instead.

## 2. Goal and decision hierarchy

**Primary research goal:** Find out whether a **predeclared conservative blend** improves pooled R² _and/or_ limits regression in overall MAE, study tails, and within-one-log coverage compared with the optimized procedure.

Do **not** optimize weights against the already seen five outer-fold outcomes. Prioritize an auditable answer over an inflated leaderboard.

A negative result is a successful experiment if it is executed and reported correctly. Keep the 0.472631 original and 0.503381 optimized procedures as immutable references regardless of the result.

## 3. Stage A — Zero-fit integrity and fixed-weight analysis (mandatory)

**A1. Load and align**:

- Read `reproduction_outputs/out_of_fold_predictions.csv` and `optimization_outputs/optimized_oof_predictions.csv` (plus associated schemas, manifests and summary artifacts).
- Verify that each prediction corresponds to the _same source row, same study group, same held-out fold, same target,_ and the labeled historical scenario. Assert one and only one OOF prediction per eligible row.
- Compare loaded baseline and optimized predictions by recomputing every reference aggregate metric to a strict numerical tolerance, documenting any rounding or precision differences.
- Verify no outer validation targets were ever used to fit blend coefficients or select a new model.

**A2. Evaluate exactly these predeclared, fixed combinations with no learning from targets**:

- `B0 = 1.00 * original + 0.00 * optimized` (original reference).
- `B25 = 0.75 * original + 0.25 * optimized`.
- `B50 = 0.50 * original + 0.50 * optimized` — **predeclared main candidate**.
- `B75 = 0.25 * original + 0.75 * optimized`.
- `B100 = 0.00 * original + 1.00 * optimized` (nested optimized reference).

All blending is on the **common log10 prediction scale**, with weights nonnegative and summing to 1. No test-row-dependent or study-specific weights; no use of test targets or validation outcomes to choose weights. These fixed alternatives are a retrospective sensitivity analysis. **Do not declare the largest-scoring one the winner.**

**A3. Compute paired, identical-row metrics** for all five:

- pooled R², pooled log10 MAE and RMSE;
- study-macro MAE, study-macro MSE, median and p90 study MAE, worst-study MAE;
- fraction of all rows within ±1 log10 unit of the **offset target**;
- per-fold R²/MAE/RMSE and per-study MAE/MSE, retaining small-study and one-study-fold results;
- changes relative to both original and optimized references;
- performance by method family and study size, with explicit row counts.

Include a clear paired diagnostic for folds 2 and 5 and for the 14 studies previously worse by >0.10 log10 MAE under optimization; **these are post-hoc diagnostics only, never a feature, weight, threshold or selection rule**.

**A4. Main-candidate interpretation:** B50 is a fixed hypothesis, not chosen from the outer outcome leaderboard. Show its trade-off with B0 and B100. A B50 improvement is still exploratory due to previously inspected evaluation data.

## 4. Stage B — Decide whether any new fitting is worthwhile

After Stage A, produce a summary table and apply this **precommitted screening test to B50 vs B100**:

- pooled R² degradation is no more than **0.015** (or pooled R² improves);
- study-macro MAE does not increase;
- at least **two** of the following three optimized-model weaknesses improve: pooled MAE, p90 study MAE, within-one-log fraction;
- worst-study MAE does not increase by more than **0.10 log10**;
- no new arithmetic or data-integrity issue.

This is a _cost-control gate_, **not confirmation of transfer**. If the gate fails, **STOP after Stage A and the report**; do not train a new model or search other weights to rescue the score. Explain exactly which conditions failed.

If B50 passes, Stage C is optional but authorized only if the estimated compute fits the budget and all grouped partition checks pass.

## 5. Stage C — Prospective training-only blend selection (conditional, tightly bounded)

The zero-fit Stage A is not sufficient to evaluate a _learned_ blend. If Stage B passes, implement a **strictly nested, study-grouped outer-fold procedure**:

- For each of the five existing outer folds, **do not consult that fold's labels in any model/weight-selection step**.
- Inside its outer-training studies, generate genuine held-out predictions for both base learners. The fixed original LightGBM may be fitted directly on each inner-training subset. The optimized base is itself a _selection procedure_; select its configuration using **only the training studies of that inner split** (inner-inner study-grouped selection). Never select a configuration using labels from the split being predicted.
- All numeric imputation, encoding, feature-range clipping and support estimates must fit within each relevant training partition. No accidental training/validation study overlap, including nested levels.
- If the nested group count is too low for valid inner-inner validation, use the original fixed LightGBM as a documented predefined fallback for that partition; never relax leakage requirements or make up a configuration.
- Use only train-side cross-fitted predictions to choose among the fixed weights `{0.25, 0.50, 0.75}` on the optimized base (original weight `1 − w`). Also allow endpoints 0 and 1 as safe references. Choose by a prespecified, group-aware objective: minimize study-macro MAE subject to pooled inner R² being within 0.015 of the best admissible option. Tie-break to B50, then toward more original-model weight. The objective and tie-break must be frozen before any new fits.
- Refit the outer-fold optimized procedure using only outer-training data and its inner-only choice; refit the baseline on outer-training data. Apply the inner-selected weight to _outer-held-out_ predictions.
- Compare this **nested weight-selection procedure** with the original and optimized OOF references on identical outer-fold rows, but identify it as _exploratory on reused folds_.
- Save a per-fold audit of every split, selection, fallback, fit count and study identifier. No adaptive weight selection based on outer errors, no stacking model, no test-target-driven gating.

**Hard compute cap:** at most **100 newly started LightGBM fits** for conditional Stage C, counting inner-inner and selected outer fits. Before launching, calculate a reliable upper bound from the planned folds. If the design exceeds 100 fits, **skip Stage C and stop**, documenting why. Do not silently cut folds, loosen nesting, reuse predictions from an incompatible training set, or exceed the cap. No additional model families, no Optuna, no deep learning, no data acquisition or hyperparameter searches.

## 6. Tests and reproducibility

Implement focused unit/integration tests that verify:

- exact OOF row alignment, fold and study identity; unique coverage and constant target/scenario;
- baseline and optimized metric reproduction; blending arithmetic and endpoint equivalence;
- study-macro metrics, factor-ten-offset tolerance calculations, and all-case evaluation;
- no outer-label use in Stage C parameter/weight fitting; no group overlap at every nesting level;
- leakage-safe preprocessors, including numeric clipping and missing categories;
- deterministic behavior with documented seeds; no hidden state or automatic fits on import;
- hard fit-cap enforcement, valid checkpoint identity and preserved original-file hashes;
- explicit distinction between completed Stage A, authorized/skipped Stage C and any failures.

Create an autonomous executed Jupyter notebook using the same reusable source modules as the CLI. **Write, execute and debug every required code cell yourself**; do not ask the user to paste code or click Run. Reruns must reuse checkpoints only when hashes of all training data, source code, configurations and study splits match. Report tests that actually ran, not intended tests.

## 7. Required outputs

Save under `blending_outputs/`:

1. `BLENDING_AND_ROBUSTNESS_REPORT.md` — findings, both references, fixed blend table, fold/study harms, Stage B gate, Stage C status, limitations and decision.
2. `LightGBM_Blend_Research_EXECUTED.ipynb` — fully executed notebook.
3. `source/` — importable, documented source files and notebook runner.
4. `fixed_blend_metrics.csv`, `paired_fold_metrics.csv`, `paired_study_metrics.csv`, `method_and_study_size_metrics.csv`.
5. `aligned_oof_predictions.csv` — exact source-row/study/fold/target/baseline/optimized and each fixed blend; include scenario label.
6. `stage_b_gate.json`, `stage_c_status.json` and, if Stage C ran, `nested_weight_selection.csv`, `nested_oof_predictions.csv`, `nested_fit_audit.csv`.
7. `figures/` — compact pooled-metric and fold/study-comparison plots.
8. `integrity_manifest.json`, `tests.txt`, `environment.json`, `fit_log.jsonl`, `final_status.json`.

**Final response from Codex must contain**: (a) B50 versus both references on pooled R², pooled MAE, study-macro MAE, p90/worst study MAE and within-one-log fraction; (b) Stage B pass/fail and whether Stage C actually ran; (c) new model-fit count, executed notebook-cell count and passed tests; (d) exact saved report path; (e) a plain statement that original files remain unchanged.

## 8. Hard stop and honesty requirements

- Do not overwrite or promote the original reference, the optimized nested procedure, or a deployable model.
- Do not claim that a higher previously observed pooled R² proves independently better generalization. The future chronological results from other phases must **not** influence weights, parameters, or selection.
- Do not change concentration targets, use unverified adjustments as certified truth, remove difficult studies, or optimize on the reported five outer folds.
- If no fixed blend improves the reliability trade-off, say so and stop. A clear negative result is preferable to another unnecessary research phase.
- **Stop immediately after producing the required report. Do not start another phase automatically.**
