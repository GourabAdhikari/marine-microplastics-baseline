Stage 4 — Study-Robust LightGBM Selection (bounded, autonomous Codex task)
Mission
Continue the existing cloned marine-microplastics-baseline repository, not the separate microplastic-2 project. The already executed reproduction, nested LightGBM optimization, and fixed blending analyses are complete. Do not redo them.

This stage tests one specific hypothesis: Can a more conservative, study-error-aware rule for selecting among the SAME eight previously evaluated LightGBM settings retain most of the nested optimized R² while reducing difficult-study prediction errors? This is not a new architecture search. No new features, data corrections, models, or blend-weight searches are authorized.

Do not ask the user to write code, run cells, or intervene with routine debugging. Implement the workflow, create the notebook, execute all its cells, run tests, and write a factual final report yourself. Stop after reporting results.
Immutable reference and scientific scope

- Repository code and data/noaa_microplastics.csv are read-only.
- Preserve reproduction_outputs/, optimization_outputs/, blending_outputs/, .git/, all earlier models and reports, and all original source files. Do not overwrite or rename anything in them.
- Write all new files under study_robustness_outputs/ only.
- Reference original fixed LightGBM pooled OOF R² = 0.4726308398591973.
- Reference previously nested-optimized procedure pooled OOF R² = 0.5033812847921568.
- Historical protocol: 16,988 ocean-water rows, 72 provenance-linked studies, target log10(concentration + 0.001), original feature definitions and preprocessing, same exact 5 GroupKFold outer memberships and row order as the frozen baseline.
- All comparisons must retain exactly the original historical_hypothetical_unit_scenario: the Barrows/Dehm ×10^6 values are historically assumed, not scientifically verified corrections. Never describe modeled values as verified true synthetic microplastic abundance. Do not modify targets or filter studies.
- The five outer folds and their labels have been inspected many times. Any improvement here is exploratory on reused benchmark data, even if the new selection logic uses training-only metrics. Do not assert independent generalization or production readiness.
  Source artifacts to inspect first — strictly read-only

1. reproduction_outputs/out_of_fold_predictions.csv, reproduction_outputs/summary.json, reproduction_outputs/model_parameters.json.
2. optimization_outputs/optimized_oof_predictions.csv, optimization_outputs/inner_metrics_by_outer_fold.csv, optimization_outputs/candidate_registry.json, optimization_outputs/outer_selections.csv, optimization_outputs/frozen_splits.json, and any relevant code or manifest needed to reconstruct the exact existing eight candidates.
3. blending_outputs/BLENDING_AND_ROBUSTNESS_REPORT.md for descriptive context only. Do not use outer-fold blending outcomes to choose any Stage 4 rule. No blending in this stage.
4. Verify hashes/scenario/row identifiers/fold membership. If essential artifacts are absent or disagree, stop cleanly with an actionable diagnosis; do not invent or silently recompute mismatched historical results.
   Computational budget and two-stage stop rules

- Stage A: zero new LightGBM fits. Audit, define two fixed selectors, and determine exact per-fold selections from the already saved original inner-CV candidate summaries. This phase should be cheap.
- Stage B: maximum 10 new outer-fold LightGBM fits (two selector rules × five outer folds; fewer when valid identical OOF predictions or fully matching model checkpoints may be reused). No new inner fitting and no hyperparameter search. Hard-cap calls to LGBMRegressor.fit and log each fit with input data fingerprint, selected configuration and fold.
- Stage C: metrics, paired diagnostics, notebook execution and testing. Do not trigger any new model runs after seeing Stage B outer metrics.
- If the original inner metrics lack any mandatory fields needed for a rule, either compute that field from saved inner training-only OOF predictions if those exist and are correctly keyed, or mark the specific rule not executable. Do not refit a new inner grid or substitute outer-label metrics.
- Do not launch the historical chronological test, any external data downloads, ensemble/stacking search, neural networks, CatBoost/XGBoost search, or further numbered phases.
  Stage A — define and freeze two conservative selectors
  Read the saved inner-CV table as is: for each outer fold and each of the eight previously fixed configurations, use only original inner pooled R², study-macro MAE, p90 study MAE and worst-study MAE. Verify that these values were calculated from inner validation predictions generated without access to the outer test outcomes, and that every eligible study is counted.

Call the original 500-tree default LightGBM configuration C0. Define training-only per-outer-fold reference metrics r2_0, macro_0, p90_0, and worst_0 from C0's saved inner-CV result. Handle zero/undefined denominators explicitly; fail safely. All quantities refer to log10 target metrics.

Selector S1 — Tail-first guarded:

- A candidate is eligible only when all hold: inner_R2 >= r2_0 - 0.010, inner_macro_MAE <= 1.02 _ macro_0, inner_p90_MAE <= p90_0, and inner_worst_MAE <= 1.05 _ worst_0.
- C0 is always eligible by construction.
- From eligible candidates, select by lexicographic order: lowest inner_p90_MAE, lowest inner_macro_MAE, highest inner_R2, then smallest candidate ID. No using outer targets or fold outcomes.
  Selector S2 — Balanced guarded:
- A candidate is eligible only when all hold: inner_R2 >= r2_0 - 0.010, inner_macro_MAE <= 1.03 _ macro_0, inner_p90_MAE <= 1.03 _ p90_0, and inner_worst_MAE <= 1.10 \* worst_0.
- C0 is always eligible by construction.
- Determine the highest eligible inner_R2; restrict to eligible candidates within 0.010 R² of that value; choose lowest inner_macro_MAE, then lowest inner_p90_MAE, then highest inner_R2, then smallest candidate ID.
  Exactly these two rules; do not alter thresholds or introduce an additional selector after inspecting held-out results. Use a stable, deterministic ID tie-break. Freeze for each fold the selected candidate, all relevant inner metrics and eligibility reasons in a versioned locked_selections.json before any new outer prediction is scored. Preserve an auditable pre-fit timestamp/hash and test that the selection code never reads outer targets/outer OOF outcomes.
  Reuse and outer refitting
  For each of the 5 original outer folds, train the configuration selected by S1/S2 on that fold's original training rows only using exact original target transformations, feature preprocessing, model parameters, random seed and held-out fold membership. Match the same scientific hypothetical unit scenario.
- If a selected fold configuration is identical to the original C0 model, reuse its frozen OOF predictions after verifying row IDs, model parameters, preprocessing, target/scenario and split identity. Do not train again.
- If it is identical to the previously optimized procedure's selected configuration for that exact fold, reuse its OOF predictions after the same strict verification.
- Otherwise perform one new outer refit for that fold/selector. Cache by complete fingerprints (source, data, folds, scenario, hyperparameters, training row IDs and labels, environment). Reuse across selectors when selections coincide.
- No fitting on outer-held-out labels. No per-study routing using study identity, held-out outcome, fold number as feature, or observed test errors.
- Assert one OOF prediction per row, 16,988 total, 72 linked studies and original fold test sizes 6135, 2714, 2712, 2714, 2713. No missing/extra/duplicate rows, silent join, changed target or missing prediction substitution.
  Stage B/C — evaluate and explain, without reselecting
  Compute from the frozen OOF predictions for four separate evaluated procedures: original C0, previously nested optimized, S1, S2. All 4 use identical row IDs, groups, target, folds and correction scenario. Do not call this a comparison of independently trained test sets.

Report, for each:

- Pooled R², pooled log10 MAE and RMSE.
- Study-macro MAE and MSE (equal weight for each of 72 studies); median, p90 and worst-study MAE.
- Percent of predictions within ±1 log10 unit of concentration + 0.001.
- Arithmetic mean fold R² and all five fold-specific R²/MAE/RMSE.
- Study-level deltas versus both original and previously optimized procedures, counts of studies worsened, counts worsened by >0.10 log10 MAE; identify worst harmed studies descriptively.
- Method-family summaries (net, pump, bulk water) and study-size summaries. Show whether the largest study disproportionately affects pooled R².
- Actual number of fresh fits, cached fits, tests executed, notebook cells executed, warnings and failures.
  Frozen descriptive screening rule (do not use to tune again): qualify a selector as a promising exploratory candidate only if its pooled R² is at least 0.5033812847921568 - 0.015, its study-macro MAE and p90-study MAE are no worse than the optimized model, and its worst-study MAE does not exceed the optimized model's. Also show pooled MAE, tolerance coverage and mean fold R² even if they worsen. A pass is not confirmation and does not authorize overwriting baseline or optimizing based on these outer outcomes.

If neither selector meets the rule, state clearly that no improved reliability procedure was found and retain original/optimized results. If a selector passes, call it an exploratory candidate only. In all cases do not select a new production model, do not repeat this experiment with revised thresholds, and stop.
Scientific and statistical guardrails

- The dataset is dominated by a 6,135-row study; keep every row and disclose the imbalance.
- Never train a meta-model on original outer OOF predictions; any future blending would require inner cross-fitting and is explicitly out of scope here.
- Note that 0.503381 is a previously inspected pooled benchmark and the historical unit multipliers are hypothetical.
- Do not confuse negative individual-fold R² with negative pooled R², or R² with accuracy.
- Do not claim independent study generalization based only on these reused folds. New unseen studies with verified methods are ultimately needed.
- Avoid changing the reported target or removing difficult studies to raise metrics. No outcome-based edits.
  Implementation and autonomous execution — mandatory
  Create under study_robustness_outputs/:
- source/ containing modular audit, selector, fit-budget, evaluation, notebook-builder/runner and reporting code.
- Study_Robustness_EXECUTED.ipynb (actually executed in a fresh kernel, not merely saved).
- locked_selections.json, selector_eligibility.csv, model_fit_log.jsonl, all_procedure_oof_predictions.csv with original row identifiers, summary_metrics.csv, paired_fold_metrics.csv, paired_study_metrics.csv, and method/size analyses.
- tests/ with focused tests for: protected paths unchanged, exact OOF alignment, inner-only selector inputs, S1/S2 arithmetic and ties, C0 fallback, correct caching, deterministic selections, fit cap, target/scenario equality, missing values, no outer-data leakage, and independent metric recomputation.
- EXECUTION_STATUS.json with actual notebook cell outcomes, fit counts, tests and hashes.
- STUDY_ROBUSTNESS_REPORT.md explaining methods, actual results, harms, limitations and recommended next scientific step.
  Create all needed code and notebook cells yourself. Run them. If a test fails, debug the implementation and rerun. Never silently edit rules, input data or labels to force a test to pass. Preserve pre-existing code and files by a complete read-only hash manifest and recheck at the end. Use the existing reproducible virtual environment and avoid any unnecessary new dependencies or downloads.

Mandatory final response from Codex: give a short table of the four procedures, pass/fail of the frozen screening rule, actual new fit count (max 10), test count, notebook execution status, and exact report path. Stop without launching follow-on experiments.
