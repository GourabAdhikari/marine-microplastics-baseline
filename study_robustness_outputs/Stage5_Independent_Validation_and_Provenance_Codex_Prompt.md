STAGE 5 — Independent-study validation readiness, provenance audit, and frozen research-model protocol
Mission
You are Codex working inside the existing cloned marine-microplastics-baseline repository. Finish this bounded Stage 5 autonomously: inspect the actual repository, audit the scientifically unresolved concentration transformations, freeze an external-study evaluation protocol, and create a fully tested ready-to-run validation pipeline. Do not run another model search on the five historical outer folds. Do not present a new higher historical R².

Scientific motivation: the most recent Stage 4 report (study_robustness_outputs/STUDY_ROBUSTNESS_REPORT.md) showed exploratory, already-inspected historical grouped-CV results:
Frozen historical procedure Pooled OOF R² Study-macro MAE Worst-study MAE
Original C0 LightGBM 0.4726308398591973 1.060942164 4.943934893
Prior optimized selector 0.5033812847921568 1.001411331 5.295836118
Stage 4 S1, tail-first 0.5194916146841499 0.980533172 5.244593780
Stage 4 S2, balanced guarded 0.5256280075581679 0.999353649 5.165887296

All these results use the same 16,988 observations from 72 linked studies, the exact historical 5 outer study-held-out folds, the exact original feature pipeline and log10(concentration + 0.001) target, and the historical_hypothetical_unit_scenario. The hypothesized ×10^6 multipliers for Barrows (1,389 rows) and Dehm (81 rows) are unverified; they may not be labeled ground-truth corrections. S1/S2 still have negative mean outer-fold R², nonuniform study performance, and have repeatedly been compared on the already-inspected historical benchmark. These numbers are exploratory, not an independent test.

Main objective: establish whether and how S2 could be evaluated on genuinely unseen, scientifically comparable studies, without contaminating the test and without inventing a validation score. Secondary objective: optionally train one frozen research-only full-development-data S2 pipeline only after the independent-evaluation protocol and eligibility gates are locked. No deployed model is authorized by this task.
Absolute protections and working location

1. Read all relevant existing reports, candidate registry, frozen preprocessing code, source-row mappings, OOF files, inner-validation summaries, and checkpoints read-only. Understand the exact schema before coding.
2. Never edit, overwrite, rename, move or delete the raw dataset, .git/, microplastics.ipynb, microplastics.py, gamm.R, lockfile, reproduction_outputs/, optimization_outputs/, blending_outputs/, study_robustness_outputs/, or their contents. Do not edit the previous Stage 5-freezing files. Hash and verify preservation.
3. All newly created files, notebooks, logs, checkpoints and reports go only under independent_validation_outputs/. Never write package caches or notebook runtime files outside this folder if controllable. Reuse the established Python environment; do not download libraries or change dependencies unless indispensable, and document any exceptional inability.
4. The original 0.47263084 and S2 0.52562801 historical reference results are IMMUTABLE. Do not replace them with any new score or silently change their scenario. No leaderboard, hyperparameter expansion, blend/stack tuning, feature engineering search, chronological experiments, or cherry-picking datasets by results.
5. Stop at the stated deliverables; report unavailable evidence honestly. Never claim scientific provenance was resolved or an independent R² was measured unless directly supported by inspectable sources and a truly eligible previously untouched external dataset.
   Stage A — Verify and freeze the benchmark identity (zero model fits)

- Verify the frozen data fingerprint and original row-order/source-row IDs, 16,988 eligible model rows, 72 linked studies, target formula and offset, exact features/units and transformations, the 5 outer fold memberships, S1/S2 selection algorithms, and historical hypotheses. Cross-check the cited values against actual files, not this prompt alone. If inconsistent, stop score-related work and report discrepancies.
- Verify original preprocessing: training-fold-only imputer/scaler/encoder, numeric clipping behavior at inference, category handling, method-family definitions, depth/mesh transforms, cyclical month, and any known missing features. Check that DOI/accession/study ID, outcomes, and fold IDs are never predictive inputs.
- Record file hashes, package/runtime versions, data counts, and the exact definition of study grouping. Create baseline_identity.json and protected_hash_manifest_before.json.
- Write down the critical historical-selection caveat: S2 was identified after multiple analyses of the same five test folds; selecting it as a future research candidate is permissible, but scoring it again on those same folds does not produce new independent evidence.
  Stage B — Scientific provenance audit (zero model fits)
  Produce PROVENANCE_AUDIT.md and provenance_row_audit.csv (one record or well-documented aggregate per relevant group) with three levels: verified from primary evidence, supported but ambiguous, unresolved. Never turn lack of evidence into a verified fact.

For Barrows and Dehm, independently document, wherever the available source material allows:

- Exact citation/DOI, accessible primary-source location, dates and page/table/equation/figure for every physical-unit claim; make citations/URLs navigable where possible. If full text is unavailable, record not_accessible rather than guessing.
- Published unit vs original CSV unit and field description; particles per volume or area, mass vs particle count, m³ vs L vs km², raw/missing zeros, sampling/filtration volume and volume calculation, subsampling and recovery factors, extraction protocol, synthetic verification status.
- Explain whether the mathematical unit factor can be derived from original volume units; do not accept a blanket ×10^6 just because it increases R². For Barrows, distinguish membrane filter pore size from actual net mesh. For Dehm, distinguish an aggregate plausibility argument from row-specific traceability.
- For any row-specific correction, require source measurement, exact calculation, explicit conversion factor and evidence reference. If not auditable, retain the original values and label the historical correction hypothesis_only. Do not overwrite the CSV.
- Identify other obvious discrepancies, duplicates, untraceable studies, ambiguous zero values, missing volume, and unclear synthetic-only verification. Do not silently drop records or change the frozen original benchmark.
  Network access is optional. If internet is available and permitted, conduct only narrowly scoped source verification for the two named studies and any immediately necessary primary methodological documentation; cap public-source lookups at six targeted lookups, record URLs/access dates and do not make claims based on snippets alone. If offline, use repository evidence and create a source-request checklist. Do not invent article content, page numbers, peer-reviewed evidence, or new data.

Deliver a candid verdict: historical scenario reproducible is not the same as physical concentration labels verified. Flag whether there is enough provenance to proceed to comparable external concentration evaluation. A failed provenance gate must block any scientifically validated result, not erase the exploratory benchmark.
Stage C — External-study validation protocol and eligibility gates (zero model fits)
Write and freeze before reading external test outcomes or training a final model:

- EXTERNAL_VALIDATION_PROTOCOL.md (a complete pre-analysis plan), external_schema.json, source_manifest_template.csv, external_data_dictionary.md, and protocol_lock.json (SHA-256 content hashes and time of lock).
- Define intended prediction use case, target units and type, log transformation/offset, eligible environments (marine ocean-water observations per m³; explain exclusions), plastic verification, method family/mesh/depth metadata requirements and mismatch-handling. Do not pretend method harmonization is solved.
- Require independent publications/DOIs and no overlapping accession, source observations, derived records or near-duplicates with the development 72 studies; warn when the same campaign appears in multiple papers. Include DOI/metadata overlap audit and source-row fingerprints, with documented limitations.
- External data must have source provenance, unambiguous concentration units, signed sampling method context, and compatible model input features. Predeclare how missing inputs/categorical levels are handled using the frozen training pipeline, how invalid/incompatible studies are flagged, and do not impute or delete evaluation outcomes. Separate eligible from ineligible with explicit reasons. Never select studies by their observed error or targets.
- Predeclare all reports: pooled R² (only if finite and target varies), MAE, RMSE, study-macro MAE/MSE, median/p90/worst study MAE, % within ±1 log of (concentration + 0.001), counts and uncertainty. Show each study's size and error; a single-study external validation is not representative of broad generalization. If R² denominator is zero, report undefined, never zero.
- Predeclare study-cluster bootstrap uncertainty for MAE/RMSE and, when meaningful, R². If too few independent studies, report uncertainty as unstable/insufficient; do not present false precision. No resampling individual rows as if they were independent studies. Use fixed seed; document number of bootstrap replicates (e.g., 1,000) and bootstrap failure rules.
- Predeclare a descriptive comparison to the prior frozen exploratory benchmark, acknowledging it is a different population. Do not label that as paired or directly comparable; if both S1 and S2 are to be tested externally, freeze both models and a single primary endpoint beforehand, and adjust claims for multiple inspected candidates. Default: S2 only as the primary future candidate, selected based on past development work, with S1 only an explicitly labeled secondary candidate if independent data volume and compute justify it. No outcome-driven switch from S2 to S1.
- Define out-of-distribution and missing-method metadata flags from training input distributions only; do not tune thresholds against external outcomes.
- List strict statuses: NO_EXTERNAL_DATA, UNVERIFIED_TARGET_PROVENANCE, OVERLAP_FOUND, SCHEMA_INCOMPATIBLE, ELIGIBLE_UNSCORED, INDEPENDENT_EVALUATION_COMPLETED. Lack of independent data must produce NO_EXTERNAL_DATA, not an invented external score.
- Confirm freeze/lock: evaluate candidates only after protocol hash is recorded. Once results are viewed, that dataset is no longer an untouched test for more tuning.
  Stage D — Implement a ready-to-run external validation pipeline
  Create reusable, importable, testable Python modules under independent_validation_outputs/source/, and an executed notebook Stage5_Validation_Readiness_EXECUTED.ipynb (fresh kernel). The CLI and notebook must call the same implementations, without hidden fitting on import.

Implement separate commands:

1. --audit: benchmark + provenance checks; no training or scoring external outcomes.
2. --prepare-external PATH: inspect optional explicitly supplied external data, validate schema and physical units, duplicate/overlap leakage audit, produce eligibility and rejection reasons, and log whether targets are present without revealing or using them for model selection. Do not fetch or synthesize external observations. Treat missing path as NO_EXTERNAL_DATA (normal outcome).
3. --fit-frozen-research-model: only after writing/locking the protocol and satisfying the training authorization gate below. This is an optional request-specific research fit, not a default side effect.
4. --evaluate-external PATH: only when the protocol was locked before inspecting the dataset's labels, the independent dataset passes eligibility, the frozen full-data research model already exists, and all provenance concerns/target comparability are explicitly documented. Evaluate ONCE, save immutable detailed predictions and diagnostics; fail closed otherwise. Never tune, calibrate, pick weights or change filters based on its outcomes.
5. --tests: offline tests with tiny synthetic fixtures only; no model fits except an explicitly counted tiny smoke test if indispensable (prefer mocks). Run protections, scenario mismatch, study overlap, target-perturbation, finite/constant-target, unseen category, missing column, unit mismatch, leakage, and lock tests.
   Implement an explicit read-only mode when no external file exists; the default execution must still finish, execute all notebook cells, run tests, write manifests and deliver the report. Never use toy/synthetic data to report an external R². Source evidence cannot be fabricated.
   Stage E — Optional training: full-development S2 research model (bounded)
   A final single all-data model cannot be obtained simply by copying one of the five distinct Stage 4 S2 fold-selected models. Their training sets and selected configurations differ.

Default: skip fitting; create the exact plan and CLI only. Authorize the fit only if (i) this protocol is already locked, (ii) an identified external validation set or realistic evaluation campaign exists with its intended target definition frozen, (iii) no overlap or known incompatibility is detected, and (iv) it is clearly labeled a research_only_historical_hypothetical_unit_scenario model. If any gate fails, output a machine-readable NOT_TRAINED status with reasons; do not ask the user to confirm in the middle of the run.

If authorized, freeze S2's rule exactly from Stage 4 rather than inventing another selector: using only the full 72-study development set, create a new GroupKFold(3) inner partition. Evaluate only the original eight C0–C7 fixed LightGBM configurations with exactly the same preprocessing behavior, target, seed and package environment. Select the one final configuration from those inner OOF predictions using S2's existing inner-only guard / 0.010 near-best-band / tie-break rule. Then fit one model on all 16,988 development rows. Fit transforms on development data only; serialize the entire transformer+estimator and unambiguous schema and metadata. Save selected candidate, inner metrics, provenance hashes and training-only diagnostics; never claim training-set fit metrics are independent performance. Model artifact must explicitly carry the historical hypothesis warning.

Hard training ceiling: 24 candidate inner LightGBM fits + 1 full-data selected refit = 25 new LightGBM fits total. Verify cached fit identity if resuming; no other fit jobs are permitted. Never reuse prior fold checkpoints trained on a different set as if they were full-data fits. Maximum threads=4. Record all fit start/finish/failure events and ensure hard caps and crash-safe resume. No model selection using external study labels. If an optional run is unavailable, leave exact, executable CLI instructions for later.
Required integrity checks

- Verify all untouched files retain their original SHA-256 hashes; save before/after manifests.
- Validate the original reference values from actual preserved outputs to available precision. Do not rerun historical model training.
- Validate exactly one source-row identity per OOF reference, exact group membership, no accidental join duplications, labels and scenario alignment.
- Strict separation of train-only selection and external target evaluation; test that changing external labels cannot change selected model, eligibility filtering, imputers or any hyperparameters.
- Reject test-study DOI/accession/measurement duplicates even if DOI strings use different formatting; document what metadata matching cannot detect.
- No synthetic result reported as empirical; no invented or scraped pseudo-observations presented as independent study measurements.
- All CLI options documented and executed with tests and a fresh-kernel notebook. If an article cannot be accessed or metadata needed for verification is missing, report the blocker precisely without failing unrelated pipeline tests.
  Deliverables — save only under independent_validation_outputs/
- STAGE5_REPORT.md (write last): plain-English decision, exact audit coverage, provenance verdicts and links, lock status, independent-data status, number of new fits, and candid next action.
- PROVENANCE_AUDIT.md, provenance_row_audit.csv, source_request_checklist.md.
- EXTERNAL_VALIDATION_PROTOCOL.md, protocol_lock.json, external_schema.json, external_data_dictionary.md, source_manifest_template.csv.
- baseline_identity.json, protected_hash_manifest_before.json, protected_hash_manifest_after.json, integrity_result.json.
- source/ (importable CLI and modules), Stage5_Validation_Readiness_EXECUTED.ipynb, notebook_execution_status.json, tests.txt.
- external_eligibility_audit.csv and external_status.json only if external data is supplied; otherwise external_status.json with NO_EXTERNAL_DATA.
- full_data_model_status.json, and full_data_research_model/ only if training authorization gate passes; otherwise keep no model and say NOT_TRAINED.
- fit_log.jsonl and environment.json.
- external_predictions.csv and EXTERNAL_TEST_REPORT.md only if a real eligible new-study evaluation actually takes place; never produce these by filling in fabricated values.
  Operating procedure

1. Inspect protected files and freeze input hashes. Detect existing external independent data, but do not search indiscriminately or load its outcome column for model development.
2. Perform provenance audit with the evidence actually available; distinguish verified/ambiguous/unresolved clearly.
3. Write and cryptographically lock the prespecified external eligibility and metric plan.
4. Implement CLI and notebook, execute all cells and offline tests, validate protected-file equality.
5. With no eligible external file: report NO_EXTERNAL_DATA, do not initiate any LightGBM training, save the readiness deliverables and stop.
6. With a suitable, demonstrably independent, provenance-compatible external file: follow the locked protocol and optional fitting gate; no hyperparameter search beyond the frozen full-development S2 training-only selector. If evaluating, reveal outcomes only once at the end and explicitly label independent data status/limitations.
7. Write the report LAST. Do not initiate Stage 6 or another tuning cycle.
   Exact final summary Codex must return
   Give a concise summary:

- Historical reference protected: yes/no, showing original and Stage 4 S2 reference metrics.
- Provenance for Barrows and Dehm: verified / ambiguous / unresolved, with primary-source evidence limits.
- Independent external study set: available and eligible / unavailable / ineligible, number of unique studies and observations when available.
- External protocol: locked/not locked, hash and file path.
- New LightGBM fits: exact count 0 by default (or up to 25 only if gated conditions pass).
- Actual independent held-out R²: numerical only if truly measured; otherwise NOT AVAILABLE.
- Tests and notebook: passed/failed, actual execution.
- independent_validation_outputs/STAGE5_REPORT.md path and the next scientifically meaningful action.
  STOP. Keep 0.5256280075581679 as an exploratory reused-benchmark S2 result. A better unseen-study result must be demonstrated on genuinely new, traceable observations—not declared from further reuse of the historical five folds.
