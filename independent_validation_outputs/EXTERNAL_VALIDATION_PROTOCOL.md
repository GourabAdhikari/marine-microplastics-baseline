# Locked independent-study external validation protocol — version 1

## Purpose and candidate

Evaluate **S2 only** as a research candidate for genuinely unseen marine ocean-water studies with traceable synthetic-particle count concentration in pieces/m³. No secondary S1 model, switching, blending or multiple-candidate leaderboard is authorized. S2 was chosen after many analyses of the same historical five folds; historical R² 0.5256280075581679 is exploratory, not a new holdout. Historical training scenario remains `historical_hypothetical_unit_scenario` and any fitted artifact is labeled `research_only_historical_hypothetical_unit_scenario`. Independent observations have their own `traceable_synthetic_particle_count_per_m3` provenance label; similar numerical units do not certify target equivalence.

Primary external endpoint: **study-macro MAE on log10(concentration + 0.001)**, fixed before outcomes. Secondary endpoints: pooled R², log10 MAE/RMSE, study-macro MSE, median/p90/worst study MAE, and within-one-log fraction. No success threshold is learned from external labels. Describe comparison to historical numbers as **unpaired, different-population** only; never claim a pooled-score difference proves improvement.

## Eligibility before outcomes

Accept original, untouched source observations from marine ocean water, with count-per-volume synthetic-particle outcomes, documented volume computation and synthetic-verification method, signed sampling/filtration context, and all model features. Exclude sediment, beach sand, freshwater, search-count nurdles, pellets per time, mass, per-area counts, mixed environmental media, unresolved fibre identity, and ambiguous conversions. No automatic unit conversions. Do not claim mesh/filter/method harmonization is solved: membrane pores and actual net meshes differ and require explicit compatible-context review.

Each supplied study needs the schema and signed manifest defined in external_schema.json and external_data_dictionary.md. The original source record, sample volume, correction/subsample/recovery/blank factors, zero/detection-limit meaning, sampling date, locations and plastic verification must be inspectable. Named reviewers and evidence paths/hashes are mandatory; flags alone do not establish truth. Missing historical development provenance blocks scientifically validated scoring even if external metadata look plausible.

Prepare reads metadata columns only, logs whether the outcome header exists and hashes the source bytes without exposing outcome values. The same metadata with perturbed labels must have identical eligibility decisions, features and model selection. A changed file checksum still invalidates an existing receipt as a tampering safeguard. No outcomes enter study inclusion, support thresholds or model selection. No silent deletion, imputation or substitution of evaluation outcomes.

A study with any invalid row is quarantined as a whole with explicit reasons. Valid and invalid studies are listed separately; a supplied collection containing any quarantined study is not scored. Assemble a new explicitly documented eligible collection before any outcome access; never remove difficult studies after seeing errors. Unseen numeric feature ranges are flags, not an outcome-driven exclusion. Missing depth uses the training median; missing date/location/mesh/method is incompatible. Historical unknown-family encoding is retained in the pipeline, but unknown sampling methods require source review and are rejected by this version, rather than silently labeled bulk water.

## Independence and leakage audit

Require publication/DOI, accession if assigned, campaign/cruise identity, original source-row identities, derivation ancestry and signed independence review. Normalize DOI case, doi: prefix, URL/dx.doi.org form and percent-encoding; normalize numeric accession zero padding. Block overlap of DOI, accession, source IDs, ancestor IDs/DOIs/accessions or campaign aliases with development sources. Build target-free fingerprints from sampling date, latitude/longitude rounded to six decimals and normalized method. Exact matching fingerprints and same-day/same-method points within **1 km** of a development sample are flagged as potential overlap and block scoring pending source resolution, never dismissed based on errors.

The historical export has no complete campaign/ancestry inventory. Automated checks cannot prove independence or detect all spatially/time-shifted, relabeled or derived measurements. Conservative near-duplicate flags can include genuine adjacent independent samples; source review is mandatory and this locked version does not automatically waive them. Shared external DOI/accession/campaign/source identities are grouped into connected components for study-level metrics, rather than counted as independent clusters.

## Transformations and support

Use the frozen original eight predictors: latitude, longitude, year, month sine/cosine (2πm/12), log10(mesh_mm), log1p(depth_m), method family. Never include study IDs, DOI, accession, campaign, observed concentration, fold or outcome-derived summaries. Reuse original method-family mapping and explicit known bulk-water methods.

Numeric imputation, scaling, dense one-hot encoding with unknown categories ignored and range clipping are fitted on the relevant development training partition only. For inference, numeric features outside full-development min/max receive OOD flags and the model's frozen numeric clipping; original missing depth receives its learned median. Additional flags: unknown category, missing depth, and each out-of-range numeric predictor. Thresholds use development inputs only, never external outcomes; no new support estimate is trained on test data.

## Model authorization and selection

Default: no fitting. An explicit --fit-frozen-research-model request must follow this lock, an identified metadata-eligible untouched external dataset with intended target fixed, and a signed inspectable development/target-comparability review. No overlap, known incompatibility, prior outcome inspection or provenance blocker may be present. The command otherwise writes NOT_TRAINED with reasons. A hypothetical future campaign with no traceable metadata is insufficient for this implementation.

If authorized: one new GroupKFold(3) over all 72 development study groups, the exact original eight C0–C7 settings, 24 training-only candidate fits and one selected full-data fit. Exact Stage 4 S2 rule: R²≥C0−0.010; macro≤1.03 C0; p90≤1.03 C0; worst≤1.10 C0; within 0.010 of best eligible R², choose macro then p90 then descending R² then ID. C0 eligible; nonfinite metrics fail. No early stopping or new setting. Hard ceiling **25 newly started fits**, maximum four threads. Crash-safe fit-start ledger; resume only completed exact-identity checkpoints. A started fit without a valid completed artifact is blocked, not silently repeated. No previous outer-fold model is a full-data research model.

Serialize original transformer+estimator, numeric clipping ranges, feature schema, final parameter choice, training-only diagnostics and complete code/data/config/split/environment/scenario identities. No training-set score is independent performance. Scientific evidence must support the comparison, but the immutable historical scenario must never be relabeled certified truth.

## One-time outcome access and reports

A protocol lock must predate dataset preparation and any external outcome read. The source file/manifest/evidence hashes, review receipts, pipeline code, selected model and eligibility must match at scoring. Write an exclusive outcome-access marker **before** reading concentration values. This consumes that dataset for this protocol even if scoring fails; no error-driven retries, new models, calibrations or thresholds. Already inspected sources cannot be restored to untouched status by moving or renaming files. Exact-file, metadata/source and study-identity checks are recorded; deliberate altered identifiers cannot be solved automatically.

Read every eligible target once; invalid/missing/nonfinite/negative targets abort the complete evaluation and report the failure, not a partial score. No clipping of observed outcomes. Score log10(y+0.001). Pooled R² is undefined (JSON null with explanation) if one observation or zero target variance; never force zero. MAE/RMSE and all per-study errors remain reported, including all small studies. Include all source IDs, predictions, errors, method/size/coverage/support flags, counts and uncertainty. Single-study validation describes only that study, not broad new-study generalization.

## Predeclared uncertainty

Study-cluster bootstrap, seed **0**, **1,000** replicates: sample independent connected study components with replacement and include all rows of each sampled component. A drawn component twice counts as two replicate clusters. Report percentile 95% intervals for MAE, RMSE, study-macro MAE and R² when defined. Never bootstrap individual rows. With fewer than **5** components: intervals insufficient, do not compute; 5–9: label unstable/few studies. If fewer than 90% of replicate R² values are defined, suppress that interval and report failure counts. No resampling to manufacture finite variance or precision. Intervals are descriptive and do not correct past model-selection multiplicity or measurement bias.

## Statuses and hard stop

Exact statuses: NO_EXTERNAL_DATA, UNVERIFIED_TARGET_PROVENANCE, OVERLAP_FOUND, SCHEMA_INCOMPATIBLE, ELIGIBLE_UNSCORED, INDEPENDENT_EVALUATION_COMPLETED. NO_EXTERNAL_DATA is a normal read-only outcome, not a score. Development provenance can separately be UNVERIFIED_TARGET_PROVENANCE while no external dataset exists. ELIGIBLE_UNSCORED is a schema/provenance gate status, not performance. INDEPENDENT_EVALUATION_COMPLETED requires actual eligible new observations and complete one-time evaluation; no synthetic fixture is empirical data. Failed provenance/overlap/schema/model/lock gates block outcome scoring.

No data downloads, historical fold tuning, chronological evaluation, automatic fitting on import, or follow-on phase. Produce the readiness notebook/tests and report even with no external set. Protocol version, content hashes and lock time are immutable. New evidence can be appended as signed review artifacts; a changed endpoint/schema/rule requires a separate future preregistered version before any new labels, not editing this lock.
