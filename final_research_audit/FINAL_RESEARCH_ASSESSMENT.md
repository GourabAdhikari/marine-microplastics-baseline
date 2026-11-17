# Final research assessment

## Verdict

**Qualified readiness for a transparent exploratory methods/reproducibility manuscript, not for a validated marine-concentration prediction or generalization claim.** The project is auditable and can report useful negative and positive findings, but a publishable scientific argument must foreground unresolved labels, study confounding and repeated benchmark inspection. This is not an acceptance guarantee or proof of novelty. The modest same-family selector comparisons alone do not establish a new architecture or broad predictive superiority.

The defensible contribution is a reproducible study-aware benchmark and an analysis of reliability trade-offs under an explicitly hypothetical historical unit scenario. Stronger environmental or deployment claims are **not ready**. Make the scenario caveat prominent in title/abstract/methods/results, correct any original notebook wording that presents unit hypotheses as established fact in the manuscript (the protected source itself is unchanged), and publish complete folds, failures and provenance limitations.

## Four preserved held-out procedures

Scenario: `historical_hypothetical_unit_scenario`; identical 16,988 rows and 72 studies. These are existing grouped cross-validation results, not four independent datasets.

| procedure | heldout_pooled_R2 | heldout_pooled_MAE | heldout_pooled_RMSE | heldout_pooled_study_macro_MAE | mean_fold_testing_R2 | p90_study_MAE | worst_study_MAE | within_1_log_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_C0 | 0.472630840 | 0.996377423 | 1.418197731 | 1.060942164 | -0.071234992 | 1.813391314 | 4.943934893 | 63.939251236 |
| previously_optimized | 0.503381285 | 1.009342570 | 1.376229816 | 1.001411331 | -0.080676231 | 1.901217782 | 5.295836118 | 61.007770191 |
| S1 | 0.519491615 | 0.995282641 | 1.353723309 | 0.980533172 | -0.055044517 | 1.833019340 | 5.244593780 | 62.220390864 |
| S2 | 0.525628008 | 0.991546912 | 1.345051587 | 0.999353649 | -0.037263820 | 1.786402450 | 5.165887296 | 62.832587709 |

S2 improves pooled R² by 0.052997168 over original C0 and 0.022246723 over the prior optimized procedure. Study-macro MAE is best for S1, not S2; S2's macro gain over optimized is small (~0.00206 log10). Both S1 and S2 pass the prior frozen descriptive screen but neither dominates every endpoint. Compared with original, S2 has worse worst-study MAE (5.165887 versus 4.943935) and lower within-one-log coverage (62.83% versus 63.94%). Compared with optimized, S2 still worsens 26 of 72 studies, including 10 by >0.10 log10 MAE; versus original it worsens 33, including 17 by that threshold (`study_robustness_outputs/study_harm_counts.csv`). This is not universal study-level robustness.

## Seen-data fit, held-out CV, and independent testing are distinct

1. **Training:** S2's mean fold resubstitution R² is 0.782873031. These predictions reuse training observations and fitted preprocessing, so this is optimistic goodness-of-fit, not validation.
2. **Historical grouped CV:** S2 pooled R² is 0.5256280075581679; mean fold R² is -0.037263820. Fold-wise gaps and MAE/RMSE are in `TRAINING_VS_TESTING.md`. All four procedures have negative test R² on folds 1 and 4; optimized/S1/S2 also have negative fold 2. All four mean fold R² values remain negative. Positive pooled R² partly reflects between-study concentration variation and cannot mean reliable prediction for every new study.
3. **Independent external testing:** **NOT AVAILABLE for all four procedures.** Stage 5 status is NO_EXTERNAL_DATA, provenance remains unresolved, and no all-development research model or external evaluation exists. Do not relabel a re-scored historical fold, a training score or the 0.764 random-row OOF result as independent external validation.

The large seen/unseen gaps support concern about overfitting or recognition of study-specific conditions. Distribution shifts, different fold variance denominators and hidden measurement protocols also contribute; the gaps do not quantify the contribution of each mechanism.

## Integrity strengths and selection-bias limits

- Exact grouping through DOI/accession connected components and fixed outer study partitions; no study crosses an outer train/test boundary. Each test row is scored once. Median imputation, scaling, one-hot encoding and clipping use training data only; identifiers/outcomes/fold IDs are not predictors.
- Prior optimization selected configurations using inner study-held-out metrics; Stage 4 S1/S2 selected among the same eight configurations using only saved inner summaries with auditable locked rules. This final audit performs no new selection and reproduces S2's saved test predictions exactly.
- However, the outer results guided multiple rounds of research interpretation and candidate consideration. Nested selection within each procedure does **not** undo this between-experiment selection bias. S2 was named the final research candidate after repeated inspection of these same five folds. Improvements are exploratory; no valid independent significance or generalization claim follows from their rank.
- The 6,135-row study is 36.11% of the data and entirely determines fold 1. The Stage 4 decomposition shows only ~15.18% of S2's gain over optimized comes from that largest study, so its gain is not solely due to that study. Nonetheless pooled weighting and sparse environmental/method coverage constrain interpretation; macro/tail metrics must accompany pooled R².

## Measurement quality and scientific provenance

The NOAA export mixes media/units; the fixed benchmark filters ocean water in pieces/m³. Stage 5 documented 53 eligible rows missing depth, 4,958 numerical zeros, 643 rows sharing a measurement signature and 2,378 sharing date/location/method. Repeated signatures may be real replicates, not proven duplicate errors; no rows were removed. Zero interpretation, detection limits, blanks, recovery, filtered volume and synthetic-polymer confirmation are not adequately represented in the export.

**Barrows (1,389 rows) and Dehm (81 rows) ×10^6 multipliers remain unverified hypotheses.** SI conversion of particles/L to particles/m³ is ×1000; an extra factor of 1000 requires independently documented erroneous prior division and row-level source mapping. Neither predictive score improvement nor a plausible study mean establishes that. Primary methods were not accessible in the bounded Stage 5 audit. Barrows' grab-sample mesh-field value may describe a membrane pore, not actual net mesh; synthetic-only comparability and Dehm row-specific traceability remain unresolved. The 33 bongo-net observations are also flagged as unverified, not deleted or rescaled.

Thus the target is a **historical hypothetical numerical scenario**, not certified physical ground truth or verified true synthetic microplastic abundance. Apparent model skill may reflect differences in study protocol, particle detection/size range and units as well as environmental variation. Feature associations are not causal effects and cannot be claimed as maps of true marine plastic burden.

## Defensible claims and prerequisites for stronger claims

**Defensible now:** computational reproduction of stated scores and transformations; study-disjoint CV performance under this scenario; exact seen-data/held-out gaps; conditional exploratory trade-offs among fixed LightGBM configurations/selectors; unresolved scientific-provenance and external-validation limitations. Report every negative fold and harmed study and avoid describing R² as percent accuracy.

**Not defensible now:** independently superior new-study generalization; verified real-world concentration forecasts; certified unit corrections; causal geography/protocol effects; global monitoring/deployment readiness; a claim that all studies improve; or external R² of any value.

**Next scientific actions (recommendations only, not executed):** obtain primary Barrows/Dehm articles, supplements, original observations and NCEI ingest mappings; audit counts/volumes, synthetic confirmation, filter/mesh context, blanks and nondetects; resolve duplicate/campaign provenance; obtain genuinely new comparable studies and follow the locked Stage 5 protocol before outcomes are examined. If evidence changes the target definition, preregister a separate scientific scenario rather than rewriting this benchmark. Then one frozen S2 full-development model can be assessed on a truly independent set with study-cluster uncertainty, not another score-driven historical-fold iteration.

## Audit artifacts and limits

`METRICS_COMPARISON.csv` explicitly marks unmeasured training metrics for original/optimized/S1 and every external R² as NOT_AVAILABLE. The original notebook's identifiable 0.764 is rounded random-row five-fold held-out R², not a verified training score. No original model was retrained. All new calculations use the established environment and exact original pipeline; five new S2 fits were necessary because old checkpoint files saved predictions only.

**Actual new fits: 5; no tuning, searches, downloads, external datasets or follow-on phase.** Original files and all previous results remain unchanged. This audit ends here.
