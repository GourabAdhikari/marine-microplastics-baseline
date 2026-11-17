# Stage 5 — independent-study validation readiness and provenance

## Plain-English decision

**Readiness work completed; independent performance is NOT AVAILABLE.** No eligible independent external study set was supplied or found in the bounded repository inventory. External status: **NO_EXTERNAL_DATA**. Observations/studies available for independent evaluation: **0 / 0**. No external concentration outcomes were read, no external R² was invented, and no independent predictions/test report were generated.

The historical references remain protected and unchanged:

| Preserved procedure | Historical pooled R² | Study-macro MAE | Worst-study MAE |
| --- | ---: | ---: | ---: |
| Original C0 | 0.4726308398591973 | 1.060942164 | 4.943934893 |
| Prior optimized | 0.5033812847921568 | 1.001411331 | 5.295836118 |
| Stage 4 S1 | 0.51949161468415 | 0.980533172 | 5.244593780 |
| Stage 4 S2 | 0.5256280075581679 | 0.999353649 | 5.165887296 |

These numbers are **existing exploratory reused-benchmark results**, not Stage 5 independent evidence. S2 was identified after repeated inspection of the same five outer folds. Its negative mean-fold R² (-0.037263820), heterogeneous study errors and unresolved physical labels remain relevant. The target is `log10(concentration + 0.001)` under `historical_hypothetical_unit_scenario`, not verified true synthetic microplastic abundance.

## Exact benchmark audit coverage

Verified the raw CSV fingerprint and 29,076 ×35 structure; exact 16,988 eligible source-row identities and order; 72 DOI/accession-linked study groups; all 5 outer memberships (6,135 / 2,714 / 2,712 / 2,714 / 2,713); common target/offset; exact eight predictors; original method families, mesh/depth/month transforms; training-only preprocessing and inference clipping; original C0 parameters and eight-candidate registry; and immutable S1/S2 code/locked decisions. No DOI, accession, group ID, fold or outcome is a predictor. Original raw missingness is recorded in `baseline_identity.json`.

Recomputed preserved-reference metrics from existing OOF predictions solely to verify their identity, not to propose a new score. Cross-checked prior manifests and **125** already existing checkpoint hashes; no historical models were rerun. The development identity index is target-free. Baseline identity: `baseline_identity.json`; raw data SHA-256 `5c4634c7c419f8a2c4b03adac4081e869bc59b650d86745ea66872388890a702`.

## Scientific provenance verdict

- **Barrows: unresolved**. DOI [10.1016/j.envpol.2018.02.062](https://doi.org/10.1016/j.envpol.2018.02.062), 1,389 affected rows.
- **Dehm: unresolved**. DOI [10.1016/j.marpolbul.2020.111239](https://doi.org/10.1016/j.marpolbul.2020.111239), 81 affected rows.
- Published title/publisher location was accessible through DOI redirect metadata. Primary article methods/full text were **not_accessible** (publisher URLs returned 403); NCEI source landing requests timed out. **6 targeted public-source lookups**, at the six-lookup cap. No snippet-based physical claims, guessed page numbers or invented data. Dates, URLs, bounded responses and hashes are saved under `evidence/`.
- The CSV's fields and historical multiplication code are reproducible, but row-specific source measurements, original volumes, aliquot/recovery/extraction/blank calculations, zero meanings and synthetic-only confirmation remain unresolved. The old notebook's litre-scale means are not treated as independently verified evidence.
- Mathematically, particles/L converts to particles/m³ by **×1000**. **×10^6** would require proof that the archived number was already divided by 1000 relative to the original per-L quantity. The printed CSV unit and a plausible mean do not prove that ingest error. No row-specific correction was authorized.
- Barrows is labeled grab sampling, not net sampling. The 0.00045-mm field equals 0.45 µm dimensionally; it may be membrane-pore information, but is not verified actual net mesh. Dehm's aggregate plausibility does not establish traceability of 81 observations.
- Other flags: 643 eligible rows share a full measurement signature; 2378 share date/location/method; 4958 zeros; 0 non-DOI rows; no explicit volume, polymer-confirmation, blank-correction or detection-limit fields. These are investigation flags, not authorized removals or evidence that all repeated signatures are duplicates. All historical records remain unchanged.

See `PROVENANCE_AUDIT.md`, `provenance_row_audit.csv` (two well-documented aggregates with every relevant source-row index), `duplicate_signature_audit.csv`, and `source_request_checklist.md`.

**Provenance gate: UNVERIFIED_TARGET_PROVENANCE.** Reproducible historical numbers are not physically verified labels. This blocks a scientifically validated comparable-concentration result, without erasing the historical benchmark.

## External protocol frozen before any external outcome access

- Protocol: `independent_validation_outputs/EXTERNAL_VALIDATION_PROTOCOL.md`.
- Lock: `independent_validation_outputs/protocol_lock.json`; locked **2026-11-16T13:28:10.839511+00:00**.
- Combined protocol-content SHA-256: **`19f3090443de03d63dfae71d973375717c0f9e76a786b61afd01b661142574ad`**.
- Protocol document SHA-256: `863b511de3119af33e39e295035a4c00e27ebc66eb8a3e5771f3e1faf15d835b`.
- Primary candidate **S2 only**; primary endpoint **study-macro log10 MAE**. No switch to S1, blending, threshold optimization or additional candidate search.

The locked plan defines physical/environment/verification/method eligibility; explicit row/study rejection reasons; canonical DOI/accession/source/ancestry overlap and target-free fingerprint checks; same-day/same-method 1-km near-duplicate blocking; campaign-alias limitations; connected external study groups; exact predictor transforms and training-only support flags; missing depth and unknown-method behavior; and no outcome-dependent exclusions. Missing campaign inventory in the development data means automated checks cannot prove independence; signed source review is mandatory.

Predeclared external metrics include pooled R² only when defined (constant/single targets produce null, not zero), MAE/RMSE, all study-macro/tail metrics, tolerance coverage, individual-study errors, counts and uncertainty. Whole-study cluster bootstrap: seed 0, 1,000 replicates, fewer than five groups insufficient, 5–9 groups unstable, <90% valid R² replicates suppresses the interval. No individual-row bootstrap pretending clustered records are independent. Historical/external comparisons are unpaired and different-population.

The pipeline verifies the lock and model/source/provenance receipt before one-time outcome access. An exclusive marker is written before reading concentration; failure consumes the dataset, and further tuning or re-evaluation is blocked. No externally supplied pickle is loaded. Finite/nonnegative outcomes are required without imputation/deletion; scientific provenance is distinct from mere schema validity.

## Pipeline and optional model status

**NOT_TRAINED**, zero new LightGBM fits (**actual fit starts: 0**). No full-data model directory was created. Reasons: ["No identified metadata-eligible independent set with resolved target comparability", "development_unit_and_synthetic_provenance_unresolved"].

The exact optional plan is executable and frozen: all 72 development studies, new GroupKFold(3), the same eight settings, exact Stage 4 S2 selection using training-only predictions, 24 candidate fits plus one all-data refit (hard cap 25, four threads). Five different old outer-fold models are **not** misrepresented as one final model. This optional training is not authorized now. A future request must supply a real metadata-eligible independent set, inspectable signed development/target-comparability evidence including row-specific source calculations, and an untouched outcome receipt before training. The resulting artifact remains research-only with the historical-hypothesis warning.

All five CLI commands were executed in the notebook: `--audit`, `--prepare-external` without data, `--fit-frozen-research-model` (blocked), `--evaluate-external` without data, and `--tests`. Commands and future exact usage are documented in `CLI_COMMANDS.md`. Default audit is read-only with respect to every prior artifact and never trains or scores external labels. Positive real-data training/evaluation paths remain **unexecuted**; no claim of empirical external success is made from the mocks or readiness tests.

## Tests, notebook and preservation

- **20 tests passed**, 0 failures, 0 errors, 0 skipped. Tiny synthetic fixtures stay in offline tests; no synthetic observations or scores are presented as independent data. Tests cover immutable references, lock tampering, DOI/accession normalization, observation/derivation overlap, missing columns, units/scenario, whole-study quarantine, target perturbation, fixed preprocessing/clipping/unseen categories, support flags, finite/constant-target metrics, cluster-bootstrap failure rules, training guards/cap, outcome-access blocking and original-file protection.
- **9 notebook code cells executed** in a fresh kernel. `Stage5_Validation_Readiness_EXECUTED.ipynb`, `notebook_execution_status.json`, `tests.txt` and per-command logs record actual execution. Any preparation/debug failures remain in execution logs; only the final successful test status is quoted here.
- **27311 pre-existing paths unchanged**, including `.git/`, original code/data/notebooks/models, lockfile and every previous output directory. Complete before/after hash manifests and `integrity_result.json` are saved. Symlinks are recorded without following outside interpreter links.
- Existing reproducible environment reused; no library downloads or dependency changes. Six narrow source-location requests only, no external-data acquisition. All new files and controllable caches/runtime files remain under `independent_validation_outputs/`. No new optimization, architecture, blend, chronological experiment or later phase.

## Next scientifically meaningful action

Obtain the two primary articles/supplements and NCEI original submission mappings, trace each adjusted row to counts and volume, document synthetic confirmation and detection/blank/recovery treatment, and resolve comparability without changing the preserved historical targets. Then secure genuinely new, untouched independent study/campaign observations with source evidence and the locked metadata schema. Apply the existing gates before explicitly authorizing research fitting and a single external evaluation. If evidence instead defines a different outcome, it needs a separate preregistered scientific protocol—not a silent correction or another historical-fold score.

**Actual independent held-out R²: NOT AVAILABLE. Stop here; no Stage 6 or further tuning is launched.**
