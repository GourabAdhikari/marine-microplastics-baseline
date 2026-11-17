# External data dictionary

CSV is an explicitly supplied study collection, not auto-discovered test data. `concentration` is the optional sequestered outcome column and is loaded only once by authorized evaluation; prepare reads its header only. Outcomes must be original independently traced nonnegative finite particle counts per m³, not model predictions or imputed values.

- `source_row_id`: original observation identity, globally source-qualified; unique; never a predictor.
- `study_id`, `doi`, `accession`, `campaign_id`: provenance only. DOI must be a real DOI; accession may be blank if unassigned but original source identity is required. Campaign/cruise identity and aliases must be independently reviewed.
- `date`: ISO YYYY-MM-DD sampling date. Required along with finite `lat` in [-90,90], `lon` in [-180,180].
- `method`: explicit sampling method matching a historical method after lowercase/hyphen normalization. Never infer method from outcomes.
- `mesh_mm`: finite positive sampling selectivity field, with `mesh_kind` equal to net_mesh for surface nets and membrane_pore_size otherwise. This naming is not proof of equivalent recovery or size spectrum: signed method-context evidence is mandatory.
- `depth_m`: nonnegative or blank; blank uses development median through frozen pipeline. Missing other numeric predictors or missing method is incompatible.
- `medium`: exactly Ocean water. `unit`: exactly pieces/m3. `target_type`: exactly synthetic_particle_count. `scenario`: traceable_synthetic_particle_count_per_m3. No automatic per-L, per-area or mass conversion.
- Outcome `concentration`: optional for preparation/training campaign registration, mandatory finite nonnegative for actual evaluation. Empty/invalid outcomes block the entire evaluation; no imputation/deletion.

Sidecar `PATH.manifest.json`: {"studies": [objects with source_manifest_template.csv fields]}. One object per study. `source_document_path` is repository-local inspectable evidence (copied only under this stage directory), with SHA-256; `source_file_sha256` binds the supplied CSV. Signed review fields are an auditable named-reviewer attestation, not cryptographic proof of scientific truth. No placeholder is accepted as reviewed. `campaign_aliases`, `ancestor_source_ids`, `ancestor_dois`, `ancestor_accessions` are JSON arrays.

A separate `development_provenance_review.json` must identify the immutable raw-data SHA-256, preserve the historical scenario, include signed reviewer/date/signature and repository-local evidence path/hash, and explicitly resolve development unit traceability, synthetic-target compatibility, and external target comparability without silently altering labels. A false assertion is not a source: evidence must be inspectable and reviewed before test outcomes. Currently this file is intentionally absent because the primary measurements are unresolved.

No synthetic data files are supplied as observations. Test-only fixtures are kept in tests and never become empirical predictions.
