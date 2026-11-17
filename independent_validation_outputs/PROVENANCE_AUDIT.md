# Provenance audit — physical labels are not verified

The historical scenario is computationally reproducible. That does **not** verify physical concentration labels, sampling comparability or synthetic-only identification. No CSV values were edited and no new unit multiplier is authorized. All historical multipliers remain `hypothesis_only`.

## Evidence levels

1. **Verified from inspectable primary evidence:** the publisher DOI linking pages identify the named article titles and publisher locations only. The original NOAA export is direct evidence of what the archived fields contain (not proof of their physical correctness). Exact file/response hashes and source-row mappings are saved. Mathematical SI dimensional conversion is exact, but its applicability to these records is not verified.
2. **Supported but ambiguous:** the existing notebook describes litre-scale concentrations and assumes an inverted conversion. These are prior project assertions, not independent primary-source verification. Agreement of an aggregate mean with a hypothesized scale is plausibility evidence at most.
3. **Unresolved:** both studies' row-specific original counts/volumes, original published concentration units, blank/recovery/subsampling treatment, actual filter protocol and confirmed synthetic fraction. Primary article content was not obtained; no page numbers or quoted methods are invented.

## Bounded primary-source attempts

Six targeted lookups only, no search-engine snippets, new observation downloads or broad literature search. The two DOI URLs returned HTTP 200 **redirect HTML**, not articles. The two article URLs returned HTTP 403; the two NCEI landing URLs timed out. Network was available for DOI resolution, but scientific content was not accessible in this run. Saved response bodies (403 bodies bounded at 200 kB) are not treated as article evidence.

- [barrows_doi](https://doi.org/10.1016/j.envpol.2018.02.062), accessed 2026-11-16T13:06:49.642352+00:00: HTTP 200; publisher linking/redirect metadata only; saved `evidence/barrows_doi_response.txt`.
- [dehm_doi](https://doi.org/10.1016/j.marpolbul.2020.111239), accessed 2026-11-16T13:06:51.407001+00:00: HTTP 200; publisher linking/redirect metadata only; saved `evidence/dehm_doi_response.txt`.
- [barrows_archive](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:211009), accessed 2026-11-16T13:06:53.311061+00:00: HTTP None; TimeoutError('The read operation timed out'); saved `evidence/barrows_archive_response.txt`.
- [dehm_archive](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:294676), accessed 2026-11-16T13:07:19.018995+00:00: HTTP None; TimeoutError('The read operation timed out'); saved `evidence/dehm_archive_response.txt`.
- [barrows_publisher](https://www.sciencedirect.com/science/article/pii/S0269749117349400), accessed 2026-11-16T13:08:13.413599+00:00: HTTP 403; HTTP Error 403: Forbidden; saved `evidence/barrows_publisher_response.txt`.
- [dehm_publisher](https://www.sciencedirect.com/science/article/pii/S0025326X2030357X), accessed 2026-11-16T13:08:13.813751+00:00: HTTP 403; HTTP Error 403: Forbidden; saved `evidence/dehm_publisher_response.txt`.

### Barrows

Barrows, A.P.W., S.E. Cathey, C.W. Petersen. 2018. Marine environment microfiber contamination: global patterns and the diversity of microparticle origins. Environ. Pollut. 237, 275–284. [DOI](https://doi.org/10.1016/j.envpol.2018.02.062); [NCEI archive](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:211009). Citation text comes from the CSV; the publisher DOI landing page confirms the title, not the methods.

CSV audit: 1389 rows; Grab sample; reported field unit `pieces/m3`; raw mean 0.012777537797, median 0.005; 241 zeros; 0 missing concentrations. Mesh field 0.00045 mm; depth 0.45 m. Row IDs are enumerated in `provenance_row_audit.csv`.

**Verdict: unresolved / hypothesis_only.** Primary full text, supporting data, method page/table/equation, sample volumes, volume computation, aliquot/recovery factors, extraction, blank handling, and synthetic-polymer identification are `not_accessible` or unresolved. No row-specific source-to-CSV mapping supports an authorized correction.

### Dehm

Dehm, J., S. Singh, M. Ferreira, S. Piovano. 2020. Microplastics in subsurface coastal waters along the southern coast of Viti Levu in Fiji, South Pacific. Marine Pollution Bulletin 156, 111239. https://doi.org/10.1016/j.marpolbul.2020.111239. [DOI](https://doi.org/10.1016/j.marpolbul.2020.111239); [NCEI archive](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:294676). Citation text comes from the CSV; the publisher DOI landing page confirms the title, not the methods.

CSV audit: 81 rows; Niskin bottle; reported field unit `pieces/m3`; raw mean 0.00176296296296, median 0.0013; 18 zeros; 0 missing concentrations. Mesh field 0.00045 mm; depth 0.6 m. Row IDs are enumerated in `provenance_row_audit.csv`.

**Verdict: unresolved / hypothesis_only.** Primary full text, supporting data, method page/table/equation, sample volumes, volume computation, aliquot/recovery factors, extraction, blank handling, and synthetic-polymer identification are `not_accessible` or unresolved. No row-specific source-to-CSV mapping supports an authorized correction.


## What ×10^6 would require mathematically

One litre is 0.001 m³. If an independently documented concentration is q particles/L, it converts to **1000q particles/m³**. A CSV value would require multiplication by **10^6** only if it were independently shown that the archived value was **q/1000**, rather than q or 1000q. Units printed as pieces/m³ cannot establish that earlier inversion. Both an original source measurement and the precise erroneous ingest calculation are required per record. Score improvements cannot supply this evidence.

The old code claims Barrows mean 11.8 particles/L and Dehm 1.6–2.0 particles/L. These claims were **not independently verified here** and are not adopted as measured truth. Dehm's aggregate plausibility does not map 81 source measurements to CSV rows. The audit's raw means are calculated from the untouched CSV, not from a paper.

For Barrows, the CSV method is **grab sample**, not a net. Its `Mesh Size (mm)` value 0.00045 equals 0.45 µm dimensionally. It could describe a membrane pore size, but primary filtration documentation is unavailable. It must not be presented as verified actual net mesh or interchangeable sampling selectivity. Dehm uses the same mesh-field value but a Niskin-bottle label; the same ambiguity applies.

## Other data-quality flags (descriptive only)

`dataset_provenance_flags.json` records: {
  "raw_exact_duplicate_rows": 0,
  "eligible_measurement_signature_duplicate_rows": 643,
  "eligible_same_date_location_method_rows": 2378,
  "zero_concentration_rows": 4958,
  "missing_concentration_rows": 0,
  "missing_depth_rows": 53,
  "missing_mesh_rows": 0,
  "non_DOI_rows": 0,
  "sampling_volume_field_present": false,
  "polymer_confirmation_field_present": false,
  "detection_limit_field_present": false,
  "blank_correction_field_present": false,
  "interpretation": "Duplicate signatures are flags, not proof of duplicate measurements; no records removed. Zero may mean nondetect, true zero or rounded/blank-adjusted zero; unresolved without source records."
}

Repeated observation signatures may be legitimate replicates or linked/duplicated source observations; no automatic deletion. Original DOI/accession linking remains unchanged. Non-DOI publication identifiers, absence of campaign IDs, and missing volume/polymer/blank/detection-limit fields limit traceability. The 33 bongo-net records noted in the prior project context remain unverified; no additional source lookup or multiplier was attempted. No unresolved issue is resolved by model fit.

The two local PDFs (`docs/project-synopsis.pdf`, `docs/review-paper.pdf`) are project/secondary documents, not copies of the two named primary articles or row-level laboratory records; they are not used to verify physical-unit claims. No new PDF parser/library was downloaded.

## Gate verdict

**UNVERIFIED_TARGET_PROVENANCE.** There is not enough evidence to certify a comparable external synthetic-particle concentration evaluation against the historical training labels. This blocks a scientifically validated result, not preservation of the exploratory benchmark. An external dataset can be schema-screened while the development-label provenance blocker remains. Future authorization needs inspectable, pre-outcome signed evidence reviews; no Boolean flag or dataset score alone constitutes scientific verification.
