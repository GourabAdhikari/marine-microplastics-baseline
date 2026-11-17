"""Stage A and B: read-only benchmark identity and evidence-limited provenance."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from validation_core import *
from metrics import scores


def audit_benchmark(out):
    root=out.parent
    paths=['data/noaa_microplastics.csv','microplastics.py','microplastics.ipynb','gamm.R','uv.lock',
           'reproduction_outputs/out_of_fold_predictions.csv','reproduction_outputs/summary.json','reproduction_outputs/model_parameters.json','reproduction_outputs/unit_adjustments.json',
           'optimization_outputs/optimized_oof_predictions.csv','optimization_outputs/inner_metrics_by_outer_fold.csv','optimization_outputs/candidate_registry.json','optimization_outputs/frozen_splits.json',
           'study_robustness_outputs/all_procedure_oof_predictions.csv','study_robustness_outputs/summary_metrics.csv','study_robustness_outputs/locked_selections.json',
           'study_robustness_outputs/source/selection_rules.py','study_robustness_outputs/verified_prior_checkpoint_hashes.json']
    for p in paths:require((root/p).is_file(),'Missing essential protected artifact: '+p)
    for manifest,key,base in [(root/'reproduction_outputs/input_hashes.json',None,root),(root/'optimization_outputs/artifact_checksums.json',None,root),
                              (root/'study_robustness_outputs/EXECUTION_STATUS.json','artifact_sha256',root/'study_robustness_outputs')]:
        m=json.loads(manifest.read_text());m=m[key] if key else m
        require(all(sha(base/p)==h for p,h in m.items()),'Prior artifact checksum disagreement: '+str(manifest))
    cached=json.loads((root/'study_robustness_outputs/verified_prior_checkpoint_hashes.json').read_text())
    require(all(sha(root/p)==h for p,h in cached.items()),'Prior checkpoint hash disagreement')
    base=csv(root/'reproduction_outputs/out_of_fold_predictions.csv');oof=csv(root/'study_robustness_outputs/all_procedure_oof_predictions.csv')
    require(len(base)==len(oof)==16988 and oof.study.nunique()==72,'Benchmark counts changed')
    require(base.water_row.is_unique and base.source_row.is_unique and oof.water_row.is_unique and oof.source_row.is_unique,'Duplicate identities')
    for col in ['water_row','source_row','study','fold','target_log10']:require(np.array_equal(base[col],oof[col]),'Changed benchmark '+col)
    require(oof.scenario.eq(SCENARIO).all(),'Changed target scenario')
    ns=development(root);df=ns['df'];raw=ns['raw']
    require(np.array_equal(df.y,oof.target_log10) and np.array_equal(df.study,oof.study),'Original transformation/grouping mismatch')
    require(ns['FEATURES']==FEATURES and ns['C']==.001,'Changed predictors or offset')
    mask=(ns['data'].medium=='Ocean water')&(ns['data'].unit=='pieces/m3')
    require(np.array_equal(ns['data'].index[mask],base.source_row),'Changed source-row filter/order')
    splits=json.loads((root/'optimization_outputs/frozen_splits.json').read_text())
    for f,(tr,te) in enumerate(splits['outer'],1):
        require(np.array_equal(te,np.flatnonzero(oof.fold==f)) and np.array_equal(tr,np.flatnonzero(oof.fold!=f)),'Fold mismatch')
        require(not set(df.study.iloc[tr])&set(df.study.iloc[te]),'Study leakage')
    saved=csv(root/'study_robustness_outputs/summary_metrics.csv').set_index('procedure');references={}
    for name in ['original_C0','previously_optimized','S1','S2']:
        m=scores(oof.target_log10,oof[name],oof.study)
        for key,val in m.items():
            if key=='R2_status':continue
            require(np.isclose(val,saved.loc[name,key],rtol=1e-12,atol=1e-12),'Reference metric mismatch '+name+'/'+key)
        references[name]={key:float(saved.loc[name,key]) for key in ['R2','MAE','RMSE','study_macro_MAE','study_macro_MSE','p90_study_MAE','worst_study_MAE','within_1_log','mean_fold_R2']}
    require(references['original_C0']['R2']==.4726308398591973 and references['S2']['R2']==.5256280075581679,'Immutable reference values disagree')
    lock=json.loads((root/'study_robustness_outputs/locked_selections.json').read_text())
    require(sha(root/'study_robustness_outputs/locked_selections.json')==json.loads((root/'study_robustness_outputs/lock_sha256.json').read_text())['sha256'],'Stage 4 lock altered')
    # Load the exact pure Stage 4 rules, not its runner or training code.
    sys.path.append(str(root/'study_robustness_outputs/source'))
    selection=load_module('stage5_frozen_selectors',root/'study_robustness_outputs/source/selection_rules.py')
    inner=csv(root/'optimization_outputs/inner_metrics_by_outer_fold.csv')
    chosen,_=selection.select_all(inner[selection.FIELDS])
    require(chosen==lock['payload']['selections'],'Frozen selector algorithms no longer match locked decisions')
    # Audit original preprocessing behavior without LightGBM fits.
    for tr,te in splits['outer']:
        train=df.iloc[tr];nums=ns['NUMERIC'];pre=ns['preprocessor']().fit(train[FEATURES]);num=pre.named_transformers_['num']
        require(np.allclose(num.named_steps['simpleimputer'].statistics_,train[nums].median()),'Training median mismatch')
        filled=train[nums].fillna(train[nums].median()).to_numpy()
        require(np.allclose(num.named_steps['standardscaler'].mean_,filled.mean(axis=0)),'Training scaler mismatch')
        probe=df.iloc[te[:2]][FEATURES].copy()
        for col in nums:probe[col]=[train[col].min()-1000,train[col].max()+1000]
        probe.method_family='__unseen__';clipped=ns['clip_to_range'](train)(probe)
        require(np.allclose(clipped[nums].iloc[0],train[nums].min()) and np.allclose(clipped[nums].iloc[1],train[nums].max()),'Clipping mismatch')
        require((pre.transform(clipped)[:,len(nums):]==0).all(),'Unknown-category policy mismatch')
    identity={'scenario':SCENARIO,'raw_rows':len(raw),'raw_columns':len(raw.columns),'eligible_rows':len(df),'study_count':df.study.nunique(),
              'target':'log10(concentration + 0.001)','offset':.001,'features':FEATURES,'method_family_map':ns['METHOD_FAMILY'],
              'method_names':sorted(df.method.unique()),'grouping':'Connected components through shared DOI OR accession, including transitive links; exact historical labels preserved.',
              'preprocessing':'Training-only median imputer, StandardScaler, dense OneHotEncoder(handle_unknown=ignore); inference numeric clipping to training ranges.',
              'feature_units':{'lat':'degrees north','lon':'degrees east','year':'calendar year','month':'sin/cos(2*pi*month/12)','mesh':'log10(mesh_mm)','depth':'log1p(depth_m)'},
              'numeric_missing_counts':df[ns['NUMERIC']].isna().sum().to_dict(),'unit_hypotheses':ns['UNIT_FIXES'],
              'fold_test_sizes':[len(te) for tr,te in splits['outer']],'historical_references':references,
              'input_sha256':{p:sha(root/p) for p in paths},'prior_checkpoints_verified':len(cached),
              'selection_caveat':'S2 was identified after repeated inspection of the same five outer folds. Re-scoring them is not new independent evidence.',
              'new_model_fits':0}
    dump(out/'baseline_identity.json',identity)
    # Target-free identity index: every original source row, not only altered-label observations.
    index=pd.DataFrame({'source_row':base.source_row,'source_observation_id':raw.loc[mask,'Unique ID'].astype(str).to_numpy(),
                        'study':df.study,'doi_normalized':df.doi.map(normalize_doi),'accession_normalized':df.accession.map(normalize_accession),
                        'date':df.date.dt.strftime('%Y-%m-%d'),'lat':df.lat,'lon':df.lon,'method':df.method})
    index['observation_fingerprint']=[observation_fingerprint(r.date,r.lat,r.lon,r.method) for r in index.itertuples()]
    index.to_csv(out/'development_identity_index.csv',index=False)
    # Training-only support thresholds, never external-target driven.
    dump(out/'training_support.json',{'numeric_min':df[ns['NUMERIC']].min().to_dict(),'numeric_max':df[ns['NUMERIC']].max().to_dict(),
                                     'known_methods':sorted(df.method.unique()),'known_families':sorted(df.method_family.unique()),'feature_names':FEATURES})
    return identity


def provenance_audit(out):
    root=out.parent;raw=pd.read_csv(root/'data/noaa_microplastics.csv',low_memory=False);identity=json.loads((out/'baseline_identity.json').read_text())
    mask=(raw['Marine Setting']=='Ocean water')&(raw.Unit=='pieces/m3');water=raw.loc[mask]
    evidence=json.loads((out/'evidence/source_lookup_log.json').read_text());require(evidence['targeted_lookups']<=6,'Source lookup budget exceeded')
    groups=[]
    for label,doi in [('Barrows','10.1016/j.envpol.2018.02.062'),('Dehm','10.1016/j.marpolbul.2020.111239')]:
        x=water[water.DOI.map(normalize_doi)==doi];v=x['Microplastics Measurement']
        groups.append({'group':label,'doi':doi,'rows':len(x),'source_rows_zero_based':json.dumps(x.index.tolist()),
                       'citation_from_csv':x['Long Reference'].iloc[0],'csv_unit':'pieces/m3','csv_field':'Microplastics Measurement',
                       'raw_mean':float(v.mean()),'raw_median':float(v.median()),'raw_min':float(v.min()),'raw_max':float(v.max()),'zero_rows':int((v==0).sum()),'missing_rows':int(v.isna().sum()),
                       'method_from_csv':';'.join(sorted(x['Sampling Method'].unique())),'mesh_field_mm':';'.join(map(str,sorted(x['Mesh Size (mm)'].unique()))),
                       'depth_m':';'.join(map(str,sorted(x['Water Sample Depth (m)'].unique()))),
                       'accession':str(x['NCEI Accession Number'].iloc[0]),'archive_url':x['NCEI Accession Link'].iloc[0],
                       'historical_multiplier':1e6,'correction_status':'hypothesis_only','physical_unit_verdict':'unresolved',
                       'primary_full_text':'not_accessible','published_units_verified':'not_accessible','page_table_equation':'not_accessible',
                       'volume_litres':'unresolved','volume_calculation':'unresolved','subsampling_recovery':'unresolved','extraction_protocol':'unresolved','synthetic_only_verification':'unresolved',
                       'row_specific_source_measurement_mapping':'unavailable','authorized_row_correction':False})
    pd.DataFrame(groups).to_csv(out/'provenance_row_audit.csv',index=False)
    key=['Sample Date','Latitude (degree)','Longitude (degree)','Sampling Method','Mesh Size (mm)','Water Sample Depth (m)','Microplastics Measurement','Unit']
    duplicate=water.duplicated(key,keep=False)
    spatial=water.duplicated(['Sample Date','Latitude (degree)','Longitude (degree)','Sampling Method'],keep=False)
    notes={'raw_exact_duplicate_rows':int(raw.duplicated(keep=False).sum()),'eligible_measurement_signature_duplicate_rows':int(duplicate.sum()),
           'eligible_same_date_location_method_rows':int(spatial.sum()),'zero_concentration_rows':int((water['Microplastics Measurement']==0).sum()),
           'missing_concentration_rows':int(water['Microplastics Measurement'].isna().sum()),'missing_depth_rows':int(water['Water Sample Depth (m)'].isna().sum()),
           'missing_mesh_rows':int(water['Mesh Size (mm)'].isna().sum()),'non_DOI_rows':int((~water.DOI.map(normalize_doi).str.match(r'^10\.\d{4,9}/\S+$')).sum()),
           'sampling_volume_field_present':False,'polymer_confirmation_field_present':False,'detection_limit_field_present':False,
           'blank_correction_field_present':False,'interpretation':'Duplicate signatures are flags, not proof of duplicate measurements; no records removed. Zero may mean nondetect, true zero or rounded/blank-adjusted zero; unresolved without source records.'}
    dump(out/'dataset_provenance_flags.json',notes)
    water.loc[duplicate,key+['DOI','NCEI Accession Number','Unique ID']].assign(source_row=water.index[duplicate]).to_csv(out/'duplicate_signature_audit.csv',index=False)
    access='\n'.join(f"- [{r['id']}]({r['url']}), accessed {r['access_utc']}: HTTP {r.get('http_status')}; {r.get('error','publisher linking/redirect metadata only')}; saved `{r['response_file']}`." for r in evidence['records'])
    citations='\n'.join(f"### {g['group']}\n\n{g['citation_from_csv']}. [DOI](https://doi.org/{g['doi']}); [NCEI archive]({g['archive_url']}). Citation text comes from the CSV; the publisher DOI landing page confirms the title, not the methods.\n\nCSV audit: {g['rows']} rows; {g['method_from_csv']}; reported field unit `{g['csv_unit']}`; raw mean {g['raw_mean']:.12g}, median {g['raw_median']:.12g}; {g['zero_rows']} zeros; {g['missing_rows']} missing concentrations. Mesh field {g['mesh_field_mm']} mm; depth {g['depth_m']} m. Row IDs are enumerated in `provenance_row_audit.csv`.\n\n**Verdict: unresolved / hypothesis_only.** Primary full text, supporting data, method page/table/equation, sample volumes, volume computation, aliquot/recovery factors, extraction, blank handling, and synthetic-polymer identification are `not_accessible` or unresolved. No row-specific source-to-CSV mapping supports an authorized correction.\n" for g in groups)
    text=f'''# Provenance audit — physical labels are not verified

The historical scenario is computationally reproducible. That does **not** verify physical concentration labels, sampling comparability or synthetic-only identification. No CSV values were edited and no new unit multiplier is authorized. All historical multipliers remain `hypothesis_only`.

## Evidence levels

1. **Verified from inspectable primary evidence:** the publisher DOI linking pages identify the named article titles and publisher locations only. The original NOAA export is direct evidence of what the archived fields contain (not proof of their physical correctness). Exact file/response hashes and source-row mappings are saved. Mathematical SI dimensional conversion is exact, but its applicability to these records is not verified.
2. **Supported but ambiguous:** the existing notebook describes litre-scale concentrations and assumes an inverted conversion. These are prior project assertions, not independent primary-source verification. Agreement of an aggregate mean with a hypothesized scale is plausibility evidence at most.
3. **Unresolved:** both studies' row-specific original counts/volumes, original published concentration units, blank/recovery/subsampling treatment, actual filter protocol and confirmed synthetic fraction. Primary article content was not obtained; no page numbers or quoted methods are invented.

## Bounded primary-source attempts

Six targeted lookups only, no search-engine snippets, new observation downloads or broad literature search. The two DOI URLs returned HTTP 200 **redirect HTML**, not articles. The two article URLs returned HTTP 403; the two NCEI landing URLs timed out. Network was available for DOI resolution, but scientific content was not accessible in this run. Saved response bodies (403 bodies bounded at 200 kB) are not treated as article evidence.

{access}

{citations}

## What ×10^6 would require mathematically

One litre is 0.001 m³. If an independently documented concentration is q particles/L, it converts to **1000q particles/m³**. A CSV value would require multiplication by **10^6** only if it were independently shown that the archived value was **q/1000**, rather than q or 1000q. Units printed as pieces/m³ cannot establish that earlier inversion. Both an original source measurement and the precise erroneous ingest calculation are required per record. Score improvements cannot supply this evidence.

The old code claims Barrows mean 11.8 particles/L and Dehm 1.6–2.0 particles/L. These claims were **not independently verified here** and are not adopted as measured truth. Dehm's aggregate plausibility does not map 81 source measurements to CSV rows. The audit's raw means are calculated from the untouched CSV, not from a paper.

For Barrows, the CSV method is **grab sample**, not a net. Its `Mesh Size (mm)` value 0.00045 equals 0.45 µm dimensionally. It could describe a membrane pore size, but primary filtration documentation is unavailable. It must not be presented as verified actual net mesh or interchangeable sampling selectivity. Dehm uses the same mesh-field value but a Niskin-bottle label; the same ambiguity applies.

## Other data-quality flags (descriptive only)

`dataset_provenance_flags.json` records: {json.dumps(notes,indent=2)}

Repeated observation signatures may be legitimate replicates or linked/duplicated source observations; no automatic deletion. Original DOI/accession linking remains unchanged. Non-DOI publication identifiers, absence of campaign IDs, and missing volume/polymer/blank/detection-limit fields limit traceability. The 33 bongo-net records noted in the prior project context remain unverified; no additional source lookup or multiplier was attempted. No unresolved issue is resolved by model fit.

The two local PDFs (`docs/project-synopsis.pdf`, `docs/review-paper.pdf`) are project/secondary documents, not copies of the two named primary articles or row-level laboratory records; they are not used to verify physical-unit claims. No new PDF parser/library was downloaded.

## Gate verdict

**UNVERIFIED_TARGET_PROVENANCE.** There is not enough evidence to certify a comparable external synthetic-particle concentration evaluation against the historical training labels. This blocks a scientifically validated result, not preservation of the exploratory benchmark. An external dataset can be schema-screened while the development-label provenance blocker remains. Future authorization needs inspectable, pre-outcome signed evidence reviews; no Boolean flag or dataset score alone constitutes scientific verification.
'''
    (out/'PROVENANCE_AUDIT.md').write_text(text)
    (out/'source_request_checklist.md').write_text('''# Evidence requests — no request fulfilled by a model score

For each named DOI obtain the primary article and supplements legally; cite access date, exact page/table/equation and original unit wording. Obtain NCEI original submission/ingest documentation and source observation IDs matching every CSV row.

- Barrows: original counts, actual sample/filtered volume, membrane material and pore size vs net mesh, fibre categories, polymer-confirmation method and fraction, blank correction, recovery/subsampling and field/laboratory duplicates.
- Dehm: original 81-row counts and sampled/filtered volumes, bottle-to-filtrate volume calculation, source-to-NCEI row mapping, original concentration units, extraction and polymer verification, blank/recovery factors.
- Both: distinguish numeric zero from nondetect/rounded/corrected zero and missing result; document detection limits. Derive every proposed row factor from source measurements; reject unsupported blanket corrections.
- Other studies: identify non-DOI/unpublished sources, campaign/cruise aliases, duplicate signatures and unknown synthetic-only verification; resolve the bongo-net unit ambiguity independently.
- External campaign: untouched study/campaign identities, source measurements and volume, canonical units, synthetic confirmation, compatible filtration context, signed metadata-only overlap review, protocol lock before outcome access. Verify comparability to the frozen historical labels rather than assuming it.

Keep all original records and hypotheses unchanged. If evidence implies a different target, it requires a separate scientific protocol, not a silent change to this benchmark.
''')
    state={'Barrows':'unresolved','Dehm':'unresolved','historical_corrections':'hypothesis_only','comparable_concentration_gate':'UNVERIFIED_TARGET_PROVENANCE',
           'public_source_lookups':evidence['targeted_lookups'],'primary_methods_accessible':False,'row_specific_corrections_authorized':0}
    dump(out/'provenance_status.json',state);return state
