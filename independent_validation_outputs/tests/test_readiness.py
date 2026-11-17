"""Offline synthetic-fixture tests only; no empirical external score or model fit."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from validation_core import *
from protocol import verify_lock,REQUIRED
from external import inspect_metadata,features,linked_external_groups,development_review,prepare_external
from metrics import scores,cluster_bootstrap
from model import training_gate,fit_cap,fit_frozen,FrozenResearchModel
from evaluation import evaluate_external


def fixture():
    meta=pd.DataFrame([{'source_row_id':'external-new-1','study_id':'novel','doi':'10.9999/new','accession':'9999999','campaign_id':'new-independent-cruise',
                        'date':'2030-01-01','lat':20.,'lon':30.,'method':'manta net','mesh_mm':.33,'mesh_kind':'net_mesh','depth_m':np.nan,
                        'medium':'Ocean water','unit':'pieces/m3','target_type':'synthetic_particle_count','scenario':EXTERNAL_SCENARIO}])
    dev=pd.DataFrame([{'source_observation_id':'dev-1','doi_normalized':'10.1000/development','accession_normalized':'123','date':'2000-01-01','lat':0.,'lon':0.,'method':'manta net',
                       'observation_fingerprint':observation_fingerprint('2000-01-01',0,0,'manta net')}])
    support={'known_methods':['manta net','grab sample'],'numeric_min':dict(zip(FEATURES[:-1],[-90,-180,1972,-1,-1,-4,0])),
             'numeric_max':dict(zip(FEATURES[:-1],[90,180,2026,1,1,0,5]))}
    m={'study_id':'novel','provenance_verified':True,'independence_reviewed':True,'method_context_signed':True,'previously_used_for_tuning':False,'original_unit':'pieces/m3',
       'reviewer':'TEST FIXTURE','reviewed_utc':'2030-01-01T00:00:00+00:00','review_signature':'TEST_ONLY','citation':'synthetic test only','source_url':'https://example.invalid/fixture',
       'volume_calculation_reference':'test','synthetic_verification_reference':'test','zero_and_detection_limit_reference':'test','method_context_reference':'test',
       'campaign_aliases':[],'ancestor_source_ids':[],'ancestor_dois':[],'ancestor_accessions':[]}
    return meta,dev,support,{'manta net':'surface net'},[m]


class ReadinessTests(unittest.TestCase):
    out=None

    def test_historical_identity_and_no_reference_change(self):
        i=json.loads((self.out/'baseline_identity.json').read_text())
        self.assertEqual(i['eligible_rows'],16988);self.assertEqual(i['study_count'],72)
        self.assertEqual(i['historical_references']['original_C0']['R2'],.4726308398591973)
        self.assertEqual(i['historical_references']['S2']['R2'],.5256280075581679)
        self.assertEqual(i['features'],FEATURES);self.assertEqual(i['scenario'],SCENARIO)

    def test_protocol_lock_and_tamper_detection(self):
        lock=verify_lock(self.out);self.assertEqual(lock['candidate'],'S2');self.assertTrue(lock['no_external_outcomes_inspected'])
        with patch('protocol.sha',return_value='tampered'):
            with self.assertRaises(ValueError):verify_lock(self.out)
        self.assertEqual(lock['content_sha256'],digest(lock['files']))

    def test_DOI_and_accession_normalization(self):
        for v in ['https://doi.org/10.1000/Development','HTTP://DX.DOI.ORG/10.1000/development','doi:10.1000/DEVELOPMENT','https://doi.org/10.1000%2Fdevelopment']:
            self.assertEqual(normalize_doi(v),'10.1000/development')
        for v in ['000123','123.0','gov.noaa.nodc:000123']:self.assertEqual(normalize_accession(v),'123')

    def test_study_overlap_even_identifier_format_changes(self):
        for field,value in [('doi','DOI:10.1000/DEVELOPMENT'),('accession','0000123'),('source_row_id','dev-1')]:
            m,d,s,f,manifest=fixture();m.loc[0,field]=value
            audit=inspect_metadata(m,d,s,f,manifest)
            self.assertFalse(audit.eligible_metadata.iloc[0]);self.assertIn('overlap',audit.reasons.iloc[0])

    def test_derived_records_and_campaign_alias_overlap(self):
        for field,value in [('ancestor_source_ids','dev-1'),('ancestor_dois','doi:10.1000/DEVELOPMENT'),('ancestor_accessions','000123'),('campaign_aliases','123')]:
            m,d,s,f,manifest=fixture();manifest[0][field]=[value]
            self.assertFalse(inspect_metadata(m,d,s,f,manifest).eligible_metadata.iloc[0])

    def test_exact_and_near_duplicate_measurements(self):
        m,d,s,f,manifest=fixture();m.loc[0,['date','lat','lon']]=['2000-01-01',0.,0.]
        audit=inspect_metadata(m,d,s,f,manifest);self.assertIn('exact_measurement_metadata_overlap',audit.reasons.iloc[0])
        m.loc[0,'lat']=.001
        self.assertIn('near_duplicate',inspect_metadata(m,d,s,f,manifest).reasons.iloc[0])

    def test_external_groups_link_shared_provenance(self):
        m,_,_,_,_=fixture();m=pd.concat([m]*3,ignore_index=True)
        m['study_id']=['a','b','c'];m['campaign_id']=['x','y','z'];m['accession']=['','',None];m['doi']=['10.1/a','10.1/a','10.1/c']
        groups=linked_external_groups(m);self.assertEqual(groups[0],groups[1]);self.assertNotEqual(groups[0],groups[2])

    def test_target_perturbation_leaves_eligibility_and_features_identical(self):
        m,d,s,f,manifest=fixture();a=m.assign(concentration=0.);b=m.assign(concentration=1e12)
        pd.testing.assert_frame_equal(inspect_metadata(a,d,s,f,manifest),inspect_metadata(b,d,s,f,manifest))
        pd.testing.assert_frame_equal(features(a,f),features(b,f))
        # Pure S2 selection accepts inner training summaries only, not external labels.
        root=self.out.parent;sys.path.append(str(root/'study_robustness_outputs/source'))
        rules=load_module('stage5_test_S2',root/'study_robustness_outputs/source/selection_rules.py')
        inner=csv(root/'optimization_outputs/inner_metrics_by_outer_fold.csv');fold=inner[inner.outer==1][rules.FIELDS]
        self.assertEqual(rules.select_fold(fold,'S2')[0],rules.select_fold(fold.copy(),'S2')[0])
        with self.assertRaises(ValueError):rules.select_fold(fold.assign(external_target=10),'S2')

    def test_missing_metadata_unit_and_scenario_mismatch(self):
        m,d,s,f,manifest=fixture()
        with self.assertRaises(ValueError):inspect_metadata(m.drop(columns='method'),d,s,f,manifest)
        for col,value in [('unit','pieces/L'),('scenario',SCENARIO),('medium','Beach'),('target_type','mass'),('method','unknown sampler'),('mesh_mm',0),('lat',91)]:
            bad=m.copy();bad.loc[0,col]=value
            self.assertFalse(inspect_metadata(bad,d,s,f,manifest).eligible_metadata.iloc[0])

    def test_whole_study_quarantine_no_outcome_based_deletion(self):
        m,d,s,f,manifest=fixture();m=pd.concat([m,m],ignore_index=True);m.loc[1,'source_row_id']='external-new-2';m.loc[1,'date']='2030-02-01';m.loc[1,'unit']='pieces/kg'
        audit=inspect_metadata(m,d,s,f,manifest);self.assertEqual(len(audit),2);self.assertFalse(audit.eligible_metadata.any())
        self.assertIn('study_quarantined',audit.reasons.iloc[0])

    def test_training_only_support_flags_and_missing_depth(self):
        m,d,s,f,manifest=fixture();audit=inspect_metadata(m,d,s,f,manifest)
        self.assertTrue(audit.eligible_metadata.iloc[0]);self.assertIn('OOD_year',audit['flags'].iloc[0]);self.assertIn('missing_depth',audit['flags'].iloc[0])
        self.assertTrue(pd.isna(features(m,f).log_depth.iloc[0]))

    def test_frozen_preprocessing_unknown_category_and_clipping(self):
        ns=development(self.out.parent);df=ns['df'];train=df.iloc[:100][FEATURES].copy()
        pre=ns['preprocessor']().fit(train);test=train.iloc[:2].copy();test['method_family']='__unknown__'
        test['lat']=[1000,-1000];test['log_depth']=np.nan
        clipped=ns['clip_to_range'](train)(test)
        self.assertEqual(clipped.lat.iloc[0],train.lat.max());self.assertEqual(clipped.lat.iloc[1],train.lat.min())
        transformed=pre.transform(clipped);self.assertTrue(np.isfinite(transformed).all());self.assertTrue((transformed[:,7:]==0).all())

    def test_finite_constant_target_and_offset_metrics(self):
        m=scores([1,1],[1,2],['a','b']);self.assertIsNone(m['R2']);self.assertEqual(m['MAE'],.5)
        m=scores([-3,0,1],[-2,1,2.000001],['a','b','c']);self.assertEqual(m['within_1_log'],2/3)
        for y,p in [([1,np.nan],[1,2]),([1,2],[1,np.inf])]:
            with self.assertRaises(ValueError):scores(y,p,['a','b'])

    def test_study_cluster_bootstrap_not_rows_and_failure_rules(self):
        self.assertEqual(cluster_bootstrap([0,1,2],[1,1,2],['a','a','a'])['status'],'insufficient_independent_studies')
        y=np.arange(10.);p=y+.1;g=np.repeat(np.arange(5),2)
        a=cluster_bootstrap(y,p,g,replicates=30);b=cluster_bootstrap(y,p,g,replicates=30)
        self.assertEqual(a,b);self.assertEqual(a['resampling_unit'],'whole_study');self.assertEqual(a['status'],'unstable_few_studies')
        c=cluster_bootstrap(np.ones(10),p,g,replicates=20);self.assertIsNone(c['intervals']['R2']['95_percent_interval']);self.assertEqual(c['intervals']['R2']['failed_replicates'],20)

    def test_default_provenance_and_training_gate_closed(self):
        ok,reasons=development_review(self.out);self.assertFalse(ok);self.assertTrue(reasons)
        ok,reasons,_=training_gate(self.out);self.assertFalse(ok)
        with patch('model.package_metadata.version',return_value='changed-runtime'):
            ok,reasons,_=training_gate(self.out)
            self.assertFalse(ok);self.assertTrue(any('Runtime package changed' in r for r in reasons))
        with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('Unwanted fit')) as fit:
            result=fit_frozen(self.out);self.assertEqual(result['status'],'NOT_TRAINED');self.assertEqual(fit.call_count,0)
        self.assertFalse((self.out/'full_data_research_model').exists())

    def test_fit_cap_and_no_implicit_model_training(self):
        fit_cap(24)
        with self.assertRaises(ValueError):fit_cap(25)
        from model import FrozenResearchModel
        class Echo:
            def predict(self,X):return X.lat.to_numpy()
        low={c:0 for c in FEATURES[:-1]};high={c:1 for c in FEATURES[:-1]}
        model=FrozenResearchModel(Echo(),low,high,{'label':MODEL_LABEL})
        X=pd.DataFrame([{c:2 for c in FEATURES[:-1]}|{'method_family':'surface net'}])[FEATURES]
        np.testing.assert_array_equal(model.predict(X),[1])

    def test_no_external_data_normal_status_and_no_fake_predictions(self):
        result=prepare_external(self.out,None);self.assertEqual(result['status'],'NO_EXTERNAL_DATA');self.assertFalse(result['targets_read'])
        result=evaluate_external(self.out,None);self.assertEqual(result['status'],'NO_EXTERNAL_DATA')
        self.assertFalse((self.out/'external_predictions.csv').exists());self.assertFalse((self.out/'EXTERNAL_TEST_REPORT.md').exists())
        self.assertFalse((self.out/'external_eligibility_audit.csv').exists())

    def test_repeat_outcome_access_blocked_before_read(self):
        # Test the fail-closed check with an existing ledger, without creating any external observation file.
        import evaluation
        original=Path.exists
        def exists(path):
            if str(path).endswith('label_access_ledger.jsonl'):return True
            return original(path)
        with patch.object(Path,'exists',exists):
            with self.assertRaises(ValueError):evaluation.evaluate_external(self.out,self.out.parent/'data/noaa_microplastics.csv')

    def test_public_source_budget_and_honest_unresolved_verdict(self):
        p=json.loads((self.out/'provenance_status.json').read_text());self.assertEqual(p['Barrows'],'unresolved');self.assertEqual(p['Dehm'],'unresolved')
        e=json.loads((self.out/'evidence/source_lookup_log.json').read_text());self.assertLessEqual(e['targeted_lookups'],6)
        self.assertEqual(p['row_specific_corrections_authorized'],0)
        a=csv(self.out/'provenance_row_audit.csv');self.assertEqual(a.rows.tolist(),[1389,81]);self.assertTrue(a.correction_status.eq('hypothesis_only').all())

    def test_all_protected_files_unchanged(self):
        self.assertTrue(protected_check(self.out)['unchanged'])


def run_tests(out):
    ReadinessTests.out=out;stream=io.StringIO()
    with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('No LightGBM fit authorized in offline tests')) as fit:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReadinessTests))
    text=stream.getvalue();(out/'tests.txt').write_text(text)
    if not result.wasSuccessful():raise AssertionError(text)
    status={'tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'actual_model_fits_in_tests':fit.call_count,'synthetic_fixtures':'Tiny in-memory unit fixtures only; no empirical external metrics or predictions written.'}
    dump(out/'tests/test_results.json',status);print(text);return status
