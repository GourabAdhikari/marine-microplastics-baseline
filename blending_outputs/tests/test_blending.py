"""Zero-LightGBM-fit unit/integration tests; no side effects on import."""
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from blend import (SCENARIO, WEIGHTS, METRICS, blend_predictions, validate_alignment, metrics,
                   gate, enforce_budget, identity, read_csv, sha)


class BlendingTests(unittest.TestCase):
    analysis=None

    def test_alignment_and_negative_controls(self):
        a=self.analysis
        validate_alignment(a.original,a.d)
        for col in ['source_row','fold','study','target_log10']:
            bad=a.d.copy()
            bad.loc[0,col]=bad.loc[1,col] if col=='source_row' else ('incorrect' if col=='study' else bad.loc[0,col]+1)
            with self.subTest(column=col),self.assertRaises(ValueError):validate_alignment(a.original,bad)
        bad=a.d.copy();bad.loc[0,'scenario']='raw_units'
        with self.assertRaises(ValueError):validate_alignment(a.original,bad)

    def test_no_silent_nan_or_missing_case_exclusion(self):
        for y,p,g in [([1,np.nan],[1,2],['a','b']),([1,2],[np.inf,2],['a','b']),([1,2],[1,2],['a',None]),([1,2],[1],['a','b'])]:
            with self.subTest(y=y,p=p),self.assertRaises(ValueError):metrics(y,p,g)
        with self.assertRaises(ValueError):blend_predictions([1,np.nan],[1,2])
        with self.assertRaises(ValueError):metrics([1,1],[1,2],['a','b'])

    def test_endpoint_equivalence_and_convex_arithmetic(self):
        d=self.analysis.d
        result=blend_predictions(d.original_prediction_log10,d.optimized_prediction_log10)
        np.testing.assert_array_equal(result['B0'],d.original_prediction_log10)
        np.testing.assert_array_equal(result['B100'],d.optimized_prediction_log10)
        for name,w in WEIGHTS.items():
            np.testing.assert_array_equal(result[name],(1-w)*d.original_prediction_log10.to_numpy()+w*d.optimized_prediction_log10.to_numpy())
            self.assertTrue((result[name]>=np.minimum(result['B0'],result['B100'])).all())
            self.assertTrue((result[name]<=np.maximum(result['B0'],result['B100'])).all())

    def test_targets_and_units_are_frozen(self):
        a=self.analysis
        df=a.ns['df'];raw=a.ns['data']
        base=raw[(raw.medium=='Ocean water')&(raw.unit=='pieces/m3')].reset_index(drop=True)
        np.testing.assert_array_equal(df.conc,base.conc*base.doi.map(a.ns['UNIT_FIXES']).fillna(1))
        np.testing.assert_array_equal(a.d.target_log10,np.log10(df.conc+.001))
        self.assertTrue(a.d.scenario.eq(SCENARIO).all())
        for c in ['doi','accession','study']:
            self.assertFalse(a.d[c].isna().any())
            self.assertFalse(a.d[c].astype(str).str.strip().str.lower().isin(['','nan','none','unknown','n/a']).any())

    def test_independent_aggregate_metrics_all_five(self):
        a=self.analysis;d=read_csv(a.out/'aligned_oof_predictions.csv');saved=read_csv(a.out/'fixed_blend_metrics.csv').set_index('blend')
        for name in WEIGHTS:
            y,p=d.target_log10.to_numpy(),d[name].to_numpy();err=p-y
            study_mae=[];study_mse=[]
            for _,g in d.groupby('study'):
                study_mae.append(mean_absolute_error(g.target_log10,g[name]));study_mse.append(mean_squared_error(g.target_log10,g[name]))
            independent={'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
                         'study_macro_MAE':np.mean(study_mae),'study_macro_MSE':np.mean(study_mse),'median_study_MAE':np.median(study_mae),
                         'p90_study_MAE':np.quantile(study_mae,.9),'worst_study_MAE':max(study_mae),'within_1_log':np.mean(abs(err)<=1),
                         'mean_fold_R2':np.mean([r2_score(g.target_log10,g[name]) for _,g in d.groupby('fold')])}
            for k,v in independent.items():
                with self.subTest(blend=name,metric=k):self.assertAlmostEqual(saved.loc[name,k],v,places=12)
        self.assertEqual(saved.loc['B0','R2'],.4726308398591973)
        self.assertEqual(saved.loc['B100','R2'],.5033812847921568)

    def test_all_per_fold_and_study_metrics(self):
        a=self.analysis;d=a.d
        fold=read_csv(a.out/'paired_fold_metrics.csv').set_index(['fold','blend'])
        study=read_csv(a.out/'paired_study_metrics.csv').set_index(['study','blend'])
        self.assertEqual(len(fold),25);self.assertEqual(len(study),360)
        for name in WEIGHTS:
            for f,g in d.groupby('fold'):
                self.assertAlmostEqual(fold.loc[(f,name),'R2'],r2_score(g.target_log10,g[name]),places=12)
                self.assertAlmostEqual(fold.loc[(f,name),'MAE'],mean_absolute_error(g.target_log10,g[name]),places=12)
                self.assertAlmostEqual(fold.loc[(f,name),'RMSE'],np.sqrt(mean_squared_error(g.target_log10,g[name])),places=12)
            for s,g in d.groupby('study'):
                self.assertEqual(study.loc[(s,name),'rows'],len(g))
                self.assertAlmostEqual(study.loc[(s,name),'MAE'],mean_absolute_error(g.target_log10,g[name]),places=12)
                self.assertAlmostEqual(study.loc[(s,name),'MSE'],mean_squared_error(g.target_log10,g[name]),places=12)
        self.assertEqual(set(study.index.get_level_values('study')),set(d.study))
        # Real groups need not contain singletons; the synthetic macro-metric test exercises one.

    def test_study_macro_is_not_row_weighted(self):
        m=metrics([0,0,0,2],[1,1,1,5],['large','large','large','small'])
        self.assertEqual(m['MAE'],1.5);self.assertEqual(m['study_macro_MAE'],2.)
        self.assertEqual(m['study_macro_MSE'],5.);self.assertEqual(m['worst_study_MAE'],3.)

    def test_factor_ten_offset_boundary_inclusive(self):
        # Zero target = log10(.001) = -3; prediction -2 corresponds to concentration .009.
        y=np.array([-3.,0.,1.,2.]);p=np.array([-2.,1.,-.0000001,2.])
        m=metrics(y,p,['a','b','c','d'])
        self.assertEqual(m['within_1_log'],.75)
        ratio=10**(p-y)
        np.testing.assert_array_equal(abs(p-y)<=1,(ratio>=.1)&(ratio<=10))
        self.assertAlmostEqual(10**p[0]-.001,.009)

    def test_subgroup_complete_counts_and_deltas(self):
        a=self.analysis;t=a.subgroups
        self.assertFalse(t.isna().any().any())
        for grouping in ['method_family','study_size_bin']:
            for name in WEIGHTS:
                part=t[(t.grouping==grouping)&(t.blend==name)]
                self.assertEqual(part.rows.sum(),16988)
                for _,r in part.iterrows():
                    g=a.d[a.d[grouping]==r['group']]
                    self.assertAlmostEqual(r.MAE,mean_absolute_error(g.target_log10,g[name]),places=12)
        for frame,keys in [(a.pooled,[]),(a.fold,['fold']),(a.study,['study']),(a.subgroups,['grouping','group'])]:
            groups=[(None,frame)] if not keys else frame.groupby(keys)
            for _,sub in groups:
                for ref in ['B0','B100']:
                    b=sub[sub.blend==ref].iloc[0]
                    np.testing.assert_allclose(sub['delta_vs_'+ref+'__MAE'],sub.MAE-b.MAE,rtol=1e-12,atol=1e-12)

    def test_posthoc_diagnostics_never_enter_weights(self):
        a=self.analysis
        h=read_csv(a.out/'prior_14_harmed_studies_diagnostic.csv')
        self.assertEqual(len(h),70);self.assertEqual(h.study.nunique(),14)
        self.assertEqual(set(h.study),set(a.prior_harmed))
        self.assertEqual(set(read_csv(a.out/'folds_2_and_5_diagnostic.csv').fold),{2,5})
        self.assertEqual(list(inspect.signature(blend_predictions).parameters),['original','optimized'])
        before=blend_predictions(a.d.original_prediction_log10,a.d.optimized_prediction_log10)
        changed=a.d.copy();changed['target_log10']=np.arange(len(changed))*100.;changed['study']='mutated';changed['fold']=99
        after=blend_predictions(changed.original_prediction_log10,changed.optimized_prediction_log10)
        for key in WEIGHTS:np.testing.assert_array_equal(before[key],after[key])

    def test_gate_predicates_and_failure_controls(self):
        b={'R2':.5,'MAE':1.,'RMSE':1.4,'study_macro_MAE':1.,'study_macro_MSE':2.,'median_study_MAE':.9,'p90_study_MAE':2.,'worst_study_MAE':5.,'within_1_log':.6}
        good=b|{'MAE':.9,'p90_study_MAE':1.9}
        self.assertTrue(gate(good,b)['passed'])
        for k,v in [('R2',.484),('study_macro_MAE',1.00001),('worst_study_MAE',5.10001)]:
            self.assertFalse(gate(good|{k:v},b)['passed'])
        self.assertFalse(gate(b,b)['passed']);self.assertFalse(gate(good,b,False)['passed'])
        actual=self.analysis.pooled.set_index('blend')
        self.assertEqual(self.analysis.gate,gate(actual.loc['B50'],actual.loc['B100']))

    def test_outer_and_existing_inner_study_disjointness(self):
        a=self.analysis;d=a.d
        for f,(tr,te) in enumerate(a.splits['outer'],1):
            self.assertFalse(set(d.study.iloc[tr])&set(d.study.iloc[te]))
            self.assertEqual(set(tr)|set(te),set(range(16988)))
            inner=a.splits['inner'][str(f)]
            np.testing.assert_array_equal(np.sort(np.concatenate([b for _,b in inner])),tr)
            for x,y in inner:
                self.assertEqual(set(x)|set(y),set(tr));self.assertFalse(set(d.study.iloc[x])&set(d.study.iloc[y]))
        if a.gate['passed']:
            plan=json.loads((a.out/'stage_c_planned_splits.json').read_text())
            self.assertEqual(len(plan),45)
            for p in plan:
                self.assertFalse(set(p['train_study_ids'])&set(p['validation_study_ids']))
                self.assertGreater(len(p['train_study_ids']),1)

    def test_original_preprocessing_training_only_without_model_fits(self):
        a=self.analysis;df=a.ns['df'];nums=a.ns['NUMERIC'];features=a.ns['FEATURES']
        self.assertEqual(features,['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family'])
        parts=list(a.splits['outer'])+[p for folds in a.splits['inner'].values() for p in folds]
        for tr,te in parts:
            train=df.iloc[tr][features];pre=a.ns['preprocessor']().fit(train)
            num=pre.named_transformers_['num'];imp=num.named_steps['simpleimputer'];scale=num.named_steps['standardscaler']
            np.testing.assert_allclose(imp.statistics_,train[nums].median())
            filled=train[nums].fillna(train[nums].median()).to_numpy()
            np.testing.assert_allclose(scale.mean_,filled.mean(axis=0));np.testing.assert_allclose(scale.var_,filled.var(axis=0))
            enc=pre.named_transformers_['cat'];np.testing.assert_array_equal(enc.categories_[0],np.sort(train.method_family.unique()))
            test=df.iloc[te[:3]][features].copy()
            for c in nums:test[c]=[train[c].min()-1000,train[c].max()+1000,np.nan]
            test['method_family']='__missing_from_training__'
            clipped=a.ns['clip_to_range'](train)(test)
            np.testing.assert_allclose(clipped[nums].iloc[0],train[nums].min());np.testing.assert_allclose(clipped[nums].iloc[1],train[nums].max())
            transformed=pre.transform(clipped)
            self.assertTrue(np.isfinite(transformed).all());self.assertTrue((transformed[:,len(nums):]==0).all())
            np.testing.assert_allclose(scale.mean_,filled.mean(axis=0))

    def test_budget_fail_closed_no_new_lightgbm_fits(self):
        self.assertTrue(enforce_budget(100))
        for count in [101,360,400]:
            with self.assertRaises(RuntimeError):enforce_budget(count)
        for count in [-1,1.5]:
            with self.assertRaises(ValueError):enforce_budget(count)
        a=self.analysis;self.assertFalse(a.stage_c['ran']);self.assertEqual(a.stage_c['new_fits_started'],0)
        events=[json.loads(l) for l in (a.out/'fit_log.jsonl').read_text().splitlines()]
        self.assertFalse(any(e['event'] in ['started','fit_started','completed'] for e in events))
        self.assertFalse((a.out/'nested_oof_predictions.csv').exists())
        if a.gate['passed']:
            self.assertEqual(a.stage_c['status'],'skipped_budget_exceeded');self.assertEqual(a.stage_c['upper_bound_new_fits'],400)
        else:self.assertEqual(a.stage_c['status'],'skipped_gate_failed')

    def test_import_is_side_effect_free_and_deterministic(self):
        a=self.analysis
        code="""from unittest.mock import patch
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'blending_outputs/source')
with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('Unexpected fit on import')) as fit:
 import blend
 assert fit.call_count==0
 assert blend.WEIGHTS['B50']==0.5
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=a.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout,'')
        first=blend_predictions(a.d.original_prediction_log10,a.d.optimized_prediction_log10)
        second=blend_predictions(a.d.original_prediction_log10,a.d.optimized_prediction_log10)
        for name in WEIGHTS:np.testing.assert_array_equal(first[name],second[name])

    def test_checkpoint_identity_and_input_hashes(self):
        a=self.analysis;p=json.loads((a.out/'checkpoints/analysis_identity.json').read_text());key=p.pop('identity_sha256')
        self.assertEqual(key,identity(p))
        changed=json.loads(json.dumps(p));changed['plan_sha256']='changed'
        self.assertNotEqual(key,identity(changed))
        for file,h in p['input_hashes'].items():self.assertEqual(sha(a.root/file),h)
        for file,h in p['source'].items():self.assertEqual(sha(a.out/file),h)
        for file,h in p['test_source'].items():self.assertEqual(sha(a.out/file),h)
        self.assertFalse(list((a.out/'checkpoints').glob('*.npz')))

    def test_persisted_prediction_roundtrip_and_strict_references(self):
        a=self.analysis;d=read_csv(a.out/'aligned_oof_predictions.csv')
        for col in ['water_row','source_row','study','fold','target_log10',*WEIGHTS]:np.testing.assert_array_equal(d[col],a.d[col])
        audit=read_csv(a.out/'reference_metric_audit.csv')
        self.assertTrue(np.allclose(audit.recomputed,audit.reference,rtol=1e-12,atol=1e-12))
        reference=json.loads((a.root/'reproduction_outputs/summary.json').read_text())
        self.assertEqual(len(audit),2*(len(METRICS)+1)+len(reference['pooled']))

    def test_protected_files(self):
        self.assertGreater(self.analysis.verify_protected(),27000)


def run_suite(analysis):
    BlendingTests.analysis=analysis
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BlendingTests)
    with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('This zero-fit phase must not start LightGBM')) as fit:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        assert fit.call_count==0
    text=stream.getvalue()
    (analysis.out/'tests.txt').write_text(text+'\nAdditional data/audit assertions:\n'+'\n'.join('PASS '+c for c in analysis.checks)+'\n')
    if not result.wasSuccessful():raise AssertionError(text)
    status={'unit_integration_tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped),'skipped':len(result.skipped),
            'failures':len(result.failures),'errors':len(result.errors),'data_audit_assertions_passed':len(analysis.checks),
            'lightgbm_fit_calls_in_tests':fit.call_count,
            'not_applicable':['Actual Stage C model/weight fitting and its outer-label invariance: Stage C did not run. Fixed-blend target invariance and existing outer/inner split checks were tested; new nested split checks apply only if the Stage B gate passes.']}
    (analysis.out/'tests/test_results.json').write_text(json.dumps(status,indent=2))
    print(text)
    return status
