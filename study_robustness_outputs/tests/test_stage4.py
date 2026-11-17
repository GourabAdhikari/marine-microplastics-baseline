"""Focused selector, provenance, leakage, fit-budget and saved-output tests."""
import inspect
import io
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from common import SCENARIO,ORIGINAL,OPTIMIZED,PROCEDURES,METRIC_NAMES,csv,sha,fingerprint,metrics
from selection_rules import FIELDS,select_fold,select_all
from fit_budget import ensure_available,cache_identity,read_checkpoint,array_hash
from evaluation import screening


def synthetic():
    rows=[]
    for i in range(8):
        rows.append({'outer':1,'candidate':'C0_original' if i==0 else f'C{i}_test','scenario':SCENARIO,'R2':.5 if i==0 else -1.,
                     'study_macro_MAE':1. if i==0 else 10.,'p90_study_MAE':2. if i==0 else 10.,'worst_study_MAE':4. if i==0 else 10.})
    return pd.DataFrame(rows)[FIELDS]


def setrow(t,i,r2,macro,p90,worst):
    t.loc[i,['R2','study_macro_MAE','p90_study_MAE','worst_study_MAE']]=[r2,macro,p90,worst]


class Stage4Tests(unittest.TestCase):
    ctx=None;budget=None

    def test_exact_oof_alignment_and_coverage(self):
        c=self.ctx;d=csv(c.out/'all_procedure_oof_predictions.csv')
        self.assertEqual(len(d),16988);self.assertEqual(d.study.nunique(),72)
        self.assertTrue(d.water_row.is_unique and d.source_row.is_unique)
        for col in ['water_row','source_row','study','fold','target_log10']:np.testing.assert_array_equal(d[col],c.base[col])
        self.assertEqual(d.groupby('fold').size().tolist(),[6135,2714,2712,2714,2713])
        self.assertTrue(d.scenario.eq(SCENARIO).all());self.assertTrue(np.isfinite(d[PROCEDURES]).all().all())
        np.testing.assert_array_equal(d[ORIGINAL],c.base.prediction_log10);np.testing.assert_array_equal(d[OPTIMIZED],c.opt.optimized_prediction_log10)

    def test_independent_s1_s2_arithmetic_and_order(self):
        c=self.ctx
        for rule in ['S1','S2']:
            for fold in range(1,6):
                t=c.saved_inner[c.saved_inner.outer==fold][FIELDS];b=t[t.candidate=='C0_original'].iloc[0]
                chosen,flags=select_fold(t,rule)
                allowed=[]
                for _,r in t.iterrows():
                    ok=(r.R2>=b.R2-.010 and r.study_macro_MAE<=(1.02 if rule=='S1' else 1.03)*b.study_macro_MAE
                        and r.p90_study_MAE<=(1. if rule=='S1' else 1.03)*b.p90_study_MAE
                        and r.worst_study_MAE<=(1.05 if rule=='S1' else 1.10)*b.worst_study_MAE)
                    self.assertEqual(ok,bool(flags.loc[flags.candidate==r.candidate,'eligible'].iloc[0]))
                    if ok:allowed.append(r)
                if rule=='S1':expected=min(allowed,key=lambda r:(r.p90_study_MAE,r.study_macro_MAE,-r.R2,r.candidate))
                else:
                    best=max(r.R2 for r in allowed);band=[r for r in allowed if r.R2>=best-.010]
                    expected=min(band,key=lambda r:(r.study_macro_MAE,r.p90_study_MAE,-r.R2,r.candidate))
                self.assertEqual(chosen['candidate'],expected.candidate)

    def test_s1_lexicographic_ties(self):
        t=synthetic();setrow(t,1,.60,.9,1.5,4.);setrow(t,2,.70,.9,1.5,4.)
        self.assertEqual(select_fold(t,'S1')[0]['candidate'],'C2_test')
        setrow(t,1,.70,.9,1.5,4.)
        self.assertEqual(select_fold(t,'S1')[0]['candidate'],'C1_test')
        setrow(t,2,.60,.8,1.5,4.)
        self.assertEqual(select_fold(t,'S1')[0]['candidate'],'C2_test')
        setrow(t,1,.50,1.,1.4,4.)
        self.assertEqual(select_fold(t,'S1')[0]['candidate'],'C1_test')

    def test_s2_band_and_ties(self):
        t=synthetic();setrow(t,1,.70,.9,1.9,4.);setrow(t,2,.695,.8,2.,4.);setrow(t,3,.68,.7,1.5,3.)
        self.assertEqual(select_fold(t,'S2')[0]['candidate'],'C2_test')
        setrow(t,1,.695,.8,2.,4.)
        self.assertEqual(select_fold(t,'S2')[0]['candidate'],'C1_test')
        setrow(t,2,.695,.8,1.9,4.)
        self.assertEqual(select_fold(t,'S2')[0]['candidate'],'C2_test')

    def test_C0_fallback_zero_errors_and_nonfinite(self):
        t=synthetic()
        for rule in ['S1','S2']:
            self.assertEqual(select_fold(t,rule)[0]['candidate'],'C0_original')
            zero=t.copy();zero.loc[0,['study_macro_MAE','p90_study_MAE','worst_study_MAE']]=0.
            self.assertEqual(select_fold(zero,rule)[0]['candidate'],'C0_original')
            for col in FIELDS[3:]:
                bad=t.copy();bad.loc[1,col]=np.nan
                with self.assertRaises(ValueError):select_fold(bad,rule)
        missing=t.drop(columns='worst_study_MAE')
        with self.assertRaises(ValueError):select_fold(missing,'S1')

    def test_inclusive_guard_boundaries(self):
        t=synthetic();setrow(t,1,.5-.010,1.02,2.,1.05*4.)
        self.assertTrue(select_fold(t,'S1')[1].loc[1,'eligible'])
        t.loc[1,'worst_study_MAE']+=1e-9
        self.assertFalse(select_fold(t,'S1')[1].loc[1,'eligible'])
        setrow(t,1,.5-.010,1.03,1.03*2.,1.10*4.)
        self.assertTrue(select_fold(t,'S2')[1].loc[1,'eligible'])
        t.loc[1,'R2']-=1e-9
        self.assertFalse(select_fold(t,'S2')[1].loc[1,'eligible'])

    def test_inner_only_inputs_no_io_and_deterministic(self):
        c=self.ctx;t=c.saved_inner[FIELDS].copy()
        self.assertEqual(list(inspect.signature(select_all).parameters),['inner'])
        with patch.object(Path,'read_text',side_effect=AssertionError('Selector tried reading external data')),patch('pandas.read_csv',side_effect=AssertionError('Selector tried reading CSV')):
            first,_=select_all(t);second,_=select_all(t.sample(frac=1,random_state=0))
        self.assertEqual(first,second);self.assertEqual(first,c.locked['payload']['selections'])
        with self.assertRaises(ValueError):select_all(t.assign(outer_target=123.))
        with self.assertRaises(ValueError):select_all(t.assign(outer_R2=1.))

    def test_outer_label_perturbation_does_not_affect_selection_or_fit_identity(self):
        c=self.ctx;original=c.y
        for f,(tr,te) in enumerate(c.outer,1):
            cid=c.selections.loc[(c.selections.rule=='S1')&(c.selections.outer==f),'candidate'].iloc[0]
            before=cache_identity(c,f,cid);selected=select_all(c.saved_inner[FIELDS])[0]
            perturbed=original.copy();perturbed[te]=np.arange(len(te))*1000.+99999.
            try:
                c.y=perturbed
                self.assertEqual(before,cache_identity(c,f,cid));self.assertEqual(selected,select_all(c.saved_inner[FIELDS])[0])
            finally:c.y=original
        self.assertIs(c.y,original)

    def test_original_target_scenario_and_predictor_whitelist(self):
        c=self.ctx;raw=c.ns['data'];water=raw[(raw.medium=='Ocean water')&(raw.unit=='pieces/m3')].reset_index(drop=True)
        adjusted=water.conc*water.doi.map(c.ns['UNIT_FIXES']).fillna(1)
        np.testing.assert_array_equal(c.y,np.log10(adjusted+.001))
        self.assertEqual(list(c.X.columns),['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family'])
        self.assertEqual(c.splits['scenario'],SCENARIO)
        self.assertEqual(c.registry['C0_original'],c.params0)

    def test_audited_inner_predictions_count_all_studies(self):
        c=self.ctx;table=csv(c.out/'inner_cache_audit.csv')
        self.assertEqual(len(table),40);self.assertLessEqual(table.max_absolute_metric_difference.max(),1e-12)
        hashes=json.loads((c.out/'verified_prior_checkpoint_hashes.json').read_text());self.assertEqual(len(hashes),125)
        for f,(tr,te) in enumerate(c.outer,1):
            part=table[table.outer==f]
            self.assertEqual(len(part),8);self.assertTrue(part.evaluated_rows.eq(len(tr)).all());self.assertTrue(part.evaluated_studies.eq(len(set(c.groups[tr]))).all())
            self.assertFalse(set(c.groups[tr])&set(c.groups[te]))
            np.testing.assert_array_equal(np.sort(np.concatenate([b for a,b in c.inner[f]])),tr)
            for a,b in c.inner[f]:
                self.assertFalse(set(c.groups[a])&set(c.groups[b]));self.assertEqual(set(a)|set(b),set(tr))

    def test_pre_fit_lock_timestamp_hash_and_source(self):
        c=self.ctx;lock=json.loads((c.out/'locked_selections.json').read_text())
        self.assertEqual(lock['payload_sha256'],fingerprint(lock['payload']))
        self.assertEqual(sha(c.out/'locked_selections.json'),json.loads((c.out/'lock_sha256.json').read_text())['sha256'])
        self.assertEqual((c.out/'locked_selections.json').read_bytes(),(c.out/'locked_selections_v1.json').read_bytes())
        for p,h in lock['payload']['source_hashes'].items():self.assertEqual(sha(c.out/'source'/p),h)
        locktime=datetime.fromisoformat(lock['created_utc'])
        for e in self.budget.events():
            if e['event']=='fit_started':self.assertGreaterEqual(datetime.fromisoformat(e['timestamp_utc']),locktime)
        self.assertFalse(lock['payload']['outer_labels_or_outer_predictions_used_for_selection'])

    def test_fit_cap_and_no_post_evaluation_fit_allowed(self):
        self.assertTrue(ensure_available(9))
        for started in [10,11,100]:
            with self.assertRaises(RuntimeError):ensure_available(started)
        events=self.budget.events();started=[e for e in events if e['event']=='fit_started'];completed=[e for e in events if e['event']=='fit_completed']
        self.assertLessEqual(len(started),10);self.assertEqual(len(started),len(completed));self.assertFalse(self.budget.allow_new)
        before=len(events)
        with self.assertRaises(ValueError):LGBMRegressor().fit(np.zeros((2,2)),np.zeros(2))
        self.assertEqual(len(self.budget.events()),before)
        for e in started:
            tr,te=self.ctx.outer[e['outer']-1]
            self.assertEqual(e['training_labels_sha256'],array_hash(self.ctx.y[tr]));self.assertEqual(e['parameters'],self.ctx.registry[e['candidate']])
            self.assertNotEqual(e['candidate'],'C0_original')

    def test_checkpoint_identity_and_corruption_rejection(self):
        c=self.ctx
        for metadata_path in (c.out/'checkpoints').glob('*.json'):
            meta=json.loads(metadata_path.read_text())
            if 'identity' not in meta:continue
            payload=meta['identity'];path=metadata_path.with_suffix('.npz');f=payload['fold'];te=c.outer[f-1][1]
            p=read_checkpoint(path,metadata_path,cache_identity(c,f,payload['candidate']),te)
            self.assertTrue(np.isfinite(p).all())
            bad=json.loads(json.dumps(payload));bad['parameters']['n_estimators']+=1
            with self.assertRaises(ValueError):read_checkpoint(path,metadata_path,bad,te)
            with self.assertRaises(ValueError):read_checkpoint(path,metadata_path,payload,te[::-1])
            self.assertNotEqual(fingerprint(payload),fingerprint(bad))

    def test_actual_fitted_preprocessing_training_only(self):
        c=self.ctx;nums=c.ns['NUMERIC']
        for path in (c.out/'checkpoints').glob('*.json'):
            meta=json.loads(path.read_text())
            if 'preprocessor_audit' not in meta:continue
            p=meta['preprocessor_audit'];train=c.X.iloc[p['train_rows']]
            median=train[nums].median();filled=train[nums].fillna(median).to_numpy()
            np.testing.assert_allclose(p['imputer_medians'],median);np.testing.assert_allclose(p['scaler_mean'],filled.mean(axis=0));np.testing.assert_allclose(p['scaler_var'],filled.var(axis=0))
            np.testing.assert_allclose(p['numeric_min'],train[nums].min());np.testing.assert_allclose(p['numeric_max'],train[nums].max())
            self.assertEqual(p['category_values'][0],sorted(train.method_family.unique()))
        # No model fitting: original clipping and missing/unseen transformation verified on every outer training set.
        for tr,te in c.outer:
            train=c.X.iloc[tr];prep=c.ns['preprocessor']().fit(train);test=c.X.iloc[te[:3]].copy()
            for col in nums:test[col]=[train[col].min()-1000,train[col].max()+1000,np.nan]
            test['method_family']='__unseen__';clipped=c.ns['clip_to_range'](train)(test)
            np.testing.assert_allclose(clipped[nums].iloc[0],train[nums].min());np.testing.assert_allclose(clipped[nums].iloc[1],train[nums].max())
            transformed=prep.transform(clipped);self.assertTrue(np.isfinite(transformed).all());self.assertTrue((transformed[:,len(nums):]==0).all())

    def test_reuse_only_exact_same_fold_and_configuration(self):
        c=self.ctx;d=csv(c.out/'all_procedure_oof_predictions.csv');routes=csv(c.out/'prediction_reuse_audit.csv')
        self.assertEqual(len(routes),10)
        for _,r in routes.iterrows():
            te=c.outer[int(r.fold)-1][1]
            locked=c.selections[(c.selections.rule==r.rule)&(c.selections.outer==r.fold)].iloc[0]
            self.assertEqual(r.candidate,locked.candidate)
            if r.route=='original_frozen_OOF':
                self.assertEqual(r.candidate,'C0_original');np.testing.assert_array_equal(d[r.rule].iloc[te],c.base.prediction_log10.iloc[te])
            elif r.route=='optimized_frozen_OOF':
                self.assertEqual(r.candidate,c.previous_selections.loc[c.previous_selections.outer==r.fold,'candidate'].iloc[0]);np.testing.assert_array_equal(d[r.rule].iloc[te],c.opt.optimized_prediction_log10.iloc[te])
            else:
                path=c.out/'checkpoints'/f'{r.key}.npz';p=read_checkpoint(path,path.with_suffix('.json'),cache_identity(c,int(r.fold),r.candidate),te)
                np.testing.assert_array_equal(d[r.rule].iloc[te],p)
        for f in range(1,6):
            same=c.selections[c.selections.outer==f].candidate.nunique()==1
            if same:np.testing.assert_array_equal(d.S1[d.fold==f],d.S2[d.fold==f])

    def test_independently_recomputed_aggregate_and_fold_metrics(self):
        c=self.ctx;d=csv(c.out/'all_procedure_oof_predictions.csv');summary=csv(c.out/'summary_metrics.csv').set_index('procedure');fold=csv(c.out/'paired_fold_metrics.csv').set_index(['procedure','fold'])
        for name in PROCEDURES:
            y=d.target_log10.to_numpy();p=d[name].to_numpy()
            ma=[mean_absolute_error(g.target_log10,g[name]) for _,g in d.groupby('study')]
            ms=[mean_squared_error(g.target_log10,g[name]) for _,g in d.groupby('study')]
            independent={'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
                         'study_macro_MAE':np.mean(ma),'study_macro_MSE':np.mean(ms),'median_study_MAE':np.median(ma),'p90_study_MAE':np.quantile(ma,.9),'worst_study_MAE':max(ma),
                         'within_1_log':np.mean(abs(y-p)<=1),'mean_fold_R2':np.mean([r2_score(g.target_log10,g[name]) for _,g in d.groupby('fold')])}
            for k,v in independent.items():self.assertAlmostEqual(summary.loc[name,k],v,places=12)
            for f,g in d.groupby('fold'):
                self.assertAlmostEqual(fold.loc[(name,f),'R2'],r2_score(g.target_log10,g[name]),places=12)
                self.assertAlmostEqual(fold.loc[(name,f),'MAE'],mean_absolute_error(g.target_log10,g[name]),places=12)
                self.assertAlmostEqual(fold.loc[(name,f),'RMSE'],np.sqrt(mean_squared_error(g.target_log10,g[name])),places=12)
        self.assertEqual(summary.loc[ORIGINAL,'R2'],.4726308398591973);self.assertEqual(summary.loc[OPTIMIZED,'R2'],.5033812847921568)

    def test_paired_study_metrics_and_all_case_subgroup_counts(self):
        c=self.ctx;d=csv(c.out/'all_procedure_oof_predictions.csv');s=csv(c.out/'paired_study_metrics.csv').set_index(['procedure','study'])
        self.assertEqual(len(s),4*72)
        for name in PROCEDURES:
            for study,g in d.groupby('study'):
                row=s.loc[(name,study)]
                self.assertEqual(row.rows,len(g));self.assertAlmostEqual(row.MAE,mean_absolute_error(g.target_log10,g[name]),places=12);self.assertAlmostEqual(row.MSE,mean_squared_error(g.target_log10,g[name]),places=12)
                for ref in [ORIGINAL,OPTIMIZED]:self.assertAlmostEqual(row['delta_vs_'+ref+'__MAE'],row.MAE-s.loc[(ref,study),'MAE'],places=12)
        groups=csv(c.out/'method_and_study_size_metrics.csv')
        self.assertTrue(groups.groupby(['procedure','grouping']).rows.sum().eq(16988).all())
        self.assertFalse(groups.isna().any().any())

    def test_metric_missing_values_and_offset_tolerance(self):
        with self.assertRaises(ValueError):metrics([1,np.nan],[1,2],['a','b'])
        with self.assertRaises(ValueError):metrics([1,2],[1,np.inf],['a','b'])
        with self.assertRaises(ValueError):metrics([1,2],[1,2],['a',None])
        with self.assertRaises(ValueError):metrics([1,1],[1,1],['a','b'])
        m=metrics([-3,0,1],[-2,1,2.00001],['a','b','c'])
        self.assertEqual(m['within_1_log'],2/3);self.assertAlmostEqual(10**(-2)-.001,.009)
        m=metrics([0,0,0,2],[1,1,1,5],['a','a','a','b'])
        self.assertEqual(m['study_macro_MAE'],2.);self.assertEqual(m['MAE'],1.5)

    def test_frozen_screen_and_largest_study_contribution(self):
        c=self.ctx;t=csv(c.out/'summary_metrics.csv').set_index('procedure');screens=json.loads((c.out/'screening.json').read_text())
        for name in ['S1','S2']:self.assertEqual(screens[name],screening(t.loc[name],t.loc[OPTIMIZED]))
        inf=json.loads((c.out/'largest_study_influence.json').read_text());self.assertEqual(inf['rows'],6135)
        for name in PROCEDURES:
            for ref in [ORIGINAL,OPTIMIZED]:
                total=inf[name]['delta_R2_vs_'+ref+'_from_largest']+inf[name]['delta_R2_vs_'+ref+'_from_remaining']
                self.assertAlmostEqual(total,t.loc[name,'R2']-t.loc[ref,'R2'],places=12)

    def test_protected_paths_unchanged(self):
        self.assertGreater(self.ctx.verify_protected(),27000)


def run_tests(ctx,budget):
    Stage4Tests.ctx=ctx;Stage4Tests.budget=budget
    start=len(budget.events());stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Stage4Tests))
    require_no_fits=len(budget.events())==start
    text=stream.getvalue();(ctx.out/'tests.txt').write_text(text+'\nData/provenance assertions:\n'+'\n'.join('PASS '+c for c in ctx.checks)+'\n')
    if not result.wasSuccessful() or not require_no_fits:raise AssertionError(text+'\nUnexpected fits: '+str(not require_no_fits))
    status={'tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'data_provenance_assertions_passed':len(ctx.checks),'new_fits_during_tests':0}
    (ctx.out/'tests/test_results.json').write_text(json.dumps(status,indent=2));print(text);return status
