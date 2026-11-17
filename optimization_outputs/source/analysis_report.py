"""Saved-prediction analysis, SVG diagnostics and report. No fitting here."""
import json
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from experiment import SCENARIO, REFERENCE, BASE_NAME, OPT_NAME, metric, dump, sha

COLS={BASE_NAME:'original_prediction_log10',OPT_NAME:'optimized_prediction_log10'}


def table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f'{x:.6f}'
        return str(x).replace('|',' / ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def analyze(exp):
    d=pd.read_csv(exp.out/'optimized_oof_predictions.csv',float_precision='round_trip')
    summary={'scenario':SCENARIO,'reference_R2':REFERENCE,'rows':len(d),'studies':d.study.nunique()}
    for name,col in COLS.items():
        summary[name]=metric(d.target_log10,d[col],d.study)
        summary[name]['mean_fold_R2']=float(np.mean([metric(g.target_log10,g[col],g.study)['R2'] for _,g in d.groupby('fold')]))
    summary['delta_R2']=summary[OPT_NAME]['R2']-REFERENCE
    summary['delta_study_macro_MAE']=summary[OPT_NAME]['study_macro_MAE']-summary[BASE_NAME]['study_macro_MAE']
    summary['study_macro_MAE_change_percent']=100*(summary[OPT_NAME]['study_macro_MAE']/summary[BASE_NAME]['study_macro_MAE']-1)
    for grouping,filename in [('fold','paired_per_fold_metrics.csv'),('study','paired_per_study_metrics.csv')]:
        rows=[]
        for key,g in d.groupby(grouping,sort=True):
            rec={grouping:key,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO}
            if grouping=='study':
                rec.update(fold=int(g.fold.iloc[0]),method_families='; '.join(sorted(g.method_family.unique())),study_size_bin=g.study_size_bin.iloc[0])
            for name,col in COLS.items():
                rec.update({name+'__'+k:v for k,v in metric(g.target_log10,g[col],g.study).items()})
            rec['delta_MAE']=rec[OPT_NAME+'__MAE']-rec[BASE_NAME+'__MAE']
            rec['delta_R2']=(rec[OPT_NAME+'__R2']-rec[BASE_NAME+'__R2']) if rec[BASE_NAME+'__R2'] is not None else None
            rec['material_MAE_deterioration']=rec['delta_MAE']>.10
            rows.append(rec)
        pd.DataFrame(rows).to_csv(exp.out/filename,index=False)
    rows=[]
    for grouping in ['method_family','study_size_bin']:
        for key,g in d.groupby(grouping,sort=True):
            for name,col in COLS.items():
                rows.append({'grouping':grouping,'group':key,'procedure':name,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO,**metric(g.target_log10,g[col],g.study)})
    pd.DataFrame(rows).to_csv(exp.out/'performance_by_method_and_study_size.csv',index=False)
    studies=pd.read_csv(exp.out/'paired_per_study_metrics.csv')
    summary['studies_MAE_worse']=int((studies.delta_MAE>0).sum())
    summary['studies_material_MAE_worse']=int(studies.material_MAE_deterioration.sum())
    largest=d.study.value_counts().index[0]
    mask=d.study==largest
    influence={'scenario':SCENARIO,'study':largest,'rows':int(mask.sum()),'row_fraction':float(mask.mean()),'note':'Exclusion is a descriptive influence diagnostic on existing predictions, not a new benchmark or refit.'}
    for name,col in COLS.items():
        se=(d[col]-d.target_log10)**2
        influence[name]={'largest_study_SSE':float(se[mask].sum()),'total_SSE':float(se.sum()),'largest_study_SSE_fraction':float(se[mask].sum()/se.sum()),
                         'largest_study':metric(d.loc[mask,'target_log10'],d.loc[mask,col],d.loc[mask,'study']),
                         'excluding_largest':metric(d.loc[~mask,'target_log10'],d.loc[~mask,col],d.loc[~mask,'study'])}
    delta_sse=(d.original_prediction_log10-d.target_log10)**2-(d.optimized_prediction_log10-d.target_log10)**2
    influence['largest_contribution_to_global_delta_R2']=float(delta_sse[mask].sum()/((d.target_log10-d.target_log10.mean())**2).sum())
    influence['remaining_contribution_to_global_delta_R2']=float(delta_sse[~mask].sum()/((d.target_log10-d.target_log10.mean())**2).sum())
    dump(exp.out/'largest_study_influence.json',influence)
    dump(exp.out/'optimization_summary.json',summary)
    return pd.DataFrame({name:summary[name] for name in COLS})


def plots(exp):
    """Small static SVG diagnostics avoid extra plotting-library downloads."""
    out=exp.out/'figures';out.mkdir(exist_ok=True)
    folds=pd.read_csv(exp.out/'paired_per_fold_metrics.csv')
    studies=pd.read_csv(exp.out/'paired_per_study_metrics.csv')
    def svg(title,labels,vals1,vals2,path,axis):
        width,height=920,140+32*len(labels)
        low=min(0,min(vals1),min(vals2));high=max(max(vals1),max(vals2),.01)
        span=high-low
        x=lambda v: 265+600*(v-low)/span
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="15" y="25" font-size="18">{escape(title)}</text>',
               f'<text x="15" y="47" font-size="10">{SCENARIO}</text>',
               '<text x="15" y="66" font-size="12">Blue: original; orange: nested procedure. Lines pair the same observations.</text>',
               f'<line x1="{x(0)}" x2="{x(0)}" y1="80" y2="{height-45}" stroke="#aaa"/>']
        for i,(label,a,b) in enumerate(zip(labels,vals1,vals2)):
            y=100+i*32
            parts.extend([f'<text x="15" y="{y+4}" font-size="12">{escape(str(label))}</text>',
                          f'<line x1="{x(a)}" x2="{x(b)}" y1="{y}" y2="{y}" stroke="#999"/>',
                          f'<circle cx="{x(a)}" cy="{y}" r="4" fill="#2563ac"/><circle cx="{x(b)}" cy="{y}" r="4" fill="#d95c16"/>'])
        for v in np.linspace(low,high,6):
            parts.append(f'<text x="{x(v)-15}" y="{height-25}" font-size="11">{v:.2f}</text>')
        parts.append(f'<text x="420" y="{height-5}" font-size="12">{escape(axis)}</text></svg>')
        path.write_text('\n'.join(parts))
    svg('All outer fold R² (including negative values)',[f'Fold {f}' for f in folds.fold],folds[BASE_NAME+'__R2'],folds[OPT_NAME+'__R2'],out/'fold_r2.svg','R² on log10 target')
    studies=studies.sort_values('delta_MAE',ascending=False)
    svg('All 72 study MAEs, ordered by deterioration',[f'Study {i+1}, n={int(r.rows)}' for i,r in studies.reset_index(drop=True).iterrows()],studies[BASE_NAME+'__MAE'],studies[OPT_NAME+'__MAE'],out/'all_study_mae.svg','Study MAE (log10 units)')
    studies[['study','rows','delta_MAE']].to_csv(out/'all_study_mae_order.csv',index=False)
    return [out/'fold_r2.svg',out/'all_study_mae.svg']


def report(exp):
    s=json.loads((exp.out/'optimization_summary.json').read_text())
    influence=json.loads((exp.out/'largest_study_influence.json').read_text())
    folds=pd.read_csv(exp.out/'paired_per_fold_metrics.csv')
    studies=pd.read_csv(exp.out/'paired_per_study_metrics.csv')
    groups=pd.read_csv(exp.out/'performance_by_method_and_study_size.csv')
    selections=pd.read_csv(exp.out/'outer_selections.csv')
    registry=json.loads((exp.out/'candidate_registry.json').read_text())
    events=[json.loads(l) for l in (exp.out/'fit_log.jsonl').read_text().splitlines()]
    counts={stage:sum(e['stage']==stage and e['event']=='completed' for e in events) for stage in ['inner','outer','verification']}
    checks=json.loads((exp.out/'reproduction_and_integrity_checks.json').read_text()) if (exp.out/'reproduction_and_integrity_checks.json').exists() else None
    s['actual_completed_fits']=counts
    s['total_completed_fits']=sum(counts.values())
    s['actual_started_fits']=sum(e['event']=='started' for e in events)
    s['selected_candidates']=dict(zip(selections.outer.astype(str),selections.candidate))
    s['tests_passed']=checks['passed'] if checks else 'pending final tests'
    dump(exp.out/'optimization_summary.json',s)
    metrics=pd.DataFrame({'metric':list(s[BASE_NAME]),BASE_NAME:list(s[BASE_NAME].values()),OPT_NAME:list(s[OPT_NAME].values())})
    sel_rows=[]
    for _,row in selections.iterrows():
        sel_rows.append({'fold':row.outer,'candidate':row.candidate,'differences from original':json.dumps(registry['differences_from_baseline'][row.candidate],sort_keys=True)})
    fold_table=folds[['fold','rows','studies',BASE_NAME+'__R2',OPT_NAME+'__R2','delta_R2',BASE_NAME+'__MAE',OPT_NAME+'__MAE','delta_MAE']]
    worst=studies.sort_values('delta_MAE',ascending=False).head(10)[['study','rows','fold',BASE_NAME+'__MAE',OPT_NAME+'__MAE','delta_MAE']]
    all_bad=studies[studies.material_MAE_deterioration][['study','rows','fold','delta_MAE']]
    decision=('Pooled R² improved.' if s['delta_R2']>0 else 'The predeclared nested procedure did NOT improve pooled R². Preserve the original reproducible reference.')
    env=json.loads((exp.out/'environment.json').read_text())
    text=f'''# Bounded LightGBM optimization report

**Scenario for every table and result: `{SCENARIO}`.** Historical ×10^6 unit multipliers are frozen hypotheses, scientifically unresolved—not certified corrections or physical predictions of true synthetic microplastic concentration.

## Outcome

Original reproduced pooled R²: **{REFERENCE:.16f}**. Nested optimized pooled R²: **{s[OPT_NAME]['R2']:.16f}**. Exact difference (optimized − reference): **{s['delta_R2']!r}**.

{decision} Study-macro MAE changed by **{s['delta_study_macro_MAE']:+.9f}** log10 units (**{s['study_macro_MAE_change_percent']:+.3f}%**). The inner 5% safety rule is not a guarantee of outer safety. {s['studies_MAE_worse']} of 72 studies have worse MAE; {s['studies_material_MAE_worse']} worsen by more than the predeclared descriptive threshold of 0.10 log10 units. No configuration is automatically promoted.

This is a **mixed improvement, not a uniformly better model**: pooled MAE increases from {s[BASE_NAME]['MAE']:.6f} to {s[OPT_NAME]['MAE']:.6f}; within-one-log coverage falls from {100*s[BASE_NAME]['within_1_log']:.2f}% to {100*s[OPT_NAME]['within_1_log']:.2f}%; p90 study MAE rises from {s[BASE_NAME]['p90_study_MAE']:.6f} to {s[OPT_NAME]['p90_study_MAE']:.6f}, and worst-study MAE from {s[BASE_NAME]['worst_study_MAE']:.6f} to {s[OPT_NAME]['worst_study_MAE']:.6f}. Mean fold R² is also worse. Folds 2 and 5 exceed the descriptive 0.10-log MAE deterioration flag. The desired pooled-R² and average-study-error goals improved, but tail safety did not. No outer-tail acceptability threshold was specified in advance, so these harms cannot be declared acceptable after the fact. Retain both results and the unchanged reproducible reference; do not deploy or promote this procedure on this evidence alone.

R² is explained variance on `log10(concentration + 0.001)`, not accuracy. Within ±1 log measures a factor-ten tolerance on concentration **plus the offset**, not an unqualified percentage correct. All negative fold results remain visible.

## Pooled and study-balanced metrics

{table(metrics)}

## Selection and paired outer folds

{table(pd.DataFrame(sel_rows))}

{table(fold_table)}

Full fold metrics including RMSE, all study-balanced metrics and tolerances: `paired_per_fold_metrics.csv`.

## Predeclared design

Exactly eight configurations from `optimization.md`, recorded before any tuned outer outcomes in `candidate_registry.json`. Baseline defaults are loaded from `reproduction_outputs/model_parameters.json`. No changed features or units, no early stopping, no model search beyond the registry. All other LightGBM parameters are unchanged; seed 0 and four learner threads on {env['cpu_count']} logical CPUs. We preserve the baseline settings rather than adding different LightGBM deterministic/column-wise flags. A small repeated fit tests practical determinism on this environment, not cross-platform bitwise equivalence.

Fixed outer row memberships come directly from the baseline OOF CSV, never from a replacement split. Each outer training set has GroupKFold(3); `inner_split_counts.csv` records all 15 partition counts. The selector receives only outer-training targets. Candidates are ranked by pooled inner OOF R², not fold-mean R². Nonbaseline eligibility requires strictly better R² than C0 and study-macro MAE <=105% of C0. C0 remains admissible. Among candidates within 0.005 of the best admissible R², lower macro MAE wins; exact remaining ties use ID order. This precise tie interpretation was frozen before fits. `inner_metrics_by_outer_fold.csv` contains all 40 candidate summaries, including p90 and worst-study MAE; `outer_selections.csv` records reasons. All five selections were completed before any outer refit was scored.

Only the selected procedure was evaluated on outer rows, not eight candidate outer scores. This is nested predeclared-procedure performance, not a post-hoc best candidate. The outer folds and 0.473 headline were already inspected, so even correctly nested improvement is **exploratory**. Independent prospective performance requires new studies.

## Study-level deterioration

Ten largest MAE increases (all studies remain in the full CSV):

{table(worst)}

All studies flagged as materially worse (>0.10 log10 MAE):

{table(all_bad)}

The full `paired_per_study_metrics.csv` preserves every study, including any undefined within-study R² for single-observation or constant-target studies (blank, not dropped). Macro MAE and MSE include all studies equally. Quantiles use pandas' default linear interpolation. Method-family groups are row-based; a study may contribute to multiple families. Study-size bins are fixed at 1–50, 51–200, 201–1000 and 1001+ eligible observations, never target-derived.

## Performance by method family and study size

{table(groups[['grouping','group','procedure','rows','studies','R2','MAE','RMSE','study_macro_MAE','p90_study_MAE']])}

## Influence of the largest study

`{influence['study']}` contains **{influence['rows']} rows ({100*influence['row_fraction']:.2f}%)**, comprising the entire first outer fold.

- Its fraction of total squared error: original **{100*influence[BASE_NAME]['largest_study_SSE_fraction']:.2f}%**, optimized **{100*influence[OPT_NAME]['largest_study_SSE_fraction']:.2f}%**.
- Its contribution to global ΔR²: **{influence['largest_contribution_to_global_delta_R2']:+.9f}**; remaining studies contribute **{influence['remaining_contribution_to_global_delta_R2']:+.9f}**. These add to the overall change.
- Descriptively excluding this study from scoring the existing predictions: original R² **{influence[BASE_NAME]['excluding_largest']['R2']:.9f}**, optimized R² **{influence[OPT_NAME]['excluding_largest']['R2']:.9f}**. This changes the evaluation population and denominator, is not the benchmark, and was not used for selection; no refits or row exclusions occurred.

## Integrity, environment, execution and fit accounting

- Original eligible data: 16,988 rows, 72 study groups, 73 accessions and 73 publications; 29,076 original rows. The independent grouping audit rejects missing/placeholder identifiers; no missing-identifier bridge occurs in this dataset.
- Original preprocessing cells and helper functions are extracted verbatim read-only, with original row order, median imputation, scaling, one-hot unknown handling and test clipping. Preprocessing fits inside every training fold. No identifiers or targets enter predictors.
- Barrows 1,389 rows and Dehm 81 rows retain ×10^6; other measurements are unchanged. Exact target and row equality with the frozen reproduction are tested.
- Core packages: {json.dumps(env['packages'],sort_keys=True)}. Python {env['python'].split()[0]}, {env['platform']}. Same environment as reproduction; original historical notebook metadata had Python 3.13.5. No downloads were needed.
- Actual completed LightGBM fits: **{counts['inner']} inner + {counts['outer']} selected outer + {counts['verification']} essential deterministic verification = {sum(counts.values())}**. Started fits: {s['actual_started_fits']}. C0 inner fits are required candidate comparisons, not regenerated baseline OOF. Selected C0 outer fits, if present, are explicitly logged in `fit_log.jsonl`. No full baseline rerun, no early stopping, no deployable full-data model, no extra phase.
- Tests passed: **{s['tests_passed']}**. `tests.txt` details every assertion. `reproduction_and_integrity_checks.json` records protected-file preservation, including the original dataset, original notebooks, lockfile, `.git/` and the complete reproduction directory. SHA-256 manifest: `protected_hashes_before.json`; symlinks are recorded as link targets without traversing external interpreters. No protected path was added or removed.
- Cache identities include core experiment code, data, original source, input artifacts, folds, all configurations, scenario and package versions, plus exact per-fit training labels and row IDs. Checkpoints are atomic and fit events logged. Changing the fit identity does not bypass the hard inner-fit cap. Checkpoint-only reruns do not repeat valid fits.
- The executed notebook invokes importable modules; imports have no fit side effects. `notebook_execution_status.json` records completed cells and outputs. The runner always uses a fresh kernel; runtime and IPython files are isolated under this run directory.

## Reproduce and test (from repository root)

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python optimization_outputs/source/run.py
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python optimization_outputs/source/run.py --tests
```

The first command resumes this frozen experiment from hash-verified checkpoints and executes the notebook; it does not start another experiment. Both commands are tested. For a different experiment, use a separate versioned directory rather than replacing this registry.

## Files

- `LightGBM_Optimization_EXECUTED.ipynb`: executed cells and outputs.
- `optimized_oof_predictions.csv`: exact source/water row IDs, studies, folds, common targets, both predictions and scenario label.
- `optimization_summary.json`, `paired_per_fold_metrics.csv`, `paired_per_study_metrics.csv`, `performance_by_method_and_study_size.csv`.
- `candidate_registry.json`, `inner_metrics_by_outer_fold.csv`, `inner_split_counts.csv`, `outer_selections.csv`, `frozen_splits.json`.
- `figures/fold_r2.svg`, `figures/all_study_mae.svg` (all 72 studies; order key saved alongside).
- `input_artifacts.json`, `environment.json`, `checkpoint_identity.json`, `fit_log.jsonl`, `tests.txt`, `reproduction_and_integrity_checks.json`.

**Stop:** this bounded experiment is complete. No independent validation claim and no scientifically verified unit correction is made. No automatic next phase.
'''
    (exp.out/'OPTIMIZATION_REPORT.md').write_text(text)
    return s
