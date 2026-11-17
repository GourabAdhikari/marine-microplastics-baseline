"""Static diagnostics and report rendering; no fitting or weight selection."""
import json
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from blend import SCENARIO, WEIGHTS, METRICS, read_csv, dump


def markdown_table(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.9f}'
        return str(v).replace('|',' / ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def make_figures(a):
    colors={'B0':'#2967a0','B50':'#16806a','B100':'#cf5b22'}
    def chart(title,labels,values,path,unit):
        width=980;height=135+len(labels)*34
        allvals=np.concatenate([np.asarray(v) for v in values.values()]);lo=min(0.,float(allvals.min()));hi=float(allvals.max());span=hi-lo or 1.
        x=lambda v:290+620*(v-lo)/span
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',f'<text x="12" y="24" font-size="17">{escape(title)}</text>',
             f'<text x="12" y="44" font-size="10">{SCENARIO}</text>',
             '<text x="12" y="65" font-size="12">Blue: B0 original | Green: B50 predeclared | Orange: B100 optimized</text>',
             f'<line x1="{x(0)}" x2="{x(0)}" y1="80" y2="{height-45}" stroke="#ccc"/>']
        for i,label in enumerate(labels):
            yy=95+i*34
            svg.append(f'<text x="12" y="{yy+4}" font-size="12">{escape(str(label))}</text>')
            for name,vals in values.items():svg.append(f'<circle cx="{x(vals[i])}" cy="{yy}" r="4" fill="{colors[name]}"/>')
        for tick in np.linspace(lo,hi,6):svg.append(f'<text x="{x(tick)-20}" y="{height-25}" font-size="11">{tick:.3f}</text>')
        svg.append(f'<text x="410" y="{height-5}" font-size="12">{escape(unit)}</text></svg>')
        path.write_text('\n'.join(svg))
    t=a.pooled.set_index('blend');names=list(colors)
    chart('Pooled metrics: B50 is predefined, not selected from scores',['R²','MAE','RMSE','Study-macro MAE','P90 study MAE','Worst-study MAE'],
          {n:[t.loc[n,k] for k in ['R2','MAE','RMSE','study_macro_MAE','p90_study_MAE','worst_study_MAE']] for n in names},a.out/'figures/pooled_metrics.svg','Metric value (R² dimensionless; errors log10)')
    chart('All five fold R² values, including negative values',[f'Fold {f}' for f in range(1,6)],
          {n:a.fold[a.fold.blend==n].sort_values('fold').R2.tolist() for n in names},a.out/'figures/fold_r2.svg','R² (each fold uses its own variance)')
    harmed=a.study[a.study.prior_harmed_diagnostic_only]
    order=sorted(a.prior_harmed)
    chart('Previously harmed studies: post-hoc diagnostic only',[f'{i+1}. n={int(harmed[harmed.study==s].rows.iloc[0])}' for i,s in enumerate(order)],
          {n:harmed[harmed.blend==n].set_index('study').loc[order,'MAE'].tolist() for n in names},a.out/'figures/prior_harmed_studies.svg','Study MAE (log10 units)')
    pd.DataFrame({'plot_row':range(1,15),'study':order}).to_csv(a.out/'figures/prior_harmed_order.csv',index=False)
    return [a.out/'figures'/f for f in ['pooled_metrics.svg','fold_r2.svg','prior_harmed_studies.svg']]


def compose_report(a,test_status,cell_count):
    t=a.pooled.set_index('blend');b=t.loc['B0'];m=t.loc['B50'];o=t.loc['B100']
    contrasts=pd.DataFrame([{'metric':k,'B0_original':b[k],'B50_main':m[k],'B100_optimized':o[k],
                             'B50_minus_B0':m[k]-b[k],'B50_minus_B100':m[k]-o[k]} for k in METRICS+['mean_fold_R2']])
    gate=a.gate;c=a.stage_c
    conditions=pd.DataFrame([{'condition':key,'passed':value} for key,value in gate['conditions'].items()])
    prior=a.study[a.study.prior_harmed_diagnostic_only]
    prior_table=prior.pivot(index=['study','rows','fold'],columns='blend',values='MAE').reset_index()[['study','rows','fold','B0','B50','B100']]
    prior_table['B50_minus_B0']=prior_table.B50-prior_table.B0;prior_table['B50_minus_B100']=prior_table.B50-prior_table.B100
    all50=a.study[a.study.blend=='B50']
    harms=[]
    for ref in ['B0','B100']:
        delta=all50['delta_vs_'+ref+'__MAE']
        harms.append({'comparison':'B50 vs '+ref,'studies_MAE_worse':int((delta>0).sum()),'studies_MAE_worse_by_over_0.10':int((delta>.10).sum()),
                      'studies_MAE_better':int((delta<0).sum()),'studies_MAE_equal':int((delta==0).sum())})
    harm_table=pd.DataFrame(harms)
    harm_table.to_csv(a.out/'b50_study_harm_counts.csv',index=False)
    worst=all50.sort_values('delta_vs_B0__MAE',ascending=False).head(10)[['study','rows','fold','MAE','delta_vs_B0__MAE','delta_vs_B100__MAE']]
    fold=a.fold[a.fold.blend.isin(['B0','B50','B100'])][['fold','blend','rows','studies','R2','MAE','RMSE','delta_vs_B0__MAE','delta_vs_B100__MAE']]
    groups=a.subgroups[a.subgroups.blend.isin(['B0','B50','B100'])][['grouping','group','blend','rows','studies','R2','MAE','study_macro_MAE','p90_study_MAE','within_1_log']]
    max_precision=float(abs(read_csv(a.out/'reference_metric_audit.csv').difference).max())
    if gate['passed']:
        screening='**PASS.** This authorizes only a compute-feasibility assessment; it does not confirm transfer.'
    else:screening='**FAIL.** The required stop rule applies. Failed conditions: '+', '.join(gate['failed_conditions'])+'. No alternative weight was searched to rescue the gate.'
    if c['status']=='skipped_budget_exceeded':
        stage_c=f"""**Not run: budget exceeded.** Preserve the original five outer and three inner grouped folds, and use three inner-inner grouped folds over the eight frozen configurations. The reliable no-cache upper bound is **{c['upper_bound_new_fits']} newly started fits**, versus the hard cap of **100**:

- 5 × 3 × 3 × 8 = **360** inner-inner candidate fits.
- 5 × 3 × 2 = **30** inner cross-fit base-learner refits.
- 5 × 2 = **10** outer base-learner refits.

Even setting compatible prior base predictions/refits to zero would leave the planned **360** inner-inner fits over budget. We did not reduce folds, discard candidates, or reuse optimized inner predictions selected using their own validation outcomes. All 45 planned inner-inner partitions were audited without fitting; their study IDs and row IDs are saved in `stage_c_planned_splits.json`, with group counts in `stage_c_planned_partition_counts.csv`. No insufficient-group fallback was needed. These are planned partitions, not a completed learned-blending experiment.

The unexecuted training-only weight rule was frozen before metrics: minimize study-macro MAE among fixed weights whose pooled inner R² is within 0.015 of the best option; exact ties prefer B50, then more original weight. No weights or model configurations were learned in this phase."""
    else:stage_c='**Not run: Stage B failed.** No Stage C fit was authorized or started. Nested parameter/weight selection and its label-independence tests are not claimed as completed. The existing original/optimized validation splits were audited, but no new nesting was needed.'
    decision=('The predefined B50 shows a useful retrospective trade-off and passes the cost-control screen, but it is not promoted and no independent generalization claim is made.' if gate['passed'] else 'The predefined B50 does not meet the full reliability screen. Preserve both immutable references and stop; do not promote a blend or tune weights to rescue the result.')
    env=json.loads((a.out/'environment.json').read_text())
    text=f'''# Fixed LightGBM blending and study robustness

**Scenario: `{SCENARIO}` for every result.** Historical ×10^6 multipliers for Barrows (1,389 rows) and Dehm (81 rows) are **unverified hypotheses**, not certified corrections. All predictions and errors use `log10(concentration + 0.001)`; they are not certified estimates of true synthetic-only marine microplastic concentration.

## Result and decision

The fixed original and optimized references remain **0.4726308398591973** and **0.5033812847921568** pooled R². The predeclared 50/50 blend is **{m.R2:.16f}** (Δ versus original **{m.R2-b.R2:+.16f}**, versus optimized **{m.R2-o.R2:+.16f}**).

{decision}

B50 pooled MAE is **{m.MAE:.9f}**, study-macro MAE **{m.study_macro_MAE:.9f}**, p90 study MAE **{m.p90_study_MAE:.9f}**, worst-study MAE **{m.worst_study_MAE:.9f}**, and within-one-log fraction **{m.within_1_log:.9f} ({100*m.within_1_log:.2f}%)**. All reference contrasts, including deteriorations, follow below. R² is not a percentage of correct predictions. Within-one-log is a factor-ten tolerance on **concentration plus 0.001**, not plain concentration.

## Stage A: five fixed combinations, zero fits

The registry was written to `predeclared_plan.json` before calculating blend outcomes. B50 is the **predeclared main candidate**, not the highest outer-score candidate selected after seeing results. B25/B75 are retrospective fixed-weight sensitivity analyses, not a weight search or a leaderboard winner. Each weight is constant across all rows and studies, nonnegative, and sums with its original-model weight to one. Blending occurs on the common log10 prediction scale, not on raw concentrations.

{markdown_table(a.pooled[['blend','weight_optimized','R2','MAE','RMSE','study_macro_MAE','study_macro_MSE','median_study_MAE','p90_study_MAE','worst_study_MAE','within_1_log','mean_fold_R2']])}

### Main candidate versus both immutable references

{markdown_table(contrasts)}

Full-precision changes for every blend relative to **both** references are saved in every paired metric CSV. Positive error changes mean deterioration; positive R²/coverage changes mean improvement. Study quantiles use linear interpolation. Pooled R² scores all 16,988 held-out predictions together; arithmetic mean fold R² is a separate quantity.

## Stage B: precommitted cost-control gate

{screening}

{markdown_table(conditions)}

Improvement checks among pooled MAE, p90 study MAE and within-one-log coverage: `{json.dumps(gate['weakness_improvements'])}`. Exact B50-minus-B100 deltas are in `stage_b_gate.json`. No outer labels fit coefficients or choose weights; labels are used only to score the fixed blends and apply the explicitly requested cost-control gate. That gate is itself retrospective and must not be treated as independent confirmation.

## Stage C: conditional training-only blend selection

{stage_c}

**New LightGBM fits: 0.** No prior phase was restarted; none of its 127 fits was repeated. `fit_log.jsonl` contains a not-run decision event, not a fit start. No stacking, learned gate, support estimate, new features, target changes, data acquisition, chronological-score consultation, or deployable model was used.

## All outer folds and post-hoc folds 2/5 diagnostic

{markdown_table(fold)}

Folds 2 and 5 were singled out by the preceding optimization results, not by this blend. Their diagnostics do not enter weights or the screening rule. `folds_2_and_5_diagnostic.csv` includes all five fixed combinations. The first outer fold contains a single 6,135-row study (36.11% of all observations); it is retained. Negative R² values are neither clipped nor omitted.

## Study-level robustness and harms

{markdown_table(harm_table)}

All **72 studies**, including difficult and very small studies (minimum **{int(a.d.groupby('study').size().min())} rows**), are retained. Study MAE/MSE do not require within-study target variance. We report these well-defined errors rather than within-study R²; no observations are removed. Synthetic singleton cases are tested, but there are no one-row study groups in this dataset. Macro metrics average each study equally. The following 14 studies were previously harmed by >0.10 log10 MAE under optimization; membership is copied from the protected prior table and used **only for description**:

{markdown_table(prior_table)}

Ten largest B50 MAE increases relative to the original (full 360 blend-study rows in `paired_study_metrics.csv`):

{markdown_table(worst)}

No harmed-study flag, study identity, fold ID, sample outcome, or test-specific threshold is a predictor or a blend weight.

## Method family and study-size diagnostics

{markdown_table(groups)}

The complete CSV also includes B25/B75 and every metric/delta. Study-size bins remain 1–50, 51–200, 201–1000 and 1001+ eligible rows; no target bins. Family slices are row-based, so a study can contribute to several method families. Every grouping covers all 16,988 rows for each blend. Rows and studies are explicitly counted.

## Integrity, precision, execution and tests

- Exact 16,988 modelling rows, 72 linked DOI/accession studies, original 29,076-row dataset, original source-row order, folds and target values preserved. Test sizes **6,135 / 2,714 / 2,712 / 2,714 / 2,713**. Each row has exactly one outer OOF prediction. No silent joins, NaN dropping, imputation of predictions or unsupported-row exclusions.
- The baseline OOF file has no scenario column; its scenario is established from the frozen original source, saved historical unit adjustments and exact target equality, then explicitly labeled here. The optimized file already carries the same scenario.
- Existing input manifests, source/configuration/split identities and artifact checksums were verified before analysis. CSVs were read with `float_precision='round_trip'`. Every original and optimized reference aggregate was recomputed; maximum absolute difference **{max_precision:.3g}**, under `atol=rtol=1e-12`. Full differences: `reference_metric_audit.csv`. Raw-scale baseline metrics were checked only as preservation diagnostics, never compared with a different target scenario.
- Original preprocessing and study grouping were reconstructed by read-only calls to the prior loader, never its training routines. The original median/scaler/encoder/clipping behavior was checked on all five outer and 15 original inner training partitions, including missing numeric values and unseen categories. These are preprocessing tests, **not LightGBM training**.
- **{test_status['passed']} unit/integration tests passed**, **{test_status['data_audit_assertions_passed']} additional data/audit assertions passed**, no failures. Tests include deliberately mismatched rows/labels/scenarios, missing predictions, independent metric recomputation, fixed-weight target perturbation, all-case evaluation, factor-ten boundaries, original preprocessing, split disjointness, deterministic arithmetic, import-with-no-fit enforcement, budget blocking, identity invalidation and protected-file hashes. `tests.txt` records what actually ran. Actual Stage C model/weight fitting was not run and its learned-selection invariance is not claimed.
- The initial preparation run exposed two incorrect test assumptions: it expected a real singleton study and hard-coded 29 reference-audit rows instead of the actual 27. The tests were repaired to verify complete group membership and derive the audit count from the metric schema; singleton behavior is covered synthetically. No data, folds, weights, metrics or gate thresholds were changed. The subsequent full fresh-kernel execution passed. The initial failure log remains in `preparation_execution.log`.
- **{cell_count} notebook code cells executed** in order in a fresh kernel; cell statuses and output types are captured in `notebook_execution_status.json`. Every cell uses the same importable modules as the CLI. LightGBM `fit` is patched to fail during this zero-fit run and tests, preventing accidental prior-phase execution.
- Environment: Python {env['python'].split()[0]}, {env['platform']}; packages `{json.dumps(env['packages'],sort_keys=True)}`. Seed 0 documented; fixed arithmetic is deterministic and does not use randomness. CPU/thread cap 4, no downloads. Runtime/IPython/cache files are isolated under `blending_outputs/`.
- `integrity_manifest.json` records before/after SHA-256 values for **every pre-existing file**, including the raw data, original source, `.git/`, and complete reproduction/optimization directories. Symlinks are recorded as link targets without following external interpreters. **All protected files remain unchanged.** The newly supplied prompt inside `optimization_outputs/` is also protected.
- No new model checkpoints exist. `checkpoints/analysis_identity.json` keys the zero-fit analysis to input/data hashes, all analysis/test source, frozen plan/configuration, split identities and versions. Reruns verify protected references and recompute cheap arithmetic; they never use incompatible model predictions or start training.

## Reproduce (from the repository root)

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python blending_outputs/source/run.py --prepare
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python blending_outputs/source/run.py
```

The first command executes the notebook and all tests without creating the final report; the second executes the same zero-fit analysis and writes this report **last**. These commands do not restart earlier experiments. Sources are importable and have no side-effect execution on import. All new outputs are confined to `blending_outputs/`.

## Outputs and limitations

- `LightGBM_Blend_Research_EXECUTED.ipynb`, `aligned_oof_predictions.csv`.
- `fixed_blend_metrics.csv`, `paired_fold_metrics.csv`, `paired_study_metrics.csv`, `method_and_study_size_metrics.csv`.
- `stage_b_gate.json`, `stage_c_status.json`, `final_status.json`, `tests.txt`, `fit_log.jsonl`, `environment.json`, `integrity_manifest.json`.
- `figures/pooled_metrics.svg`, `figures/fold_r2.svg`, `figures/prior_harmed_studies.svg`; extra diagnostic row keys and tables accompany these.

The five outer folds and outcomes have already been inspected, including before this phase. This analysis is **exploratory on reused historical-scenario targets**, not an independent holdout, prospective test, certified physical prediction, production-readiness assessment, or proof of new-study generalization. Neither a higher R² nor passing the cost-control gate removes these limitations. No reference is overwritten or promoted.

**Stop: the requested bounded experiment ends with this report. No further phase is started.**
'''
    return text
