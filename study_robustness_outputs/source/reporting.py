"""Descriptive reporting only. No selection, model fitting, or threshold changes."""
import json
from html import escape
import numpy as np
import pandas as pd
from common import SCENARIO,ORIGINAL,OPTIMIZED,PROCEDURES,METRIC_NAMES,csv,dump,sha


def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.9f}'
        return str(x).replace('|',' / ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def fit_accounting(ctx):
    events=[json.loads(l) for l in (ctx.out/'model_fit_log.jsonl').read_text().splitlines()]
    initial=csv(ctx.out/'initial_prediction_reuse_audit.csv');current=csv(ctx.out/'prediction_reuse_audit.csv')
    return {'new_fits_started_total':sum(e['event']=='fit_started' for e in events),'new_fits_completed_total':sum(e['event']=='fit_completed' for e in events),
            'failed_fits':sum(e['event']=='fit_failed' for e in events),'initial_routes':initial.route.value_counts().to_dict(),
            'current_routes':current.route.value_counts().to_dict(),'warnings':json.loads((ctx.out/'fit_warnings.json').read_text())}


def figures(ctx):
    f=csv(ctx.out/'paired_fold_metrics.csv');colors={ORIGINAL:'#346aaf',OPTIMIZED:'#ca602c','S1':'#14816e','S2':'#8a50a0'}
    values=f.R2;low=min(-.01,values.min());high=max(.01,values.max());x=lambda v:220+650*(v-low)/(high-low)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="365" viewBox="0 0 980 365">','<rect width="100%" height="100%" fill="white"/>',
           '<text x="12" y="24" font-size="18">All five fold R²: no negative values hidden</text>',f'<text x="12" y="43" font-size="10">{SCENARIO}</text>']
    for i,(name,color) in enumerate(colors.items()):parts.append(f'<text x="{12+i*240}" y="65" font-size="12" fill="{color}">{escape(name)}</text>')
    parts.append(f'<line x1="{x(0)}" x2="{x(0)}" y1="85" y2="315" stroke="#aaa"/>')
    for fold in range(1,6):
        y=110+(fold-1)*45;parts.append(f'<text x="12" y="{y+4}" font-size="13">Fold {fold}</text>')
        for j,(name,color) in enumerate(colors.items()):
            value=f.loc[(f.fold==fold)&(f.procedure==name),'R2'].iloc[0]
            parts.append(f'<circle cx="{x(value)}" cy="{y+(j-1.5)*7}" r="4" fill="{color}"/>')
    for tick in np.linspace(low,high,6):parts.append(f'<text x="{x(tick)-15}" y="345" font-size="12">{tick:.2f}</text>')
    parts.append('</svg>');path=ctx.out/'figures/fold_r2.svg';path.write_text('\n'.join(parts));return path


def compose(ctx,test_status,cell_count):
    summary=csv(ctx.out/'summary_metrics.csv');t=summary.set_index('procedure')
    folds=csv(ctx.out/'paired_fold_metrics.csv');studies=csv(ctx.out/'paired_study_metrics.csv');groups=csv(ctx.out/'method_and_study_size_metrics.csv')
    harms=csv(ctx.out/'study_harm_counts.csv');screens=json.loads((ctx.out/'screening.json').read_text());inf=json.loads((ctx.out/'largest_study_influence.json').read_text())
    account=fit_accounting(ctx);lock=ctx.locked
    qualifying=[s for s in ['S1','S2'] if screens[s]['passed']]
    decision=('Promising exploratory candidate(s) under the frozen screen: **'+', '.join(qualifying)+'**. This is not independent confirmation or authorization to deploy/promote a model.' if qualifying else '**Neither S1 nor S2 passes the frozen descriptive screen. No improved reliability procedure was found under the prespecified criteria. Retain the original and previously optimized references unchanged.**')
    selections=[]
    for r in lock['payload']['selections']:
        selections.append({'selector':r['rule'],'fold':r['outer'],'candidate':r['candidate'],'eligible':r['eligible_count'],
                           'inner_R2':r['inner_metrics']['R2'],'inner_macro_MAE':r['inner_metrics']['study_macro_MAE'],
                           'inner_p90_MAE':r['inner_metrics']['p90_study_MAE'],'inner_worst_MAE':r['inner_metrics']['worst_study_MAE']})
    screenrows=[]
    for name in ['S1','S2']:
        screenrows.append({'selector':name,**screens[name]['conditions'],'passes':screens[name]['passed']})
    harmblocks=[]
    for name in ['S1','S2']:
        for ref in [ORIGINAL,OPTIMIZED]:
            part=studies[studies.procedure==name].sort_values('delta_vs_'+ref+'__MAE',ascending=False).head(5)
            harmblocks.append(f'### {name}: five largest MAE increases vs {ref}\n\n'+table(part[['study','rows','fold','MAE','delta_vs_'+ORIGINAL+'__MAE','delta_vs_'+OPTIMIZED+'__MAE']]))
    influence=[]
    for name in PROCEDURES:
        influence.append({'procedure':name,'largest_study_R2':inf[name]['largest_study_R2'],'largest_SSE_fraction':inf[name]['largest_SSE_fraction'],
                          'R2_excluding_largest_descriptive_only':inf[name]['excluding_largest_R2'],
                          'delta_R2_vs_optimized_from_largest':inf[name]['delta_R2_vs_'+OPTIMIZED+'_from_largest'],
                          'delta_R2_vs_optimized_from_remaining':inf[name]['delta_R2_vs_'+OPTIMIZED+'_from_remaining']})
    prepaudit=json.loads((ctx.out/'inner_audit_summary.json').read_text());max_ref=csv(ctx.out/'reference_metric_audit.csv').difference.abs().max()
    lines=[]
    for name in ['S1','S2']:
        lines.append(f"- **{name}**: pooled R² **{t.loc[name,'R2']:.16f}**; Δ original **{t.loc[name,'R2']-.4726308398591973:+.16f}**, Δ optimized **{t.loc[name,'R2']-.5033812847921568:+.16f}**. Screen: **{'PASS' if screens[name]['passed'] else 'FAIL'}**; failed conditions: {', '.join(screens[name]['failed_conditions']) or 'none'}.")
    error_note='The initial test run hit a read-only pandas/NumPy target view in the outer-label perturbation test. The test was repaired to perturb a writable copy and restore the original reference, not mutate source measurements. This was a test-fixture error: no selector, guard threshold, training target or model prediction changed, and no fit was repeated. The initial failure is retained in `preparation_execution.log` and `execution_attempts.jsonl`; the subsequent fresh-kernel run used only saved checkpoints.'
    text=f'''# Stage 4: study-robust selection among the same eight LightGBM settings

**Scenario throughout: `{SCENARIO}`.** Barrows (1,389 rows) and Dehm (81 rows) retain the historical ×10^6 multipliers solely to match the benchmark. These are unverified hypotheses, not certified corrections or verified true synthetic microplastic abundance. No target, study, feature or measurement was changed.

## Result and decision

Immutable original pooled R²: **0.4726308398591973**. Immutable previously optimized pooled R²: **0.5033812847921568**.

{chr(10).join(lines)}

{decision}

Passing this screen does **not** mean every study or reference comparison improved. Relative to the original baseline, both selectors retain higher worst-study MAE and lower within-one-log coverage; S1 also has a slightly higher p90 study MAE. S2 has a higher median study MAE than the previously optimized procedure. Against the optimized reference, {int(harms[(harms.procedure=='S1') & (harms.reference==OPTIMIZED)].studies_worse.iloc[0])} studies worsen under S1 and {int(harms[(harms.procedure=='S2') & (harms.reference==OPTIMIZED)].studies_worse.iloc[0])} under S2. All three negative-R² outer folds remain visible, and both mean-fold R² values remain negative. Neither strategy is automatically selected for deployment.

All four procedures use the **same** 16,988 observations, 72 linked studies, log10(concentration + 0.001) targets and five frozen outer folds. They are not independent test sets. The labels/outcomes have been inspected repeatedly; these are exploratory reused-benchmark comparisons, not independent study generalization or production readiness.

## Complete pooled and study-balanced metrics

{table(summary[['procedure','R2','MAE','RMSE','study_macro_MAE','study_macro_MSE','median_study_MAE','p90_study_MAE','worst_study_MAE','within_1_log_percent','mean_fold_R2']])}

All errors are on the log10 target. R² is not accuracy. The tolerance percentage is the fraction within ±1 log10 unit of **concentration + 0.001**, not an unqualified percentage correct. Pooled R² uses all OOF predictions together; mean fold R² uses five different fold-specific denominators. Study-macro MAE/MSE weight all 72 studies equally; median/p90 use linear-interpolated quantiles.

## Frozen selectors and audit trail

Only the original eight C0–C7 configurations were available. Full parameters remain in the protected `optimization_outputs/candidate_registry.json`; its hash, exact original feature pipeline and seed/thread settings are recorded in each fit identity. No new parameter settings were invented.

**S1 — Tail-first guarded:** eligible iff inner R² ≥ C0 R² − 0.010; macro MAE ≤1.02×C0; p90 MAE ≤C0; worst MAE ≤1.05×C0. Order: lowest p90, lowest macro MAE, highest R², then smallest candidate ID.

**S2 — Balanced guarded:** eligible iff inner R² ≥ C0 R² −0.010; macro MAE ≤1.03×C0; p90 MAE ≤1.03×C0; worst MAE ≤1.10×C0. Keep candidates within 0.010 of the highest eligible inner R², then order by lowest macro MAE, lowest p90, highest R², smallest ID.

C0 is always eligible for valid finite inputs. There is no division by a baseline error: a zero C0 error implies a zero permitted error bound; undefined/nonfinite or negative errors fail safely. Exact comparisons, no outcome-dependent epsilon or threshold adjustment. Stable candidate-ID tie-breaks are tested.

{table(pd.DataFrame(selections))}

`selector_eligibility.csv` records **all 80** selector/fold/candidate cases, each component guard, band eligibility and selection/rejection reason. `locked_selections.json` and its versioned copy `locked_selections_v1.json` were frozen **{lock['created_utc']}**, before any new fit or new outer score. Lock SHA-256: `{sha(ctx.out/'locked_selections.json')}`. Payload SHA-256: `{lock['payload_sha256']}`. The lock is immutable on resume.

### Inner-only provenance verification — zero new inner fits

All **120 original inner prediction checkpoints** were verified using the original source/data/split/configuration/package-version fingerprint plus exact training row IDs and **training-only** target bytes. Their held-out row IDs and finite predictions match the recorded partitions. Reconstructing inner OOF predictions covers every eligible outer-training observation once and every training study; the original 40 candidate summaries agree with maximum absolute metric difference **{prepaudit['max_metric_difference']:.3g}** (tolerance 1e-12). See `inner_cache_audit.csv`, `inner_audit_summary.json`, and `verified_prior_checkpoint_hashes.json`.

The five prior optimized outer checkpoints were also verified against their exact selected configurations and OOF rows. The original default model parameters, original preprocessing code, target/scenario and frozen split identity were verified before baseline reuse. No inner grid was refit, no selection used an outer-test measurement, and the preceding blending report was descriptive context only. The selector module accepts **only** whitelisted inner-summary fields and does not read files or outer OOF outcomes. Perturbing outer-held-out labels leaves selections and the relevant training identity unchanged.

## Exact outer fitting and reuse

{table(csv(ctx.out/'initial_prediction_reuse_audit.csv')[['rule','fold','candidate','route','rows']])}

Actual cumulative new LightGBM calls: **{account['new_fits_started_total']} started, {account['new_fits_completed_total']} completed, {account['failed_fits']} failed**, hard cap **10**. Initial routes: `{json.dumps(account['initial_routes'])}`. Latest notebook routes: `{json.dumps(account['current_routes'])}`. Reference reuse and reuse of an identical fold/configuration checkpoint do not count as fits. These are predictions from the same full outer-training rows, not incompatible models trained on a different subset.

Each fresh call is intercepted at `LGBMRegressor.fit`, budget-checked and logged before execution with configuration, input fingerprint and pre-fit lock hash. Calls outside an authorized locked outer refit fail. The original wrapper fits train-only median imputation, scaling and categorical encoding, then clips only prediction features to training numeric ranges. Training uses features plus training y; prediction receives only test features. No DOI, accession, study label, fold ID or test outcome is a predictor. Actual fitted imputer/scaler/category/range state is saved for auditing; seed 0 and n_jobs=4 are preserved.

Each checkpoint includes complete source/data/fold/scenario/hyperparameter/environment identities, exact train/test/source row IDs, training label/feature hashes and test-feature hashes, plus prediction-file SHA-256. Identical S1/S2 choices share a checkpoint. After Stage B finishes, a persisted marker and fit guard prohibit any new fits on rerun or after scoring; only verified checkpoints/reference OOF can be reused. No deployable all-data model is produced.

## All five outer folds, including negative R²

{table(folds[['fold','procedure','rows','studies','R2','MAE','RMSE']])}

Test sizes remain **6,135 / 2,714 / 2,712 / 2,714 / 2,713**. Each row appears exactly once per procedure; no NaNs or difficult studies are dropped. Negative individual-fold R² and negative pooled R² are different quantities; neither is hidden or clipped.

## Frozen descriptive screening rule

A new selector qualifies only if pooled R² ≥ **0.4883812847921568**, study-macro MAE and p90 study MAE are no worse than the optimized reference, and worst-study MAE is no worse. Pooled MAE, tolerance coverage and mean fold R² are still reported even when they deteriorate. This screen is applied only **after** locked prediction generation; it cannot reselect a configuration.

{table(pd.DataFrame(screenrows))}

A pass would be a promising exploratory candidate only, never independent confirmation or permission to overwrite a reference. A failure does not trigger another rule or search.

## Paired study-level harms

{table(harms.drop(columns='scenario'))}

“Worse” means any positive MAE difference; “worse over 0.10” uses the fixed descriptive 0.10-log10 threshold. Equal predictions from reused configurations can produce exact ties. All study-level deltas versus **both** references are in `paired_study_metrics.csv`; all 72 studies remain. Within-study R² is not needed for these errors, including the smallest groups (minimum {int(ctx.df.groupby('study').size().min())} rows).

{chr(10).join(harmblocks)}

These study identities and errors are descriptive, never routing features or selection inputs. The worst harmed studies remain visible regardless of pooled-score changes.

## Method-family and study-size analyses

{table(groups[['grouping','group','procedure','rows','studies','R2','MAE','study_macro_MAE','p90_study_MAE','within_1_log']])}

Original method families are surface net, pump and bulk water. Study-size bins remain 1–50, 51–200, 201–1000 and 1001+ eligible observations. Each grouping covers all 16,988 rows per procedure; a study can span method families. No target-derived strata or selective exclusions.

## Largest-study influence

`{inf['study']}` contains **{inf['rows']} rows ({100*inf['row_fraction']:.2f}%)**, constituting outer fold 1. Its dominance affects pooled sample-weighted metrics but counts once in study-macro metrics.

{table(pd.DataFrame(influence))}

The last two columns decompose each global ΔR² versus optimized into squared-error changes on the largest study and on the rest, using the unchanged global variance denominator. They sum to the overall ΔR². The largest study contributes {100*inf['S1']['delta_R2_vs_'+OPTIMIZED+'_from_largest']/(t.loc['S1','R2']-t.loc[OPTIMIZED,'R2']):.2f}% of S1's improvement and {100*inf['S2']['delta_R2_vs_'+OPTIMIZED+'_from_largest']/(t.loc['S2','R2']-t.loc[OPTIMIZED,'R2']):.2f}% of S2's improvement—less than its 36.11% row share. The gains are therefore not solely an improvement on this dominant study; the imbalance still strongly affects the overall pooled benchmark. Excluding-largest R² is **only a descriptive influence calculation on existing predictions**, with a different denominator/population, not the benchmark or a refit. Primary results retain every row.

## Execution, tests, preservation and warnings

- **{test_status['passed']} unit/integration tests passed**, **{test_status['data_provenance_assertions_passed']} data/provenance assertions passed**; no new fits in tests. Tests verify exact row/target/scenario alignment; all inner studies counted; S1/S2 arithmetic, boundaries, zero-error behavior, C0 fallback and ties; deterministic selection with shuffled input; rejection of outer-data columns; no selector file I/O; outer-label perturbation; train-only actual fitted preprocessing; cache identity/corruption checks; reference reuse; fit-cap and no-post-score-fit guards; independent aggregate, fold and study metrics; and complete protected-file equality.
- **{cell_count} code cells executed** in a fresh kernel. The executed notebook and CLI call the same importable modules. `EXECUTION_STATUS.json` contains actual cell results, tests, fit counts and file hashes. `tests.txt` contains actual test output. Maximum frozen-reference aggregate discrepancy: **{max_ref:.3g}**.
- Model-fit warnings captured: **{len(account['warnings'])}** (`fit_warnings.json`). Fit failures: **{account['failed_fits']}**. Runtime/Jupyter warnings, if any, remain in execution logs and are separate from model warnings. {error_note}
- All original source, raw dataset, models/notebooks, `.git/`, and **all three previous output directories** remain byte-identical. Complete before/after read-only hash manifests and `integrity_result.json` prove preservation; symlinks are recorded by target without following external interpreter links. The Stage 4 prompt itself is protected.
- Existing reproducible environment reused, no downloads: Python {ctx.environment['python'].split()[0]}, {ctx.environment['platform']}, packages `{json.dumps(ctx.environment['packages'],sort_keys=True)}`. Four learner threads; no GPU, new features, new architecture, new hyperparameters, chronological experiment, blending, meta-model, CatBoost/XGBoost search or later phase.

## Reproduction commands (repository root)

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python study_robustness_outputs/source/run.py --prepare
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python study_robustness_outputs/source/run.py
```

The first executes the locked experiment/notebook/tests but defers the final report. The second reruns only read-only audits, cached predictions, evaluation and tests, then writes the final report **last**. The persisted completion marker prevents new fitting on a completed experiment. No historical experiment is restarted.

## Files and recommended scientific next step

- `Study_Robustness_EXECUTED.ipynb`, `locked_selections.json`, `locked_selections_v1.json`, `selector_eligibility.csv`, `selected_candidates.csv`.
- `all_procedure_oof_predictions.csv`, `summary_metrics.csv`, `paired_fold_metrics.csv`, `paired_study_metrics.csv`, `method_and_study_size_metrics.csv`, `study_harm_counts.csv`, `largest_study_influence.json`, `screening.json`.
- `model_fit_log.jsonl`, `prediction_reuse_audit.csv`, `initial_prediction_reuse_audit.csv`, `checkpoints/`, `EXECUTION_STATUS.json`, `tests.txt`, protected-file manifests, `environment.json`.

Recommended next **scientific** step, not executed here: independently resolve concentration-unit and sampling-method provenance (especially the hypothetical Barrows/Dehm multipliers and unverified extreme studies), then preregister evaluation on genuinely unseen studies with verified methods. Do not treat repeated tuning of these already inspected five folds as fresh evidence of generalization.

**Stop. Both immutable references are preserved; no production model is selected and no additional phase is launched.**
'''
    return text
