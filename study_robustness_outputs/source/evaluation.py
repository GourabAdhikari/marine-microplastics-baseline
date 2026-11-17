"""Paired post-lock evaluation; never calls selectors or model fitting."""
import json
import numpy as np
import pandas as pd
from common import SCENARIO,ORIGINAL,OPTIMIZED,PROCEDURES,METRIC_NAMES,csv,metrics,dump,require


def add_deltas(table,keys,metric_names):
    table=table.copy()
    for ref in [ORIGINAL,OPTIMIZED]:
        reference=table[table.procedure==ref]
        for k in metric_names:
            if not keys:baseline=reference[k].iloc[0]
            else:
                index=pd.MultiIndex.from_frame(table[keys]) if len(keys)>1 else table[keys[0]]
                baseline=reference.set_index(keys)[k].reindex(index).to_numpy()
            table['delta_vs_'+ref+'__'+k]=table[k]-baseline
    require(not table.isna().any().any(),'Missing paired metric: refusing silent alignment loss')
    return table


def screening(candidate,optimized):
    checks={'pooled_R2_at_least_0.4883812847921568':candidate['R2']>=.5033812847921568-.015,
            'study_macro_MAE_no_worse':candidate['study_macro_MAE']<=optimized['study_macro_MAE'],
            'p90_study_MAE_no_worse':candidate['p90_study_MAE']<=optimized['p90_study_MAE'],
            'worst_study_MAE_no_worse':candidate['worst_study_MAE']<=optimized['worst_study_MAE']}
    return {'passed':all(checks.values()),'conditions':{k:bool(v) for k,v in checks.items()},'failed_conditions':[k for k,v in checks.items() if not v],
            'interpretation':'Descriptive screen on reused outcomes only; no reselection or promotion authorized.'}


def evaluate(ctx):
    d=csv(ctx.out/'all_procedure_oof_predictions.csv')
    require(len(d)==16988 and d.water_row.is_unique and d.source_row.is_unique and d.study.nunique()==72,'Incomplete/duplicate OOF')
    for col in ['water_row','source_row','study','fold','target_log10']:
        require(np.array_equal(d[col],ctx.base[col]),'Changed common OOF field '+col)
    require(d.scenario.eq(SCENARIO).all() and np.isfinite(d[PROCEDURES].to_numpy()).all(),'Nonfinite OOF or wrong scenario')
    require(np.array_equal(d[ORIGINAL],ctx.base.prediction_log10) and np.array_equal(d[OPTIMIZED],ctx.opt.optimized_prediction_log10),'Frozen reference predictions changed')
    rows=[]
    for name in PROCEDURES:
        rows.append({'procedure':name,'rows':len(d),'studies':d.study.nunique(),'scenario':SCENARIO,**metrics(d.target_log10,d[name],d.study),
                     'mean_fold_R2':float(np.mean([metrics(g.target_log10,g[name],g.study)['R2'] for _,g in d.groupby('fold')]))})
    summary=add_deltas(pd.DataFrame(rows),[],METRIC_NAMES+['mean_fold_R2'])
    summary['within_1_log_percent']=100*summary.within_1_log
    summary.to_csv(ctx.out/'summary_metrics.csv',index=False)
    ref=json.loads((ctx.root/'optimization_outputs/optimization_summary.json').read_text());audit=[]
    for name,old in [(ORIGINAL,'original_reproduced_fixed'),(OPTIMIZED,'nested_optimized_procedure')]:
        row=summary.set_index('procedure').loc[name]
        for k,value in ref[old].items():
            require(np.isclose(row[k],value,rtol=1e-12,atol=1e-12),'Frozen reference metric mismatch '+name+'/'+k)
            audit.append({'procedure':name,'metric':k,'recomputed':row[k],'prior':value,'difference':row[k]-value})
    pd.DataFrame(audit).to_csv(ctx.out/'reference_metric_audit.csv',index=False)
    folds=[];studies=[];subgroups=[]
    for f,g in d.groupby('fold'):
        for name in PROCEDURES:folds.append({'fold':f,'procedure':name,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO,**metrics(g.target_log10,g[name],g.study)})
    for s,g in d.groupby('study'):
        for name in PROCEDURES:
            e=g[name].to_numpy()-g.target_log10.to_numpy()
            studies.append({'study':s,'fold':int(g.fold.iloc[0]),'procedure':name,'rows':len(g),'method_families':'; '.join(sorted(g.method_family.unique())),
                            'scenario':SCENARIO,'MAE':float(abs(e).mean()),'MSE':float(np.mean(e**2)),'RMSE':float(np.sqrt(np.mean(e**2))),'within_1_log':float((abs(e)<=1).mean())})
    for kind in ['method_family','study_size_bin']:
        for value,g in d.groupby(kind):
            for name in PROCEDURES:subgroups.append({'grouping':kind,'group':value,'procedure':name,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO,**metrics(g.target_log10,g[name],g.study)})
    fold_table=add_deltas(pd.DataFrame(folds),['fold'],METRIC_NAMES);study_table=add_deltas(pd.DataFrame(studies),['study'],['MAE','MSE','RMSE','within_1_log'])
    group_table=add_deltas(pd.DataFrame(subgroups),['grouping','group'],METRIC_NAMES)
    fold_table.to_csv(ctx.out/'paired_fold_metrics.csv',index=False);study_table.to_csv(ctx.out/'paired_study_metrics.csv',index=False)
    group_table.to_csv(ctx.out/'method_and_study_size_metrics.csv',index=False)
    harms=[]
    for name in ['S1','S2']:
        part=study_table[study_table.procedure==name]
        for reference in [ORIGINAL,OPTIMIZED]:
            delta=part['delta_vs_'+reference+'__MAE']
            harms.append({'procedure':name,'reference':reference,'studies_worse':int((delta>0).sum()),'studies_worse_over_0_10':int((delta>.10).sum()),
                          'studies_better':int((delta<0).sum()),'studies_equal':int((delta==0).sum()),'scenario':SCENARIO})
    pd.DataFrame(harms).to_csv(ctx.out/'study_harm_counts.csv',index=False)
    t=summary.set_index('procedure');screens={s:screening(t.loc[s],t.loc[OPTIMIZED]) for s in ['S1','S2']}
    dump(ctx.out/'screening.json',screens)
    largest=d.study.value_counts().index[0];mask=d.study==largest;sst=float(((d.target_log10-d.target_log10.mean())**2).sum())
    influence={'scenario':SCENARIO,'study':largest,'rows':int(mask.sum()),'row_fraction':float(mask.mean()),'diagnostic_only':'No exclusion from primary benchmark and no refitting.'}
    for name in PROCEDURES:
        se=(d[name]-d.target_log10)**2
        influence[name]={'total_SSE':float(se.sum()),'largest_SSE_fraction':float(se[mask].sum()/se.sum()),
                         'largest_study_R2':metrics(d.loc[mask,'target_log10'],d.loc[mask,name],d.loc[mask,'study'])['R2'],
                         'excluding_largest_R2':metrics(d.loc[~mask,'target_log10'],d.loc[~mask,name],d.loc[~mask,'study'])['R2']}
        for reference in [ORIGINAL,OPTIMIZED]:
            improvement=(d[reference]-d.target_log10)**2-se
            influence[name]['delta_R2_vs_'+reference+'_from_largest']=float(improvement[mask].sum()/sst)
            influence[name]['delta_R2_vs_'+reference+'_from_remaining']=float(improvement[~mask].sum()/sst)
    dump(ctx.out/'largest_study_influence.json',influence)
    ctx.summary=summary;ctx.screens=screens
    return summary
