"""Pure inner-summary selectors. No file I/O, fit, outer label or OOF argument."""
import numpy as np
import pandas as pd
from common import SCENARIO, require

FIELDS=['outer','candidate','scenario','R2','study_macro_MAE','p90_study_MAE','worst_study_MAE']
RULES={
    'S1':{'name':'Tail-first guarded','r2_slack':.010,'macro_factor':1.02,'p90_factor':1.,'worst_factor':1.05,
          'order':['p90_study_MAE','study_macro_MAE','R2 descending','candidate ascending']},
    'S2':{'name':'Balanced guarded','r2_slack':.010,'macro_factor':1.03,'p90_factor':1.03,'worst_factor':1.10,
          'best_eligible_r2_band':.010,'order':['study_macro_MAE','p90_study_MAE','R2 descending','candidate ascending']}}


def select_fold(inner,rule):
    require(rule in RULES,'Only S1 and S2 authorized')
    require(set(inner.columns)==set(FIELDS),'Selector accepts only whitelisted inner-summary fields; outer outcomes forbidden')
    require(len(inner)==8 and inner.candidate.is_unique and inner.outer.nunique()==1,'Expected eight unique candidates for one outer fold')
    require(sorted(c.split('_')[0] for c in inner.candidate)==[f'C{i}' for i in range(8)],'Frozen candidate IDs C0-C7 required')
    require(inner.scenario.eq(SCENARIO).all(),'Wrong selector scenario')
    columns=FIELDS[3:]
    require(np.isfinite(inner[columns].to_numpy()).all(),'Missing or undefined inner metrics: selector not executable')
    require((inner[columns[1:]]>=0).all().all(),'MAE metrics cannot be negative')
    baseline=inner.loc[inner.candidate=='C0_original']
    require(len(baseline)==1,'Missing original C0 reference')
    b=baseline.iloc[0];r=RULES[rule];t=inner.copy()
    # Multiplication rather than ratios handles zero C0 errors safely: only zero error meets that bound.
    t['r2_pass']=t.R2>=b.R2-r['r2_slack']
    t['macro_pass']=t.study_macro_MAE<=r['macro_factor']*b.study_macro_MAE
    t['p90_pass']=t.p90_study_MAE<=r['p90_factor']*b.p90_study_MAE
    t['worst_pass']=t.worst_study_MAE<=r['worst_factor']*b.worst_study_MAE
    t['eligible']=t[['r2_pass','macro_pass','p90_pass','worst_pass']].all(axis=1)
    require(bool(t.loc[t.candidate=='C0_original','eligible'].iloc[0]),'Finite C0 must be eligible by construction')
    t['best_r2_band_pass']=True
    if rule=='S2':
        best=t.loc[t.eligible,'R2'].max()
        t['best_r2_band_pass']=t.R2>=best-r['best_eligible_r2_band']
    contenders=t[t.eligible & t.best_r2_band_pass]
    order=['p90_study_MAE','study_macro_MAE','R2','candidate'] if rule=='S1' else ['study_macro_MAE','p90_study_MAE','R2','candidate']
    chosen=contenders.sort_values(order,ascending=[True,True,False,True],kind='stable').iloc[0]
    t['rule']=rule;t['selected']=t.candidate.eq(chosen.candidate)
    def reason(row):
        failed=[c for c in ['r2_pass','macro_pass','p90_pass','worst_pass'] if not row[c]]
        return 'ineligible: '+','.join(failed) if failed else ('eligible; outside best-R2 band' if not row.best_r2_band_pass else ('selected by frozen lexicographic order' if row.selected else 'eligible; loses frozen lexicographic order'))
    t['reason']=t.apply(reason,axis=1)
    selection={'rule':rule,'outer':int(b.outer),'candidate':chosen.candidate,'scenario':SCENARIO,
               'inner_metrics':{c:float(chosen[c]) for c in columns},'C0_inner_reference':{c:float(b[c]) for c in columns},
               'eligible_count':int(t.eligible.sum()),'contender_count':len(contenders),'reason':' / '.join(RULES[rule]['order'])}
    return selection,t


def select_all(inner):
    require(set(inner.columns)==set(FIELDS),'Nonwhitelisted selector input')
    require(len(inner)==40 and set(inner.outer)==set(range(1,6)),'All 40 saved inner summaries required')
    selections=[];eligibility=[]
    for rule in RULES:
        for fold in range(1,6):
            selection,table=select_fold(inner[inner.outer==fold],rule)
            selections.append(selection);eligibility.append(table)
    return selections,pd.concat(eligibility,ignore_index=True)
