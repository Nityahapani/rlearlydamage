"""Conservative interim-look sensitivity for the previously completed n=32 predictor study.

Post-hoc only. Reuses the exact NumPy predictor function definitions from ponr_predictors.py without
importing its plotting stack. No JAX and no Phase 12 in-progress outcomes are loaded.
Computes all four pre-specified predictors at n=32 and after excluding 330; Holm across four, then ×2
for the two unblinded looks. The combined exclusion + two-look result qualifies, but never replaces,
the originally registered full-sample decision.
"""
from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from posthoc_p12_audit import holm

PRED=['kl_clean','cka_clean','move_gap_rel','value_err_clean']


def pure_predictor_functions():
    path=Path('scripts/ponr_predictors.py')
    tree=ast.parse(path.read_text())
    names={'load_params','seed_slice','relu','actor','critic','log_softmax','linear_cka','predictors'}
    ns={'np':np}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(funcs)==len(names)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),ns)
    return ns


def main():
    pp=pure_predictor_functions()
    files=sorted(glob.glob('results/summary/*p9-ponr-confirm2-*.json'))
    rows=[]
    for f in files:
        s=json.loads(Path(f).read_text())
        assert s['onset']==100 and 300 in s['durations'] and s['recovery']==250
        E=s['protocol']['eval_every'];K=s['recovery']//E
        C=np.asarray(s['C']['eval_return_mean'],float)
        D=np.asarray(s['arms']['300']['D_online']['eval_return_mean'],float)
        A=(D[:,1:]-C[:,1:K+1]).mean(1)
        root=Path('results')/Path(f).stem
        bank=np.load(root/'bank.npz')['bank']
        pC=pp['load_params'](root/'C_u400.npz');pD=pp['load_params'](root/'D_u400.npz')
        n_act=pC['Dense_0']['kernel'].shape[-1]
        masked=[a for a in range(n_act) if a not in s['protocol']['allowed']]
        for i,seed in enumerate(s['seeds']):
            pred=pp['predictors'](pp['seed_slice'](pD,i),pp['seed_slice'](pC,i),bank[i],masked)
            rows.append({'seed':seed,'A_300':float(A[i]),**pred})
        print('Loaded completed Phase 9 chunk',s['seeds'],flush=True)
    rows.sort(key=lambda r:r['seed'])
    assert [r['seed'] for r in rows]==list(range(300,332))
    results={}
    for label,rs in [('all32',rows),('without330',[r for r in rows if r['seed']!=330])]:
        Y=np.array([r['A_300'] for r in rs]);n=len(rs)
        vals=[]
        for k in PRED:
            r=float(stats.spearmanr([row[k] for row in rs],Y).statistic)
            t=r*np.sqrt((n-2)/max(1e-12,1-r*r))
            p=float(stats.t.cdf(t,n-2))
            vals.append({'predictor':k,'rho':r,'p_one_sided':p})
        h=holm([v['p_one_sided'] for v in vals])
        for v,p in zip(vals,h):
            v['Holm_four']=p
            v['Holm_four_times_two_looks']=min(1,2*p)
        results[label]={'n':n,'predictors':vals}
    kl=results['all32']['predictors'][0]
    assert abs(kl['rho']-(-.48))<.01
    assert abs(kl['Holm_four']-.0103)<.00006, 'Must reproduce existing primary report'
    L=['# Interim-look sensitivity of the completed Phase 9 predictor study (post-hoc)', '',
       'The n=32 primary is unchanged. Exclusion of seed 330 is post-hoc; the combined exclusion + two-look adjustment is a stricter sensitivity, not a replacement decision.', '',
       '| sample | n | KL rho | one-sided p | Holm over four | ×2 looks | clears combined 0.05 threshold? |',
       '|---|---|---|---|---|---|---|']
    for label,r in results.items():
        v=r['predictors'][0]
        L.append(f"| {label} | {r['n']} | {v['rho']:+.4f} | {v['p_one_sided']:.6f} | {v['Holm_four']:.6f} | "
                 f"{v['Holm_four_times_two_looks']:.6f} | {v['Holm_four_times_two_looks']<.05} |")
    out={'status':'post-hoc sensitivity; original full-sample registered decision unchanged',
         'source_function_file':'scripts/ponr_predictors.py','source_summaries':files,
         'results':results,'per_seed':rows}
    S=Path('results/summary')
    (S/'p12_interim_predictors-posthoc.json').write_text(json.dumps(out,indent=2)+'\n')
    (S/'p12_interim_predictors-posthoc.md').write_text('\n'.join(L)+'\n')
    print('\n'.join(L),flush=True)


if __name__=='__main__':
    main()
