"""Frozen prospective comparison for Phase 13 D. `prepare` uses only old seeds 300–331.
`evaluate` refuses incomplete validation cohorts; no test-set fitting or model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from posthoc_p12_interim import pure_predictor_functions

MODELS={'training_mean':[], 'acute_gap':['acute_gap'], 'removal_entropy':['entropy'],
        'KL':['KL'], 'acute_gap_plus_KL':['acute_gap','KL'], 'removal_score':['removal_score']}


def extract(files):
    pp=pure_predictor_functions();rows=[]
    for path in files:
        d=json.loads(Path(path).read_text());n=len(d['seeds'])
        if d.get('prospective_features'):
            rows.extend(d['prospective_features']);continue
        assert d['onset']==100 and d['durations']==[300] and d['recovery']==250
        D=np.asarray(d['arms']['300']['D_online']['eval_return_mean'],float)
        C=np.asarray(d['C']['eval_return_mean'],float)
        k=d['recovery']//d['protocol']['eval_every']
        Y=(D[:,1:]-C[:,1:k+1]).mean(1)
        C_at=np.asarray(d['C']['eval_return_mean'])[:,d['C']['updates'].index(400)]
        root=Path('results')/Path(path).stem
        bank=np.load(root/'bank.npz')['bank']
        CD=pp['load_params'](root/'D_u400.npz');CC=pp['load_params'](root/'C_u400.npz')
        for i,seed in enumerate(d['seeds']):
            pd,pc=pp['seed_slice'](CD,i),pp['seed_slice'](CC,i)
            _,ld=pp['actor'](pd,bank[i].astype(np.float64));_,lc=pp['actor'](pc,bank[i].astype(np.float64))
            lpD,lpC=pp['log_softmax'](ld),pp['log_softmax'](lc)
            rows.append(dict(seed=seed,A_300=float(Y[i]),acute_gap=float(C_at[i]-D[i,0]),removal_score=float(D[i,0]),
                             entropy=float(-np.sum(np.exp(lpD)*lpD,axis=1).mean()),
                             KL=float(np.sum(np.exp(lpC)*(lpC-lpD),axis=1).mean())))
    rows.sort(key=lambda r:r['seed']);assert len({r['seed'] for r in rows})==len(rows)
    return rows


def design(rows,features,mu=None,sd=None):
    if not features:return np.ones((len(rows),1)),[],[]
    X=np.array([[r[k] for k in features] for r in rows],float)
    mu=X.mean(0) if mu is None else np.asarray(mu)
    sd=X.std(0) if sd is None else np.asarray(sd);sd=np.maximum(sd,1e-12)
    return np.column_stack([np.ones(len(rows)),(X-mu)/sd]),mu.tolist(),sd.tolist()


def prepare():
    files=sorted(Path('results/summary').glob('*p9-ponr-confirm2-*.json'))
    rows=extract(files);assert [r['seed'] for r in rows]==list(range(300,332))
    Y=np.array([r['A_300'] for r in rows]);models={}
    for name,features in MODELS.items():
        X,mu,sd=design(rows,features)
        models[name]=dict(features=features,mean=mu,sd=sd,coefficients=np.linalg.lstsq(X,Y,rcond=None)[0].tolist())
    payload=dict(status='FROZEN before all held-out seeds 900–915',training_seeds=list(range(300,332)),
                 validation_seeds=list(range(900,916)),training_rows=rows,models=models,
                 primary_comparison=['acute_gap','acute_gap_plus_KL'],primary_metric='paired held-out squared-error reduction',
                 practical_gain_threshold=.10,source_files=[str(f) for f in files],
                 source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                 score_only_baseline='removal_score is an additional pre-validation secondary baseline; no oracle C required')
    out=Path('prereg/p13_frozen_predictors.json');assert not out.exists(),'Do not refit frozen coefficients'
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print('Frozen',len(models),'OLS baselines on old 32 seeds; no validation data read')
    print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest())


def evaluate():
    frozen=Path('prereg/p13_frozen_predictors.json');F=json.loads(frozen.read_text())
    files=sorted(Path('results/summary').glob('*p13-D-s*-*.json'))
    full=[];seen=[]
    for f in files:
        d=json.loads(f.read_text())
        if 'D_online' in d.get('arms',{}).get('300',{}):full.append(f);seen+=d['seeds']
    assert set(seen)==set(range(900,916)) and len(seen)==16,'Hold primary until all 16 validation seeds finish'
    rows=extract(full);Y=np.array([r['A_300'] for r in rows]);pred={};loss={}
    for name,m in F['models'].items():
        X,_,_=design(rows,m['features'],m['mean'],m['sd']);pred[name]=X@np.array(m['coefficients']);loss[name]=(pred[name]-Y)**2
    delta=loss['acute_gap']-loss['acute_gap_plus_KL']
    gain=float(1-loss['acute_gap_plus_KL'].mean()/loss['acute_gap'].mean())
    se=delta.std(ddof=1)/4;h=stats.t.ppf(.975,15)*se
    p=float(stats.ttest_1samp(delta,0,alternative='greater').pvalue)
    res=dict(n=16,validation_rows=rows,frozen_model_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
             source_files=[str(f) for f in full],MSE={m:float(v.mean()) for m,v in loss.items()},
             predictions={m:v.tolist() for m,v in pred.items()},losses={m:v.tolist() for m,v in loss.items()},
             primary_loss_difference=[float(delta.mean()),float(delta.mean()-h),float(delta.mean()+h)],
             primary_one_sided_p=p,primary_gain=gain,practical_threshold=.10,
             rank={m:float(stats.spearmanr(pred[m],Y).statistic) if np.std(pred[m])>0 else None for m in pred},
             feature_rank={k:float(stats.spearmanr([r[k] for r in rows],Y).statistic) for k in ['KL','acute_gap','entropy','removal_score']},
             status='Independent prospective cohort; global Phase 13 Holm applied separately; not new-task or causal validation')
    out=Path('results/summary/p13_D_prospective-analysis.json');out.write_text(json.dumps(res,indent=2)+'\n')
    L=['# Phase 13 D — frozen prospective diagnostic comparison','',res['status'],'',
       '| frozen model | held-out MSE | rank correlation (secondary) |','|---|---|---|']
    for m in pred:L.append(f"| {m} | {res['MSE'][m]:.3f} | {res['rank'][m]} |")
    L+=['',f'Primary gain vs acute gap: {100*gain:+.1f}%; paired loss difference {res["primary_loss_difference"]}; one-sided p={p:.6g}.',
        'Practical gain threshold = 10%, combined with global Phase 13 multiplicity requirement. No coefficients were fitted to this cohort.']
    Path('results/summary/p13_D_prospective-analysis.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','evaluate']);a=ap.parse_args()
    prepare() if a.mode=='prepare' else evaluate()
