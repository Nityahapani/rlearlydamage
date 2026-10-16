"""One fixed Phase-14 claim, n=32 only; no setting/endpoint selection or interim primary inference.

The primary is an intersection-union of four practical-effect hypotheses at alpha .01, plus
prespecified seedwise consistency. Robustness outputs never rescue a failed primary.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

S=Path('results/summary')


def tci(x,level=.95):
    x=np.asarray(x,float);assert np.isfinite(x).all()
    m=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));h=float(stats.t.ppf((1+level)/2,len(x)-1)*se)
    return [m,m-h,m+h]


def directional(x,shift,direction):
    z=np.asarray(x)+shift
    if np.std(z)==0:return 0. if (z.mean()<0 if direction=='less' else z.mean()>0) else 1.
    return float(stats.ttest_1samp(z,0,alternative=direction).pvalue)


def tost(x,delta):
    return max(directional(x,delta,'greater'),directional(x,-delta,'less'))


def bootstrap_ci(x,level=.95,reps=20000):
    x=np.asarray(x,float);r=np.random.default_rng(140032);ix=r.integers(0,len(x),(reps,len(x)))
    v=x[ix].mean(1);q=(1-level)/2
    return [float(np.quantile(v,q)),float(np.quantile(v,1-q))]


def select_complete():
    p=json.loads(Path('prereg/phase14.json').read_text());wanted=p['training_seeds']
    bygroup={}
    for f in sorted(S.glob('*p14-actor-repair-s*-Asterix-MinAtar-*.json')):
        d=json.loads(f.read_text())
        if d['protocol'].get('smoke'):continue
        group=tuple(d['seeds']);assert set(group)<=set(wanted)
        if group in bygroup:
            f0,old=bygroup[group]
            assert old['state_fingerprints']==d['state_fingerprints'],'Resume cannot change donor states'
            for k in set(old.get('cells',{}))&set(d.get('cells',{})):
                assert old['cells'][k]==d['cells'][k],f'Conflicting cell {k}'
            if len(old.get('cells',{}))>len(d.get('cells',{})):continue
        bygroup[group]=(f,d)
    full=[];seen=[]
    expected={'/'.join(c[k] for k in ['reference','context','opt','code']) for c in p['cells']}
    for group,(f,d) in bygroup.items():
        if d.get('status')=='complete' and set(d['cells'])==expected:full.append((f,d));seen+=list(group)
    if sorted(seen)!=wanted or len(seen)!=p['n']:
        raise SystemExit(f'Primary WITHHELD: {len(seen)} complete seeds of fixed {p["n"]}; no interim result computed')
    assert len(seen)==len(set(seen));full.sort(key=lambda t:t[1]['seeds'][0])
    return p,full


def main():
    P,files=select_complete();ds=[d for _,d in files];seeds=sum((d['seeds'] for d in ds),[])
    assert all(d['protocol']==ds[0]['protocol'] for d in ds), 'All chunks must use identical scientific protocol'
    assert all(d['checks']['DDD_equals_untouched_natural_D'] and d['checks']['CCC_equals_untouched_natural_C'] for d in ds)
    names=list(ds[0]['cells'])
    M={k:np.concatenate([np.asarray(d['cells'][k]['eval_return_mean'],float) for d in ds]) for k in names}
    alt={k:np.concatenate([np.asarray(d['cells'][k]['eval_alt_return_mean'],float) for d in ds]) for k in P['second_eval_cells']}
    def historical(key='primary'):
        rows=[]
        for d in ds:
            fk=d['fork_eval'] if key=='primary' else d['fork_eval_alt']
            bank={u:np.asarray(v) for u,v in zip(fk['updates'],fk['C'])}
            cc=np.asarray(d['cells']['fork/clean/donor/CCC']['eval_return_mean' if key=='primary' else 'eval_alt_return_mean'])
            bank.update({u:cc[:,i] for i,u in enumerate(d['eval_updates'])})
            L=d['protocol']['window'][1]-d['protocol']['window'][0]
            rows.extend(np.stack([bank[u-L] for u in d['eval_updates']],axis=1))
        return np.asarray(rows)
    pause=historical();pause2=historical('alt')
    A=lambda X:X[:,1:].mean(1)
    nat=M['pause/damaged/donor/DDD'];resc=M['pause/damaged/donor/CDD'];ref=M['pause/damaged/donor/CCC'];bad=M['pause/damaged/donor/DCC']
    primary={'native_AUC_harm':A(nat)-A(pause),'native_late_harm':(nat[:,-5:]-pause[:,-5:]).mean(1),
             'onset_actor_rescue':A(resc)-A(nat),'reciprocal_actor_transfer':A(bad)-A(ref)}
    ps={c['name']:directional(primary[c['name']],c['shift'],c['direction']) for c in P['primary']['contrasts']}
    consistency=dict(native_AUC_negative=int((primary['native_AUC_harm']<0).sum()),actor_rescue_positive=int((primary['onset_actor_rescue']>0).sum()))
    success=max(ps.values())<.01 and min(consistency.values())>=24
    q=float(stats.t.ppf(1-.05/4,31));sim={}
    for c in P['primary']['contrasts']:
        x=primary[c['name']];se=x.std(ddof=1)/np.sqrt(32)
        sim[c['name']]=dict(one_sided_95_family_bound=float(x.mean()+(q*se if c['direction']=='less' else -q*se)),direction=c['direction'])
    z={c:M['pause/damaged/zero/'+c] for c in ['CCC','CDC','DCC','DDC']}
    za=(A(z['DCC'])+A(z['DDC'])-A(z['CCC'])-A(z['CDC']))/2
    zc=(A(z['CDC'])+A(z['DDC'])-A(z['CCC'])-A(z['DCC']))/2
    zero_rescue=A(z['CDC'])-A(z['DDC'])
    reset_rescue=A(M['pause/reset/donor/CDD'])-A(M['pause/reset/donor/DDD'])
    residual=A(resc)-A(ref);residual_hist=A(resc)-A(pause)
    state_hist=A(ref)-A(pause)
    a_nat=alt['pause/damaged/donor/DDD'];a_resc=alt['pause/damaged/donor/CDD']
    alt_metrics=dict(native_AUC_harm=tci(A(a_nat)-A(pause2)),native_late_harm=tci((a_nat[:,-5:]-pause2[:,-5:]).mean(1)),onset_actor_rescue=tci(A(a_resc)-A(a_nat)))
    bootR=np.random.default_rng(140033);ix=bootR.integers(0,32,(20000,32))
    ratios=primary['onset_actor_rescue'][ix].mean(1)/(-primary['native_AUC_harm'][ix].mean(1))
    ratio=float(primary['onset_actor_rescue'].mean()/(-primary['native_AUC_harm'].mean()))
    result=dict(status='fixed n=32 complete; one registered claim, all robustness retained',n=32,seeds=seeds,
                prereg_sha256=hashlib.sha256(Path('prereg/phase14.json').read_bytes()).hexdigest(),
                multiplicity_scope='one fixed Phase14 compound at alpha .01; not retrospective paper-wide control',
                summaries=[str(f) for f,_ in files],source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f,_ in files},
                primary_p_values=ps,primary_compound_p=max(ps.values()),alpha=.01,primary_success=bool(success),seed_consistency=consistency,
                primary={k:dict(CI95=tci(v),CI99=tci(v,.99),bootstrap95=bootstrap_ci(v),per_seed=v.tolist()) for k,v in primary.items()},
                simultaneous_one_sided_bounds=sim,
                repaired_fraction_point=ratio,repaired_fraction_bootstrap95=[float(v) for v in np.quantile(ratios,[.025,.975])],
                residual_vs_statepause=dict(CI95=tci(residual),TOST_pm_030=tost(residual,.30)),
                residual_vs_historicalpause=dict(CI95=tci(residual_hist),TOST_pm_030=tost(residual_hist,.30)),
                fresh_Adam=dict(actor_main=tci(za),critic_main=tci(zc),actor_rescue=tci(zero_rescue),
                                actor_in_onset_critic=tci(A(z['DCC'])-A(z['CCC'])),
                                actor_in_damaged_critic=tci(A(z['DDC'])-A(z['CDC'])),
                                critic_in_onset_actor=tci(A(z['CDC'])-A(z['CCC'])),
                                critic_in_damaged_actor=tci(A(z['DDC'])-A(z['DCC'])),
                                total=tci(A(z['DDC'])-A(z['CCC']))),
                statepause_minus_historical=dict(CI95=tci(state_hist),TOST_pm_030=tost(state_hist,.30)),
                reset_minus_natural=dict(native_AUC=tci(A(M['pause/reset/donor/DDD'])-A(nat)),
                                         actor_rescue=tci(reset_rescue-primary['onset_actor_rescue'])),
                reset_context_actor_rescue=tci(reset_rescue),independent_eval_key=54321,alternate_evaluation=alt_metrics,
                sign_tests={k:dict(negative=int((x<0).sum()),positive=int((x>0).sum()),p_two_sided=float(stats.binomtest(int((x>0).sum()),int((x!=0).sum()),.5).pvalue) if (x!=0).sum() else 1.) for k,x in primary.items()},
                cell_AUC={k:tci(A(v)) for k,v in M.items()},historical_pause_curve=pause.tolist(),cells={k:v.tolist() for k,v in M.items()})
    S.joinpath('p14_single_result-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    def fmt(v):return f'{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]'
    L=['# Phase 14 — one focused independent actor-repair result','',result['status'],'',
       f'Primary alpha=.01; compound p={max(ps.values()):.8g}; prespecified single claim success: **{success}**.',
       f'Seedwise consistency: {consistency}.', '', '| primary contrast | effect [95% CI] | 99% CI | threshold-shifted p |','|---|---|---|---|']
    for k,x in primary.items():L.append(f'| {k} | {fmt(tci(x))} | {fmt(tci(x,.99))} | {ps[k]:.7g} |')
    L+=['',f'Point fraction of native historical-pause AUC cost repaired: {ratio:.1%}; paired-bootstrap95 interval {result["repaired_fraction_bootstrap95"]}. Not an information fraction.',
         '',f'Residual vs matched state-pause: {result["residual_vs_statepause"]}.',f'Residual vs historical pause: {result["residual_vs_historicalpause"]}.',
         f'Common fresh-Adam checks: {result["fresh_Adam"]}.',f'Reset-context actor rescue: {fmt(tci(reset_rescue))}.',
         f'Independent evaluation key54321: {alt_metrics}.','',
         'All CIs pair training seeds, not episodes or checkpoints. Secondary equivalence/robustness does not rescue a failed primary. The result is scoped to Asterix PPO, this impairment and finite horizon; it does not identify a unique naturally encoded representation or prove irreversibility.']
    S.joinpath('p14_single_result-analysis.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


if __name__=='__main__':main()
