"""Phase 13 registered analysis. Default withholds incomplete cohorts.
Five prespecified primary claim p-values use a fixed Holm family; unrun/incomplete = 1.
Legacy claims are not retroactively given global error control. Context checks remain secondary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from posthoc_p12_audit import tci,holm,shapley,tost

S=Path('results/summary');CODES=['CCC','CCD','CDC','CDD','DCC','DCD','DDC','DDD'];CARR=['actor','critic','opt']


def pneg(x):
    x=np.asarray(x,float)
    if np.std(x)==0:return 0.0 if x.mean()<0 else 1.0
    return float(stats.ttest_1samp(x,0,alternative='less').pvalue)


def fmt(ci):
    e,lo,hi=ci;return f'{e:+.2f} [{lo:+.2f}, {hi:+.2f}]'+('*' if lo>0 or hi<0 else '')


def load_study(gid,npartial=False):
    files=sorted(S.glob(f'*p13-{gid}-s*-*.json'));ds=[];seen=[]
    for f in files:
        d=json.loads(f.read_text())
        if d.get('status')=='complete':ds.append(d);seen+=d['seeds']
    expected=json.loads(Path('prereg/phase13.json').read_text())['studies'][gid]['seeds']
    if sorted(seen)!=expected and not npartial:return None
    if len(seen)<2:return None
    assert len(set(seen))==len(seen)
    keys=set(ds[0]['cells']);assert all(set(d['cells'])==keys for d in ds)
    M={k:np.concatenate([np.asarray(d['cells'][k]['eval_return_mean'],float) for d in ds]) for k in sorted(keys)}
    checks=all(all(d['checks'].values()) for d in ds)
    assert checks,'Natural-continuation identity checks failed'
    nativeDkey=next(k for k in M if k.endswith('/damaged/donor/DDD'))
    # Natural clean is unbroken at and after onset; this reconstructs the true historical pause.
    pause=[]
    for d in ds:
        bank={u:np.asarray(v) for u,v in zip(d['fork_eval']['updates'],d['fork_eval']['C'])}
        cc=np.asarray(d['cells']['fork/clean/donor/CCC']['eval_return_mean'])
        bank.update({u:cc[:,i] for i,u in enumerate(d['eval_updates'])})
        w0,w1=d['protocol']['window'];L=w1-w0
        pause.extend(np.stack([bank[u-L] for u in d['eval_updates']],axis=1))
    pause=np.asarray(pause)
    native=(M[nativeDkey][:,1:]-pause[:,1:]).mean(1)
    out=dict(study=gid,n=len(seen),seeds=seen,complete=sorted(seen)==expected,natural_identity_checks_pass=True,
             native_damage=tci(native),native_damage_per_seed=native.tolist(),groups={},contexts={},cells={k:v.tolist() for k,v in M.items()})
    for prefix in sorted({k.rsplit('/',1)[0] for k in M}):
        subset={k.rsplit('/',1)[1]:v[:,1:].mean(1) for k,v in M.items() if k.startswith(prefix+'/')}
        if all(c in subset for c in CODES):
            total=subset['DDD']-subset['CCC']
            main={c:np.mean([v for k,v in subset.items() if k[i]=='D'],axis=0)-np.mean([v for k,v in subset.items() if k[i]=='C'],axis=0) for i,c in enumerate(CARR)}
            ph=shapley(subset)
            primaryparts={'total_negative':pneg(total),'actor_main_negative':pneg(main['actor']),
                          'actor_phi_below_critic':pneg(ph['actor']-ph['critic']),'actor_phi_below_Adam':pneg(ph['actor']-ph['opt'])}
            out['groups'][prefix]=dict(total=tci(total),effects={k:tci(x) for k,x in main.items()},
                shapley={k:tci(x) for k,x in ph.items()},shares={k:float(x.mean()/total.mean()*100) for k,x in ph.items()} if tci(total)[2]<0 else None,
                actor_in_clean_background=tci(subset['DCC']-subset['CCC']),actor_rescue_damaged_background=tci(subset['CDD']-subset['DDD']),
                compound_parts=primaryparts,compound_p=max(primaryparts.values()),
                per_seed={'total':total.tolist(),**{k:main[k].tolist() for k in CARR}})
        # Four actor corners: report both backgrounds even without a full factorial.
        if all(c in subset for c in ['CCC','DCC','CDD','DDD']):
            out['contexts'][prefix]=dict(total=tci(subset['DDD']-subset['CCC']),
                 bad_actor_into_reference=tci(subset['DCC']-subset['CCC']),
                 good_actor_into_damaged=tci(subset['CDD']-subset['DDD']))
        # Common zero-Adam 2x2, not a donor optimizer effect.
        if prefix.endswith('/zero') and all(c in subset for c in ['CCC','CDC','DCC','DDC']):
            act=(subset['DCC']+subset['DDC']-subset['CCC']-subset['CDC'])/2
            crit=(subset['CDC']+subset['DDC']-subset['CCC']-subset['DCC'])/2
            out['contexts'][prefix]=dict(total=tci(subset['DDC']-subset['CCC']),actor=tci(act),critic=tci(crit),interpretation='all moments reset; not the natural old optimizer')
    ref='pause' if gid in ['A','B'] else 'fork';g=out['groups'][ref+'/clean/donor']
    parts=dict(g['compound_parts'])
    if gid in ['A','B']:parts['natural_historical_pause_harm']=pneg(native)
    out['primary_parts']=parts;out['primary_p']=max(parts.values()) if out['complete'] else 1.0
    # Natural D versus standardized C-context DDD; nuisance-scale uncertainty is retained in TOST.
    stdkey=ref+'/clean/donor/DDD';delta=(M[stdkey][:,1:]-M[nativeDkey][:,1:]).mean(1)
    clean=M['fork/clean/donor/CCC'][:,1:].mean(1)
    plo=pneg(-(delta+.05*clean));phi=pneg(delta-.05*clean)
    out['natural_vs_standardized']=dict(difference=tci(delta),margin_rule='5% of mean native clean AUC',margin=float(.05*clean.mean()),TOST_joint=max(plo,phi),
                                       natural_D_AUC=tci(M[nativeDkey][:,1:].mean(1)),standardized_DDD_AUC=tci(M[stdkey][:,1:].mean(1)))
    return out


def exposure():
    fs=sorted(S.glob('*p13-E-s*-*.json'));ds=[json.loads(p.read_text()) for p in fs]
    ds=[d for d in ds if d.get('status')=='complete' and not d['protocol'].get('smoke')]
    seeds=sum((d['seeds'] for d in ds),[])
    if sorted(seeds)!=list(range(1000,1008)):return None
    for d in ds:
        c=d['checks']
        assert c['initial_only_reward_labels_differ'] and c['fixed_behavior_env_rng_equal']
        assert c['post_washout_all_nonlearner_states_equal'] and c['old_era_fraction_after_K']==0
    M={k:np.concatenate([np.asarray(d['arms'][k]['eval_return_mean']) for d in ds])[:,1:].mean(1) for k in ['UU','RU','UR','RR','U0','R0']}
    total=M['RR']-M['UU'];param=(M['RU']+M['RR']-M['UU']-M['UR'])/2
    opt=(M['UR']+M['RR']-M['UU']-M['RU'])/2
    return dict(n=8,seeds=seeds,complete=True,total=tci(total),primary_p=pneg(total),practical_delta=1,
                parameters=tci(param),moments=tci(opt),interaction=tci(M['RR']-M['UR']-M['RU']+M['UU']),
                common_zero_moment_difference=tci(M['R0']-M['U0']),TOST_1=tost(total,1),
                state_matching_verified=True,old_era_fraction_after_K=0,
                per_seed={'total':total.tolist(),'parameters':param.tolist(),'moments':opt.tolist()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-partial',action='store_true');a=ap.parse_args()
    R={k:load_study(k,a.include_partial) for k in ['A','B','C']};R['E']=exposure()
    dj=S/'p13_D_prospective-analysis.json';R['D']=json.loads(dj.read_text()) if dj.exists() else None
    p={k:(r.get('primary_p',r.get('primary_one_sided_p',1)) if r is not None else 1) for k,r in R.items()}
    order=['A','B','C','E','D'];hp=holm([p[k] for k in order])
    for k,h in zip(order,hp):
        if R[k] is not None:R[k]['phase13_Holm_p']=h
    out=dict(family=order,missing_or_incomplete_p=1,scope='five NEW Phase 13 claims only, not retrospective paper-wide control',studies=R)
    (S/'p13_confirmatory-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    L=['# Phase 13 — registered claim-family analysis','',out['scope'],'',
       '| study | n | primary p (compound where specified) | Phase-13 Holm p | status |','|---|---|---|---|---|']
    for k,h in zip(order,hp):
        r=R[k];status='not run / incomplete (no primary inference)' if r is None else ('planned cohort complete' if r.get('complete',True) else 'descriptive partial')
        if k=='E' and r is not None:status+='; substantial harm requires mean <= -1'
        if k=='D' and r is not None:status+='; usefulness also requires gain >=10%'
        L.append(f"| {k} | {r.get('n',0) if r else 0} | {p[k]:.6g} | {h:.6g} | {status} |")
    for k in ['A','B','C']:
        r=R[k]
        if r is None:continue
        L += ['',f'## {k}: natural benchmark and contexts (n={r["n"]})','',f'Native damage beyond historical pause: {fmt(r["native_damage"])}.',
              f'Natural-state identity checks: {r["natural_identity_checks_pass"]}.',
              f'Context shift, standardized DDD minus natural D: {fmt(r["natural_vs_standardized"]["difference"])}; joint-scale TOST p={r["natural_vs_standardized"]["TOST_joint"]:.4g}.',
              '', '| reference/context/moments | total gap | actor effect / bad-actor transfer | good-actor rescue |','|---|---|---|---|']
        for c,x in r['contexts'].items():
            L.append(f"| {c} | {fmt(x['total'])} | {fmt(x.get('actor',x.get('bad_actor_into_reference')))} | {fmt(x['good_actor_into_damaged']) if 'good_actor_into_damaged' in x else 'zero-moment 2x2'} |")
        for g,x in r['groups'].items():
            L += ['',f'{g}: actor/critic/Adam main effects '+', '.join(f'{n} {fmt(x["effects"][n])}' for n in CARR)+'.',
                  f'Shapley shares (only if total CI <0): {x["shares"]}.']
    (S/'p13_confirmatory-analysis.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


if __name__=='__main__':main()
