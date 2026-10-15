"""Phase 13: fresh-seed fork/pause-reference surgery, natural continuations and context checks.

Protocols and confirmatory decisions are fixed in prereg/phase13.json. Primary inference is withheld
until every planned seed in a study has all requested cells. No scores are printed during execution.
The damaged-context all-D cell is asserted to equal the untouched damaged runner state, including
its environment, observation, RNG, update counter, optimizer and all ancillary state.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rled.agents.ppo import make,make_eval
from rled.provenance import build_manifest,write_manifest
from rled.surgery import compose_ppo,all_codes,PPO_CARRIERS

warnings.filterwarnings('ignore')
CODES=all_codes(PPO_CARRIERS)
CORNERS=['CCC','DCC','CDD','DDD']
ZERO=['CCC','CDC','DCC','DDC']


def cells_for(study):
    cells=[]
    def add(ref,context,codes,opt='donor'):
        cells.extend(dict(reference=ref,context=context,opt=opt,code=c) for c in codes)
    if study=='A':
        for ref in ['fork','pause']:
            for ctx in ['clean','damaged']:add(ref,ctx,CODES)
        add('pause','clean',ZERO,'zero')
        add('pause','reset',CORNERS)
    elif study=='B':
        add('fork','clean',['CCC'])  # unbroken clean trace for historical pause benchmark
        add('pause','clean',CODES)
        add('pause','damaged',CORNERS)
    elif study=='C':
        add('fork','clean',CODES)
        add('fork','damaged',CORNERS)
        add('fork','clean',ZERO,'zero')
    else:raise ValueError(study)
    # Always run the natural clean continuation first.
    first=next(c for c in cells if c==dict(reference='fork',context='clean',opt='donor',code='CCC'))
    cells.remove(first);return [first]+cells


def cell_name(c):
    return '/'.join(c[k] for k in ['reference','context','opt','code'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',choices=['A','B','C'],required=True)
    ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    plan=json.loads(Path('prereg/phase13.json').read_text())
    p=dict(plan['studies'][a.study])
    if a.smoke:p.update(env='CartPole-v1',window=[10,20],recovery=20)
    E=10;w0,w1=p['window'];R=p['recovery'];spu=8192;n=len(a.seeds)
    dp={'prob':1.0} if p['deficit']=='reward_flip' else {'prob':.5}
    base=dict(ENV_NAME=p['env'],HIDDEN=128,DEFICIT=p['deficit'],DEFICIT_PARAMS=dp,TOTAL_TIMESTEPS=(w1+R)*spu)
    cells=cells_for(a.study)
    if a.smoke:
        keep=[('fork','clean','donor','CCC'),('pause','clean','donor','DDD'),('fork','damaged','donor','DDD'),
              ('pause','clean','zero','CCC'),('pause','reset','donor','DDD')]
        cells=[dict(zip(['reference','context','opt','code'],x)) for x in keep]
    protocol=dict(**p,steps_per_update=spu,eval_every=E,eval_envs=64,eval_seed=10000,
                  cells=cells,fresh_reset_seed_offset=900000,
                  context_definition='all non-learner fields from C or D; reset uses C RNG/counter with newly reset env/obs',
                  pause_definition='parameters and Adam state from shared onset; common future counter from context',
                  zero_opt_definition='all Adam states freshly initialized identically by procedure for all actor/critic cells',
                  no_score_logging=True,smoke=a.smoke)
    manifest=build_manifest('p13-'+a.tag+'-'+p['env'],{**base,'protocol':protocol},a.seeds)
    mp=write_manifest(manifest);stem=mp.stem
    out=Path('results/summary')/(stem+'.json')
    cfg,init,run=make(base)
    init_v=jax.jit(jax.vmap(init))
    run_v=jax.jit(jax.vmap(lambda r:run(r,E)))
    ev_v=jax.jit(jax.vmap(make_eval(cfg,num_envs=64),in_axes=(0,None)))
    ek=jax.random.PRNGKey(10000)
    def ev(rs):return np.asarray(jax.device_get(ev_v(rs['train_state'].params,ek)['ret'])).mean(1)
    def bounds(b0,b1):return jnp.tile(jnp.array([b0*spu,b1*spu],jnp.int32),(n,1))
    def equal(x,y):
        lx,tx=jax.tree.flatten(x);ly,ty=jax.tree.flatten(y)
        return tx==ty and all(np.array_equal(np.asarray(u),np.asarray(v)) for u,v in zip(lx,ly))
    t=time.time();print('manifest',mp,flush=True)
    rs=init_v(jnp.stack([jax.random.PRNGKey(s) for s in a.seeds]))
    rs=dict(rs,deficit_bounds=bounds(0,0))
    for _ in range(w0//E):rs,_=run_v(rs)
    P=rs;C=rs;D=dict(rs,deficit_bounds=bounds(w0,w1))
    fork=dict(updates=[w0],C=[ev(C).tolist()],D=[ev(D).tolist()])
    for k in range((w1-w0)//E):
        C,_=run_v(C);D,_=run_v(D)
        fork['updates'].append(w0+(k+1)*E)
        fork['C'].append(ev(C).tolist());fork['D'].append(ev(D).tolist())
    D=dict(D,deficit_bounds=bounds(0,0))
    refP=dict(C,train_state=C['train_state'].replace(params=P['train_state'].params,opt_state=P['train_state'].opt_state))
    fresh=init_v(jnp.stack([jax.random.PRNGKey(900000+s) for s in a.seeds]))
    reset=dict(C,env_state=fresh['env_state'],obs=fresh['obs']);del fresh
    zero=jax.vmap(lambda pp:C['train_state'].tx.init(pp))(C['train_state'].params)
    summary=dict(manifest=mp.name,seeds=a.seeds,protocol=protocol,eval_updates=list(range(w1,w1+R+1,E)),
                 fork_eval=fork,cells={},checks={},status='running')
    save=lambda:out.write_text(json.dumps(summary,indent=1))
    caches={}
    for k,c in enumerate(cells):
        name=cell_name(c);ref=C if c['reference']=='fork' else refP
        context={'clean':C,'damaged':D,'reset':reset}[c['context']]
        composed=compose_ppo(ref,D,c['code'])
        ts=composed['train_state']
        if c['opt']=='zero':ts=ts.replace(opt_state=zero)
        z=dict(context,train_state=ts,deficit_bounds=bounds(0,0))
        if c['context']=='damaged' and c['code']=='DDD' and c['opt']=='donor':
            identical=equal(z,D)
            summary['checks'][name+'_equals_natural_D']=identical
            assert identical,'D-context DDD must be the exact natural damaged continuation'
        if c==dict(reference='fork',context='clean',opt='donor',code='CCC'):
            assert equal(z,C);summary['checks']['clean_CCC_equals_natural_C']=True
        # DDD is identical across donor reference choices at a fixed context.
        alias=(c['context'],c['opt'],'DDD') if c['code']=='DDD' and c['opt']=='donor' else None
        if alias in caches:
            summary['cells'][name]=dict(summary['cells'][caches[alias]],alias_of=caches[alias])
        else:
            vals=[ev(z)]
            for _ in range(R//E):z,_=run_v(z);vals.append(ev(z))
            arr=np.stack(vals,axis=1)
            summary['cells'][name]=dict(eval_return_mean=arr.tolist())
            if alias is not None:caches[alias]=name
        summary['wallclock_s']=round(time.time()-t,1);save()
        print(f'cell {k+1}/{len(cells)} {name} saved; elapsed {time.time()-t:.0f}s (scores withheld)',flush=True)
    summary['status']='complete';save()
    print('complete',out,'seconds',round(time.time()-t,1),flush=True)


if __name__=='__main__':main()
