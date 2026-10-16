"""One focused Phase-14 actor-repair result; target outcomes are withheld until fixed n=32.

Native D is checked over its full runner tree. The actor donor is the shared onset state, not a
clean fork with extra progress. Completed cells can resume only after exact state/config checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rled.agents.ppo import make,make_eval
from rled.provenance import build_manifest,write_manifest
from rled.surgery import compose_ppo

warnings.filterwarnings('ignore')


def name(c):return '/'.join(c[k] for k in ['reference','context','opt','code'])


def fingerprint(rs):
    h=hashlib.sha256()
    for x in jax.tree.leaves(rs):
        v=np.asarray(x)
        h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def equal(x,y):
    a,ta=jax.tree.flatten(x);b,tb=jax.tree.flatten(y)
    return ta==tb and all(np.array_equal(np.asarray(u),np.asarray(v)) for u,v in zip(a,b))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    p=json.loads(Path('prereg/phase14.json').read_text())
    env=p['env'];w0,w1=p['window'];R=p['recovery'];E=p['eval_every'];n=len(a.seeds);spu=8192
    if a.smoke:env='CartPole-v1';w0,w1,R=10,20,20
    base=dict(ENV_NAME=env,HIDDEN=128,DEFICIT='reward_flip',DEFICIT_PARAMS={'prob':1.},TOTAL_TIMESTEPS=(w1+R)*spu)
    cells=p['cells'];altset=set(p['second_eval_cells'])
    proto=dict(schema='p14-v1',env=env,window=[w0,w1],recovery=R,eval_every=E,eval_envs=64,
               eval_seed=10000,second_eval_seed=54321,second_eval_cells=sorted(altset),cells=cells,
               steps_per_update=spu,smoke=a.smoke,actor_donor='shared onset; no extra clean progress',
               context='natural D for main swaps; reset changes only env/obs, keeps D RNG/counter',
               optimizer_control='common fresh Adam for all4 actor x critic combinations')
    previous=[]
    for f in sorted(Path('results/summary').glob(f'*p14-{a.tag}-{env}-*.json')):
        old=json.loads(f.read_text())
        if old['seeds']==a.seeds and old['protocol']==proto:previous.append((f,old))
    oldpath,old=(max(previous,key=lambda t:len(t[1].get('cells',{}))) if previous else (None,None))
    manifest=build_manifest('p14-'+a.tag+'-'+env,{**base,'protocol':proto},a.seeds)
    mp=write_manifest(manifest);out=Path('results/summary')/(mp.stem+'.json')
    cfg,init,run=make(base)
    iv=jax.jit(jax.vmap(init));rv=jax.jit(jax.vmap(lambda r:run(r,E)))
    evv=jax.jit(jax.vmap(make_eval(cfg,num_envs=64),in_axes=(0,None)))
    def ev(rs,key):return np.asarray(jax.device_get(evv(rs['train_state'].params,jax.random.PRNGKey(key))['ret'])).mean(1)
    def bounds(x,y):return jnp.tile(jnp.array([x*spu,y*spu],jnp.int32),(n,1))
    t=time.time();print('manifest',mp,flush=True)
    z=iv(jnp.stack([jax.random.PRNGKey(s) for s in a.seeds]));z=dict(z,deficit_bounds=bounds(0,0))
    for _ in range(w0//E):z,_=rv(z)
    P=z;C=z;D=dict(z,deficit_bounds=bounds(w0,w1))
    fork=dict(updates=[w0],C=[ev(C,10000).tolist()],D=[ev(D,10000).tolist()])
    fork2=dict(updates=[w0],C=[ev(C,54321).tolist()])
    for j in range((w1-w0)//E):
        C,_=rv(C);D,_=rv(D)
        fork['updates'].append(w0+(j+1)*E);fork['C'].append(ev(C,10000).tolist());fork['D'].append(ev(D,10000).tolist())
        fork2['updates'].append(w0+(j+1)*E);fork2['C'].append(ev(C,54321).tolist())
    D=dict(D,deficit_bounds=bounds(0,0))
    hashes={k:fingerprint(v) for k,v in [('C',C),('D',D),('P',P)]}
    if old:
        assert old['state_fingerprints']==hashes,'Do not reuse cells with different donor states'
        assert old['fork_eval']==fork and old['fork_eval_alt']==fork2,'Fork trajectories must reproduce exactly'
    pref=dict(D,train_state=D['train_state'].replace(params=P['train_state'].params,opt_state=P['train_state'].opt_state))
    new=iv(jnp.stack([jax.random.PRNGKey(900000+s) for s in a.seeds]))
    reset=dict(D,env_state=new['env_state'],obs=new['obs']);del new
    zero=jax.vmap(lambda pp:D['train_state'].tx.init(pp))(D['train_state'].params)
    summary=dict(manifest=mp.name,seeds=a.seeds,protocol=proto,eval_updates=list(range(w1,w1+R+1,E)),
                 fork_eval=fork,fork_eval_alt=fork2,state_fingerprints=hashes,cells={},checks={},status='running',
                 resumed_from=str(oldpath) if oldpath else None)
    def save():
        tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(summary,indent=1));tmp.replace(out)
    for j,c in enumerate(cells):
        k=name(c);ref=C if c['reference']=='fork' else pref
        ctx={'clean':C,'damaged':D,'reset':reset}[c['context']]
        ts=compose_ppo(ref,D,c['code'])['train_state']
        if c['opt']=='zero':ts=ts.replace(opt_state=zero)
        r=dict(ctx,train_state=ts,deficit_bounds=bounds(0,0))
        if c['context']=='damaged' and c['opt']=='donor' and c['code']=='DDD':
            assert equal(r,D);summary['checks']['DDD_equals_untouched_natural_D']=True
        if k=='fork/clean/donor/CCC':
            assert equal(r,C);summary['checks']['CCC_equals_untouched_natural_C']=True
        if old and k in old.get('cells',{}):
            v=old['cells'][k]
            assert np.asarray(v['eval_return_mean']).shape==(n,R//E+1)
            if k in altset:assert np.asarray(v['eval_alt_return_mean']).shape==(n,R//E+1)
            # Reuse only full cells; the donor hashes and full fork measurements matched above.
            summary['cells'][k]=v
            status='resumed verified cell'
        else:
            vals=[ev(r,10000)];vals2=[ev(r,54321)] if k in altset else None
            for _ in range(R//E):
                r,_=rv(r);vals.append(ev(r,10000))
                if vals2 is not None:vals2.append(ev(r,54321))
            arr=np.stack(vals,axis=1);assert np.isfinite(arr).all()
            v=dict(eval_return_mean=arr.tolist())
            if vals2 is not None:v['eval_alt_return_mean']=np.stack(vals2,axis=1).tolist()
            summary['cells'][k]=v;status='saved'
        summary['wallclock_s']=round(time.time()-t,1);save()
        print(f'cell {j+1}/{len(cells)} {k}: {status}; elapsed {time.time()-t:.0f}s; outcomes withheld',flush=True)
    summary['status']='complete';save();print('complete',out,flush=True)


if __name__=='__main__':main()
