"""Phase 13 E: same transitions, different initial labels, controlled common exposure and washout.

After K fixed-behavior updates the full buffer and every non-learner state must be byte-identical.
Naturally matched parameter/moment continuations RR and UU are primary; the crossed and common-zero
moment continuations are secondary causal-state probes. No outcomes are printed during execution.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rled.agents.pqn_exposure import make,make_eval
from rled.provenance import build_manifest,write_manifest

warnings.filterwarnings('ignore')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if not a.smoke:
        flag=Path('/home/user/p13logs/exposure-software-smoke.done')
        if not flag.exists():
            log=Path('/home/user/p13logs/exposure-software-smoke.log')
            with log.open('w') as f:
                r=subprocess.run([sys.executable,'-u',__file__,'--tag','software-smoke-E','--seeds','99991','--smoke'],stdout=f,stderr=subprocess.STDOUT)
            assert r.returncode==0,'Exposure software smoke failed; inspect log before any target data'
            flag.touch()
    plan=json.loads(Path('prereg/phase13.json').read_text())['studies']['E'];n=len(a.seeds)
    w0,w1=plan['window'];K=25;R=300;E=10
    base=dict(ENV_NAME='SpaceInvaders-MinAtar',REPLAY=True,BUFFER_SIZE=100000,DEFICIT='reward_flip',EXPOSURE_BEHAVIOR=True)
    if a.smoke:
        w0,w1,K,R=2,4,2,20
        base.update(NUM_ENVS=4,NUM_STEPS=4,NUM_EPOCHS=1,NUM_MINIBATCHES=2,REPLAY_BATCH=8,
                    BUFFER_SIZE=32,HIDDEN=16,EPS_START=.05,EPS_FINISH=.05,EPS_DECAY_STEPS=1)
    base['TOTAL_TIMESTEPS']=(w1+K+R)*base.get('NUM_ENVS',128)*base.get('NUM_STEPS',32)
    cfg,init,run=make(base);spu=cfg['STEPS_PER_UPDATE']
    proto=dict(window=[w0,w1],exposure_updates=K,recovery=R,eval_every=E,eval_envs=64,eval_eps=.05,
               steps_per_update=spu,buffer_size=cfg['BUFFER_SIZE'],teacher='frozen clean fork parameters at removal; epsilon floor unchanged',
               arms=['UU','RU','UR','RR','U0','R0'],reward_repair='only sign of original-window D rewards is restored',smoke=a.smoke)
    m=build_manifest('p13-'+a.tag+'-pqnr-exposure',{**cfg,'protocol':proto},a.seeds)
    mp=write_manifest(m);out=Path('results/summary')/(mp.stem+'.json')
    iv=jax.jit(jax.vmap(init));rv=jax.jit(jax.vmap(lambda r:run(r,1)))
    evv=jax.jit(jax.vmap(make_eval(cfg,num_envs=64,eps=.05),in_axes=(0,None)))
    ek=jax.random.PRNGKey(10000)
    def ev(r):return np.asarray(jax.device_get(evv(r['train_state'].params,ek)['ret'])).mean(1)
    def bounds(x,y):return jnp.tile(jnp.array([x*spu,y*spu],jnp.int32),(n,1))
    def same(x,y):
        lx,tx=jax.tree.flatten(x);ly,ty=jax.tree.flatten(y)
        return tx==ty and all(np.array_equal(np.asarray(v),np.asarray(w)) for v,w in zip(lx,ly))
    t=time.time();print('manifest',mp,flush=True)
    z=iv(jnp.stack([jax.random.PRNGKey(s) for s in a.seeds]));z=dict(z,deficit_bounds=bounds(0,0))
    for _ in range(w0):z,_=rv(z)
    C=z;D=dict(z,deficit_bounds=bounds(w0,w1));del z
    for _ in range(w1-w0):C,_=rv(C);D,_=rv(D)
    teacher=C['train_state'].params
    clean_removal=ev(C).tolist();del C
    D=dict(D,deficit_bounds=bounds(0,0),behavior_params=teacher,behavior_until=jnp.full((n,),w1+K,jnp.int32))
    tag=D['buffer']['tag'];rew=D['buffer']['reward'];era=(tag>=w0)&(tag<w1)
    restored=jnp.where(era,-rew,rew)
    U=dict(D,buffer=dict(D['buffer'],reward=restored));RR=D;del D
    # Sole initial difference is the old reward-label array.
    initial_other_equal=same({k:v for k,v in RR.items() if k!='buffer'},{k:v for k,v in U.items() if k!='buffer'})
    assert initial_other_equal
    assert same({k:v for k,v in RR['buffer'].items() if k!='reward'},{k:v for k,v in U['buffer'].items() if k!='reward'})
    exposure=dict(updates=[0],R=[ev(RR).tolist()],U=[ev(U).tolist()])
    for k in range(K):
        RR,_=rv(RR);U,_=rv(U)
        # Both collectors must have identical genuine environment transitions despite different learners.
        assert same(RR['env_state'],U['env_state']) and same(RR['obs'],U['obs'])
        assert same(RR['rng'],U['rng'])
        if (k+1)%5==0 or k+1==K:
            exposure['updates'].append(k+1);exposure['R'].append(ev(RR).tolist());exposure['U'].append(ev(U).tolist())
    nonR={k:v for k,v in RR.items() if k!='train_state'}
    nonU={k:v for k,v in U.items() if k!='train_state'}
    assert same(nonR,nonU),'All non-learner states must match after complete overwrite'
    remaining=np.asarray((RR['buffer']['tag']>=w0)&(RR['buffer']['tag']<w1)).mean()
    assert remaining==0,'Old labels must be fully gone'
    summary=dict(manifest=mp.name,seeds=a.seeds,protocol=proto,checks=dict(initial_only_reward_labels_differ=initial_other_equal,
                 fixed_behavior_env_rng_equal=True,post_washout_all_nonlearner_states_equal=True,old_era_fraction_after_K=float(remaining)),
                 clean_removal=clean_removal,exposure=exposure,eval_updates=list(range(0,R+1,E)),arms={},status='running')
    zero=jax.vmap(lambda pp:RR['train_state'].tx.init(pp))(RR['train_state'].params)
    for arm in proto['arms']:
        srcP=RR if arm[0]=='R' else U
        srcO=RR if arm[1]=='R' else U
        ts=RR['train_state'].replace(params=srcP['train_state'].params,opt_state=zero if arm[1]=='0' else srcO['train_state'].opt_state)
        r=dict(RR,train_state=ts)
        if arm=='RR':assert same(r,RR)
        if arm=='UU':assert same(r,U)
        vals=[ev(r)]
        for _ in range(R//E):
            for __ in range(E):r,_=rv(r)
            vals.append(ev(r))
        summary['arms'][arm]=dict(eval_return_mean=np.stack(vals,axis=1).tolist())
        summary['wallclock_s']=round(time.time()-t,1);out.write_text(json.dumps(summary,indent=1))
        print('saved arm',arm,'elapsed',round(time.time()-t,1),'(outcomes withheld)',flush=True)
    summary['status']='complete';out.write_text(json.dumps(summary,indent=1));print('complete',out,flush=True)


if __name__=='__main__':main()
