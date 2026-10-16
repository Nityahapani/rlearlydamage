"""Independent arithmetic/source audit of the completed fixed-n result, not a new analysis rule.

Uses scalar Python rows and the t CDF formula, separately from the analysis implementation.
Never computes a result on a partial cohort.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

import numpy as np
from scipy import stats


def main():
    f=Path('results/summary/p14_single_result-analysis.json')
    if not f.exists():raise SystemExit('Audit withheld: complete n=32 primary artifact not yet available')
    r=json.loads(f.read_text());p=json.loads(Path('prereg/phase14.json').read_text())
    assert r['n']==p['n']==32 and r['seeds']==p['training_seeds']
    for src in ['prereg/phase14.json','scripts/p14_actor_repair.py','src/rled/agents/ppo.py','src/rled/surgery.py']:
        assert Path(src).read_bytes()==subprocess.check_output(['git','show','6262b2b:'+src]),src
    rows={k:[] for k in r['primary']};seen=[]
    for name in r['summaries']:
        raw=Path(name);assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['source_hashes'][name]
        d=json.loads(raw.read_text());assert d['status']=='complete' and len(d['cells'])==11
        assert all(d['checks'][k] for k in ['DDD_equals_untouched_natural_D','CCC_equals_untouched_natural_C'])
        seen+=d['seeds']
        for i,seed in enumerate(d['seeds']):
            def curve(key):return d['cells'][key]['eval_return_mean'][i]
            table=dict(zip(d['fork_eval']['updates'],[v[i] for v in d['fork_eval']['C']]))
            table.update(dict(zip(d['eval_updates'],curve('fork/clean/donor/CCC'))))
            lag=d['protocol']['window'][1]-d['protocol']['window'][0]
            H=[table[u-lag] for u in d['eval_updates']]
            D=curve('pause/damaged/donor/DDD');R=curve('pause/damaged/donor/CDD')
            B=curve('pause/damaged/donor/DCC');P=curve('pause/damaged/donor/CCC')
            rows['native_AUC_harm'].append(statistics.mean(D[1:])-statistics.mean(H[1:]))
            rows['native_late_harm'].append(statistics.mean([a-b for a,b in zip(D[-5:],H[-5:])]))
            rows['onset_actor_rescue'].append(statistics.mean(R[1:])-statistics.mean(D[1:]))
            rows['reciprocal_actor_transfer'].append(statistics.mean(B[1:])-statistics.mean(P[1:]))
    assert seen==p['training_seeds'] and len(set(seen))==32
    ps=[]
    for c in p['primary']['contrasts']:
        k=c['name'];x=rows[k];mean=statistics.mean(x);sd=statistics.stdev(x);se=sd/math.sqrt(32)
        assert np.allclose(x,r['primary'][k]['per_seed'],atol=1e-12,rtol=0)
        if se:
            T=(mean+c['shift'])/se
            pv=float(stats.t.cdf(T,31) if c['direction']=='less' else stats.t.sf(T,31))
        else:
            shifted=mean+c['shift']
            pv=0. if (shifted<0 if c['direction']=='less' else shifted>0) else 1.
        ps.append(pv)
        assert math.isclose(pv,r['primary_p_values'][k],rel_tol=1e-10,abs_tol=1e-15)
        for level,label in [(.95,'CI95'),(.99,'CI99')]:
            half=float(stats.t.ppf((1+level)/2,31))*se
            assert np.allclose([mean,mean-half,mean+half],r['primary'][k][label],atol=1e-12,rtol=0)
    counts=dict(native_AUC_negative=sum(v<0 for v in rows['native_AUC_harm']),actor_rescue_positive=sum(v>0 for v in rows['onset_actor_rescue']))
    assert counts==r['seed_consistency']
    assert r['primary_success']==(max(ps)<.01 and min(counts.values())>=24)
    print('PASS: all32 raw paired rows, historical lag, practical-effect p-values,95/99% CIs, seed consistency, source/data hashes and compound verdict independently reproduce')


if __name__=='__main__':main()
