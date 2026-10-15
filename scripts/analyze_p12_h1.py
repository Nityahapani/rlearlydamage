"""Independent-seed H1 CartPole replication. Registered in 8adf203 after the positive G1 result.

Primary analysis only at the planned fresh n=8 (600–607); no G1 pooling or optional seed expansion.
Incomplete analyses require --include-partial at budget stop and are explicitly descriptive.
Calls the unchanged surgery analysis and records exact raw-data decisions in a JSON sidecar.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from posthoc_p12_audit import CARR,CODES,holm,read_test,scaled_tost,shapley,tci,tost


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--include-partial',action='store_true')
    ap.add_argument('--test',choices=['h1','h2'],default='h1')
    a=ap.parse_args()
    name,base,tag=('h1-cartpole-fresh',600,'p12_h1_cartpole_fresh') if a.test=='h1' else ('h2-si-actnoise-fresh',700,'p12_h2_si_actnoise_fresh')
    seeds,M,onset,U,ds,files=read_test(name)
    complete=set(seeds)==set(range(base,base+8))
    if not complete and not a.include_partial:
        raise SystemExit(f'{a.test.upper()} analysis withheld: expected all fresh seeds {base}–{base+7}')
    if len(seeds)<2:
        raise SystemExit('No completed fresh-replication chunk')
    A={c:m[:,1:].mean(1) for c,m in M.items()}
    total=A['DDD']-A['CCC']
    fx={n:np.mean([A[c] for c in CODES if c[i]=='D'],axis=0)-np.mean([A[c] for c in CODES if c[i]=='C'],axis=0)
        for i,n in enumerate(CARR)}
    ph=shapley(A)
    margin=float(.1*abs(A['CCC'].mean()))
    ps=[float(stats.ttest_1samp(fx[n],0).pvalue) for n in CARR]
    hp=holm(ps)
    tc=tci(total)
    verdict=('NO GAP TO LOCALIZE' if tc[2]>=0 else ('REPLICATES' if tci(fx['actor'])[2]<0 and min(CARR,key=lambda n:ph[n].mean())=='actor' else 'DOES NOT REPLICATE'))
    status=f'Registered independent-seed replication {a.test.upper()} (n=8, seeds {base}–{base+7})' if complete else 'DESCRIPTIVE ONLY (incomplete planned fresh n=8)'
    subprocess.run([sys.executable,'scripts/analyze_p8_surgery.py','--summaries',*files,'--tag',tag,
                    '--sesoi',str(margin),'--fork-eval','--title',status],check=True,capture_output=True)
    res={'status':status,'complete':complete,'seeds':seeds,'n':len(seeds),'summaries':files,
         'total':tc,'effects':{n:tci(fx[n]) for n in CARR},'shapley':{n:tci(ph[n]) for n in CARR},
         'shares':{n:float(ph[n].mean()/total.mean()*100) for n in CARR} if tc[2]<0 else None,
         'raw_p':dict(zip(CARR,ps)),'holm':dict(zip(CARR,hp)),'margin':margin,
         'tost':{n:tost(fx[n],margin) for n in CARR},'scaled_tost_posthoc':{n:scaled_tost(fx[n],A['CCC']) for n in CARR},
         'verdict':verdict,'g1_pooled':False,'independent_seed_ids':True}
    Path(f'results/summary/{tag}-analysis.json').write_text(json.dumps(res,indent=2)+'\n')
    print(status,verdict)
    print('Total',tc,'actor',res['effects']['actor'],'shares',res['shares'],'Holm',res['holm'])


if __name__=='__main__':
    main()
