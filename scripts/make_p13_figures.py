"""Native versus standardized continuation curves; only complete registered cohorts are shown."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

S=Path('results/summary')
def main():
 r=json.loads((S/'p13_confirmatory-analysis.json').read_text())['studies']
 studies=[k for k in ['A','B'] if r.get(k) and r[k].get('complete')]
 if not studies:return
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
 f,axs=plt.subplots(1,len(studies),figsize=(5.8*len(studies),5.8),squeeze=False,constrained_layout=True)
 for ax,k in zip(axs[0],studies):
  z=r[k];cells=z['cells'];x=np.arange(len(z['historical_pause_curve'][0]))*10
  def draw(M,label,color,ls='-'):
   M=np.asarray(M);mu=M.mean(0);h=stats.t.ppf(.975,len(M)-1)*M.std(0,ddof=1)/np.sqrt(len(M))
   ax.plot(x,mu,label=label,color=color,ls=ls,lw=1.8);ax.fill_between(x,mu-h,mu+h,color=color,alpha=.08)
  draw(z['historical_pause_curve'],'Historical pause','#222222','--')
  draw(cells['pause/damaged/donor/DDD'],'Natural D','#B54C42')
  draw(cells['pause/clean/donor/DDD'],'Standardized DDD','#D29528',':')
  draw(cells['pause/clean/donor/CCC'],'Onset learner / C context','#216F89')
  draw(cells['pause/damaged/donor/CCC'],'Onset learner / D context','#7462A2','-.')
  ax.set_title(('A  CartPole' if k=='A' else 'B  Asterix')+f' (independent n={z["n"]})',weight='bold')
  ax.set_xlabel('Updates after removal');ax.set_ylabel('Unrestricted true-reward return');ax.grid(alpha=.14)
  ax.legend(fontsize=11.5,loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,ncol=1)
 f.savefig('figures/p13_natural_contexts.png',dpi=200,facecolor='white')
 f.savefig('figures/p13_natural_contexts.pdf',facecolor='white')
 print('Rendered natural/context curves:',studies)
if __name__=='__main__':main()
