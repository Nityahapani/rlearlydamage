"""Render the single Phase-14 result from complete n=32 analysis; no selective endpoint display."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

S=Path('results/summary')


def ci(v,d=2):
    return f'${v[0]:+.{d}f}\\ci{{{v[1]:+.{d}f}}}{{{v[2]:+.{d}f}}}$'


def pv(p):
    import math
    if p<=0:return r'$<10^{-12}$'
    if p<.0001:
        e=int(math.floor(math.log10(p)));return f'${p/10**e:.2f}\\times10^{{{e}}}$'
    return f'${p:.4f}$'


def main():
    r=json.loads((S/'p14_single_result-analysis.json').read_text());assert r['n']==32
    M=r['cells'];hist=np.asarray(r['historical_pause_curve']);x=np.arange(hist.shape[1])*10
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    f,(a,b)=plt.subplots(2,1,figsize=(7.5,7.8),constrained_layout=True)
    for key,label,color,ls in [
        (None,'Historical pause','#24282B','--'),
        ('pause/damaged/donor/CCC','Onset-state reference','#737B80',':'),
        ('pause/damaged/donor/DDD','Natural impaired','#B14A43','-'),
        ('pause/damaged/donor/CDD','Onset actor only','#1B7282','-'),
        ('pause/damaged/donor/DCC','Reciprocal bad actor','#CA963A',':')]:
        A=hist if key is None else np.asarray(M[key]);mu=A.mean(0);h=stats.t.ppf(.995,31)*A.std(0,ddof=1)/np.sqrt(32)
        a.plot(x,mu,label=label,color=color,ls=ls,lw=2)
        a.fill_between(x,mu-h,mu+h,color=color,alpha=.09)
    a.set_title('A  Actor rollback without extra clean progress',weight='bold',fontsize=12)
    a.set_xlabel('Unrestricted recovery updates');a.set_ylabel('True-reward evaluation return')
    a.grid(alpha=.12);a.legend(fontsize=9.5,frameon=False,loc='upper left',ncol=2)
    pairs=[('Native AUC cost',r['primary']['native_AUC_harm']['CI95'],r['primary']['native_AUC_harm']['CI99']),
           ('Late cost',r['primary']['native_late_harm']['CI95'],r['primary']['native_late_harm']['CI99']),
           ('Actor rescue',r['primary']['onset_actor_rescue']['CI95'],r['primary']['onset_actor_rescue']['CI99']),
           ('Reciprocal transfer',r['primary']['reciprocal_actor_transfer']['CI95'],r['primary']['reciprocal_actor_transfer']['CI99']),
           ('Fresh-Adam rescue',r['fresh_Adam']['actor_rescue'],None),
           ('Reset-world rescue',r['reset_context_actor_rescue'],None),
           ('New eval-key rescue',r['alternate_evaluation']['onset_actor_rescue'],None)]
    primary_order=['native_AUC_harm','native_late_harm','onset_actor_rescue','reciprocal_actor_transfer']
    for i,(lab,v,w) in enumerate(pairs):
        if i<4:
            points=np.asarray(r['primary'][primary_order[i]]['per_seed'])
            b.scatter(points,i+np.linspace(-.14,.14,32),s=12,color='#91ADB5',alpha=.55,zorder=1,edgecolors='none')
        m,lo,hi=v
        if w:b.plot(w[1:], [i,i],color='#92B3BC',lw=1.2)
        b.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#1B7282',capsize=3,markersize=5)
    b.axvline(0,color='.5',ls='--',lw=1);b.set_yticks(range(len(pairs)),[p[0] for p in pairs]);b.invert_yaxis()
    b.set_xlabel('Paired effect, return points');b.set_title('B  Paired primary effects and robustness checks',weight='bold',fontsize=12);b.grid(axis='x',alpha=.12)
    f.savefig('figures/p14_actor_repair.png',dpi=220,facecolor='white');f.savefig('figures/p14_actor_repair.pdf',facecolor='white')
    import shutil
    for ext in ['png','pdf']:shutil.copy2('figures/p14_actor_repair.'+ext,'paper/figures/p14_actor_repair.'+ext)
    rows=[]
    labels={'native_AUC_harm':'native AUC cost','native_late_harm':'late-window cost','onset_actor_rescue':'onset-actor rescue','reciprocal_actor_transfer':'reciprocal actor transfer'}
    thresholds={'native_AUC_harm':r'$<-0.50$','native_late_harm':r'$<-0.30$','onset_actor_rescue':r'$>+0.50$','reciprocal_actor_transfer':r'$<-0.50$'}
    for k in labels:
        q=r['primary'][k]
        rows.append(labels[k]+' & '+ci(q['CI95'])+' & '+ci(q['CI99'])+' & '+thresholds[k]+' & '+pv(r['primary_p_values'][k])+r' \\')
    table='\n'.join([r'\begin{table}[t]',r'\centering',
        r'\caption{Independent Asterix actor-repair confirmation, 32 training seeds. All four prespecified practical-effect components must pass through their maximum one-sided $p$ at $\alpha=.01$, plus at least 24/32 native-negative and rescue-positive seeds. Intervals pair independent training seeds conditional on the fixed primary evaluation batch, not episodes/checkpoints. Secondary checks cannot rescue a failed primary.}',
        r'\label{tab:p14solid}',r'\small',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lcccc}',r'\toprule',
        r'component & 95\% CI & 99\% CI & required mean & shifted $p$ \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])
    verdict='met' if r['primary_success'] else 'not met'
    text=r'''\section{A focused high-powered actor-repair result}\label{sec:p14solid}
\paragraph{Fixed design and causal contrast.} We chose one previously supported setting rather than screen additional games or report the best endpoint. Asterix PPO received reward sign corruption during updates 390--440 (409{,}600 environment steps), followed by 200 unrestricted updates. Thirty-two entirely unused training seeds 1200--1231 were fixed before target data; none of the earlier eight Asterix seeds enters this primary. The donor actor comes from the shared onset at 390, not the clean fork at 440, so it cannot import the clean agent's additional progress. All primary swaps use the exact natural damaged future context. Within pause-cell codes, C denotes the onset donor P, not extra clean-fork progress; D denotes impaired removal state. CDD therefore changes only actor parameters, whereas DCC transfers the impaired actor into onset critic/moments. The all-D cell is asserted identical over the full runner-state tree to untouched D; the unbroken clean trace supplies the historical pause benchmark. Recovery AUC excludes removal and averages the 20 checkpoints at +10--200; the late endpoint averages +160--200. No low-competence seed is filtered. Matching concerns the removal-time world/RNG/counter and initial non-actor learner fields; subsequent actions, data, critic and moments may diverge and are part of the total downstream intervention effect.

'''
    pclaim=r['primary_compound_p']
    import math
    pe=int(math.floor(math.log10(pclaim))) if pclaim>0 else -12
    ptext=(f'{pclaim/10**pe:.2f}\\times10^{{{pe}}}' if 0<pclaim<.0001 else f'{pclaim:.4f}')
    text+=r'\paragraph{One registered compound result.} '+ 'The single compound criterion is '+verdict+', with $p_{\\rm claim}='+ptext+r'$ at $\alpha=.01$; every component is reported in Table~\ref{tab:p14solid}. '
    text+=f'Native AUC cost is '+ci(r['primary']['native_AUC_harm']['CI95'])+', late-window cost '+ci(r['primary']['native_late_harm']['CI95'])+', and replacing only the onset actor changes AUC by '+ci(r['primary']['onset_actor_rescue']['CI95'])+'. '
    text+='The reciprocal actor transfer is '+ci(r['primary']['reciprocal_actor_transfer']['CI95'])+'. '
    text+=f'Seedwise consistency is {r["seed_consistency"]["native_AUC_negative"]}/32 native-negative and {r["seed_consistency"]["actor_rescue_positive"]}/32 rescue-positive. The repaired fraction is {100*r["repaired_fraction_point"]:.1f}\\% (paired-bootstrap 95\\% interval {100*r["repaired_fraction_bootstrap95"][0]:.1f}--{100*r["repaired_fraction_bootstrap95"][1]:.1f}\\%) by ratio of paired means, not an allocation of encoded information.\n\n'+table
    text+=r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/p14_actor_repair.png}
\caption{One prespecified actor-repair result. At removal, the impaired and actor-repaired learners have identical natural D world/RNG/counter, critic and moments; only actor parameters change. Both then learn unrestrictedly, so downstream experience and learner fields can diverge. The reciprocal contrast is against the matched onset-state reference (CCC), not an assumed-identical historical pause. Historical pause is the earlier unbroken clean trajectory. Panel A bands are nominal pointwise 99\% seed intervals. Panel B shows 95\% intervals and, for primary components, thin 99\% whiskers. Small points in the four primary rows are all 32 paired seed effects, not selected examples. Secondary robustness is not a new primary success.}
\label{fig:p14solid}
\end{figure}

'''
    text+=r'\paragraph{Prespecified robustness.} '+ 'The common-fresh-Adam actor effect is '+ci(r['fresh_Adam']['actor_main'])+', with rescue '+ci(r['fresh_Adam']['actor_rescue'])+'; reset-world rescue is '+ci(r['reset_context_actor_rescue'])+'. '
    text+='The corresponding fresh-Adam critic effect is '+ci(r['fresh_Adam']['critic_main'])+', so a smaller critic effect is not declared absent. '
    text+='The primary conditions on the registered 64-episode evaluation batch; the second key is neither selected nor pooled into it. With independent evaluation key 54321, native cost is '+ci(r['alternate_evaluation']['native_AUC_harm'])+', late cost '+ci(r['alternate_evaluation']['native_late_harm'])+' and actor rescue '+ci(r['alternate_evaluation']['onset_actor_rescue'])+'. '
    text+='\n\n'+r'\paragraph{Most of the cost, not full restoration.} '+'Residual rescue-minus-matched-pause is '+ci(r['residual_vs_statepause']['CI95'])+f' (TOST $\\pm.30$, $p={r["residual_vs_statepause"]["TOST_pm_030"]:.4f}$); versus historical pause it is '+ci(r['residual_vs_historicalpause']['CI95'])+f' ($p={r["residual_vs_historicalpause"]["TOST_pm_030"]:.4f}$). '
    text+='Matched state-pause minus historical pause is '+ci(r['statepause_minus_historical']['CI95'])+f' (TOST $\\pm.30$, $p={r["statepause_minus_historical"]["TOST_pm_030"]:.4f}$). '
    hist_status='met' if r['residual_vs_historicalpause']['TOST_pm_030']<.05 else 'not met'
    state_status='met' if r['residual_vs_statepause']['TOST_pm_030']<.05 else 'not met'
    text+='The secondary historical-reference restoration margin is '+hist_status+', whereas the matched-state-reference margin is '+state_status+'. A practical margin is not literal equality, and no full-restoration claim is made. Bootstrap, exact sign and simultaneous-bound sensitivities are in the analysis artifact. The result remains an intervention-specific, finite-horizon actor-state repair in one game and impairment family, not a unique representation mechanism, universal actor dominance or irreversible loss.\n'
    text=text.replace('A focused high-powered actor-repair result', 'An onset actor checkpoint repairs native recovery cost' if r['primary_success'] else 'A fixed 32-seed onset-actor repair test')
    S.joinpath('p14_single_result.tex').write_text(text)
    print('Rendered one-result figure/section; primary criterion',verdict)


if __name__=='__main__':main()
