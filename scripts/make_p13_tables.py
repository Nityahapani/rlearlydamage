"""Render Phase 13 analysis without hand-copied values or mixing reference tiers.
The emitted fragments are complete tables for the final paper, not new analyses.
"""
from __future__ import annotations
import json
from pathlib import Path

S=Path('results/summary')


def ci(v,d=1):
    e,lo,hi=v;star=r'\sig' if lo>0 or hi<0 else ''
    return f'${e:+.{d}f}\\ci{{{lo:+.{d}f}}}{{{hi:+.{d}f}}}{star}$'


def pv(p):
    return r'$<10^{-4}$' if p<1e-4 else f'${p:.3f}$'


def table(caption,label,columns,head,rows):
    return '\n'.join([r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',
                      r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+columns+'}',r'\toprule',head+r' \\',r'\midrule',
                      *rows,r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])


def main():
    I=json.loads((S/'p13_confirmatory-analysis.json').read_text());R=I['studies'];frags={}
    names={'A':'CartPole flip','B':'Asterix flip','C':'SI actuator noise','E':'controlled replay labels','D':'prospective diagnostic'}
    rows=[]
    for k in I['family']:
        r=R.get(k)
        if r is None:rows.append(f'{k}: {names[k]} & -- & -- & -- & not completed'+r' \\');continue
        p=r.get('primary_p',r.get('primary_one_sided_p',1))
        status='complete' if r.get('complete',True) else 'descriptive only'
        rows.append(f'{k}: {names[k]} & {r["n"]} & {pv(p)} & {pv(r["phase13_Holm_p"])} & {status}'+r' \\')
    frags['family']=table(r'Five \emph{new} Phase-13 primary claim tests. A--C use the registered intersection-union (maximum component $p$) construction, including paired Shapley dominance rather than only a largest point estimate. Holm is over this fixed family of five; unrun/incomplete tests receive $p=1$. This is not retrospective paper-wide control. E additionally requires a point effect $\leq-1$; D requires a point MSE reduction $\geq10\%$.','tab:p13family','lcccc','study & $n$ & primary $p$ & phase-wise Holm $p$ & completion',rows)
    rows=[]
    for k in ['A','B','C']:
        r=R.get(k)
        if r is None:continue
        ref='pause' if k in ['A','B'] else 'fork';g=r['groups'][ref+'/clean/donor']
        sh='--' if g['shares'] is None else f"{g['shares']['actor']:.0f}\\%"
        rows.append(f'{k}: {names[k]} & {r["n"]} & '+('Tier II' if k in ['A','B'] else 'Tier I')+' & '+ci(r['native_damage'])+' & '+ci(g['total'])+' & '+ci(g['effects']['actor'])+' & '+sh+r' \\')
    frags['primary']=table(r'Independent-seed localization at the planned sample. Native damage is natural D minus the unbroken historical clean pause. The surgery gap uses the indicated reference under the clean future context; it is not automatically identical to that native estimand. Pause-reference rows target Tier II, but support is asserted only when direct interventions and native-harm criteria pass; Tier-I actuator noise tests only same-update gap localization. Shares are descriptive allocations, not encoded information.','tab:p13primary','lclcccc','study & $n$ & tier/reference & native pause-relative AUC & surgery AUC gap & actor main effect & actor share',rows)
    rows=[]
    for k in ['A','B','C']:
        r=R.get(k)
        if r is None:continue
        q=r['natural_vs_standardized']
        rows.append(f'{k}: {names[k]} & '+ci(q['natural_D_AUC'])+' & '+ci(q['standardized_DDD_AUC'])+' & '+ci(q['difference'])+' & '+f"$\\pm{q['margin']:.2f}$"+' & '+pv(q['TOST_joint'])+r' \\')
    frags['natural']=table(r'Natural damaged continuation versus standardized-context DDD, paired over independent seeds. The natural cell is asserted equal over the full runner tree, not merely the parameters. Context-equivalence margin: $5\%$ of native clean mean AUC; TOST retains uncertainty/covariance in that scale. Nonsignificance alone is not evidence of context neutrality.','tab:p13natural','lccccc','study & natural D AUC & standardized DDD AUC & standardized $-$ natural & margin & TOST $p$',rows)
    rows=[]
    for k in ['A','B','C']:
        r=R.get(k)
        if r is None:continue
        for name,x in r['contexts'].items():
            a=ci(x.get('actor',x.get('bad_actor_into_reference')))
            rescue=ci(x['good_actor_into_damaged']) if 'good_actor_into_damaged' in x else '--'
            rows.append(f'{k}: '+name.replace('_',r'\_')+' & '+ci(x['total'])+' & '+a+' & '+rescue+r' \\')
    frags['contexts']=table(r'Secondary context and compatibility checks, always reported. An actor-transfer contrast is not a unique natural decomposition. In zero-moment rows all cells restart Adam, and the actor column is the two-factor main effect; this is an alternative update procedure, not an untouched natural optimizer. All intervals are nominal paired $95\%$ $t$-intervals.','tab:p13contexts','lccc','study: reference/context/moments & total AUC gap & bad actor effect / zero-moment main effect & good actor rescue',rows)
    e=R.get('E')
    if e:
        rows=[n+' & '+ci(e[k],2)+r' \\' for n,k in [('naturally matched R--U total','total'),('parameter main effect','parameters'),('moment main effect','moments'),('interaction','interaction'),('R--U with common fresh moments','common_zero_moment_difference')]]
        frags['replay']=table(r'Controlled replay-label exposure after complete washout ($n=8$). The genuine transition tuples, ages, state/action identities, collection policy and update schedule are matched. After $K=25$ the full buffer and every non-learner field are exactly equal; remaining differences are in the learner state. The RR--UU total compares naturally matched parameter/moment histories. Crossing and fresh moments are secondary procedure-specific probes, not a unique internal encoding analysis.','tab:p13replay','lc','post-washout contrast & recovery AUC difference',rows)
    d=R.get('D')
    if d:
        rows=[]
        for m,v in d['MSE'].items():
            rr=d['rank'][m];rho='--' if rr is None else f'${rr:+.2f}$'
            rows.append(m.replace('_',r'\_')+f' & ${v:.2f}$ & '+rho+r' \\')
        frags['prediction']=table(r'Prospective held-out diagnostic performance ($n=16$). Coefficients and feature scales were frozen on the earlier 32 before any new validation run. No fitting or model selection uses these outcomes. Primary utility is acute-gap versus acute-gap+KL held-out squared error, with a $10\%$ point-gain requirement and the five-claim Holm criterion. Correlations are secondary; this is neither new-task nor failure-probability validation.','tab:p13prediction','lcc','frozen model & held-out MSE & rank correlation',rows)
    out=S/'p13_tables.tex';out.write_text('\n'.join('% P13 TABLE '+k+'\n'+v for k,v in frags.items()))
    print('Rendered fragments:',list(frags))


if __name__=='__main__':main()
