"""Integrate exactly one complete Phase14 result; never run for incomplete data."""
from pathlib import Path
import json


def ci(v,d=2):return f'${v[0]:+.{d}f}\\ci{{{v[1]:+.{d}f}}}{{{v[2]:+.{d}f}}}$'

def pp(p):
    if p<.0001:
        import math
        if p<=0:return r'<10^{-12}'
        e=int(math.floor(math.log10(p)));return f'{p/10**e:.2f}\\times10^{{{e}}}'
    return f'{p:.4f}'


def main():
    r=json.loads(Path('results/summary/p14_single_result-analysis.json').read_text());assert r['n']==32
    p=Path('paper/main.tex');s=p.read_text();assert '\\label{sec:p14solid}' not in s and '../results/summary/p14_single_result.tex' not in s,'Run only once on baseline manuscript'
    native=ci(r['primary']['native_AUC_harm']['CI95']);late=ci(r['primary']['native_late_harm']['CI95'])
    rescue=ci(r['primary']['onset_actor_rescue']['CI95']);transfer=ci(r['primary']['reciprocal_actor_transfer']['CI95'])
    pc=pp(r['primary_compound_p']);ok=r['primary_success'];status='met' if ok else 'not met'
    old=r'Independent Asterix ($n=8$) has native pause-relative AUC $-1.6\ci{-1.8}{-1.3}$ and actor allocation 98\%; its compound criterion passes the new five-claim family.'
    new='A focused, separately registered Asterix confirmation ($n=32$ unused seeds) has native pause-relative AUC '+native+', late cost '+late+' and onset-actor-only rescue '+rescue+'. The single four-component practical-effect criterion is '+status+' ($p_{\\rm claim}='+pc+r'$, $\alpha=.01$); the donor contains no clean-fork progress.'
    assert s.count(old)==2;s=s.replace(old,new)
    anchor=r'\paragraph{Contributions.}'
    para=r'''\paragraph{One focused test of actual beyond-time harm.}
'''
    para+='The new32-seed test in Section~\\ref{sec:p14solid} restores only the pre-impairment actor checkpoint in the natural impaired learner, leaving its critic, Adam history and future world untouched. It tests native historical-pause AUC and late deficits, actor rescue and reciprocal transfer together, rather than selecting the strongest of these endpoints. '
    para+='Its fixed compound criterion is '+status+'. '
    para+='The prior independent eight-seed Asterix cohort motivates this selected condition but is not pooled into the new primary. Common fresh moments, a reset world and a second evaluation key are prespecified robustness checks, not substitute primary tests.\n\n'
    para=para.replace('new32','new 32')
    s=s.replace(anchor,para+anchor,1)
    phrase='The new compound tests are read with their phase-wise error control, not as a uniform confirmation label.'
    s=s.replace(phrase,phrase+' A separate 32-seed confirmation concentrates on natural-context onset-actor rescue with explicit practical thresholds (Section~\\ref{sec:p14solid}).',1)
    s=s.replace('Only the five new primary claims share the Phase-13 Holm family; missing/incomplete studies receive p=1.', 'Only the five Phase-13 primary claims share that Holm family; missing/incomplete studies receive p=1. Phase~14 has one separate compound at $\\alpha=.01$.',1)
    ledger=r'D prospective diagnostic & held-out squared error & acute-gap versus acute-gap+KL & 16 fresh & same five-claim family; 10\% gain & fixed-coefficient cost prediction, not failure probability \\'
    assert ledger in s
    s=s.replace(ledger,ledger+'\n'+r'Focused actor repair (Tier II) & native pause AUC, late cost, reciprocal rescue & onset actor only and reciprocal actor & 32 fresh & one Phase-14 compound; alpha=.01 & '+status+r'; fixed practical thresholds, no pooling \\',1)
    target=r'\section{In PQN-R, the network carries most of the damage}\label{sec:pqnr}'
    assert target in s
    s=s.replace(target,r'\clearpage'+'\n'+r'\input{../results/summary/p14_single_result.tex}'+'\n'+r'\clearpage'+'\n\n'+target,1)
    target=r'\paragraph{Pathway and prospective checks do not automatically upgrade mechanisms.}'
    para=r'\paragraph{A checkpoint repair, not borrowed clean progress.}'+'\n'
    para+='Section~\\ref{sec:p14solid} is a single selected-condition confirmation on32 unused training IDs, not a new task added to the generalization count. Its actor donor is the learner\'s own shared onset checkpoint, so the intervention cannot borrow the clean fork\'s intervening improvements. '
    para+='The primary is '+status+', and the natural actor rescue is '+rescue+' versus native cost '+native+'. '
    para+='Reciprocal transfer and the specified optimizer, context and evaluation checks are reported alongside it. This is operational evidence about this actor-state intervention, not an additive partition of naturally stored information or a proof that the same rollback works on other tasks.\n\n'
    s=s.replace(target,para.replace('on32','on 32')+target,1)
    # Add one practical interpretation while keeping the original oracle-transplant limitation.
    practical='This oracle clean fork is not generally available in deployment.'
    s=s.replace(practical,practical+' The focused onset-actor test instead uses an already-existing past checkpoint, but remains one controlled assay rather than a validated automatic rollback policy.',1)
    scope='but they are not a random sample of new tasks and do not establish a prevalence law.'
    s=s.replace(scope,scope+' Phase~14 increases independent seeds for one selected Asterix condition, not environment or architecture coverage.',1)
    repl='The new compound tests are read with their phase-wise error control'
    # Reproducibility and audit trail; all legacy outcomes stay intact.
    repro='Appendix~\\ref{app:prov} maps every table to its source files.'
    s=s.replace(repro,repro+' Phase~14 archives all11 cell curves per seed, both requested evaluation-key curves and donor-state fingerprints; full transient runner trees are not saved. The source, fixed-seed protocol and software-only checks reproduce the study.',1).replace('all11','all 11')
    provenance=r'controlled label exposure & raw \url{p13-E-*.json}, state-equality checks & controlled assay; original movement mechanism not thereby proven \\'
    assert provenance in s
    s=s.replace(provenance,provenance+'\n'+r'Table~\ref{tab:p14solid}, Figure~\ref{fig:p14solid} & \url{p14_single_result-analysis.json}, \url{p14_single_result.tex}, raw \url{p14-actor-repair-*.json} & fixed32 design6262b2b; single compound alpha=.01 \\',1)
    timeline=r'\paragraph{Interim looks: what they could and could not have influenced.}'
    assert timeline in s
    note=r'\paragraph{One focused Phase-14 confirmation.}'+'\n'
    note+='The design was public before target data (6262b2b): 32 unused seeds1200--1231, all11 requested cells, fixed practical thresholds and one four-component intersection-union at $\\alpha=.01$, plus seed consistency. The analysis source and synthetic software checks were committed before any formal target inference. No target primary was computed until all32 completed, and there was no seed expansion, setting switch, key selection or secondary rescue. The primary criterion is '+status+'. Earlier Asterix cohorts are prior evidence only; this alpha level is not retrospective error control across the paper. Full runner trees are transient, while all requested curves and exact donor fingerprints are archived.\n\n'
    s=s.replace(timeline,note.replace('seeds1200','seeds 1200').replace('all11','all 11').replace('all32','all 32')+timeline,1)
    p.write_text(s)
    print('Integrated exactly one Phase14 result; criterion',status)


if __name__=='__main__':main()
