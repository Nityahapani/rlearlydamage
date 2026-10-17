"""Render appendix data/provenance from frozen results and checked release metadata only."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'results/summary/p14_single_result-analysis.json'
    data=path.read_bytes()
    assert data==subprocess.check_output(['git','show','74c6ceb:'+str(path.relative_to(ROOT))],cwd=ROOT)
    r=json.loads(data)
    out=ROOT/'paper/sections';out.mkdir(exist_ok=True)
    keys=['native_AUC_harm','native_late_harm','onset_actor_rescue','reciprocal_actor_transfer']
    lines=[r'\begin{table}[p]',r'\centering',
      r'\caption{All original primary seed effects, in average evaluation-return points. These are the32 original inferential units, not the later same-seed reconstructions. Signs are retained; no seed or large effect is excluded. Full precision is in the linked CSV/JSON.}',
      r'\label{app:seedtable}',r'\small',r'\begin{tabular}{rcccc}',r'\toprule',
      r'seed & native $\gamma_H$ & late $\ell_H$ & actor rescue $\tau_D$ & reciprocal $\delta_D$ \\',r'\midrule']
    for j,seed in enumerate(r['seeds']):
        values=[r['primary'][k]['per_seed'][j] for k in keys]
        lines.append(str(seed)+' & '+' & '.join(f'${v:+.3f}$' for v in values)+r' \\')
    lines += [r'\midrule','mean & '+' & '.join(f"${r['primary'][k]['CI95'][0]:+.3f}$" for k in keys)+r' \\',
              r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'p14_seed_rows.tex').write_text('\n'.join(lines).replace('the32','the 32')+'\n')
    full=[r'\begin{table}[ht]',r'\centering',
          r'\caption{Full numerical primary record:95/99\% nominal paired-seed intervals and secondary simultaneous one-sided95\% Bonferroni bounds. The latter uses four components, not all analyses in the paper.}',
          r'\label{app:fullnumeric}',r'\small',r'\resizebox{\linewidth}{!}{%',
          r'\begin{tabular}{lcccc}',r'\toprule',
          r'component & mean & 95\% interval & 99\% interval & simultaneous one-sided95\% bound \\',r'\midrule']
    for label,k in zip(['native AUC','late cost','actor rescue','reciprocal'],keys):
        a=r['primary'][k]['CI95'];b=r['primary'][k]['CI99'];c=r['simultaneous_one_sided_bounds'][k]
        sign=r'\leq' if c['direction']=='less' else r'\geq'
        full.append(label+f' & ${a[0]:+.3f}$ & $[{a[1]:+.3f},{a[2]:+.3f}]$ & $[{b[1]:+.3f},{b[2]:+.3f}]$ & $'+sign+f" {c['one_sided_95_family_bound']:+.3f}$"+r' \\')
    full += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    (out/'p14_primary_full_table.tex').write_text('\n'.join(full).replace('record:95','record: 95').replace('one-sided95','one-sided 95')+'\n')
    release=ROOT/'docs/p14_reconstructed_state_release.json'
    if release.exists():
        q=json.loads(release.read_text());groups=q['verified_groups']
        seeds=sum((g['seeds'] for g in groups),[])
        assert len(seeds)==len(set(seeds)) and set(seeds)<=set(range(1200,1232))
        assert all(g['primary_reference_curves_exact'] and g['public_download_url'] and g['bundle_sha256'] for g in groups)
        n=len(seeds);count=len(groups)
        text=(r'\paragraph{Checked reconstruction status.} The original experiment did not retain full donor files. '
              f'The later, separately timestamped reconstruction provides checked public P/C/D bundles for {n}/32 original seeds ({count}/8 groups). '
              r'Each published group matches the original numerical donor fingerprints and all five primary/reference cell curves, after serialization/restoration of its donors. '
              r'These are same-seed reconstructions, not new independent observations. ')
        fulln=sum(len(g['seeds']) for g in groups if g['recomputed_cell_count']==11)
        text+=f'The additional six original secondary cells were rerun for {fulln}/32 seeds; their original results remain available for all32. '
        if groups and all(g['independent_replay_pass'] for g in groups):
            text+=r'All published groups also pass fresh-process loading and both-key native/restored replay at all21 checkpoints. '
        text+=r'Bundle URLs and SHA256 values are in the \artifact{docs/p14_reconstructed_state_release.json}{checked reconstructed-state release index}. '
        if n<32:text+=r'The reconstruction is incomplete; no unverified seed group is described as available. '
    else:
        text=(r'\paragraph{Reconstruction status in this working draft.} The exact reconstruction is still running; no completed public donor bundle is asserted here. '
              r'Original mean curves and fingerprint/check records remain available for all32 primary seeds. '
              r'The final artifact status will be filled only from verified per-group reports and publication receipts, not from the intended schedule. ')
    (out/'reconstruction_status.tex').write_text(text.replace('all11','all 11').replace('all21','all 21').replace('all32','all 32')+'\n')
    print('Rendered original32 seed rows,full interval table and factual reconstruction status')


if __name__=='__main__':main()
