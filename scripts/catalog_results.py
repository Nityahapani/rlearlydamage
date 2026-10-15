"""Inventory saved experimental outputs and source-commit availability.
The catalog is descriptive metadata, not a new experiment or inference rule.
"""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path

def main():
    root=Path.cwd();files=[];runs=[]
    for base in ['results','runs']:
        for p in sorted(Path(base).rglob('*')):
            if p.is_file() and p.suffix in {'.json','.md','.npz','.tex'}:
                files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in sorted(Path('results/summary').glob('*.json')):
        d=json.loads(p.read_text())
        if 'manifest' not in d or 'seeds' not in d:continue
        m=Path('runs')/d['manifest'];md=json.loads(m.read_text()) if m.exists() else {}
        proto=d.get('protocol',{})
        expected=proto.get('arms',[])
        if isinstance(expected,str):expected=[expected]
        observed=list(d.get('arms',{}))
        if 'cells' in d and proto.get('cells'):
            expected=['/'.join(c[k] for k in ['reference','context','opt','code']) for c in proto['cells']]
            observed=list(d['cells'])
        elif d.get('durations') and isinstance(d.get('arms'),dict):
            expected=[f'{dur}/{a}' for dur in d['durations'] for a in expected]
            observed=[f'{dur}/{a}' for dur,arms in d['arms'].items() for a in arms]
        elif 'configs' in d:
            expected=list(proto.get('configs',d['configs']))
            observed=list(d['configs'])
        status=('complete requested arms' if expected and set(expected).issubset(observed) else
                'partial requested arms' if expected else 'see protocol-specific summary')
        h=md.get('git',{}).get('commit')
        present=bool(h and subprocess.run(['git','cat-file','-e',h+'^{commit}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0)
        runs.append({'summary':str(p),'manifest':str(m),'seeds':d['seeds'],'status':status,
                     'expected_arms':expected,'observed_arms':observed,'recorded_source_commit':h,'source_commit_available':present})
    dataset=Path('docs/result_inventory.json')
    dataset.write_text(json.dumps({'files':files,'run_summaries':runs},indent=2)+'\n')
    train=['src/rled/agents/ppo.py','src/rled/agents/pqn.py','scripts/surgery_ppo.py','scripts/pqn_surgery.py']
    ver='79700c2'
    checks={}
    for f in train:
        old=subprocess.check_output(['git','show',f'{ver}:{f}'])
        checks[f]={'matches_public_79700c2':old==Path(f).read_bytes(),'sha256':hashlib.sha256(Path(f).read_bytes()).hexdigest()}
    missing=sorted({r['recorded_source_commit'] for r in runs if r['recorded_source_commit'] and not r['source_commit_available']})
    P=Path('docs/result_availability.md')
    lines=['# Result archive and completion status','',
       'This inventory records saved outputs; it does not invent missing experiments. All saved run summaries, manifests, analyses and available NPZ arrays are committed. See `result_inventory.json` for file hashes and requested/observed arms.', '',
       '## Phase 12','',
       '- Complete at planned n=8: G1 CartPole flip, G2 Acrobot flip, G3 actuator noise, G5 observation noise, G6 extinction, G7 PQN-R flip.',
       '- G4 Freeway: four complete factorial seeds (0–3); partial arms for seeds 4–5 are archived but excluded from factorial aggregation. Seeds 6–7 did not run. The four-seed analysis is descriptive, never counted as a confirmatory success.',
       '- Optional independent-seed replications H1 (600–607) and H2 (700–707) did not start.', '',
       '## Raw data and large generated states','',
       'Per-seed score curves and raw derived predictor values are preserved in summary JSON/Markdown files. All NPZ arrays that survived the sandbox reset are included, including removal-parameter arrays and evaluation-state banks. Full runner-state pickles and some redundant onset/short-duration weight dumps had been placed in a temporary snapshot-excluded cache; they did not survive the reset. They are not falsely described as archived. Regenerate them from the saved seed/configuration when needed.', '',
       '## Source hashes after the sandbox reset','',
       'Some original local-only Git objects were lost before they could be pushed. Original hashes in run manifests are retained, not rewritten. The available public pre-run revision `79700c2` has the same four Phase-12 training-file contents as the recovered source; changes in later local commits were analysis, documentation or result artifacts. This is a reconstruction aid, not a claim that the unavailable whole Git trees were recovered.', '',
       '| training file | matches public 79700c2? |','|---|---|']
    lines += [f'| `{f}` | {c["matches_public_79700c2"]} |' for f,c in checks.items()]
    lines += ['', 'Recorded source commits unavailable in the restored clone: '+(', '.join('`'+h[:7]+'`' for h in missing) if missing else 'none')+'.',
              '', 'A deterministic rerun must preserve each manifest’s seed group, configuration, checkpoint grid and package versions. The original experiment code and the published recovered source are both identified; no missing outcomes are imputed.']
    lines += ['', '## Phase 13 follow-up (all planned cohorts complete)', '',
              'The next registered phase completes A/B/C/E at 8 independent seeds each and prospective D at 16. Phase-12 H1/H2 were not retroactively completed: the formerly unused IDs are used in expanded Phase-13 protocols and the cohorts are not double-counted as additional tasks.',
              'The new inference family is fixed at five claims, with registered compound tests and Holm correction. Only B (Asterix pause) and C (actuator-noise gap) meet it. A has clear finite-window native harm but its stronger compound dominance test does not pass; E and D retain their null/inconclusive primary outcomes.',
              'Natural D identity and controlled washout checks, all cell curves, frozen predictor coefficients, all prospective feature rows and surviving checkpoint arrays are archived. No fresh-seed result is inferred from an unrun or partial cohort.']
    P.write_text('\n'.join(lines)+'\n')
    print('Catalog:',len(files),'files,',len(runs),'run summaries; NPZ files:',sum(f['path'].endswith('.npz') for f in files))
    print('Phase-12 training files match public code:',all(c['matches_public_79700c2'] for c in checks.values()))
    print('Missing original local commit objects:',len(missing))
if __name__=='__main__':main()
