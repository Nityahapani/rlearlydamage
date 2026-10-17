"""Primary-first exact reconstruction after measured host runtime exceeded the initial plan.

The original scientific study is unchanged/all11 cells complete. Reuse the already-running
first-group full11 reconstruction once, then reconstruct the other groups' five primary/reference
cells and independently load/replay native+restored states for EVERY group. Same32 seeds, not N64.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive-root',default='/home/user/build/p14-reconstruction/archives')
    ap.add_argument('--log-root',default='/home/user/p14-primary-reconstruction-logs')
    ap.add_argument('--no-new-group-after',default='2026-10-17T12:20:00+00:00')
    ap.add_argument('--hard-stop',default='2026-10-17T13:20:00+00:00')
    args=ap.parse_args();logs=Path(args.log_root);logs.mkdir(parents=True,exist_ok=True)
    archive=Path(args.archive_root);cut=datetime.fromisoformat(args.no_new_group_after);end=datetime.fromisoformat(args.hard_stop)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    progress={'role':'same32-seed reconstruction, not additional empirical evidence',
              'scope':'all primary/reference cells; full11 in firstgroup only; independent primary file replay in everygroup',
              'source_worktree':str(ROOT),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
              'start_utc':datetime.now(timezone.utc).isoformat(),'groups':[],'status':'running'}
    def save():(logs/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    def run(command,filename):
        with (logs/filename).open('w') as log:
            try:return subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=max(1,(end-datetime.now(timezone.utc)).total_seconds())).returncode
            except subprocess.TimeoutExpired:return 124
    save()
    for start in range(1200,1232,4):
        seeds=list(range(start,start+4));name=f's{start}to{start+3}';bundle=archive/(name+'.zip')
        print(datetime.now(timezone.utc).isoformat(),'START',name,flush=True)
        if start==1200:
            assert bundle.is_file(),'Wait for original first-group worker; never duplicate it'
            with zipfile.ZipFile(bundle) as z:meta=json.loads(z.read('bundle_manifest.json'))
            assert meta['status']=='complete' and meta['seeds']==seeds and meta['all_cells_exactly_match_frozen_reference']
            count=11;code=0
        else:
            if datetime.now(timezone.utc)>cut:
                progress['status']='incomplete: operational new-group cutoff';save();return
            code=run([sys.executable,'-u','scripts/p14_checkpoint_reproduction.py','--mode','primary',
                      '--seeds',*map(str,seeds),'--archive-root',str(archive)],name+'.log')
            count=5
        record={'seeds':seeds,'recomputed_cell_count':count,'returncode':code}
        if code:
            progress['groups'].append(record);progress['status']='STOP: reconstruction discrepancy/technical failure/time limit';save();raise SystemExit(code)
        replay=run([sys.executable,'-u','scripts/p14_checkpoint_reproduction.py','--mode','replay',
                    '--seeds',*map(str,seeds),'--bundle',str(bundle)],name+'-independent-replay.log')
        record.update(independent_replay_returncode=replay,finished_utc=datetime.now(timezone.utc).isoformat())
        progress['groups'].append(record);save()
        if replay:
            progress['status']='STOP: saved-state replay discrepancy or time limit';save();raise SystemExit(replay)
        print(record['finished_utc'],'DONE',name,'donors/primary curves and independent file replay EXACT',flush=True)
    progress['status']='complete: all32 donor/primary/reference checks and independent native/restored file replays exact; extra6 controls rerun only in firstgroup'
    progress['end_utc']=datetime.now(timezone.utc).isoformat();save();print(progress['status'],flush=True)


if __name__=='__main__':main()
