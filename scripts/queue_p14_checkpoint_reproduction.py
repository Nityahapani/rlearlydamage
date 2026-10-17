"""Fixed operational reproduction schedule; no new empirical hypothesis or sample extension."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive-root',default='/home/user/build/p14-reconstruction/archives')
    ap.add_argument('--log-root',default='/home/user/p14-reconstruction-logs')
    ap.add_argument('--no-new-group-after',default='2026-10-17T12:20:00+00:00')
    ap.add_argument('--hard-stop',default='2026-10-17T13:20:00+00:00')
    args=ap.parse_args()
    logs=Path(args.log_root);logs.mkdir(parents=True,exist_ok=True)
    archive=Path(args.archive_root);archive.mkdir(parents=True,exist_ok=True)
    cut=datetime.fromisoformat(args.no_new_group_after);end=datetime.fromisoformat(args.hard_stop)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    progress={'role':'same-seed reconstruction; not additional independent evidence','source_worktree':str(ROOT),
              'start_utc':datetime.now(timezone.utc).isoformat(),'groups':[],'status':'running'}
    def save():
        (logs/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    save()
    for start in range(1200,1232,4):
        if datetime.now(timezone.utc)>cut:
            progress['status']='incomplete: operational new-group cutoff';save();return
        seeds=list(range(start,start+4));name=f's{start}to{start+3}'
        print(datetime.now(timezone.utc).isoformat(),'START',name,flush=True)
        command=[sys.executable,'-u','scripts/p14_checkpoint_reproduction.py','--mode','full',
                 '--seeds',*map(str,seeds),'--archive-root',str(archive)]
        with (logs/(name+'.log')).open('w') as log:
            try:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 timeout=max(1,(end-datetime.now(timezone.utc)).total_seconds()))
            except subprocess.TimeoutExpired:
                progress['status']='incomplete: hard time limit';save();raise SystemExit(2)
        record={'seeds':seeds,'returncode':r.returncode,'finished_utc':datetime.now(timezone.utc).isoformat()}
        if r.returncode:
            progress['groups'].append(record);progress['status']='STOP: reproduction discrepancy or software failure';save();raise SystemExit(r.returncode)
        if start==1200:
            # Fixed first group, two primary cells, independent fresh template/process from saved files.
            with (logs/(name+'-independent-replay.log')).open('w') as log:
                replay=subprocess.run([sys.executable,'-u','scripts/p14_checkpoint_reproduction.py','--mode','replay',
                    '--seeds',*map(str,seeds),'--bundle',str(archive/(name+'.zip'))],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,(end-datetime.now(timezone.utc)).total_seconds()))
            record['independent_first_group_replay_returncode']=replay.returncode
            if replay.returncode:
                progress['groups'].append(record);progress['status']='STOP: independent saved-state replay discrepancy';save();raise SystemExit(replay.returncode)
        progress['groups'].append(record);save()
        print(datetime.now(timezone.utc).isoformat(),'DONE',name,'exact full-cell/serialization checks passed',flush=True)
    progress['status']='complete: all32 same seeds/all11 cells matched; first-group independent file replay passed'
    progress['end_utc']=datetime.now(timezone.utc).isoformat();save()
    print(progress['status'],flush=True)


if __name__=='__main__':main()
