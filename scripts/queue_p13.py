"""Sequential Phase 13 queue. Fixed cohort-level gates; progress does not expose outcomes.
All requested results are committed after each completed chunk. Analysis is not run mid-cohort.
"""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOG=Path('/home/user/p13queue.log');L=Path('/home/user/p13logs');L.mkdir(exist_ok=True)


def say(msg):
    line=f'{datetime.now(timezone.utc):%H:%M:%S} {msg}'
    with LOG.open('a') as f:f.write(line+'\n')
    print(line,flush=True)


def commit_outputs(study,seeds):
    subprocess.run(['git','add','results/summary','runs'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=Nityahapani','-c','user.email=118457303+Nityahapani@users.noreply.github.com',
                    'commit','-qm',f'Phase 13 {study}: saved chunk {seeds[0]}–{seeds[-1]} (no interim primary analysis)'],cwd=ROOT,check=True)


def main():
    p=json.loads((ROOT/'prereg/phase13.json').read_text())
    end=datetime.fromisoformat(p['budget_end_utc'])
    cut=datetime.fromisoformat(p['no_new_chunk_after_utc'])
    for study in p['queue_priority']:
        if datetime.now(timezone.utc)>=end: say('budget ended; remaining studies not run');break
        gate=p['whole_study_start_cutoffs_utc'].get(study)
        if gate and datetime.now(timezone.utc)>datetime.fromisoformat(gate):say(f'SKIP study {study}: whole-study budget gate');continue
        if study=='D' and not (ROOT/'prereg/p13_frozen_predictors.json').exists():
            say('SKIP study D: prediction coefficients not frozen');continue
        S=p['studies'][study]['seeds'];ok=True
        for j in range(0,len(S),4):
            seeds=S[j:j+4];tag=f'p13-{study}-s{seeds[0]}to{seeds[-1]}'
            if (L/f'{tag}.done').exists():continue
            if datetime.now(timezone.utc)>cut: say(f'SKIP remaining {study}: new-chunk cutoff');ok=False;break
            if study in 'ABC':
                cmd=[sys.executable,'-u','scripts/p13_surgery.py','--study',study,'--tag',tag,'--seeds',*map(str,seeds)]
            elif study=='E':
                cmd=[sys.executable,'-u','scripts/p13_replay_exposure.py','--tag',tag,'--seeds',*map(str,seeds)]
            else:
                cmd=[sys.executable,'-u','scripts/ponr_sweep.py','--onset','100','--durations','300','--recovery','250',
                     '--arms','D_online','--tag',tag,'--seeds',*map(str,seeds)]
            say(f'start study {study}, chunk {seeds[0]}–{seeds[-1]}')
            with (L/f'{tag}.log').open('w') as f:
                r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:
                say(f'FAILED study {study}, chunk {seeds[0]}–{seeds[-1]} code {r.returncode}; stop this study')
                ok=False;break
            commit_outputs(study,seeds)
            (L/f'{tag}.done').touch()
            say(f'finished study {study}, chunk {seeds[0]}–{seeds[-1]}')
        say(('complete study ' if ok else 'incomplete study ')+study)
    say('Phase 13 queue finished')


if __name__=='__main__':main()
