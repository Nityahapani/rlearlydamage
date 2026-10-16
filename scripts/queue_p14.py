"""One registered n=32 cohort, eight sequential four-seed jobs, no outcome peeking or seed extension."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
L=Path('/home/user/p14logs');L.mkdir(exist_ok=True)
Q=Path('/home/user/p14queue.log')


def say(msg):
    text=f'{datetime.now(timezone.utc):%H:%M:%S} {msg}'
    with Q.open('a') as f:f.write(text+'\n')
    print(text,flush=True)


def main():
    p=json.loads((ROOT/'prereg/phase14.json').read_text());cut=datetime.fromisoformat(p['no_new_chunk_after_utc'])
    for i in range(0,len(p['training_seeds']),4):
        seeds=p['training_seeds'][i:i+4];tag=f'actor-repair-s{seeds[0]}to{seeds[-1]}'
        if (L/(tag+'.done')).exists():continue
        if datetime.now(timezone.utc)>cut:say('new-chunk budget cutoff: incomplete cohort, no primary verdict');return
        say(f'start chunk {seeds[0]}–{seeds[-1]} of fixed32')
        with (L/(tag+'.log')).open('w') as f:
            r=subprocess.run([sys.executable,'-u','scripts/p14_actor_repair.py','--tag',tag,'--seeds',*map(str,seeds)],
                             cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        subprocess.run(['git','add','results/summary','runs'],cwd=ROOT,check=True)
        subprocess.run(['git','-c','user.name=Nityahapani','-c','user.email=118457303+Nityahapani@users.noreply.github.com',
                        'commit','-qm',f'Phase14 single result: archive chunk {seeds[0]}–{seeds[-1]} (primary withheld)'],cwd=ROOT,check=True)
        if r.returncode:say(f'FAILED chunk {seeds[0]}–{seeds[-1]}, retained outputs; no primary inference');return
        (L/(tag+'.done')).touch();say(f'finished chunk {seeds[0]}–{seeds[-1]}')
    say('fixed32 cohort complete; primary analysis may now run')


if __name__=='__main__':main()
