"""Budget-gated independent-seed CartPole replication (H1), registered before these seeds exist.

Runs only after the main Phase 12 queue is finished and only if the whole test can start before
2026-10-15 00:00 UTC. Both four-seed chunks are then run, without looking at either chunk's outcomes.
Also enforces the previously stated Freeway new-chunk cutoff at 23:30 UTC, including across midnight.
Training logs are never read by this orchestrator.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
L=Path('/home/user/p12logs')
Q=Path('/home/user/p12queue.log')
FREEWAY_CUTOFF=datetime(2026,10,14,23,30,tzinfo=timezone.utc).timestamp()
START_CUTOFF=datetime(2026,10,15,0,0,tzinfo=timezone.utc).timestamp()
H2_START_CUTOFF=datetime(2026,10,14,23,25,tzinfo=timezone.utc).timestamp()


def log(message):
    with Q.open('a') as f:
        f.write(f"{datetime.now(timezone.utc):%H:%M:%S} {message}\n")
    print(message,flush=True)


def main():
    L.mkdir(exist_ok=True,parents=True)
    timer_applied=False
    while 'queue v3 finished' not in (Q.read_text() if Q.exists() else ''):
        if time.time()>=FREEWAY_CUTOFF and not timer_applied:
            # The current main queue checks this marker before starting each Freeway chunk.
            (L/'freeway-rflip.abandoned').touch()
            log('Freeway start cutoff reached; no additional G4 chunks (active chunk may finish)')
            timer_applied=True
        time.sleep(15)
    if time.time()>START_CUTOFF:
        log('SKIP H1 independent CartPole replication (whole-test start cutoff)')
        return
    for seeds in [list(range(600,604)),list(range(604,608))]:
        name=f'h1-cartpole-fresh-s{seeds[0]}'
        if (L/f'{name}.done').exists():
            continue
        log('start '+name)
        cmd=[sys.executable,'-u','scripts/surgery_ppo.py','--env','CartPole-v1','--deficit','reward_flip',
             '--flip-prob','1.0','--window','10','60','--recovery','150','--eval-fork',
             '--arms','CCC','DCC','CDC','CCD','DDC','DCD','CDD','DDD',
             '--seeds',*[str(s) for s in seeds],'--tag',f'p12-{name}']
        with (L/f'{name}.log').open('w') as f:
            res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if res.returncode:
            log('FAILED '+name+'; incomplete H1 is descriptive only')
            return
        (L/f'{name}.done').touch()
        log('done '+name)
    log('H1 replication finished')
    if time.time()>H2_START_CUTOFF:
        log('SKIP H2 independent actuator-noise replication (whole-test start cutoff 23:25)')
        return
    for seeds in [list(range(700,704)),list(range(704,708))]:
        name=f'h2-si-actnoise-fresh-s{seeds[0]}'
        if (L/f'{name}.done').exists():
            continue
        log('start '+name)
        cmd=[sys.executable,'-u','scripts/surgery_ppo.py','--env','SpaceInvaders-MinAtar','--deficit','action_noise',
             '--noise-prob','0.5','--window','100','150','--recovery','150','--eval-fork',
             '--arms','CCC','DCC','CDC','CCD','DDC','DCD','CDD','DDD',
             '--seeds',*[str(s) for s in seeds],'--tag',f'p12-{name}']
        with (L/f'{name}.log').open('w') as f:
            res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if res.returncode:
            log('FAILED '+name+'; incomplete H2 is descriptive only')
            return
        (L/f'{name}.done').touch()
        log('done '+name)
    log('H2 replication finished')


if __name__=='__main__':
    main()
