"""Operational post-queue build. Blocks on a Linux process-exit event, not outcome/file polling.

The registered analysis itself refuses anything short of all32 complete seeds. Outputs are
NOT automatically committed, pushed or presented: a human/assistant quality review follows.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import os
import select
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue-pid',required=True,type=int);a=ap.parse_args()
    try:
        fd=os.pidfd_open(a.queue_pid,0)
    except ProcessLookupError:fd=None
    print('Waiting for queue exit; no target inference or JAX concurrency',flush=True)
    if fd is not None:
        poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    print('Queue exited at',datetime.now(timezone.utc).isoformat(),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    for s in ['analyze_p14.py','make_p14_result.py']:
        subprocess.run([sys.executable,'-u','scripts/'+s],check=True,env=env)
    if '../results/summary/p14_single_result.tex' not in Path('paper/main.tex').read_text():
        subprocess.run([sys.executable,'-u','scripts/integrate_p14.py'],check=True,env=env)
    subprocess.run([sys.executable,'-u','scripts/catalog_results.py'],check=True,env=env)
    tex=Path('/home/user/build/texbin/tectonic')
    with Path('/home/user/p14paper-build.log').open('w') as f:
        subprocess.run([str(tex),'-k','--keep-logs','main.tex'],cwd='paper',stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    print('Built paper/main.pdf at',datetime.now(timezone.utc).isoformat(),'; review before publication',flush=True)


if __name__=='__main__':main()
