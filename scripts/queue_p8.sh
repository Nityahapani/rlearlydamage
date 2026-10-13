#!/usr/bin/env bash
# Phase 8 queue: sequential, resumable (a job is skipped once its done-marker exists).
set -u
cd /home/user/rlearlydamage
. .venv/bin/activate
L=/home/user/p8logs; mkdir -p $L
run() {  # name, command...
  local name=$1; shift
  if [ -f $L/$name.done ]; then echo "skip $name"; return; fi
  echo "=== start $name $(date)"; "$@" > $L/$name.log 2>&1 && touch $L/$name.done && echo "=== done $name $(date)" || echo "=== FAILED $name $(date)"
}
REF=results/summary/20261011-p2-recovery-move-ppo-SpaceInvaders-MinAtar-fea755e.json
run J1a python -u scripts/surgery_ppo.py --seeds 4 5 6 7 --reference $REF --arms CCC CCD CDC CDD DCC DCD DDC DDD CCC+bD DDD+bC CCC+bDr DDD+bCr --tag p8-surgery-move-late-s4to7
run J1b python -u scripts/surgery_ppo.py --seeds 8 9 10 11 --reference $REF --arms CCC CCD CDC CDD DCC DCD DDC DDD CCC+bD DDD+bC CCC+bDr DDD+bCr --tag p8-surgery-move-late-s8to11
run J2 python -u scripts/surgery_ppo.py --seeds 0 1 2 3 --reference $REF --arms DDD+bC DDD+bCr CCC+bDr --tag p8-dataarms-move-late-s0to3
run J4a python -u scripts/pqn_surgery.py --late-onset 250 --duration 100 --recovery 650 --seeds 0 1 2 3 --arms DDD CDD DCD --save-states none --tag p8-split-late
run J4b python -u scripts/pqn_surgery.py --late-onset 250 --duration 100 --recovery 650 --seeds 4 5 6 7 --arms DDD CDD DCD --save-states none --tag p8-split-late
run J3a python -u scripts/surgery_ppo.py --env Asterix-MinAtar --allowed 0 2 4 --window 390 440 --recovery 200 --seeds 0 1 2 3 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-asterix-s0to3
run J3b python -u scripts/surgery_ppo.py --env Asterix-MinAtar --allowed 0 2 4 --window 390 440 --recovery 200 --seeds 4 5 6 7 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-asterix-s4to7
run J5a python -u scripts/ponr_sweep.py --durations 300 --seeds 100 101 102 103 --tag p8-ponr-confirm-s100to103
run J5b python -u scripts/ponr_sweep.py --durations 300 --seeds 104 105 106 107 --tag p8-ponr-confirm-s104to107
run J6a python -u scripts/surgery_ppo.py --deficit reward_flip --flip-prob 1.0 --seeds 0 1 2 3 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-rflip-s0to3
run J6b python -u scripts/surgery_ppo.py --deficit reward_flip --flip-prob 1.0 --seeds 4 5 6 7 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-rflip-s4to7
run J7a python -u scripts/surgery_ppo.py --hidden 256 --seeds 0 1 2 3 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-h256-s0to3
run J7b python -u scripts/surgery_ppo.py --hidden 256 --seeds 4 5 6 7 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p8-surgery-h256-s4to7
echo "=== queue finished $(date)"
