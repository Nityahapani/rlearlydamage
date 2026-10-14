#!/usr/bin/env bash
# Phase 10 queue (pre-registered in LOG.md 2026-10-14 Phase 10): onset-reference (pause-counterfactual) PPO surgery.
set -u
cd /home/user/rlearlydamage
. .venv/bin/activate
L=/home/user/p10logs; mkdir -p $L
run() {
  local name=$1; shift
  if [ -f $L/$name.done ]; then echo "skip $name"; return; fi
  echo "=== start $name $(date)"; "$@" > $L/$name.log 2>&1 && touch $L/$name.done && echo "=== done $name $(date)" || echo "=== FAILED $name $(date)"
}
A="CCC CCD CDC CDD DCC DCD DDC DDD"
run K1a python -u scripts/surgery_ppo.py --deficit reward_flip --flip-prob 1.0 --seeds 0 1 2 3 --arms $A --clean-source onset --tag p10-onsetref-rflip-s0to3
run K1b python -u scripts/surgery_ppo.py --deficit reward_flip --flip-prob 1.0 --seeds 4 5 6 7 --arms $A --clean-source onset --tag p10-onsetref-rflip-s4to7
run K2  python -u scripts/surgery_ppo.py --seeds 0 1 2 3 --arms $A --clean-source onset --tag p10-onsetref-move-s0to3
echo "=== queue finished $(date)"
