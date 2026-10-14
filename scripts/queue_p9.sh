#!/usr/bin/env bash
# Phase 9 queue (pre-registered in LOG.md, 2026-10-14): sequential, resumable (skips jobs with a done-marker).
set -u
cd /home/user/rlearlydamage
. .venv/bin/activate
L=/home/user/p9logs; mkdir -p $L
run() {  # name, command...
  local name=$1; shift
  if [ -f $L/$name.done ]; then echo "skip $name"; return; fi
  echo "=== start $name $(date)"; "$@" > $L/$name.log 2>&1 && touch $L/$name.done && echo "=== done $name $(date)" || echo "=== FAILED $name $(date)"
}
# E1: confirmatory recovery estimands, PPO movement late [100,150), fresh seeds 200-211
for c in "200 201 202 203" "204 205 206 207" "208 209 210 211"; do
  set -- $c; run E1-s$1 python -u scripts/recovery_test.py --configs control move_late --seeds $c --tag p9-recovery-move-late-s$1to$4
done
# E3: independent confirmatory point-of-no-return study at d = 300 + predictor test, fresh seeds 300-331
for b in 300 304 308 312 316 320 324 328; do
  run E3-s$b python -u scripts/ponr_sweep.py --durations 300 --arms D_online --seeds $b $((b+1)) $((b+2)) $((b+3)) --tag p9-ponr-confirm2-s${b}to$((b+3))
done
# E2: Asterix reward-flip 2^3 surgery, window [390,440), recovery 200, seeds 0-7
run E2-s0 python -u scripts/surgery_ppo.py --env Asterix-MinAtar --deficit reward_flip --flip-prob 1.0 --window 390 440 --recovery 200 --seeds 0 1 2 3 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p9-surgery-asterix-rflip-s0to3
run E2-s4 python -u scripts/surgery_ppo.py --env Asterix-MinAtar --deficit reward_flip --flip-prob 1.0 --window 390 440 --recovery 200 --seeds 4 5 6 7 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p9-surgery-asterix-rflip-s4to7
echo "=== queue finished $(date)"
