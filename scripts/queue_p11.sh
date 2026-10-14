#!/usr/bin/env bash
# Phase 11: complete the unfinished pre-registered Phase 9 runs (same commands; logs/markers in /home/user/p9logs).
set -u
cd /home/user/rlearlydamage
. .venv/bin/activate
L=/home/user/p9logs; mkdir -p $L
run() {
  local name=$1; shift
  if [ -f $L/$name.done ]; then echo "skip $name"; return; fi
  echo "=== start $name $(date)"; "$@" > $L/$name.log 2>&1 && touch $L/$name.done && echo "=== done $name $(date)" || echo "=== FAILED $name $(date)"
}
run E2-s4 python -u scripts/surgery_ppo.py --env Asterix-MinAtar --deficit reward_flip --flip-prob 1.0 --window 390 440 --recovery 200 --seeds 4 5 6 7 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p9-surgery-asterix-rflip-s4to7
for b in 316 320 324 328; do
  run E3-s$b python -u scripts/ponr_sweep.py --durations 300 --arms D_online --seeds $b $((b+1)) $((b+2)) $((b+3)) --tag p9-ponr-confirm2-s${b}to$((b+3))
done
echo "=== queue finished $(date)"
