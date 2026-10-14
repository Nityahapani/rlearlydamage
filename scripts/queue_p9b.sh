#!/usr/bin/env bash
# Phase 9 resume queue after the sandbox restart (LOG 2026-10-14 ~09:30 deviation): E2 seeds 0-3 first, then E3 chunks
# in pre-registered order until the budget ends. Same commands as queue_p9.sh; resumable via done-markers.
set -u
cd /home/user/rlearlydamage
. .venv/bin/activate
L=/home/user/p9logs; mkdir -p $L
run() {
  local name=$1; shift
  if [ -f $L/$name.done ]; then echo "skip $name"; return; fi
  echo "=== start $name $(date)"; "$@" > $L/$name.log 2>&1 && touch $L/$name.done && echo "=== done $name $(date)" || echo "=== FAILED $name $(date)"
}
run E2-s0 python -u scripts/surgery_ppo.py --env Asterix-MinAtar --deficit reward_flip --flip-prob 1.0 --window 390 440 --recovery 200 --seeds 0 1 2 3 --arms CCC CCD CDC CDD DCC DCD DDC DDD --eval-fork --tag p9-surgery-asterix-rflip-s0to3
for b in 304 308 312 316 320 324 328; do
  run E3-s$b python -u scripts/ponr_sweep.py --durations 300 --arms D_online --seeds $b $((b+1)) $((b+2)) $((b+3)) --tag p9-ponr-confirm2-s${b}to$((b+3))
done
echo "=== queue finished $(date)"
