#!/bin/bash
# Phase 12 generalization panel (v2: per-chunk tags so chunks never share a summary filename) (pre-registered in LOG.md). Resumable: finished chunks leave a .done marker.
# Order is the pre-registered priority order; tests not reached within the budget are reported as not run.
cd /home/user/rlearlydamage && . .venv/bin/activate
L=/home/user/p12logs; mkdir -p $L
ARMS="CCC DCC CDC CCD DDC DCD CDD DDD"
run () {  # name, then surgery_ppo args
  local name=$1; shift
  for chunk in "0 1 2 3" "4 5 6 7"; do
    local id="$name-s${chunk:0:1}"
    [ -f $L/$id.done ] && continue
    echo "$(date +%T) start $id" >> /home/user/p12queue.log
    python -u scripts/surgery_ppo.py "$@" --arms $ARMS --eval-fork --seeds $chunk --tag p12-$name-s${chunk:0:1} > $L/$id.log 2>&1 \
      && touch $L/$id.done && echo "$(date +%T) done $id" >> /home/user/p12queue.log \
      || echo "$(date +%T) FAILED $id" >> /home/user/p12queue.log
  done
}
source /home/user/rlearlydamage/scripts/queue_p12_tests.sh
echo "$(date +%T) queue finished" >> /home/user/p12queue.log
