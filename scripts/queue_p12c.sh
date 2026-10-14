#!/bin/bash
# Phase 12 G7 (pre-registered addendum 2): PQN-R reward-flip network x buffer surgery.
# Waits for the main Phase 12 queue to finish; starts a chunk only if it can plausibly finish by 23:30 (~35 min/chunk).
cd /home/user/rlearlydamage && . .venv/bin/activate
L=/home/user/p12logs
until grep -q "queue finished" /home/user/p12queue.log 2>/dev/null; do sleep 60; done
for chunk in "0 1 2 3" "4 5 6 7"; do
  id="pqnr-rflip-s${chunk:0:1}"
  [ -f $L/$id.done ] && continue
  now=$(date +%H%M); if [ $now -gt 2255 ] || [ $now -lt 1200 ]; then echo "$(date +%T) SKIP $id (start cutoff)" >> /home/user/p12queue.log; continue; fi
  echo "$(date +%T) start $id" >> /home/user/p12queue.log
  python -u scripts/pqn_surgery.py --deficit reward_flip --late-onset 250 --duration 100 --recovery 650 \
     --arms CC DC CD DD --seeds $chunk --tag p12-pqnr-rflip-s${chunk:0:1} > $L/$id.log 2>&1 \
     && touch $L/$id.done && echo "$(date +%T) done $id" >> /home/user/p12queue.log \
     || echo "$(date +%T) FAILED $id" >> /home/user/p12queue.log
done
echo "$(date +%T) G7 queue finished" >> /home/user/p12queue.log
