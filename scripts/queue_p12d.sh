#!/bin/bash
# Phase 12 follow-up queue (deviation logged 19:20): G4 Freeway rerun in 2-seed chunks after the 4-seed run was
# OOM-killed (1.14 GB RSS on a 2 GB machine), then G7 (PQN-R reward flip) under its start cutoff.
cd /home/user/rlearlydamage && . .venv/bin/activate
L=/home/user/p12logs; Q=/home/user/p12queue.log
ARMS="CCC DCC CDC CCD DDC DCD CDD DDD"
until grep -q "^.* queue finished" $Q 2>/dev/null; do sleep 60; done
for chunk in "0 1" "2 3" "4 5" "6 7"; do
  id="freeway-rflip-s${chunk:0:1}"
  [ -f $L/$id.done ] && continue
  [ -f $L/freeway-abandoned ] && break
  now=$(date +%H%M); if [ $now -gt 2240 ]; then echo "$(date +%T) SKIP $id (cutoff)" >> $Q; continue; fi
  echo "$(date +%T) start $id" >> $Q
  python -u scripts/surgery_ppo.py --env Freeway-MinAtar --deficit reward_flip --flip-prob 1.0 --window 20 70 --recovery 150 \
     --arms $ARMS --eval-fork --seeds $chunk --tag p12-freeway-rflip-s${chunk:0:1} > $L/$id.log 2>&1 \
     && touch $L/$id.done && echo "$(date +%T) done $id" >> $Q \
     || { echo "$(date +%T) FAILED $id (Freeway abandoned)" >> $Q; touch $L/freeway-abandoned; }
done
for chunk in "0 1 2 3" "4 5 6 7"; do
  id="pqnr-rflip-s${chunk:0:1}"
  [ -f $L/$id.done ] && continue
  now=$(date +%H%M); if [ $now -gt 2320 ]; then echo "$(date +%T) SKIP $id (start cutoff)" >> $Q; continue; fi
  echo "$(date +%T) start $id" >> $Q
  python -u scripts/pqn_surgery.py --deficit reward_flip --late-onset 250 --duration 100 --recovery 650 \
     --arms CC DC CD DD --seeds $chunk --tag p12-pqnr-rflip-s${chunk:0:1} > $L/$id.log 2>&1 \
     && touch $L/$id.done && echo "$(date +%T) done $id" >> $Q \
     || echo "$(date +%T) FAILED $id" >> $Q
done
echo "$(date +%T) follow-up queue finished" >> $Q
