#!/bin/bash
# Phase 12 queue v3 (deviation logged 19:20): after a dirty-tree cascade failed G4-G7 at manifest time (no runs),
# order G5, G6, G7, then G4 Freeway in 2-seed chunks (OOM at 4 seeds) as time allows. Resumable via .done markers.
cd /home/user/rlearlydamage && . .venv/bin/activate
L=/home/user/p12logs; Q=/home/user/p12queue.log
ARMS="CCC DCC CDC CCD DDC DCD CDD DDD"
ppo () {  # name, chunk-list (comma-separated chunks), then surgery args
  local name=$1 chunks=$2; shift 2
  IFS=',' read -ra CH <<< "$chunks"
  for chunk in "${CH[@]}"; do
    local id="$name-s${chunk:0:1}"
    [ -f $L/$id.done ] && continue
    [ -f $L/$name.abandoned ] && return
    now=$(date +%H%M); if [ $now -gt 2330 ]; then echo "$(date +%T) SKIP $id (cutoff)" >> $Q; continue; fi
    echo "$(date +%T) start $id" >> $Q
    python -u scripts/surgery_ppo.py "$@" --arms $ARMS --eval-fork --seeds $chunk --tag p12-$name-s${chunk:0:1} > $L/$id.log 2>&1 \
      && touch $L/$id.done && echo "$(date +%T) done $id" >> $Q \
      || { echo "$(date +%T) FAILED $id" >> $Q; touch $L/$name.abandoned; }
  done
}
ppo si-obsnoise   "0 1 2 3,4 5 6 7" --env SpaceInvaders-MinAtar --deficit obs_noise --sigma 1.0 --window 100 150 --recovery 150
ppo si-extinction "0 1 2 3,4 5 6 7" --env SpaceInvaders-MinAtar --deficit reward_scale --scale 0.0 --window 100 150 --recovery 150
for chunk in "0 1 2 3" "4 5 6 7"; do
  id="pqnr-rflip-s${chunk:0:1}"
  [ -f $L/$id.done ] && continue
  now=$(date +%H%M); if [ $now -gt 2320 ]; then echo "$(date +%T) SKIP $id (start cutoff)" >> $Q; continue; fi
  echo "$(date +%T) start $id" >> $Q
  python -u scripts/pqn_surgery.py --deficit reward_flip --late-onset 250 --duration 100 --recovery 650 \
     --arms CC DC CD DD --seeds $chunk --tag p12-pqnr-rflip-s${chunk:0:1} > $L/$id.log 2>&1 \
     && touch $L/$id.done && echo "$(date +%T) done $id" >> $Q || echo "$(date +%T) FAILED $id" >> $Q
done
ppo freeway-rflip "0 1,2 3,4 5,6 7" --env Freeway-MinAtar --deficit reward_flip --flip-prob 1.0 --window 20 70 --recovery 150
echo "$(date +%T) queue v3 finished" >> $Q
