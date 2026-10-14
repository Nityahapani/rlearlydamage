#!/bin/bash
cd /home/user/rlearlydamage && . .venv/bin/activate
for spec in "CartPole-v1 200" "Acrobot-v1 200" "Freeway-MinAtar 400" "Breakout-MinAtar 400"; do
  set -- $spec
  [ -f /home/user/p12logs/pilot-$1.done ] && continue
  echo "$(date +%T) start $1" 
  python -u scripts/recovery_test.py --env $1 --configs control --updates $2 --seeds 0 1 2 3 --tag p12-longctl --allow-dirty > /home/user/p12logs/pilot-$1.log 2>&1 && touch /home/user/p12logs/pilot-$1.done
  echo "$(date +%T) end $1"
done
