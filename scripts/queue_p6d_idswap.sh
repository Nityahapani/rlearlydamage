#!/bin/bash
# resumable: skips seeds whose summary already holds both arms + final consistency write; removes partial ones
cd /home/user/rlearlydamage && . .venv/bin/activate && export PYTHONPATH=src
for s in $(seq 0 31); do
  st=$(python - "$s" <<'PY'
import glob, json, os, sys
s = sys.argv[1]; fs = glob.glob(f"results/summary/*p6d-idswap-s{s}-pqnr-1000k-*.json")
state = "todo"
for f in fs:
    d = json.load(open(f))
    if set(d["arms"]) == {"purge_R", "purge_U"} and "wallclock_s" in d and (int(s) >= 12 or d["consistency_check"]):
        state = "done"
    else:
        os.remove(f); m = f.replace("results/summary", "runs"); os.path.exists(m) and os.remove(m)
print(state)
PY
)
  [ "$st" = "done" ] && { echo "seed $s already done"; continue; }
  extra=""; [ $s -lt 12 ] && extra="--p6b-summary $(ls results/summary/*p6b-tpurge-s$s-pqnr-1000k-*.json)"
  python -u scripts/pqn_identity_swap.py --seeds $s --tag idswap-s$s $extra
done
echo QUEUE_DONE
