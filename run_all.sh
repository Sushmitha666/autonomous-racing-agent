#!/bin/bash
# Full ablation: 5 learned variants x 3 seeds, 2 parallel workers
cd "$(dirname "$0")"
for s in 0 1 2; do for v in ppo_vision ppo_astar_only ppo_vision_astar ppo_vision_astar_lstm ppo_full_hybrid; do echo "$v $s"; done; done \
 | xargs -P2 -L1 bash -c 'python3 ppo.py $0 $1 300000 > logs/$0_s$1.log 2>&1'
echo done > logs/ALL_DONE
