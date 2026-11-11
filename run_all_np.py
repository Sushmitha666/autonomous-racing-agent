"""Cross-platform runner (Windows-friendly replacement for run_all.sh): 5 variants x 3 seeds, 2 parallel workers."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

STEPS = sys.argv[1] if len(sys.argv) > 1 else "300000"
jobs = [(v, s) for s in (0, 1, 2) for v in ("ppo_vision", "ppo_astar_only", "ppo_vision_astar", "ppo_vision_astar_lstm", "ppo_full_hybrid")]


def run(j):
    print("start", j, flush=True)
    subprocess.run([sys.executable, "ppo_np.py", j[0], str(j[1]), STEPS], check=True)


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, jobs))
print("ALL DONE")
