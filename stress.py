"""Robustness test of every trained policy: hard unseen tracks, and camera noise (10% pixel flips)."""
import glob
import json

import numpy as np
import torch

from features import FeatureBuilder
from ppo import AC, evaluate, run_classical
from planning import classical_action

HARD = list(range(2000, 2010))
CONDS = {"hard_tracks": dict(tracks=HARD), "camera_noise_10pct": dict(img_noise=0.10),
         "hard_tracks+camera_noise": dict(tracks=HARD, img_noise=0.10)}


def main():
    torch.set_num_threads(2)
    out = {"classical_astar_purepursuit": {c: {k: v for k, v in evaluate(lambda env, o, reset=False: classical_action(env), record_traj=False, **kw).items() if k != "traj"}
                                          for c, kw in CONDS.items()}}
    for f in sorted(glob.glob("results/ppo_*_s*.pt")):
        name = f.split("/")[-1][:-3]
        variant, seed = name.rsplit("_s", 1)
        fb = FeatureBuilder(variant, 1)
        net = AC(fb.dim); net.load_state_dict(torch.load(f)); net.eval()

        def pol(env, o, reset=False):
            if reset:
                fb.reset_all()
                return None
            with torch.no_grad():
                return net.dist(torch.tensor(fb([o]))).mean[0].numpy()

        out[name] = {c: {k: v for k, v in evaluate(pol, record_traj=False, **kw).items() if k != "traj"} for c, kw in CONDS.items()}
        print(name, {c: round(r["success_rate"], 2) for c, r in out[name].items()}, flush=True)
    json.dump(out, open("results/stress.json", "w"), indent=1)


if __name__ == "__main__":
    main()
