"""Torch-free robustness test of every trained policy (results/*.npz)."""
import glob
import json

from common import HARD_TRACKS, evaluate
from features_np import FeatureBuilder
from npnets import Policy
from planning import classical_action
from ppo_np import make_policy_fn

CONDS = {"hard_tracks": dict(tracks=HARD_TRACKS), "camera_noise_10pct": dict(img_noise=0.10),
         "hard_tracks+camera_noise": dict(tracks=HARD_TRACKS, img_noise=0.10)}
strip = lambda r: {k: v for k, v in r.items() if k != "traj"}


def main():
    out = {"classical_astar_purepursuit": {c: strip(evaluate(lambda env, o, reset=False: classical_action(env), record_traj=False, **kw)) for c, kw in CONDS.items()}}
    for f in sorted(glob.glob("results/ppo_*_s*.npz")):
        name = f.split("/")[-1].split("\\")[-1][:-4]
        variant = name.rsplit("_s", 1)[0]
        fb = FeatureBuilder(variant, 1)
        net = Policy(fb.dim, path=f)
        out[name] = {c: strip(evaluate(make_policy_fn(net, fb), record_traj=False, **kw)) for c, kw in CONDS.items()}
        print(name, {c: round(r["success_rate"], 2) for c, r in out[name].items()}, flush=True)
    json.dump(out, open("results/stress.json", "w"), indent=1)


if __name__ == "__main__":
    main()
