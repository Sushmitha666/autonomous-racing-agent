"""Torch-free shared pieces: track splits, evaluation loop, classical baseline."""
import numpy as np

from env import RacingEnv
from planning import classical_action

TRAIN_TRACKS = list(range(0, 24))
TEST_TRACKS = list(range(1000, 1010))
HARD_TRACKS = list(range(2000, 2010))


def evaluate(policy_fn, n_eps=20, record_traj=True, tracks=None, img_noise=0.0):
    tracks = tracks or TEST_TRACKS
    env = RacingEnv(tracks)
    env.img_noise = img_noise
    rets, lens, succ, off, laps, skids, traj = [], [], [], [], [], [], None
    for k in range(n_eps):
        env.reset(track_seed=tracks[k % len(tracks)], start_seed=5000 + k)
        policy_fn(env, None, reset=True)
        o, R, path = env.observe(), 0.0, [env.pos.copy()]
        while True:
            o, r, d, info = env.step(policy_fn(env, o))
            R += r; path.append(env.pos.copy())
            if d:
                break
        rets.append(R); lens.append(info["t"]); succ.append(info["success"]); off.append(info["off_track"]); skids.append(info["skids"])
        if info["success"]:
            laps.append(info["t"] * 0.1)
        if k == 0 and record_traj:
            traj = dict(track=int(env.seed), xy=np.array(path)[::3].round(2).tolist(), success=bool(info["success"]))
    return dict(avg_reward=float(np.mean(rets)), reward_var=float(np.var(rets)), avg_ep_len=float(np.mean(lens)),
                success_rate=float(np.mean(succ)), off_track_rate=float(np.mean(off)),
                avg_lap_time=float(np.mean(laps)) if laps else None, avg_skids=float(np.mean(skids)), traj=traj)


def run_classical():
    return evaluate(lambda env, o, reset=False: classical_action(env))
