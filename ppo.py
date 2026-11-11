"""PPO (continuous Gaussian policy, GAE) trained on vectorised RacingEnv with variant-specific features."""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from env import RacingEnv
from features import FeatureBuilder, VARIANTS
from perception import TEST_TRACKS, TRAIN_TRACKS
from planning import classical_action


class AC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.v = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.logstd = nn.Parameter(torch.full((2,), -0.7))
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.1)

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.logstd.exp())


def evaluate(policy_fn, n_eps=20, record_traj=True, tracks=None, img_noise=0.0):
    """policy_fn(env, obs)->action. Unseen tracks, fixed start seeds. Returns metrics dict."""
    tracks = tracks or TEST_TRACKS
    env = RacingEnv(tracks)
    env.img_noise = img_noise
    rets, lens, succ, off, laps, skids, traj = [], [], [], [], [], [], None
    for k in range(n_eps):
        o = env.reset(track_seed=tracks[k % len(tracks)], start_seed=5000 + k)
        policy_fn(env, None, reset=True)
        R, path = 0.0, [env.pos.copy()]
        while True:
            o, r, d, info = env.step(policy_fn(env, o))
            R += r
            path.append(env.pos.copy())
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


def train(variant, seed, total_steps, n_envs=8, T=128, model_dir="models", out_dir="results", log=print):
    torch.set_num_threads(1)
    torch.manual_seed(seed); np.random.seed(seed)
    fb = FeatureBuilder(variant, n_envs, model_dir)
    net = AC(fb.dim)
    opt = torch.optim.Adam(net.parameters(), 3e-4, eps=1e-5)
    envs = [RacingEnv(TRAIN_TRACKS, rng_seed=seed * 100 + i) for i in range(n_envs)]
    obs = [e.reset() for e in envs]
    fb.reset_all()
    x = fb(obs)
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs)
    curve, gstep, t0 = [], 0, time.time()
    n_upd = total_steps // (n_envs * T)
    for upd in range(n_upd):
        for g in opt.param_groups:
            g["lr"] = 3e-4 * (1 - upd / n_upd)
        X = np.zeros((T, n_envs, fb.dim), np.float32); A = np.zeros((T, n_envs, 2), np.float32)
        LP = np.zeros((T, n_envs), np.float32); R = np.zeros((T, n_envs), np.float32)
        D = np.zeros((T, n_envs), np.float32); V = np.zeros((T + 1, n_envs), np.float32)
        TRUNC = np.zeros((T, n_envs), np.float32)
        for t in range(T):
            with torch.no_grad():
                xt = torch.tensor(x)
                dist = net.dist(xt); a = dist.sample()
                LP[t] = dist.log_prob(a).sum(-1).numpy(); V[t] = net.v(xt).squeeze(-1).numpy()
            X[t], A[t] = x, a.numpy()
            nobs = []
            for i, e in enumerate(envs):
                o, r, d, info = e.step(a[i].numpy())
                R[t, i], D[t, i] = r, float(d)
                ep_ret[i] += r; ep_len[i] += 1
                if d:
                    TRUNC[t, i] = float(not (info["off_track"] or info["success"]))   # time-limit truncation
                    curve.append((gstep + (t + 1) * n_envs, float(ep_ret[i]), int(ep_len[i]), bool(info["success"])))
                    ep_ret[i] = 0; ep_len[i] = 0
                    o = e.reset(); fb.reset_env(i)
                nobs.append(o)
            x = fb(nobs)
        gstep += T * n_envs
        with torch.no_grad():
            V[T] = net.v(torch.tensor(x)).squeeze(-1).numpy()
        adv = np.zeros((T, n_envs), np.float32); last = 0
        for t in reversed(range(T)):
            nonterm = 1.0 - D[t]
            delta = R[t] + 0.99 * V[t + 1] * nonterm - V[t]
            last = delta + 0.99 * 0.95 * nonterm * last
            adv[t] = last
        ret = adv + V[:T]
        bx, ba, blp = torch.tensor(X.reshape(-1, fb.dim)), torch.tensor(A.reshape(-1, 2)), torch.tensor(LP.reshape(-1))
        badv, bret = torch.tensor(adv.reshape(-1)), torch.tensor(ret.reshape(-1))
        for _ in range(5):
            perm = torch.randperm(len(bx))
            for i in range(0, len(bx), 256):
                b = perm[i:i + 256]
                dist = net.dist(bx[b])
                lp = dist.log_prob(ba[b]).sum(-1)
                ratio = (lp - blp[b]).exp()
                ad = (badv[b] - badv[b].mean()) / (badv[b].std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(0.8, 1.2) * ad).mean()
                vl = 0.5 * (net.v(bx[b]).squeeze(-1) - bret[b]).pow(2).mean()
                loss = pl + 0.5 * vl
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if upd % 10 == 0 or upd == n_upd - 1:
            rec = [c for c in curve[-30:]]
            log(f"[{variant} s{seed}] step {gstep} ret {np.mean([c[1] for c in rec]) if rec else 0:.1f} "
                f"succ {np.mean([c[3] for c in rec]) if rec else 0:.2f} ({time.time() - t0:.0f}s)")
    # deterministic evaluation on unseen tracks
    fb_eval = FeatureBuilder(variant, 1, model_dir)

    def pol(env, o, reset=False):
        if reset:
            fb_eval.reset_all()
            return None
        with torch.no_grad():
            return net.dist(torch.tensor(fb_eval([o]))).mean[0].numpy()

    res = evaluate(pol)
    res.update(variant=variant, seed=seed, steps=gstep, train_time=time.time() - t0, curve=curve)
    import os
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{variant}_s{seed}.json", "w"))
    torch.save(net.state_dict(), f"{out_dir}/{variant}_s{seed}.pt")
    log(f"[{variant} s{seed}] EVAL success {res['success_rate']:.2f} reward {res['avg_reward']:.1f} lap {res['avg_lap_time']}")
    return res


if __name__ == "__main__":
    variant, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    train(variant, seed, steps)
