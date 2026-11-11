"""PPO in pure NumPy (manual backprop). Same algorithm/hyper-parameters as ppo.py.  Usage: python ppo_np.py <variant> <seed> <steps>"""
import json
import os
import sys
import time

import numpy as np

from common import TRAIN_TRACKS, evaluate
from env import RacingEnv
from features_np import FeatureBuilder
from npnets import Adam, Policy

LOG2PI = np.log(2 * np.pi)


def make_policy_fn(net, fb):
    def pol(env, o, reset=False):
        if reset:
            fb.reset_all()
            return None
        return net.pi.forward(fb([o]))[0]
    return pol


def train(variant, seed, total_steps, n_envs=8, T=128, out_dir="results", log=print):
    rng = np.random.default_rng(seed)
    fb = FeatureBuilder(variant, n_envs)
    net = Policy(fb.dim, rng)
    params = net.pi.params() + net.v.params() + [net.logstd]
    opt = Adam(params, 3e-4)
    envs = [RacingEnv(TRAIN_TRACKS, rng_seed=seed * 100 + i) for i in range(n_envs)]
    obs = [e.reset() for e in envs]
    fb.reset_all(); x = fb(obs)
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs)
    curve, gstep, t0 = [], 0, time.time()
    n_upd = total_steps // (n_envs * T)
    for upd in range(n_upd):
        opt.lr = 3e-4 * (1 - upd / n_upd)
        X = np.zeros((T, n_envs, fb.dim), np.float32); A = np.zeros((T, n_envs, 2), np.float32)
        LP = np.zeros((T, n_envs), np.float32); R = np.zeros((T, n_envs), np.float32)
        D = np.zeros((T, n_envs), np.float32); V = np.zeros((T + 1, n_envs), np.float32)
        std = np.exp(net.logstd)
        for t in range(T):
            mu = net.pi.forward(x)
            a = mu + std * rng.standard_normal(mu.shape).astype(np.float32)
            LP[t] = (-0.5 * ((a - mu) / std) ** 2 - net.logstd - 0.5 * LOG2PI).sum(-1)
            V[t] = net.v.forward(x)[:, 0]
            X[t], A[t] = x, a
            nobs = []
            for i, e in enumerate(envs):
                o, r, d, info = e.step(a[i])
                R[t, i], D[t, i] = r, float(d)
                ep_ret[i] += r; ep_len[i] += 1
                if d:
                    curve.append((gstep + (t + 1) * n_envs, float(ep_ret[i]), int(ep_len[i]), bool(info["success"])))
                    ep_ret[i] = 0; ep_len[i] = 0
                    o = e.reset(); fb.reset_env(i)
                nobs.append(o)
            x = fb(nobs)
        gstep += T * n_envs
        V[T] = net.v.forward(x)[:, 0]
        adv = np.zeros((T, n_envs), np.float32); last = 0
        for t in reversed(range(T)):
            nt = 1.0 - D[t]
            last = R[t] + 0.99 * V[t + 1] * nt - V[t] + 0.99 * 0.95 * nt * last
            adv[t] = last
        ret = adv + V[:T]
        bx, ba, blp = X.reshape(-1, fb.dim), A.reshape(-1, 2), LP.reshape(-1)
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        for _ in range(5):
            perm = rng.permutation(len(bx))
            for i in range(0, len(bx), 256):
                b = perm[i:i + 256]; B = len(b)
                mu, hs_pi = net.pi.forward(bx[b], cache=True)
                v, hs_v = net.v.forward(bx[b], cache=True)
                std = np.exp(net.logstd)
                z = (ba[b] - mu) / std
                lp = (-0.5 * z ** 2 - net.logstd - 0.5 * LOG2PI).sum(-1)
                ratio = np.exp(lp - blp[b])
                ad = (badv[b] - badv[b].mean()) / (badv[b].std(ddof=1) + 1e-8)
                active = (ratio * ad <= np.clip(ratio, 0.8, 1.2) * ad)          # unclipped branch is the min
                g_lp = np.where(active, -ad * ratio, 0.0) / B                   # d loss / d logprob
                g_mu = g_lp[:, None] * z / std                                   # dlogp/dmu = (a-mu)/std^2
                g_ls = (g_lp[:, None] * (z ** 2 - 1)).sum(0)                     # dlogp/dlogstd
                g_v = (0.5 * (v[:, 0] - bret[b]) / B)[:, None]                   # 0.5 * 0.5*(v-ret)^2 / B
                grads = net.pi.backward(hs_pi, g_mu) + net.v.backward(hs_v, g_v) + [g_ls]
                opt.step(grads)
        if upd % 10 == 0 or upd == n_upd - 1:
            rec = curve[-30:]
            log(f"[{variant} s{seed}] step {gstep} ret {np.mean([c[1] for c in rec]) if rec else 0:.1f} "
                f"succ {np.mean([c[3] for c in rec]) if rec else 0:.2f} ({time.time() - t0:.0f}s)")
    fb_eval = FeatureBuilder(variant, 1)
    res = evaluate(make_policy_fn(net, fb_eval))
    res.update(variant=variant, seed=seed, steps=gstep, train_time=time.time() - t0, curve=curve)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{variant}_s{seed}.json", "w"))
    net.save(f"{out_dir}/{variant}_s{seed}.npz")
    log(f"[{variant} s{seed}] EVAL success {res['success_rate']:.2f} reward {res['avg_reward']:.1f} lap {res['avg_lap_time']}")
    return res


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
