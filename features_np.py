"""Torch-free feature builder (same features / ablation variants as features.py)."""
import numpy as np

import rag
from npnets import NpPerception

VARIANTS = {
    "ppo_vision":            (True, False, False, False),
    "ppo_astar_only":        (False, True, False, False),
    "ppo_vision_astar":      (True, True, False, False),
    "ppo_vision_astar_lstm": (True, True, True, False),
    "ppo_full_hybrid":       (True, True, True, True),
}
_PERC = None


class FeatureBuilder:
    def __init__(self, variant, n_envs, model_dir="models"):
        global _PERC
        _PERC = _PERC or NpPerception(f"{model_dir}/np_models.npz", f"{model_dir}/norm.npz")
        self.p = _PERC
        self.use_cnn, self.use_astar, self.use_lstm, self.use_rag = VARIANTS[variant]
        self.n = n_envs
        self.need_cnn = self.use_cnn or self.use_lstm or self.use_rag
        self.tab = rag.build_hint_table() if self.use_rag else None
        self.reset_all()
        from env import RacingEnv
        e = RacingEnv([0]); o = e.reset(track_seed=0)
        self.dim = self([o] * n_envs).shape[1]
        self.reset_all()

    def reset_all(self):
        self.h, self.c = np.zeros((self.n, 64), np.float32), np.zeros((self.n, 64), np.float32)

    def reset_env(self, i):
        self.h[i] = 0; self.c[i] = 0

    def __call__(self, obs):
        speed = np.array([o["speed"] for o in obs], np.float32)
        act = np.stack([o["act"] for o in obs])
        parts = [(speed / 20)[:, None], act]
        if self.need_cnn:
            emb, logits = self.p.cnn(np.stack([o["img"] for o in obs]))
            z = (emb - self.p.mu) / self.p.sd
        if self.use_cnn:
            parts.append(np.clip(z, -5, 5).astype(np.float32))
        if self.use_astar:
            parts.append(np.stack([o["astar"] for o in obs]))
        if self.use_lstm or self.use_rag:
            x = np.concatenate([z, (speed / 20)[:, None], act], 1).astype(np.float32)
            out, self.h, self.c = self.p.lstm_step(x, self.h, self.c)
            fy = out * 10
            if self.use_lstm:
                parts.append(fy / 10)
        if self.use_rag:
            d = logits.argmax(1)
            y20 = np.abs(fy[:, 1])
            sharp = np.where(y20 < 2, 0, np.where(y20 < 6, 1, 2))
            sb = np.where(speed < 6, 0, np.where(speed < 12, 1, 2))
            h = self.tab[d, sharp, sb]
            parts.append(np.stack([h[:, 0], (h[:, 0] * 20.0 - speed) / 20, d - 1.0, h[:, 1]], 1).astype(np.float32))
        return np.concatenate(parts, 1).astype(np.float32)
