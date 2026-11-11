"""Builds the policy observation for each ablation variant (frozen CNN / LSTM / A* / RAG feature stack)."""
import numpy as np
import torch

import rag
from perception import load_models

VARIANTS = {
    # name: (cnn_emb, astar, lstm, rag)
    "ppo_vision":           (True, False, False, False),
    "ppo_astar_only":       (False, True, False, False),
    "ppo_vision_astar":     (True, True, False, False),
    "ppo_vision_astar_lstm": (True, True, True, False),
    "ppo_full_hybrid":      (True, True, True, True),
}


class FeatureBuilder:
    def __init__(self, variant, n_envs, model_dir="models"):
        self.use_cnn, self.use_astar, self.use_lstm, self.use_rag = VARIANTS[variant]
        self.n = n_envs
        self.cnn, self.lstm, self.mu, self.sd = load_models(model_dir)
        self.need_cnn = self.use_cnn or self.use_lstm or self.use_rag
        self.h = None
        self.tab = rag.build_hint_table() if self.use_rag else None
        self.reset_all()
        self.dim = self(None, probe=True).shape[1]

    def reset_all(self):
        self.h = (torch.zeros(1, self.n, 64), torch.zeros(1, self.n, 64))

    def reset_env(self, i):
        self.h[0][:, i] = 0
        self.h[1][:, i] = 0

    @torch.no_grad()
    def __call__(self, obs, probe=False):
        if probe:
            from env import RacingEnv
            e = RacingEnv([0]); o = e.reset(track_seed=0)
            obs = [o] * self.n
        n = len(obs)
        speed = np.array([o["speed"] for o in obs], np.float32)
        act = np.stack([o["act"] for o in obs])
        parts = [(speed / 20)[:, None], act]
        if self.need_cnn:
            img = torch.tensor(np.stack([o["img"] for o in obs])).unsqueeze(1)
            emb, logits, _ = self.cnn(img)
            z = (emb.numpy() - self.mu) / self.sd
        if self.use_cnn:
            parts.append(np.clip(z, -5, 5).astype(np.float32))
        if self.use_astar:
            parts.append(np.stack([o["astar"] for o in obs]))
        if self.use_lstm or self.use_rag:
            x = torch.tensor(np.concatenate([z, (speed / 20)[:, None], act], 1), dtype=torch.float32).unsqueeze(1)
            out, self.h = self.lstm(x, self.h)
            fy = out[:, 0].numpy() * 10            # predicted lateral offsets (+10,+20,+35)
            if self.use_lstm:
                parts.append(fy / 10)
        if self.use_rag:
            d = logits.argmax(1).numpy()                           # 0 left 1 straight 2 right
            y20 = np.abs(fy[:, 1])
            sharp = np.where(y20 < 2, 0, np.where(y20 < 6, 1, 2))
            sb = np.where(speed < 6, 0, np.where(speed < 12, 1, 2))
            h = self.tab[d, sharp, sb]                              # (n, 2): target factor, brake flag
            tgt = h[:, 0] * 20.0
            parts.append(np.stack([h[:, 0], (tgt - speed) / 20, d - 1.0, h[:, 1]], 1).astype(np.float32))
        return np.concatenate(parts, 1).astype(np.float32)
