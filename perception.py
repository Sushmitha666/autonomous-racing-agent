"""CNN (road perception) + LSTM (look-ahead road prediction): data collection, training, metrics."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from env import RacingEnv
from planning import classical_action

TRAIN_TRACKS = list(range(0, 24))
TEST_TRACKS = list(range(1000, 1010))          # unseen tracks
EMB = 32


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(512, EMB), nn.ReLU())
        self.cls = nn.Linear(EMB, 3)            # 0 left corner / 1 straight / 2 right corner
        self.lat = nn.Linear(EMB, 1)            # aux: lateral offset from centreline

    def forward(self, x):
        e = self.conv(x)
        return e, self.cls(e), self.lat(e).squeeze(-1)


class SeqNet(nn.Module):
    """LSTM over [CNN emb, speed, last action] -> centreline lateral offset at +10/+20/+35 units."""
    def __init__(self, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(EMB + 3, hid, batch_first=True)
        self.out = nn.Linear(hid, 3)

    def forward(self, x, h=None):
        o, h = self.lstm(x, h)
        return self.out(o), h


class FrameMLP(nn.Module):                      # no-memory baseline for the LSTM
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMB + 3, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


def collect(tracks, n_steps, seed, noise=0.25, ep_cap=400):
    """Drive with the classical controller + temporally-correlated noise; record per-step data by episode."""
    env = RacingEnv(tracks, rng_seed=seed, max_steps=ep_cap)
    rng = np.random.default_rng(seed + 7)
    eps, total = [], 0
    while total < n_steps:
        o = env.reset()
        ep = dict(img=[], vec=[], cls=[], fy=[], lat=[])
        nz = np.zeros(2)
        scale = rng.uniform(0.4, 0.8)
        while True:
            ep["img"].append(o["img"].astype(np.uint8))
            ep["vec"].append([o["speed"] / 20, *o["act"]])
            c, fy, lat = env.labels()
            ep["cls"].append(c); ep["fy"].append(fy); ep["lat"].append(lat)
            nz = 0.92 * nz + rng.normal(0, [noise * 0.4, noise * 0.5])
            a = classical_action(env, speed_scale=scale) + nz
            o, r, d, info = env.step(a)
            if d:
                break
        eps.append({k: np.array(v) for k, v in ep.items()})
        total += len(ep["cls"])
    return eps


def _stack(eps, k):
    return np.concatenate([e[k] for e in eps])


def train_cnn(train_eps, test_eps, epochs=8, log=print):
    torch.manual_seed(0)
    X = torch.tensor(_stack(train_eps, "img"), dtype=torch.float32).unsqueeze(1)
    yc = torch.tensor(_stack(train_eps, "cls")); yl = torch.tensor(_stack(train_eps, "lat"), dtype=torch.float32) / 6
    net = CNN(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    w = torch.tensor(1.0 / np.maximum(np.bincount(yc.numpy(), minlength=3), 1), dtype=torch.float32)
    w = w / w.mean()
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            _, lo, la = net(X[b])
            loss = nn.functional.cross_entropy(lo, yc[b], weight=w) + 0.5 * nn.functional.mse_loss(la, yl[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        log(f"  cnn epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    return net


@torch.no_grad()
def embed(net, eps):
    net.eval()
    out = []
    for e in eps:
        x = torch.tensor(e["img"], dtype=torch.float32).unsqueeze(1)
        out.append(net(x)[0].numpy())
    return out


@torch.no_grad()
def cnn_metrics(net, eps):
    net.eval()
    X = torch.tensor(_stack(eps, "img"), dtype=torch.float32).unsqueeze(1)
    y = _stack(eps, "cls")
    _, lo, la = net(X)
    p = lo.argmax(1).numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(accuracy=float(accuracy_score(y, p)), precision=float(pr), recall=float(rc), f1=float(f1),
                lat_mae=float(np.abs(la.numpy() * 6 - _stack(eps, "lat")).mean()),
                confusion=confusion_matrix(y, p, labels=[0, 1, 2]).tolist(), n=int(len(y)))


def _seq_batch(eps, embs, mu, sd, T=400):
    B = len(eps)
    X = np.zeros((B, T, EMB + 3), np.float32); Y = np.zeros((B, T, 3), np.float32); M = np.zeros((B, T), np.float32)
    for i, (e, z) in enumerate(zip(eps, embs)):
        n = min(len(z), T)
        X[i, :n, :EMB] = (z[:n] - mu) / sd; X[i, :n, EMB:] = e["vec"][:n]
        Y[i, :n] = e["fy"][:n] / 10; M[i, :n] = 1
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


def train_seq(model, train_eps, train_emb, mu, sd, epochs=60, lr=3e-3, seq=True):
    torch.manual_seed(0)
    X, Y, M = _seq_batch(train_eps, train_emb, mu, sd)
    opt = torch.optim.Adam(model.parameters(), lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]
            pred = model(X[b])[0] if seq else model(X[b])
            loss = (((pred - Y[b]) ** 2).mean(-1) * M[b]).sum() / M[b].sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def seq_metrics(model, eps, embs, mu, sd, seq=True, tol=1.5):
    X, Y, M = _seq_batch(eps, embs, mu, sd)
    pred = model(X)[0] if seq else model(X)
    m = M.bool()
    err = (pred - Y)[m] * 10
    return dict(mae=float(err.abs().mean()), rmse=float(err.pow(2).mean().sqrt()),
                accuracy_within_1p5=float((err.abs() < tol).float().mean()),
                mae_per_horizon=[float(v) for v in err.abs().mean(0)], n=int(m.sum()))


def build_all(out_dir="models", n_train=45000, n_test=12000, log=print):
    import os
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    log("collecting data ...")
    tr = collect(TRAIN_TRACKS, n_train, seed=1)
    te = collect(TEST_TRACKS, n_test, seed=2)
    log(f"  train {sum(len(e['cls']) for e in tr)} steps / {len(tr)} eps; test {sum(len(e['cls']) for e in te)} steps; {time.time() - t0:.0f}s")
    cnn = train_cnn(tr, te, log=log)
    cm = cnn_metrics(cnn, te)
    log("CNN (unseen tracks): " + json.dumps({k: v for k, v in cm.items() if k != 'confusion'}))
    etr, ete = embed(cnn, tr), embed(cnn, te)
    allz = np.concatenate(etr)
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    lstm = train_seq(SeqNet(), tr, etr, mu, sd)
    lm = seq_metrics(lstm, te, ete, mu, sd)
    mlp = train_seq(FrameMLP(), tr, etr, mu, sd, seq=False)
    mm = seq_metrics(mlp, te, ete, mu, sd, seq=False)
    log("LSTM: " + json.dumps(lm)); log("single-frame MLP baseline: " + json.dumps(mm))
    torch.save(cnn.state_dict(), f"{out_dir}/cnn.pt"); torch.save(lstm.state_dict(), f"{out_dir}/lstm.pt")
    np.savez(f"{out_dir}/norm.npz", mu=mu, sd=sd)
    json.dump(dict(cnn=cm, lstm=lm, mlp_baseline=mm, train_steps=sum(len(e['cls']) for e in tr)),
              open(f"{out_dir}/perception_metrics.json", "w"), indent=1)
    return cnn, lstm


def load_models(out_dir="models"):
    cnn, lstm = CNN(), SeqNet()
    cnn.load_state_dict(torch.load(f"{out_dir}/cnn.pt")); lstm.load_state_dict(torch.load(f"{out_dir}/lstm.pt"))
    nz = np.load(f"{out_dir}/norm.npz")
    return cnn.eval(), lstm.eval(), nz["mu"], nz["sd"]


if __name__ == "__main__":
    torch.set_num_threads(2)
    build_all()
