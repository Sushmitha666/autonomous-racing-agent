"""Pure-NumPy inference for the CNN, LSTM and PPO policy (no torch needed)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d(x, w, b, stride, pad):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, w.shape[2:], axis=(2, 3))[:, :, ::stride, ::stride]   # B,C,oh,ow,k,k
    return np.einsum("bchwij,ocij->bohw", win, w, optimize=True) + b[None, :, None, None]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class NpPerception:
    def __init__(self, path="models/np_models.npz", norm="models/norm.npz"):
        d = np.load(path)
        self.d = {k: d[k].astype(np.float32) for k in d.files}
        n = np.load(norm)
        self.mu, self.sd = n["mu"], n["sd"]

    def cnn(self, img):                       # img (B,32,32) -> emb (B,32), class logits (B,3)
        d, x = self.d, img[:, None].astype(np.float32)
        for i, (s, p) in zip((0, 2, 4), ((2, 2), (2, 1), (2, 1))):
            x = np.maximum(conv2d(x, d[f"cnn.conv.{i}.weight"], d[f"cnn.conv.{i}.bias"], s, p), 0)
        e = np.maximum(x.reshape(len(x), -1) @ d["cnn.conv.7.weight"].T + d["cnn.conv.7.bias"], 0)
        return e, e @ d["cnn.cls.weight"].T + d["cnn.cls.bias"]

    def lstm_step(self, x, h, c):             # x (B, 35) -> pred (B,3), h, c
        d = self.d
        g = x @ d["lstm.lstm.weight_ih_l0"].T + h @ d["lstm.lstm.weight_hh_l0"].T + d["lstm.lstm.bias_ih_l0"] + d["lstm.lstm.bias_hh_l0"]
        H = h.shape[1]
        i, f, gg, o = sigmoid(g[:, :H]), sigmoid(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), sigmoid(g[:, 3 * H:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        return h @ d["lstm.out.weight"].T + d["lstm.out.bias"], h, c


class MLP:
    """tanh MLP with manual backprop (used for PPO policy mean and value)."""
    def __init__(self, sizes, rng, last_gain=1.0):
        self.W, self.b = [], []
        for i, (a, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            bound = 1 / np.sqrt(a)                                   # torch Linear default init
            w = rng.uniform(-bound, bound, (o, a)).astype(np.float32)
            if i == len(sizes) - 2:
                w *= last_gain
            self.W.append(w); self.b.append(rng.uniform(-bound, bound, o).astype(np.float32))

    def forward(self, x, cache=False):
        hs = [x]
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w.T + b
            if k < len(self.W) - 1:
                x = np.tanh(x)
            hs.append(x)
        return (x, hs) if cache else x

    def backward(self, hs, gout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = gout
        for k in reversed(range(len(self.W))):
            gW[k], gb[k] = g.T @ hs[k], g.sum(0)
            if k > 0:
                g = (g @ self.W[k]) * (1 - hs[k] ** 2)
        return gW + gb

    def params(self):
        return self.W + self.b


class Adam:
    def __init__(self, params, lr, eps=1e-5):
        self.p, self.lr, self.eps, self.t = params, lr, eps, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, grads, max_norm=0.5):
        n = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if n > max_norm:
            grads = [g * (max_norm / (n + 1e-6)) for g in grads]
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + self.eps)


class Policy:
    def __init__(self, dim, rng=None, path=None):
        rng = rng or np.random.default_rng(0)
        self.pi, self.v = MLP([dim, 128, 128, 2], rng, 0.1), MLP([dim, 128, 128, 1], rng)
        self.logstd = np.full(2, -0.7, np.float32)
        if path:
            d = np.load(path)
            for net, pre in ((self.pi, "pi"), (self.v, "v")):
                for k, i in enumerate((0, 2, 4)):
                    net.W[k][:] = d[f"{pre}.{i}.weight"]; net.b[k][:] = d[f"{pre}.{i}.bias"]
            self.logstd[:] = d["logstd"]

    def save(self, path):
        d = {"logstd": self.logstd}
        for net, pre in ((self.pi, "pi"), (self.v, "v")):
            for k, i in enumerate((0, 2, 4)):
                d[f"{pre}.{i}.weight"], d[f"{pre}.{i}.bias"] = net.W[k], net.b[k]
        np.savez(path, **d)
