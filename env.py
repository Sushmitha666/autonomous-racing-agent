"""Top-down racing environment: kinematic car, continuous (steer, accel/brake), egocentric camera image."""
import math

import numpy as np

from planning import ALAT, DMAX, VMAX, WB, plan_path
from track import HALF_W, get_track, wrap

DT = 0.1
IMG = 32
# camera: 32 units forward (-4..28 ahead of car), +-12 lateral, one pixel = 1.0 x 0.75 units
_fx = np.linspace(28 - 0.5, -4 + 0.5, IMG)           # row 0 = far
_fy = np.linspace(12 - 0.375, -12 + 0.375, IMG)      # col 0 = left
FX, FY = np.meshgrid(_fx, _fy, indexing="ij")

_PATHS = {}


def get_path(seed):
    if seed not in _PATHS:
        _PATHS[seed] = plan_path(get_track(seed), "astar")[0]
    return _PATHS[seed]


class RacingEnv:
    def __init__(self, track_seeds, max_steps=700, rng_seed=0):
        self.seeds = list(track_seeds)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(rng_seed)
        self.img_noise = 0.0                       # fraction of camera pixels flipped (stress test)
        self._nrng = np.random.default_rng(rng_seed + 99)

    # ---------- helpers ----------
    def _frame(self, pts):
        d = pts - self.pos
        c, s = math.cos(self.th), math.sin(self.th)
        return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)

    def reset(self, track_seed=None, start_seed=None):
        self.seed = int(self.rng.choice(self.seeds)) if track_seed is None else track_seed
        r = self.rng if start_seed is None else np.random.default_rng(start_seed)
        self.track, self.cl, self.path = get_track(self.seed), get_track(self.seed).cl, get_path(self.seed)
        self.pos = self.cl.pts[0] + self.cl.tan[0] * 0 + np.array([-self.cl.tan[0, 1], self.cl.tan[0, 0]]) * r.uniform(-2, 2)
        self.th = float(self.cl.head[0] + r.uniform(-0.12, 0.12))
        self.v = float(r.uniform(0, 4))
        self.delta, self.act = 0.0, np.zeros(2, np.float32)
        self.ci, self.lat = 0, 0.0
        self.pidx = self.path.nearest(self.pos, 0)[0]
        self.progress, self.t, self.skids = 0.0, 0, 0
        self.ci, self.lat = self.cl.nearest(self.pos, 0)
        return self.observe()

    # ---------- sensing ----------
    def image(self):
        c, s = math.cos(self.th), math.sin(self.th)
        wx = self.pos[0] + FX * c - FY * s
        wy = self.pos[1] + FX * s + FY * c
        img = self.track.sample(np.stack([wx, wy], -1)).astype(np.float32)
        if self.img_noise > 0:
            flip = self._nrng.random(img.shape) < self.img_noise
            img = np.where(flip, 1 - img, img)
        return img

    def astar_feat(self):
        p = self.path
        look = np.array([p.at(self.pidx, d) for d in (5, 10, 16, 24)])
        q = self._frame(p.pts[look])
        _, lat = p.nearest(self.pos, self.pidx)
        n = int(10 + 1.6 * self.v)
        k = np.abs(p.kappa[(self.pidx + np.arange(n)) % p.N]).max()
        return np.concatenate([q[:, 0] / 20, q[:, 1] / 10, [lat / HALF_W, k * 10]]).astype(np.float32)

    def observe(self):
        return dict(img=self.image(), speed=self.v, act=self.act.copy(), astar=self.astar_feat())

    # ---------- ground truth for supervised perception ----------
    def labels(self):
        cl = self.cl
        kbar = cl.kappa[(self.ci + 1 + np.arange(20)) % cl.N].mean()
        cls = 0 if kbar > 0.02 else (2 if kbar < -0.02 else 1)    # 0 left, 1 straight, 2 right
        idx = np.array([cl.at(self.ci, d) for d in (10, 20, 35)])
        fy = self._frame(cl.pts[idx])[:, 1]
        return cls, fy.astype(np.float32), self.lat

    # ---------- dynamics ----------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.act = a.astype(np.float32)
        self.delta += 0.5 * (a[0] * DMAX - self.delta)
        acc = (5.0 * a[1] if a[1] > 0 else 12.0 * a[1]) - 0.05 * self.v
        self.v = float(np.clip(self.v + acc * DT, 0, VMAX))
        tan = math.tan(self.delta)
        skid = self.v * self.v * abs(tan) / WB > ALAT
        if skid:
            self.v *= 0.92
            self.skids += 1
        self.th += self.v / WB * tan * DT
        self.pos = self.pos + self.v * DT * np.array([math.cos(self.th), math.sin(self.th)])
        old = self.cl.s[self.ci]
        self.ci, self.lat = self.cl.nearest(self.pos, self.ci)
        self.pidx = self.path.nearest(self.pos, self.pidx)[0]
        ds = (self.cl.s[self.ci] - old + self.cl.L / 2) % self.cl.L - self.cl.L / 2
        self.progress += ds
        self.t += 1
        r = 0.1 * ds - 0.002 - (0.05 if skid else 0.0)
        off = abs(self.lat) > HALF_W
        lap = self.progress >= self.cl.L - 1
        if off:
            r -= 2.0
        if lap:
            r += 2.0
        done = off or lap or self.t >= self.max_steps
        info = dict(success=bool(lap and not off), off_track=bool(off), progress=self.progress, t=self.t, skids=self.skids)
        return self.observe(), float(r), done, info
