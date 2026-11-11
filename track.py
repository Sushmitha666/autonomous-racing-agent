"""Procedural closed race tracks + a loop-path helper (used for centreline AND the A* path)."""
import numpy as np
from scipy.spatial import cKDTree

HALF_W = 6.0      # half track width (units ~ metres)
RES = 0.5         # raster resolution of the camera mask


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resample_closed(pts, ds=1.0):
    """Resample a closed polyline to (approximately) uniform spacing ds."""
    p = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(round(L / ds)), 8)
    t = np.linspace(0, L, n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


class LoopPath:
    """Closed path with ~1 unit spacing; tangent, curvature, arclength, local nearest search."""

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.N = len(self.pts)
        nxt, prv = np.roll(self.pts, -1, 0), np.roll(self.pts, 1, 0)
        t = nxt - prv
        self.tan = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        self.head = np.arctan2(self.tan[:, 1], self.tan[:, 0])
        seg = np.linalg.norm(nxt - self.pts, axis=1)
        self.ds = float(seg.mean())
        self.s = np.concatenate([[0], np.cumsum(seg)[:-1]])
        self.L = float(seg.sum())
        dh = wrap(np.roll(self.head, -1) - np.roll(self.head, 1))
        self.kappa = dh / (2 * self.ds)

    def nearest(self, pos, hint):
        idx = (hint + np.arange(-8, 26)) % self.N
        d = self.pts[idx] - pos
        j = int(np.argmin((d * d).sum(1)))
        i = int(idx[j])
        if (d[j] ** 2).sum() > 400:           # lost: global search
            i = int(np.argmin(((self.pts - pos) ** 2).sum(1)))
        dx, dy = pos - self.pts[i]
        lat = self.tan[i, 0] * dy - self.tan[i, 1] * dx   # + = left of path
        return i, float(lat)

    def at(self, i, dist):
        return (i + int(round(dist / self.ds))) % self.N


class Track:
    def __init__(self, seed):
        self.seed = seed
        rng = np.random.default_rng(seed)
        hard = seed >= 2000                       # 'hard' unseen tracks: wigglier, tighter corners
        amp = 0.34 if hard else 0.16
        kmax = 1 / 11.0 if hard else 1 / 16.0
        for _ in range(400):
            th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            R0 = rng.uniform(62, 80)
            r = np.ones_like(th)
            for k in range(2, 7):
                r += rng.uniform(0, amp) / (k - 1) ** 0.7 * np.cos(k * th + rng.uniform(0, 2 * np.pi))
            pts = np.stack([R0 * r * np.cos(th), R0 * r * np.sin(th)], 1)
            cl = LoopPath(resample_closed(pts, 1.0))
            if np.abs(cl.kappa).max() < kmax:
                break
            amp *= 0.99
        self.cl = cl
        P = cl.pts
        m = HALF_W + 18
        self.origin = P.min(0) - m
        size = P.max(0) + m - self.origin
        self.shape = (int(size[1] / RES) + 1, int(size[0] / RES) + 1)   # (rows=y, cols=x)
        self.tree = cKDTree(P)
        gx = self.origin[0] + (np.arange(self.shape[1]) + 0.5) * RES
        gy = self.origin[1] + (np.arange(self.shape[0]) + 0.5) * RES
        G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        d, _ = self.tree.query(G)
        self.mask = (d < HALF_W).reshape(self.shape)

    def sample(self, xy):
        """xy: (...,2) world coords -> road mask value (0/1), 0 outside raster."""
        ij = np.floor((xy - self.origin) / RES).astype(np.int64)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < self.shape[1]) & (ij[..., 1] >= 0) & (ij[..., 1] < self.shape[0])
        ix = np.clip(ij[..., 0], 0, self.shape[1] - 1)
        iy = np.clip(ij[..., 1], 0, self.shape[0] - 1)
        return self.mask[iy, ix] & ok


_CACHE = {}


def get_track(seed):
    if seed not in _CACHE:
        _CACHE[seed] = Track(seed)
    return _CACHE[seed]
