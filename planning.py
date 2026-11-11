"""Search/planning: A* and BFS on the road grid, lap-level racing-line planner, pure-pursuit controller."""
import heapq
import math
import time
from collections import deque

import numpy as np

from track import HALF_W, LoopPath, resample_closed, wrap

GRID = 1.0
MARGIN = 1.5          # keep this far from the kerb
CLEAR_K = 1.0         # clearance cost weight
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NBC = [math.sqrt(2) if a and b else 1.0 for a, b in NB]


class Grid:
    def __init__(self, track):
        P = track.cl.pts
        m = HALF_W + 4
        self.o = P.min(0) - m
        size = P.max(0) + m - self.o
        self.W, self.H = int(size[0] / GRID) + 1, int(size[1] / GRID) + 1
        gx = self.o[0] + (np.arange(self.W) + 0.5) * GRID
        gy = self.o[1] + (np.arange(self.H) + 0.5) * GRID
        G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        d, _ = track.tree.query(G)
        self.allowed = (d < HALF_W - MARGIN).tolist()
        self.cost = (1.0 + CLEAR_K * (d / HALF_W) ** 2).tolist()

    def cell(self, xy):
        c = np.floor((np.asarray(xy) - self.o) / GRID).astype(int)
        return int(c[1]) * self.W + int(c[0])

    def xy(self, idx):
        return self.o + (np.array([idx % self.W, idx // self.W]) + 0.5) * GRID


def _heur(a, b, W):
    dx, dy = abs(a % W - b % W), abs(a // W - b // W)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def astar(grid, s, g):
    W = grid.W
    gs, par, open_, closed = {s: 0.0}, {s: None}, [(_heur(s, g, W), s)], set()
    exp = 0
    while open_:
        _, u = heapq.heappop(open_)
        if u in closed:
            continue
        closed.add(u)
        exp += 1
        if u == g:
            break
        ux, uy = u % W, u // W
        for (dy, dx), c in zip(NB, NBC):
            x, y = ux + dx, uy + dy
            if x < 0 or y < 0 or x >= W or y >= grid.H:
                continue
            v = y * W + x
            if not grid.allowed[v] or v in closed:
                continue
            ng = gs[u] + c * grid.cost[v]
            if ng < gs.get(v, 1e18):
                gs[v], par[v] = ng, u
                heapq.heappush(open_, (ng + _heur(v, g, W), v))
    path, u = [], g
    while u is not None:
        path.append(u)
        u = par.get(u)
    return path[::-1], exp


def bfs(grid, s, g):
    W = grid.W
    par, q, exp = {s: None}, deque([s]), 0
    while q:
        u = q.popleft()
        exp += 1
        if u == g:
            break
        ux, uy = u % W, u // W
        for dy, dx in NB:
            x, y = ux + dx, uy + dy
            if x < 0 or y < 0 or x >= W or y >= grid.H:
                continue
            v = y * W + x
            if grid.allowed[v] and v not in par:
                par[v] = u
                q.append(v)
    path, u = [], g
    while u is not None:
        path.append(u)
        u = par.get(u)
    return path[::-1], exp


def plan_path(track, method="astar", anchor=40):
    """Plan a full lap as chained segments between centreline anchors. Returns (LoopPath, stats)."""
    grid = Grid(track)
    cl = track.cl
    anchors = list(range(0, cl.N, anchor)) + [0]
    search = astar if method == "astar" else bfs
    t0 = time.perf_counter()
    cells, exp = [], 0
    for a, b in zip(anchors[:-1], anchors[1:]):
        p, e = search(grid, grid.cell(cl.pts[a]), grid.cell(cl.pts[b]))
        exp += e
        cells += p if not cells else p[1:]
    dt = time.perf_counter() - t0
    xy = np.array([grid.xy(c) for c in cells[:-1]])
    raw_len = float(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum())
    k = 9                                                     # circular moving-average smoothing
    pad = np.vstack([xy[-k:], xy, xy[:k]])
    sm = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="same") for i in (0, 1)], 1)[k:-k]
    path = LoopPath(resample_closed(sm, 1.0))
    return path, dict(expanded=exp, time=dt, length=path.L, raw_length=raw_len)


# ---- classical controller (A* path + pure pursuit + curvature speed profile) ----
WB, DMAX, VMAX, ALAT = 2.5, 0.45, 20.0, 12.0


def classical_action(env, speed_scale=0.7):
    path, pi = env.path, env.pidx
    v = env.v
    Ld = 5.0 + 0.6 * v
    tgt = path.pts[path.at(pi, Ld)]
    d = tgt - env.pos
    c, s = math.cos(env.th), math.sin(env.th)
    x, y = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
    alpha = math.atan2(y, x)
    delta = math.atan2(2 * WB * math.sin(alpha), max(Ld, 1.0))
    steer = float(np.clip(delta / DMAX, -1, 1))
    n = int(10 + 1.6 * v)
    k = np.abs(path.kappa[(pi + np.arange(n)) % path.N]).max()
    vt = min(VMAX, math.sqrt(ALAT * speed_scale / (k + 1e-6)))
    acc = float(np.clip(0.8 * (vt - v), -1, 1))
    return np.array([steer, acc], dtype=np.float32)
