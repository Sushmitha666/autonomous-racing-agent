"""Small local API for the React racing dashboard."""
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import RacingEnv
from features_np import FeatureBuilder, VARIANTS
from npnets import Policy
from planning import classical_action
from track import HALF_W

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
app = FastAPI(title="Apex Racing Lab API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class LapRequest(BaseModel):
    agent: str = "ppo_full_hybrid"
    seed: int = Field(default=0, ge=0, le=2)
    track: int = Field(default=1000, ge=0, le=9999)
    noise: float = Field(default=0.0, ge=0, le=0.1)


@app.get("/api/health")
def health():
    return {"status": "online"}


@app.get("/api/dashboard")
def dashboard():
    try:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        stress = json.loads((RESULTS / "stress.json").read_text(encoding="utf-8"))
        perception = json.loads((ROOT / "models" / "perception_metrics.json").read_text(encoding="utf-8"))
        return {"summary": summary, "stress": stress, "perception": perception}
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not load project results: {exc}")


@app.post("/api/run-lap")
def run_lap(req: LapRequest):
    env = RacingEnv([req.track], rng_seed=req.seed)
    env.img_noise = req.noise
    obs = env.reset(track_seed=req.track, start_seed=5000 + req.seed)
    if req.agent == "classical_astar_purepursuit":
        policy = lambda _env, _obs: classical_action(_env)
        name = "Classical A*"
    else:
        variant = req.agent
        if variant not in VARIANTS:
            raise HTTPException(400, "Choose one of the available trained policies.")
        weights = RESULTS / f"{variant}_s{req.seed}.npz"
        if not weights.exists():
            raise HTTPException(404, f"No trained weights found for {variant}, seed {req.seed}.")
        try:
            fb = FeatureBuilder(variant, 1, str(ROOT / "models"))
            net = Policy(fb.dim, path=str(weights))
        except Exception as exc:
            raise HTTPException(500, f"Could not load policy: {exc}")
        fb.reset_all()
        policy = lambda _env, current: net.pi.forward(fb([current]))[0]
        name = variant.replace("ppo_", "").replace("_", " ").title()

    points = [env.pos.copy().tolist()]
    speeds = [float(env.v)]
    rewards, total_reward, info = [], 0.0, {}
    for _ in range(env.max_steps):
        action = policy(env, obs)
        obs, reward, done, info = env.step(action)
        rewards.append(float(reward)); total_reward += reward
        points.append(env.pos.copy().tolist()); speeds.append(float(env.v))
        if done:
            break
    cl = env.cl
    edge_a = cl.pts + HALF_W * np.stack([-cl.tan[:, 1], cl.tan[:, 0]], axis=1)
    edge_b = cl.pts - HALF_W * np.stack([-cl.tan[:, 1], cl.tan[:, 0]], axis=1)
    return {
        "agent": name, "track": req.track, "points": points,
        "centerline": cl.pts[::3].round(2).tolist(),
        "edge_a": edge_a[::3].round(2).tolist(), "edge_b": edge_b[::3].round(2).tolist(),
        "speed": speeds, "reward_curve": np.cumsum(rewards).round(2).tolist(),
        "metrics": {"success": bool(info.get("success")), "steps": int(info.get("t", len(rewards))),
                    "lap_time": round(float(info.get("t", 0)) * 0.1, 2), "reward": round(float(total_reward), 2),
                    "skids": int(info.get("skids", 0)), "off_track": bool(info.get("off_track"))},
    }
