"""Aggregate everything into results/summary.json, results/results.md and plots."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rag
from common import TEST_TRACKS, run_classical
from planning import plan_path
from track import HALF_W, get_track

R = "results"
NAMES = {"classical_astar_purepursuit": "Classical (A* + pure pursuit)", "ppo_vision": "PPO + CNN",
         "ppo_astar_only": "PPO + A* only", "ppo_vision_astar": "PPO + CNN + A*",
         "ppo_vision_astar_lstm": "PPO + CNN + A* + LSTM", "ppo_full_hybrid": "Full hybrid (+ RAG)"}
ORDER = list(NAMES)


def convergence_episode(curve, win=20):
    ret = np.array([c[1] for c in curve])
    if len(ret) < win + 5:
        return None
    ma = np.convolve(ret, np.ones(win) / win, mode="valid")
    plateau = ret[-max(len(ret) // 10, win):].mean()
    idx = np.where(ma >= 0.9 * plateau)[0]
    return int(idx[0] + win) if len(idx) else None


def steps_to_success(curve, thr=0.8, win=20):
    s = np.array([c[3] for c in curve], float)
    st = np.array([c[0] for c in curve])
    if len(s) < win:
        return None
    ma = np.convolve(s, np.ones(win) / win, mode="valid")
    idx = np.where(ma >= thr)[0]
    return int(st[idx[0] + win - 1]) if len(idx) else None


def ms(x):
    x = [v for v in x if v is not None]
    return (float(np.mean(x)), float(np.std(x))) if x else (None, None)


def search_metrics():
    out = {"astar": [], "bfs": []}
    for s in TEST_TRACKS:
        for m in out:
            _, st = plan_path(get_track(s), m)
            out[m].append(st)
    return {m: {k: float(np.mean([r[k] for r in v])) for k in ("expanded", "time", "raw_length", "length")} for m, v in out.items()}


def main():
    runs = {}
    for f in sorted(glob.glob(f"{R}/ppo_*_s*.json")):
        d = json.load(open(f))
        runs.setdefault(d["variant"], []).append(d)
    classical = run_classical()
    runs["classical_astar_purepursuit"] = [dict(classical, curve=[], seed=0, steps=0)]
    summ = {}
    for v in ORDER:
        if v not in runs:
            continue
        rs = runs[v]
        summ[v] = dict(
            n_seeds=len(rs),
            success_rate=ms([r["success_rate"] for r in rs]), avg_reward=ms([r["avg_reward"] for r in rs]),
            reward_var=ms([r["reward_var"] for r in rs]), avg_ep_len=ms([r["avg_ep_len"] for r in rs]),
            avg_lap_time=ms([r["avg_lap_time"] for r in rs]), avg_skids=ms([r["avg_skids"] for r in rs]),
            off_track_rate=ms([r["off_track_rate"] for r in rs]),
            convergence_episodes=ms([convergence_episode(r["curve"]) for r in rs]) if rs[0]["curve"] else (None, None),
            steps_to_80pct_success=ms([steps_to_success(r["curve"]) for r in rs]) if rs[0]["curve"] else (None, None))
    perc = json.load(open("models/perception_metrics.json"))
    ragm = {k: rag.evaluate_rag(k) for k in (1, 2, 3, 4)}
    summ_all = dict(overall=summ, search=search_metrics(), cnn=perc["cnn"], lstm=perc["lstm"],
                    mlp_baseline=perc["mlp_baseline"], rag=ragm[4], rag_vs_k=ragm)
    json.dump(summ_all, open(f"{R}/summary.json", "w"), indent=1)

    # ---- plots ----
    cols = dict(zip(ORDER, ["#444", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    grid = np.linspace(0, 300000, 61)
    for v in ORDER[1:]:
        if v not in runs:
            continue
        for k, (key, lab) in enumerate([(1, "training episode return"), (3, "training success rate")]):
            ys = []
            for r in runs[v]:
                st = np.array([c[0] for c in r["curve"]]); y = np.array([float(c[key]) for c in r["curve"]])
                ys.append([y[(st > g - 15000) & (st <= g + 15000)].mean() if ((st > g - 15000) & (st <= g + 15000)).any() else np.nan for g in grid])
            m = np.nanmean(ys, 0)
            ax[k].plot(grid, m, label=NAMES[v], color=cols[v])
            ax[k].set_xlabel("environment steps"); ax[k].set_ylabel(lab)
    if "classical_astar_purepursuit" in summ:
        ax[0].axhline(summ["classical_astar_purepursuit"]["avg_reward"][0], color="#444", ls="--", label="Classical baseline (eval)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{R}/learning_curves.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    labs = [NAMES[v] for v in ORDER if v in summ]
    for a, (k, t) in zip(ax, [("success_rate", "Success rate (unseen tracks)"), ("avg_reward", "Average reward"), ("avg_lap_time", "Lap time of successful laps (s)")]):
        m = [summ[v][k][0] or 0 for v in ORDER if v in summ]; e = [summ[v][k][1] or 0 for v in ORDER if v in summ]
        a.bar(range(len(m)), m, yerr=e, color=[cols[v] for v in ORDER if v in summ], capsize=3)
        a.set_xticks(range(len(m))); a.set_xticklabels([l.replace(" + ", "\n+ ") for l in labs], fontsize=6); a.set_title(t, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{R}/ablation.png", dpi=130); plt.close()

    # trajectories on the first unseen track (episode 0 of each evaluation)
    tr = get_track(TEST_TRACKS[0]); P, T = tr.cl.pts, tr.cl.tan
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    for sgn in (-1, 1):
        e = np.vstack([P + sgn * HALF_W * N, (P + sgn * HALF_W * N)[:1]])
        ax.plot(e[:, 0], e[:, 1], color="k", lw=1)
    for v in ORDER:
        if v not in runs:
            continue
        t = runs[v][0].get("traj")
        if t:
            xy = np.array(t["xy"]); ax.plot(xy[:, 0], xy[:, 1], color=cols[v], lw=1.2, label=NAMES[v])
    ax.set_aspect("equal"); ax.legend(fontsize=6); ax.set_title(f"Evaluation laps, unseen track {TEST_TRACKS[0]}")
    plt.tight_layout(); plt.savefig(f"{R}/trajectories.png", dpi=130); plt.close()

    # ---- markdown ----
    def f(x, d=2, pct=False):
        if x is None or x[0] is None:
            return "-"
        if pct:
            return f"{100 * x[0]:.0f}% ± {100 * x[1]:.0f}"
        return f"{x[0]:.{d}f} ± {x[1]:.{d}f}"
    L = ["# Results — Autonomous Racing Agent\n", "Evaluation: 20 episodes on 10 **unseen** tracks, deterministic policy, mean ± std over training seeds.\n",
         "## Overall agent / ablation (DRL metrics)\n",
         "| Agent | Success | Avg reward | Reward var | Ep. length | Lap time (s) | Skids/ep | Convergence (episodes) | Steps to 80% success |", "|---|---|---|---|---|---|---|---|---|"]
    for v in ORDER:
        if v in summ:
            s = summ[v]
            L.append(f"| {NAMES[v]} | {f(s['success_rate'], pct=True)} | {f(s['avg_reward'], 1)} | {f(s['reward_var'], 1)} | {f(s['avg_ep_len'], 0)} | {f(s['avg_lap_time'], 1)} | {f(s['avg_skids'], 1)} | {f(s['convergence_episodes'], 0)} | {f(s['steps_to_80pct_success'], 0)} |")
    sm = summ_all["search"]
    if os.path.exists(f"{R}/stress.json"):
        st = json.load(open(f"{R}/stress.json"))
        conds = list(next(iter(st.values())).keys())
        L += ["\n## Robustness stress test (success rate; hard = tighter, wigglier unseen tracks; camera noise = 10% pixels flipped)\n",
              "| Agent | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for v in ORDER:
            ks = [k for k in st if k == v or k.rsplit("_s", 1)[0] == v]
            if ks:
                cells = []
                for c in conds:
                    x = [st[k][c]["success_rate"] for k in ks]
                    cells.append(f"{100 * np.mean(x):.0f}%" + (f" (min {100 * min(x):.0f}%)" if len(x) > 1 else ""))
                L.append(f"| {NAMES[v]} | " + " | ".join(cells) + " |")
        summ_all["stress"] = {v: {c: float(np.mean([st[k][c]["success_rate"] for k in st if k == v or k.rsplit("_s", 1)[0] == v])) for c in conds} for v in ORDER if any(k == v or k.rsplit("_s", 1)[0] == v for k in st)}
        json.dump(summ_all, open(f"{R}/summary.json", "w"), indent=1)
    L += ["\n## Search (A* vs BFS, full-lap planning, 10 unseen tracks)\n", "| Method | Path length (units) | Nodes expanded | Search time (ms) |", "|---|---|---|---|"]
    for m in ("astar", "bfs"):
        L.append(f"| {m.upper()} | {sm[m]['length']:.1f} | {sm[m]['expanded']:.0f} | {1000 * sm[m]['time']:.1f} |")
    c = summ_all["cnn"]; l = summ_all["lstm"]; mb = summ_all["mlp_baseline"]; rg = summ_all["rag"]
    L += ["\n## CNN road perception (3-class upcoming-corner classifier, unseen tracks, macro-averaged)\n",
          f"Accuracy {c['accuracy']:.3f} · Precision {c['precision']:.3f} · Recall {c['recall']:.3f} · F1 {c['f1']:.3f} · lateral-offset MAE {c['lat_mae']:.2f} units\n",
          "\n## LSTM look-ahead road prediction (lateral offset of centreline at +10/+20/+35 units)\n",
          "| Model | MAE | RMSE | Accuracy (|err|<1.5) |", "|---|---|---|---|",
          f"| LSTM | {l['mae']:.3f} | {l['rmse']:.3f} | {l['accuracy_within_1p5']:.3f} |",
          f"| Single-frame MLP (no memory) | {mb['mae']:.3f} | {mb['rmse']:.3f} | {mb['accuracy_within_1p5']:.3f} |",
          "\n## RAG advisor (k=4, 21 driving situations)\n",
          f"Retrieval precision {rg['retrieval_precision']:.2f} · recall {rg['retrieval_recall']:.2f} · grounded-answer rate {rg['grounded_answer_rate']:.2f} · hallucination rate {rg['hallucination_rate']:.2f} · correct target-speed advice {rg['correct_target_speed_rate']:.2f}\n",
          "\n![curves](learning_curves.png)\n![ablation](ablation.png)\n![traj](trajectories.png)\n"]
    open(f"{R}/results.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
