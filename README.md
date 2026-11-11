# Autonomous Racing Agent — Task 5 (CNN + LSTM + A* + RAG + PPO)

## Launch the React dashboard (Windows)

The dashboard has three pages: **Overview**, **Track Lab** (runs a real lap with the shipped NumPy policies), and **Model Insights**. It uses the existing trained `.npz` weights and result files; PyTorch is not required.

Open two PowerShell terminals in this project folder.

Terminal 1 — Python API:
```powershell
py -m pip install -r requirements.txt
py -m uvicorn api:app --reload
```

Terminal 2 — React frontend:
```powershell
cd frontend
npm install
npm run dev
```

Open the local URL Vite prints (normally `http://localhost:5173`). Leave both terminals running while using the dashboard. To stop them, press `Ctrl+C` in each terminal.

Pipeline (as required by the lab brief):
**Game → Perception/Prediction → Search/Planning → LLM/SLM + RAG → Deep RL → Action → Reward → Learning**

| Stage | File | What it does |
|---|---|---|
| Game | `track.py`, `env.py` | Procedural closed tracks, kinematic car, continuous actions `[steer, throttle/brake]`, grip limit (skid → speed loss), 32×32 egocentric camera, reward = progress − skid penalty, −2 off-track, +2 lap |
| Perception | `perception.py` | CNN: upcoming-corner class (left/straight/right) + lateral offset from the camera image |
| Prediction | `perception.py` | LSTM over `[CNN emb, speed, last action]` → centreline lateral offset at +10/+20/+35 units (beyond camera range) |
| Search/Planning | `planning.py` | A* (and BFS for comparison) on the road grid, chained into a full-lap path; pure-pursuit + curvature speed profile = classical baseline |
| RAG | `rag.py` | TF-IDF retrieval over a 16-passage racing knowledge base → advice → target-speed hint |
| Deep RL | `ppo.py`, `features.py` | PPO (Gaussian policy, GAE) on the variant-specific feature stack |
| Ablation | `run_all.sh`, `stress.py`, `report.py` | 5 learned variants × 3 seeds × 300k steps + classical baseline + stress test |

## Run on Windows / VS Code WITHOUT PyTorch (recommended if torch is blocked)
Needs only `pip install numpy scipy scikit-learn matplotlib`. Trained CNN/LSTM/PPO weights ship as `.npz`.
```powershell
python report.py            # rebuild tables + plots from shipped results   (seconds)
python stress_np.py         # re-run robustness test with the shipped policies (~4 min)
python ppo_np.py ppo_full_hybrid 0 300000   # train one PPO run in pure NumPy (~4 min)
python run_all_np.py        # all 15 PPO runs, 2 in parallel, no bash needed (~40 min)
python stress_np.py ; python report.py
```
`perception.py` (CNN/LSTM training), `ppo.py`, `features.py`, `stress.py`, `export_weights.py` use torch; everything `*_np.py`, `npnets.py`, `common.py` does not.
CNN/LSTM are *trained* with torch but *run* in NumPy (outputs match torch to ~1e-6). Retraining them needs torch (e.g. Colab).

## Run (with PyTorch)
```bash
pip install torch numpy scipy scikit-learn matplotlib
python3 perception.py          # collect data, train CNN + LSTM  (~1.5 min)  -> models/
python3 rag.py                 # RAG metrics
./run_all.sh                   # PPO ablation (~35 min on 2 CPU cores)   -> results/
python3 stress.py              # robustness test (~4 min)
python3 report.py              # results/results.md, summary.json, plots
```

## Results (see `results/results.md`)
Eval = 20 episodes on 10 unseen tracks, mean ± std over 3 seeds.

* All variants reach **100 % lap success** and ~44 reward on unseen tracks; learned agents lap in **~23 s vs 36 s** for the classical controller (which is deliberately conservative; PPO accepts ~9 skids/lap for speed).
* Sample efficiency: full hybrid reaches 80 % training success in ~17.5k steps vs ~20k for the others (small difference, 3 seeds).
* Robustness (hard tracks / 10 % camera noise): A*-only, +LSTM and full hybrid stayed at 100 %; CNN-only dropped to 82 % under noise (63 % combined) and CNN+A* to 70 % on hard tracks (one seed collapsed). CNN 95 % accuracy / F1 0.86; LSTM MAE 0.30 vs 0.42 for a no-memory MLP; A* expands ~7× fewer nodes than BFS.

## Honest caveats
* **The task is easy**: the nominal test saturates at 100 %, so the ablation separates variants only by lap time, sample efficiency (small) and the stress test. 3 seeds is too few to claim the robustness ordering is statistically solid — run more seeds / harder tracks before drawing conclusions in the report.
* **No LLM is available offline.** The "SLM" is a deterministic template composer over retrieved passages (`Advisor.generate` in `rag.py` is the one place to plug in a real LLM/SLM). Consequently grounded-answer = 100 % / hallucination = 0 % **by construction** — these RAG numbers become meaningful only with a real LLM. Retrieval precision/recall (0.67/0.62, k=4) are real.
* A* uses the known track map (HD-map assumption); CNN/LSTM/PPO encoders are trained in stages and the CNN/LSTM are frozen during PPO (not end-to-end).
* RAG hints are observation features only (no reward shaping), so the ablation stays fair.
* SAC was not implemented (task allows PPO/SAC); PPO only.
