"""Run once WHERE TORCH WORKS: converts trained .pt files to torch-free .npz (already done in the shipped zip)."""
import glob
import numpy as np
import torch

sd = {}
for name, f in (("cnn", "models/cnn.pt"), ("lstm", "models/lstm.pt")):
    for k, v in torch.load(f).items():
        sd[f"{name}.{k}"] = v.numpy()
np.savez("models/np_models.npz", **sd)
for f in glob.glob("results/ppo_*_s*.pt"):
    w = {k: v.numpy() for k, v in torch.load(f).items()}
    np.savez(f[:-3] + ".npz", **w)
print("exported", len(sd), "perception tensors and", len(glob.glob("results/*.npz")), "policies")
