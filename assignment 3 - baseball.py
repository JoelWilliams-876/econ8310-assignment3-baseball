# PyTorch DataLoader — minimal, clean example for tabular data
# Works for both regression or classification targets.
import os
os.system ('pip install torch pandas numpy')
# import subprocess, sys
# packages = ["torch", "pandas", "numpy"]
#import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

# ---- 1) Mock up some tabular data (replace this with your real DataFrame) ----
# Suppose we have 1,000 rows, 8 feature columns, and a numeric target.
df = pd.DataFrame(rng.normal(size=(1000, 8)), columns=[f"x{i}" for i in range(8)])
df["y"] = (df["x0"] * 2.0 - df["x1"] * 0.5 + rng.normal(scale=0.75, size=len(df)))  # regression target
# For classification, you could instead do: df["y"] = (df["y"] > df["y"].median()).astype(int)

FEATURE_COLS = [c for c in df.columns if c.startswith("x")]
TARGET_COL = "y"

# ---- 2) Custom Dataset ----
class TabularDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols, target_col, x_dtype=torch.float32, y_dtype=torch.float32):
        # Store as tensors for speed (avoid per-item Pandas overhead)
        x = frame[feature_cols].to_numpy(copy=True)
        y = frame[[target_col]].to_numpy(copy=True)  # keep 2D shape (N, 1)

        # Optional: standardize features (simple example). Remove if you’ll normalize elsewhere.
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - self.x_mean) / self.x_std

        self.X = torch.tensor(x, dtype=x_dtype)
        self.y = torch.tensor(y, dtype=y_dtype)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---- 3) Build the dataset and train/val splits ----
full_ds = TabularDataset(df, FEATURE_COLS, TARGET_COL)

n_total = len(full_ds)
n_val = int(0.2 * n_total)
n_train = n_total - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# ---- 4) DataLoaders ----
BATCH_SIZE = 64
NUM_WORKERS = 0  # set >0 for speed on Linux/macOS; keep 0 on Windows/Notebooks to avoid issues
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

# ---- 5) Example: iterate once (shows shapes) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    print("Batch X:", xb.shape, "Batch y:", yb.shape)
    break

# ---- OPTIONAL: If you need a custom collate_fn (e.g., variable-length items) ----
# def my_collate(batch):
#     xs, ys = zip(*batch)
#     return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
# train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=my_collate)

# ---- NOTES ----
# • Replace the mock df with your real DataFrame.
# • For classification, set y_dtype=torch.long and ensure targets are integer class IDs.
# • For images/text/audio, prefer torchvision/torchtext/torchaudio datasets+transforms.
