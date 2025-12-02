import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import config
import time
from dataset_stream import get_dataset_for_ticker, INPUT_WINDOW, NUM_FEATURES
from patchtst import PatchTST


# -----------------------------
# GPU CHECK
# -----------------------------
def gpu_mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        print(f"🟦 GPU {tag} — Allocated: {a:.2f} MB | Reserved: {r:.2f} MB")
    else:
        print("⚠ CUDA not available — running on CPU")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {DEVICE}")
gpu_mem("initial")


# -----------------------------
# TRAIN CONFIG
# -----------------------------
EPOCHS = 5
BATCH = 512          # ← Optimal for RTX 2050
LR = 3e-4

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# TRAINING
# -----------------------------
def train_one(ticker):
    print(f"\n🚀 Training PatchTST for {ticker}")

    ds = get_dataset_for_ticker(ticker)
    if ds is None:
        print(f"❌ No dataset for {ticker}")
        return

    print(f"📊 Dataset size: {len(ds)} samples")

    # Split train/val
    val_len = max(500, len(ds) // 10)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    # Windows-safe DataLoader (no multiprocessing)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model
    model = PatchTST(INPUT_WINDOW, NUM_FEATURES).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()



    for ep in range(1, EPOCHS + 1):
        epoch_start = time.time()
        start_stamp = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n🕒 Epoch {ep}/{EPOCHS} started at {start_stamp}")

        model.train()
        train_losses = []

        for X, y in train_dl:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                pred = model(X)
                val_losses.append(loss_fn(pred, y).item())

        epoch_end = time.time()
        end_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        duration = epoch_end - epoch_start

        print(
            f"📆 Epoch {ep}/{EPOCHS} completed at {end_stamp}  "
            f"| Duration: {duration:.2f} sec  "
            f"| Train={np.mean(train_losses):.6f}  "
            f"| Val={np.mean(val_losses):.6f}"
        )

        gpu_mem(f"after epoch {ep}")

    # Save model
    path = f"{MODEL_DIR}/{ticker}_patchtst_7d.pt"
    torch.save(model.state_dict(), path)
    print(f"💾 Saved model → {path}")


# -----------------------------
# RUN ALL TICKERS
# -----------------------------
if __name__ == "__main__":
    for tic in config.CRYPTOS:
        train_one(tic)

    print("\n🎉 Training completed!")
    gpu_mem("final")
