import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import config

from dataset_stream_inc import (
    get_incremental_dataset,
    INPUT_WINDOW,
    NUM_FEATURES,
)

from patchtst import PatchTST

VERSION_DIR = "models_versions"
os.makedirs(VERSION_DIR, exist_ok=True)


# -----------------------------
# DEVICE SETTINGS
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {DEVICE}")


# -----------------------------
# TRAIN CONFIG
# -----------------------------
EPOCHS = 5           # small updates (fast daily training)
BATCH = 512
LR = 2e-4

MODEL_DIR = "models_incremental"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_or_init(ticker):
    """
    Priority:
    1) Load incremental model if exists
    2) Else load full-history model if exists
    3) Else start fresh
    """
    model = PatchTST(INPUT_WINDOW, NUM_FEATURES).to(DEVICE)

    inc_path = f"models_incremental/{ticker}_patchtst_7d_incremental.pt"
    full_path = f"models/{ticker}_patchtst_7d.pt"

    # 1️⃣ Load incremental model (most recent, best)
    if os.path.exists(inc_path):
        print(f"🧠 Loading incremental model → {inc_path}")
        state = torch.load(inc_path, map_location=DEVICE)
        model.load_state_dict(state)
        return model

    # 2️⃣ Fallback: load full-history model for the first warm-start
    if os.path.exists(full_path):
        print(f"🧠 Loading full-history model → {full_path}")
        state = torch.load(full_path, map_location=DEVICE)
        model.load_state_dict(state)
        return model

    # 3️⃣ No model exists: start new model
    print("🆕 No existing model found → starting fresh PatchTST")
    return model



# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_incremental_warm(ticker):
    print(f"\n🚀 Warm-Start Incremental Training → {ticker} (180 days)")

    ds = get_incremental_dataset(ticker, days=180)
    if ds is None:
        print(f"❌ No incremental data available for {ticker}")
        return

    print(f"📊 Incremental dataset size: {len(ds)} samples")

    # Train/Val split
    val_len = max(500, len(ds) // 12)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Load existing model or initialize fresh
    model = load_or_init(ticker)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n🕒 Epoch {ep}/{EPOCHS}")

        model.train()
        train_losses = []

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_losses.append(loss_fn(pred, y).item())

        duration = time.time() - epoch_start

        print(
            f"📆 Epoch {ep}/{EPOCHS} | "
            f"Train={np.mean(train_losses):.6f} | "
            f"Val={np.mean(val_losses):.6f} | "
            f"Time={duration:.2f}s"
        )

    # -----------------------------
    # SAVE LATEST (overwrite)
    # -----------------------------
    latest_path = f"{MODEL_DIR}/{ticker}_patchtst_7d_incremental.pt"
    torch.save(model.state_dict(), latest_path)
    print(f"💾 Saved LATEST model → {latest_path}")

    # -----------------------------
    # SAVE VERSIONED COPY (timestamped)
    # -----------------------------
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    version_path = f"{VERSION_DIR}/{ticker}_{timestamp}.pt"
    torch.save(model.state_dict(), version_path)

    print(f"📦 Saved VERSIONED model → {version_path}")


# -----------------------------
# RUN ALL TICKERS
# -----------------------------
if __name__ == "__main__":
    for tic in config.CRYPTOS:
        train_incremental_warm(tic)

    print("\n🎉 Warm-Start Incremental training complete!")
