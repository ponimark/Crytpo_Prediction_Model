import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
from math import exp
import config

from patchtst import PatchTST

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
SCALER_DIR = "scalers"

INPUT_WINDOW = 256     # ✔ YOU TRAINED WITH THIS
HORIZON = 10080
NUM_FEATS = 5

app = FastAPI(title="Crypto PatchTST Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}
LATEST_WINDOWS = {}

# ---------------------------
# STARTUP
# ---------------------------
@app.on_event("startup")
def startup_load():
    print("🔄 Loading models and latest windows...")

    from sqlalchemy import create_engine

    DB_URL = (
        f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
        f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
    )
    engine = create_engine(DB_URL)

    for ticker in config.CRYPTOS:
        safe = ticker.replace("/", "")

        # Load model
        model_path = f"{MODEL_DIR}/{safe}_patchtst_7d.pt"
        if not os.path.exists(model_path):
            print(f"⚠ Missing model: {safe}")
            continue

        model = PatchTST(INPUT_WINDOW, NUM_FEATS).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        MODELS[safe] = model

        print(f"✔ Loaded model for {safe}")

        # Load latest 256 rows
        q = f"""
            SELECT scaled_close, scaled_sma_20, scaled_sma_50,
                   scaled_rsi, scaled_macd, close, date
            FROM processed_market_data
            WHERE ticker = '{safe}'
            ORDER BY date DESC
            LIMIT {INPUT_WINDOW}
        """

        df = pd.read_sql(q, engine).sort_values("date")

        if len(df) < INPUT_WINDOW:
            print(f"⚠ Not enough rows for {safe}")
            continue

        LATEST_WINDOWS[safe] = df.reset_index(drop=True)
        print(f"✔ Cached latest 256 rows for {safe}")

    print("🎉 Startup complete!\n")


# ---------------------------
# PREDICT
# ---------------------------
# ---------------------------
# PREDICT
# ---------------------------
def predict_single(ticker: str):

    ticker = ticker.upper()

    if ticker not in MODELS:
        return {"error": f"Model not loaded for {ticker}"}

    model = MODELS[ticker]
    df = LATEST_WINDOWS.get(ticker)

    if df is None:
        return {"error": f"No cached data for {ticker}"}

    # Current true market price
    current_price = float(df["close"].iloc[-1])

    # Build the 256×5 feature window
    feats = df[
        ["scaled_close", "scaled_sma_20", "scaled_sma_50",
         "scaled_rsi", "scaled_macd"]
    ].values

    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Predict log-return
    with torch.no_grad():
        log_ret = model(X).cpu().numpy()[0][0]

    # Convert log-return → future price
    predicted_price = current_price * np.exp(log_ret)

    # Direction
    diff = predicted_price - current_price
    pct = (diff / current_price) * 100

    if abs(diff) < current_price * 0.001:
        direction = "FLAT"
    elif diff > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    # -------- CLEAN OUTPUT --------
    return {
        "ticker": ticker,
        "current_price": round(float(current_price), 2),
        "predicted_price_7d": round(float(predicted_price), 2),
        "percent_change": round(float(pct), 2),
        "direction": direction
    }



# ---------------------------
# ROUTES
# ---------------------------
@app.get("/")
def home():
    return {"message": "PatchTST predictor running!"}


@app.get("/predict")
def predict_endpoint(ticker: str = Query(...)):
    return predict_single(ticker)


# ---------------------------
# LAUNCH
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting FastAPI...")
    uvicorn.run("predict_patchtst:app", host="0.0.0.0", port=8000)
