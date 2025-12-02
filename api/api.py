import os
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from patchtst import PatchTST
from sklearn.preprocessing import MinMaxScaler
import uvicorn

# ==============================
# CONFIG
# ==============================
DB_USER = "stock_admin"
DB_PASS = "stocks"
DB_HOST = "localhost"
DB_NAME = "stock"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

SCALER_DIR = r"E:\ML_AI\scalers"
MODEL_DIR = r"E:\ML_AI\models"

INPUT_LEN = 256
PRED_LEN = 15
NUM_FEATURES = 5

engine = create_engine(DB_URL)

# ==============================
# FastAPI APP
# ==============================
app = FastAPI(title="Crypto PatchTST Forecaster")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Helper: Load Scaler
# ==============================
def load_scaler(ticker):
    path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Scaler not found: {path}")

    import joblib
    return joblib.load(path)

# ==============================
# Helper: Load Model
# ==============================
def load_model(ticker, device):
    path = os.path.join(MODEL_DIR, f"{ticker}_patchtst.pt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model not found: {path}")

    model = PatchTST(
        input_len=INPUT_LEN,
        pred_len=PRED_LEN,
        num_features=NUM_FEATURES,
    ).to(device)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ==============================
# Helper: Fetch last 256 rows
# ==============================
def load_latest_data(ticker):
    query = text("""
        SELECT scaled_close, scaled_sma_20, scaled_sma_50, scaled_rsi, scaled_macd
        FROM processed_market_data
        WHERE ticker = :ticker
        ORDER BY date DESC
        LIMIT 256
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})

    if len(df) < 256:
        raise HTTPException(status_code=400, detail="Not enough history (256 required).")

    return df.iloc[::-1].values  # reverse to ascending order

# ==============================
# API ROUTES
# ==============================

@app.get("/")
def home():
    return {"status": "online", "model": "PatchTST Forecaster", "sequence_length": INPUT_LEN}


@app.get("/predict/{ticker}")
def predict_ticker(ticker: str):

    ticker = ticker.upper()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1 — Load scaler & model
    scaler = load_scaler(ticker)
    model = load_model(ticker, device)

    # Step 2 — Load last 256 feature rows
    X = load_latest_data(ticker)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # Step 3 — Predict
    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy().flatten()

    # Step 4 — Only scale back "close" (1 column)
    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # Step 5 — return ONLY the final 15th minute
    last_pred = float(pred_real[-1])

    return {
        "ticker": ticker,
        "prediction_next_15min": last_pred,
        "device_used": device,
    }

# ==============================
# Run server
# ==============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
