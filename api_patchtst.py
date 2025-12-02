from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from torch.amp import autocast
from patchtst import PatchTST
import config

app = FastAPI(title="Crypto 7-Day Forecast API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_WINDOW = 1024
NUM_FEATURES = 5

DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
)

# ===============================
# Load last 1024 rows
# ===============================
def load_last_window(ticker):
    engine = create_engine(DB_URL)

    query = """
        SELECT *
        FROM processed_market_data
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """

    df = pd.read_sql(query, engine, params=(ticker, INPUT_WINDOW))

    if len(df) < INPUT_WINDOW:
        raise ValueError(f"Not enough data for {ticker}")

    df = df.sort_values("date").reset_index(drop=True)

    window = df[
        ["scaled_close", "scaled_sma_20", "scaled_sma_50", "scaled_rsi", "scaled_macd"]
    ].values.astype(np.float32)

    return window, df.iloc[-1].to_dict()


# ===============================
# Load PatchTST Model
# ===============================
def load_model(ticker):
    model = PatchTST(
        input_len=INPUT_WINDOW,
        num_features=NUM_FEATURES,
        patch_len=16,
        stride=8,
        d_model=128,
        depth=4
    ).to(DEVICE)

    path = f"models/{ticker}_patchtst_7d.pt"

    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ===============================
# Inverse price scaling
# ===============================
def inverse_price(scaler, scaled_value, row):
    arr = np.array([[
        scaled_value,
        row["scaled_sma_20"],
        row["scaled_sma_50"],
        row["scaled_rsi"],
        row["scaled_macd"],
    ]])

    return float(scaler.inverse_transform(arr)[0][0])


# ===============================
# Predict 7-day future
# ===============================
def predict_7d_future(ticker):
    window, latest_row = load_last_window(ticker)
    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model = load_model(ticker)

    with torch.no_grad():
        with autocast("cuda", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
            pred_scaled = model(window_tensor).item()

    scaler = joblib.load(f"scalers/{ticker}_scaler.pkl")

    current_real = inverse_price(scaler, latest_row["scaled_close"], latest_row)
    pred_real = inverse_price(scaler, pred_scaled, latest_row)

    roi = ((pred_real - current_real) / current_real) * 100
    direction = "UP" if roi > 0 else "DOWN"

    return {
        "ticker": ticker,
        "current_price": round(current_real, 3),
        "prediction_price_7d": round(pred_real, 3),
        "roi_percent_7d": round(roi, 2),
        "direction": direction
    }


# ===============================
# FASTAPI ROUTE
# ===============================
@app.get("/predict/{ticker}")
def predict_endpoint(ticker: str):
    ticker = ticker.upper()

    try:
        result = predict_7d_future(ticker)
        return JSONResponse(content=result)

    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"No trained model found for {ticker}"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ===============================
# AUTO-START SERVER WHEN RUN DIRECTLY
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_patchtst:app",   # <-- filename:app
        host="0.0.0.0",
        port=8000,
        reload=True
    )
