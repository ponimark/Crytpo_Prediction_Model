import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
import config

# --------------------------------
# DB CONNECTION
# --------------------------------
DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
)
engine = create_engine(DB_URL)

SCALER_DIR = "scalers"
os.makedirs(SCALER_DIR, exist_ok=True)


# --------------------------------
# INDICATORS
# --------------------------------
def add_indicators(df):
    df = df.copy()

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    df = df.bfill().ffill()
    return df


# ========================================================
# MAIN — MEMORY EFFICIENT: READ + PROCESS + WRITE PER TICKER
# ========================================================
def preprocess_eff():
    print("📌 Getting tickers (small query)...")
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM crypto_market_data", engine)["ticker"].tolist()
    print("✔ Tickers found:", tickers)

    print("⚠ Dropping old processed_market_data")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS processed_market_data"))

    # ================================
    # PROCESS TICKERS ONE BY ONE
    # ================================
    for tic in tickers:
        print(f"\n🔹 Processing {tic}...")

        # ---- Load only THIS ticker into memory
        query = f"""
            SELECT date, ticker, close, high, low
            FROM crypto_market_data
            WHERE ticker = '{tic}'
            ORDER BY date ASC
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            print(f"⚠ No rows for {tic}, skipping.")
            continue

        df["date"] = pd.to_datetime(df["date"])

        print(f"📈 Loaded {len(df)} rows for {tic}")

        # ---- Compute indicators
        df = add_indicators(df)

        # ---- Prepare feature scaling
        feat_cols = ["close", "sma_20", "sma_50", "rsi", "macd"]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feat_cols])
        joblib.dump(scaler, f"{SCALER_DIR}/{tic}_scaler.pkl")

        for i, col in enumerate(feat_cols):
            df[f"scaled_{col}"] = scaled[:, i]

        # ---- Keep only final columns
        final = df[
            [
                "date",
                "ticker",
                "close",
                "scaled_close",
                "scaled_sma_20",
                "scaled_sma_50",
                "scaled_rsi",
                "scaled_macd",
            ]
        ]

        # ---- Append directly to DB (does NOT keep in memory)
        final.to_sql("processed_market_data", engine, index=False, if_exists="append")

        print(f"✔ Saved {tic} ({len(final)} rows)")

        # ---- Free memory
        del df
        del final

    print("\n✅ DONE — All tickers processed efficiently (low RAM).")


if __name__ == "__main__":
    preprocess_eff()
