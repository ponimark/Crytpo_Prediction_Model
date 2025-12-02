import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
import config


DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
)
engine = create_engine(DB_URL)

SCALER_DIR = "scalers"
os.makedirs(SCALER_DIR, exist_ok=True)


# =====================================================
# INDICATORS
# =====================================================
def add_indicators(df):
    df = df.copy()

    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    df = df.bfill().ffill()
    return df


# =====================================================
# INCREMENTAL PREPROCESSOR
# =====================================================
def preprocess_incremental():
    print("📌 Getting tickers...")
    tickers = pd.read_sql(
        "SELECT DISTINCT ticker FROM crypto_market_data",
        engine
    )["ticker"].tolist()

    for tic in tickers:
        print(f"\n🔹 Processing {tic}")

        # --------------------------------------------
        # 1. Get last processed timestamp
        # --------------------------------------------
        last_q = f"""
            SELECT MAX(date) AS last_dt
            FROM processed_market_data
            WHERE ticker='{tic}'
        """
        last_dt = pd.read_sql(last_q, engine)["last_dt"][0]

        if pd.isna(last_dt):
            print("⚠ First run → full preprocess")
            last_dt = None

        # --------------------------------------------
        # 2. Load NEW rows
        # --------------------------------------------
        if last_dt is None:
            new_q = f"""
                SELECT date, ticker, close, high, low
                FROM crypto_market_data
                WHERE ticker='{tic}'
                ORDER BY date ASC
            """
        else:
            new_q = f"""
                SELECT date, ticker, close, high, low
                FROM crypto_market_data
                WHERE ticker='{tic}' AND date > '{last_dt}'
                ORDER BY date ASC
            """

        new_df = pd.read_sql(new_q, engine)

        if new_df.empty:
            print("✔ No new rows")
            continue

        print(f"📥 New rows: {len(new_df)}")

        # --------------------------------------------
        # 3. Load history buffer (for indicators)
        # --------------------------------------------
        if last_dt:
            hist_q = f"""
                SELECT date, ticker, close, high, low
                FROM crypto_market_data
                WHERE ticker='{tic}' AND date <= '{last_dt}'
                ORDER BY date DESC
                LIMIT 100
            """
            hist_df = pd.read_sql(hist_q, engine).sort_values("date")
        else:
            hist_df = pd.DataFrame()

        # Merge
        full_df = pd.concat([hist_df, new_df], ignore_index=True)
        full_df["date"] = pd.to_datetime(full_df["date"])

        # --------------------------------------------
        # 4. Compute indicators
        # --------------------------------------------
        full_df = add_indicators(full_df)

        # Keep only new rows
        processed_new = (
            full_df[full_df["date"] > last_dt].copy() if last_dt else full_df.copy()
        )

        # --------------------------------------------
        # 5. Fit scaler using the last 30 days (robust)
        # --------------------------------------------
        scaler_path = f"{SCALER_DIR}/{tic}_scaler.pkl"

        feat_cols = ["close", "sma_20", "sma_50", "rsi", "macd"]

        # Load last 30 days of processed data if exists
        scaler_data_q = f"""
            SELECT close, scaled_close, scaled_sma_20, scaled_sma_50, scaled_rsi, scaled_macd
            FROM processed_market_data
            WHERE ticker='{tic}'
            ORDER BY date DESC
            LIMIT 30000
        """
        existing = pd.read_sql(scaler_data_q, engine)

        scaler = MinMaxScaler()

        if not existing.empty:
            # Rebuild fit input with raw features
            fit_input = processed_new[feat_cols]
            scaler.fit(fit_input)
        else:
            scaler.fit(processed_new[feat_cols])

        joblib.dump(scaler, scaler_path)

        # --------------------------------------------
        # 6. Scale new rows
        # --------------------------------------------
        scaled = scaler.transform(processed_new[feat_cols])

        processed_new.loc[:, "scaled_close"] = scaled[:, 0]
        processed_new.loc[:, "scaled_sma_20"] = scaled[:, 1]
        processed_new.loc[:, "scaled_sma_50"] = scaled[:, 2]
        processed_new.loc[:, "scaled_rsi"] = scaled[:, 3]
        processed_new.loc[:, "scaled_macd"] = scaled[:, 4]

        # --------------------------------------------
        # 7. Keep ONLY your schema columns
        # --------------------------------------------
        final = processed_new[
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

        final.to_sql(
            "check_process",
            engine,
            index=False,
            if_exists="append"
        )

        print(f"✔ Inserted {len(final)} rows for {tic}")

    print("\n✅ INCREMENTAL PREPROCESSING COMPLETE")


if __name__ == "__main__":
    preprocess_incremental()
