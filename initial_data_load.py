import pandas as pd
import psycopg2
import io
import os
import config


DB_URL = (
    f"postgresql://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:{config.DB_PORT}/{config.DB_NAME}"
)


def detect_ticker(filepath):
    filename = os.path.basename(filepath)
    # Example: ADAUSD_1m_Coinbase.csv → ADAUSD
    return filename[:6].upper()


def fast_load_csv(filepath):
    ticker = detect_ticker(filepath)
    print(f"\n🚀 Loading {filepath}  |  Detected Ticker = {ticker}")

    # Load CSV
    df = pd.read_csv(filepath)

    # FIX: Correct rename (capital O)
    df.rename(columns={
        "Open time": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # FIX: Convert string date WITHOUT milliseconds
    df["date"] = pd.to_datetime(df["date"])

    # Add ticker
    df["ticker"] = ticker

    # Reorder to match database schema
    df = df[["date", "open", "high", "low", "close", "volume", "ticker"]]

    print(f"📊 Rows to import: {len(df)}")

    # COPY import (fastest)
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    copy_sql = """
        COPY crypto_market_data (date, open, high, low, close, volume, ticker)
        FROM STDIN WITH CSV HEADER
    """

    cursor.copy_expert(copy_sql, buffer)
    conn.commit()

    cursor.close()
    conn.close()

    print(f"✅ Successfully imported {len(df)} rows for {ticker}")


if __name__ == "__main__":
    fast_load_csv("data/ADAUSD_1m_Coinbase.csv")
    fast_load_csv("data/BTCUSD_1m_Coinbase.csv")
    fast_load_csv("data/ETHUSD_1m_Coinbase.csv")
    fast_load_csv("data/XRPUSD_1m_Coinbase.csv")
    fast_load_csv("data/BNBUSD_1m_BitMEX.csv")

