import pandas as pd
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import config


# ---------------------------
# DATABASE CONNECTION
# ---------------------------
DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:{config.DB_PORT}/{config.DB_NAME}"
)
engine = create_engine(DB_URL)


# ---------------------------
# CREATE TABLE IF NOT EXISTS
# ---------------------------
def ensure_table_exists():
    create_sql = """
    CREATE TABLE IF NOT EXISTS crypto_market_data (
        date TIMESTAMP NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        ticker VARCHAR(20),
        PRIMARY KEY (date, ticker)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))

    print("✔ Table ensured: crypto_market_data")


# ---------------------------
# GET MOST RECENT TIMESTAMP FOR A TICKER
# ---------------------------
def get_latest_timestamp(ticker):
    query = text("SELECT MAX(date) FROM crypto_market_data WHERE ticker = :tic")
    with engine.connect() as conn:
        result = conn.execute(query, {"tic": ticker})
        return result.scalar()


# ---------------------------
# FETCH OHLC MINUTE DATA FROM KRAKEN (INCREMENTAL)
# ---------------------------
def fetch_kraken_incremental(pair, start_dt):
    url = "https://api.kraken.com/0/public/OHLC"
    all_rows = []

    # Kraken uses UNIX seconds
    since = int(start_dt.timestamp())

    while True:
        params = {
            "pair": pair,
            "interval": 1,
            "since": since,
        }

        print(f"📡 Kraken → {pair} starting @ {datetime.fromtimestamp(since)}")

        r = requests.get(url, params=params)
        data = r.json()

        if "error" in data and data["error"]:
            print(f"❌ Kraken error: {data['error']}")
            break

        # Extract returned symbol key dynamically
        result_key = [key for key in data["result"].keys() if key != "last"][0]
        candles = data["result"][result_key]
        last_timestamp = data["result"]["last"]

        if not candles:
            print("✔ No more data (finished)")
            break

        for c in candles:
            ts = datetime.fromtimestamp(c[0])
            all_rows.append([
                ts,
                float(c[1]),  # open
                float(c[2]),  # high
                float(c[3]),  # low
                float(c[4]),  # close
                float(c[6])   # volume
            ])

        # If Kraken returns same timestamp → no more data
        if last_timestamp == since:
            print("✔ Reached last timestamp")
            break

        since = last_timestamp

    df = pd.DataFrame(all_rows, columns=[
        "date", "open", "high", "low", "close", "volume"
    ])

    return df


# ---------------------------
# MAP BINANCE-LIKE SYMBOL → KRAKEN SYMBOL
# ---------------------------
def map_to_kraken(ticker):
    # BTCUSDT → BTCUSD
    return ticker.replace("USDT", "USD")


# ---------------------------
# MAIN INGEST FUNCTION
# ---------------------------
def ingest_minute_data():
    print("🚀 Starting incremental 1-minute ingestion (Kraken)…")
    ensure_table_exists()

    all_frames = []

    for ticker in config.CRYPTOS:
        print(f"\n🔹 Processing {ticker}")

        kraken_pair = map_to_kraken(ticker)
        last_ts = get_latest_timestamp(ticker)

        if last_ts is None:
            # First time loading data
            start_dt = datetime(2022, 1, 1)  # ← Your training range
            print(f"→ First-time load from {start_dt}")
        else:
            start_dt = last_ts + timedelta(minutes=1)
            print(f"→ Incremental load from {start_dt}")

        df = fetch_kraken_incremental(kraken_pair, start_dt)

        if df is None or df.empty:
            print("⚠ No new data")
            continue

        df["ticker"] = ticker

        print(f"📥 Retrieved {len(df)} new rows")
        all_frames.append(df)

    if not all_frames:
        print("\n⚠ No new candles to insert.")
        return

    final_df = pd.concat(all_frames)

    # Remove duplicates
    final_df.drop_duplicates(subset=["date", "ticker"], inplace=True)

    print(f"📌 Final rows to insert after dedupe: {len(final_df)}")

    final_df.to_sql("crypto_market_data", engine, if_exists="append", index=False)

    print("\n✅ SUCCESS: Kraken minute data ingested!")


# ---------------------------
if __name__ == "__main__":
    ingest_minute_data()
