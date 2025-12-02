import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import config

# -----------------------
# CONFIG
# -----------------------
DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
)

engine = create_engine(DB_URL)

TABLE = "check_crypto"
final_table='crypto_market_data'

# Coinbase symbols → your DB ticker names
SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ADAUSD": "ADA-USD",
    "XRPUSD": "XRP-USD"
}

# -----------------------
# GET LATEST TIMESTAMP
# -----------------------
def get_latest_dt(ticker):
    q = text(f"SELECT MAX(date) FROM {final_table} WHERE ticker=:t")
    with engine.connect() as conn:
        r = conn.execute(q, {"t": ticker}).fetchone()[0]
    return r


# -----------------------
# FETCH COINBASE WINDOW
# -----------------------
def fetch_coinbase(product_id, start_dt, end_dt):
    url = (
        f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        f"?granularity=60"
        f"&start={start_dt.isoformat()}"
        f"&end={end_dt.isoformat()}"
    )

    r = requests.get(url)
    try:
        data = r.json()
    except:
        print("   ❌ JSON decode error")
        return pd.DataFrame()

    if type(data) is dict:
        print("   ❌ Coinbase error:", data)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)

    return df[["date", "open", "high", "low", "close", "volume"]]


# -----------------------
# INGEST ONE SYMBOL
# -----------------------
def ingest(symbol, product_id):
    print(f"\n🚀 Fetching new data for {symbol} ...")

    last_dt = get_latest_dt(symbol)
    if last_dt is None:
        print("   ❌ No existing data in DB — cannot incremental sync!")
        return

    print(f"   Last DB timestamp: {last_dt}")

    # We want data from midnight of the day AFTER last_dt
    target_date = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    # ---- Hard-coded 4-hour windows ----
    windows = [
        ("00:00", "04:00"),
        ("04:00", "08:00"),
        ("08:00", "12:00"),
        ("12:00", "16:00"),
        ("16:00", "20:00"),
        ("20:00", "23:59")
    ]

    full = []
    first = True

    for s, e in windows:
        start_dt = datetime.strptime(f"{target_date} {s}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{target_date} {e}", "%Y-%m-%d %H:%M")

        # FIX: shift all except first window by +1 minute to avoid overlap
        if not first:
            start_dt += timedelta(minutes=1)
        first = False

        print(f"   ▶ Window {start_dt.time()} → {end_dt.time()}")

        df_w = fetch_coinbase(product_id, start_dt, end_dt)
        if not df_w.empty:
            full.append(df_w)

    if not full:
        print("   ℹ No new rows returned.")
        return

    df = pd.concat(full).sort_values("date")

    # Drop duplicates from Coinbase boundaries
    df = df.drop_duplicates(subset=["date"])

    # Keep only strictly newer than last_dt
    df = df[df["date"] > last_dt]

    if df.empty:
        print("   ℹ Nothing new after duplicate removal.")
        return

    # Add ticker column
    df["ticker"] = symbol

    # Insert
    print(f"   ✔ Inserting {len(df)} new rows...")

    with engine.begin() as conn:
        df.to_sql(final_table, conn, if_exists="append", index=False)

    print("   ✔ Done.")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print("🚀 Starting 24-hour windowed ingestion...")

    for ticker, product in SYMBOLS.items():
        ingest(ticker, product)

    print("\n🎉 Ingestion complete!")
