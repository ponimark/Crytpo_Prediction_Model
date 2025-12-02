import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sqlalchemy import create_engine
import config

DB_URL = (
    f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
    f"@{config.DB_HOST_OUTSIDE}:5432/{config.DB_NAME}"
)
engine = create_engine(DB_URL)

# 7-day prediction horizon
HORIZON = 7 * 24 * 60
INPUT_WINDOW = 256

FEATURE_COLS = [
    "scaled_close",
    "scaled_sma_20",
    "scaled_sma_50",
    "scaled_rsi",
    "scaled_macd",
]
NUM_FEATURES = len(FEATURE_COLS)


class FastDataset(Dataset):
    """
    Efficient window slicing dataset — perfect for incremental training.
    """

    def __init__(self, df):
        df = df.reset_index(drop=True)

        self.X_all = torch.tensor(
            df[FEATURE_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        self.close = torch.tensor(
            df["close"].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        self.max_start = len(df) - INPUT_WINDOW - HORIZON
        if self.max_start <= 0:
            raise ValueError("Not enough rows for incremental training.")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        X = self.X_all[idx : idx + INPUT_WINDOW]

        price_now = self.close[idx + INPUT_WINDOW]
        price_future = self.close[idx + INPUT_WINDOW + HORIZON]
        y = torch.log(price_future / price_now)

        return X, y.unsqueeze(0)


def get_incremental_dataset(ticker, days=180):
    """
    Load ONLY the last <days> of processed data.
    """

    q = f"""
        SELECT *
        FROM processed_market_data
        WHERE ticker = '{ticker}'
        AND date >= NOW() - INTERVAL '{days} days'
        ORDER BY date ASC
    """

    df = pd.read_sql(q, engine)

    if df.empty:
        print(f"⚠ No processed rows for {ticker}")
        return None

    try:
        return FastDataset(df)
    except Exception as e:
        print(f"⚠ Dataset error for {ticker}: {e}")
        return None
