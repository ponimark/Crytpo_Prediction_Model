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
    Windows are NOT pre-expanded.
    Instead we store the full matrix once and slice it per __getitem__.

    This is MUCH smaller memory footprint and SAFE for Windows workers.
    """

    def __init__(self, df):
        df = df.reset_index(drop=True)

        # Store full data in contiguous pinned buffer for fast slicing
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
            raise ValueError("Not enough rows")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        # FAST slicing — torch does not copy data here
        X = self.X_all[idx : idx + INPUT_WINDOW]

        price_now = self.close[idx + INPUT_WINDOW]
        price_future = self.close[idx + INPUT_WINDOW + HORIZON]
        y = torch.log(price_future / price_now)

        return X, y.unsqueeze(0)


def get_dataset_for_ticker(ticker):
    q = f"""
        SELECT *
        FROM processed_market_data
        WHERE ticker = '{ticker}'
        ORDER BY date ASC
    """

    df = pd.read_sql(q, engine)

    if df.empty:
        print(f"⚠ No processed data for {ticker}")
        return None

    try:
        return FastDataset(df)
    except Exception as e:
        print(f"⚠ Dataset error for {ticker}: {e}")
        return None
