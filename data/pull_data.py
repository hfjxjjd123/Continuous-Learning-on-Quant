import os
from typing import List

import pandas as pd

import numpy as np

from settings.default import PINNACLE_DATA_CUT, PINNACLE_DATA_FOLDER

def pull_quandl_sample_data(ticker: str) -> pd.DataFrame:
    return (
        pd.read_csv(os.path.join("data", "quandl", f"{ticker}.csv"), parse_dates=[0])
        .rename(columns={"Trade Date": "date", "Date": "date", "Settle": "close"})
        .set_index("date")
        .replace(0.0, np.nan)
    )

def pull_binance_sample_data_ft(symbol: str) -> pd.DataFrame:
    """
    Load 15-min Binance futures data for the given symbol.

    Expects a CSV at data/binance/ft/{symbol}.csv with columns:
    Time, Opn, Hgh, Low, Cls, Vol, NoT, TBV, Ret

    Returns a DataFrame indexed by datetime 'Time' with raw columns.
    """
    dtypes = {
        'Time': np.int32,
        'Opn':  np.float32,
        'Hgh':  np.float32,
        'Low':  np.float32,
        'Cls':  np.float32,
        'Vol':  np.float32,
        'NoT':  np.int32,
        'TBV':  np.float32,
        'Ret': np.float32
    }

    path = os.path.join("datasets", "online15m", f"{symbol}.csv")
    df = pd.read_csv(path, dtype = dtypes)
    
    df['Time'] = pd.to_datetime(df['Time'], unit='s', utc=True)
    df = df.set_index('Time')

    return df


def pull_pinnacle_data(ticker: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PINNACLE_DATA_FOLDER, f"{ticker}_{PINNACLE_DATA_CUT}.CSV"),
        names=["date", "open", "high", "low", "close", "volume", "open_int"],
        parse_dates=[0],
        index_col=0,
    )[["close"]].replace(0.0, np.nan)


def _fill_blanks(data: pd.DataFrame):
    return data[
        data["close"].first_valid_index() : data["close"].last_valid_index()
    ].fillna(
        method="ffill"
    )  # .interpolate()


def pull_pinnacle_data_multiple(
    tickers: List[str], fill_missing_dates=False
) -> pd.DataFrame:
    data = pd.concat(
        [pull_pinnacle_data(ticker).assign(ticker=ticker).copy() for ticker in tickers]
    )

    if not fill_missing_dates:
        return data.dropna().copy()

    dates = data.reset_index()[["date"]].drop_duplicates().sort_values("date")
    data = data.reset_index().set_index("ticker")

    return (
        pd.concat(
            [
                _fill_blanks(
                    dates.merge(data.loc[t], on="date", how="left").assign(ticker=t)
                )
                for t in tickers
            ]
        )
        .reset_index()
        .set_index("date")
        .drop(columns="index")
        .copy()
    )
