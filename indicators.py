import pandas as pd
import numpy as np


def add_indicators(df):

    close = df["Close"]

    # EMA
    df["EMA20"] = close.ewm(span=20).mean()
    df["EMA50"] = close.ewm(span=50).mean()
    df["EMA100"] = close.ewm(span=100).mean()
    df["EMA200"] = close.ewm(span=200).mean()

    # SMA
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100/(1+rs))

    # MACD

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["HIST"] = df["MACD"] - df["SIGNAL"]

    # Bollinger

    std20 = close.rolling(20).std()

    df["BB_MID"] = df["SMA20"]
    df["BB_UPPER"] = df["SMA20"] + 2 * std20
    df["BB_LOWER"] = df["SMA20"] - 2 * std20

    # ATR

    high = df["High"]
    low = df["Low"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # Ortalama Hacim

    df["VOL20"] = df["Volume"].rolling(20).mean()

    return df