"""
V101 Quantum AI
Indicators Module
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ==========================================================
# Moving Averages
# ==========================================================

def sma(series, period):
    return series.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def wma(series, period):
    w = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, w) / w.sum(),
        raw=True
    )


def hma(series, period):

    half = int(period / 2)
    sqrt = int(np.sqrt(period))

    wma1 = wma(series, half)
    wma2 = wma(series, period)

    raw = 2 * wma1 - wma2

    return wma(raw, sqrt)


def vwma(close, volume, period):

    pv = close * volume

    return (
        pv.rolling(period).sum() /
        volume.rolling(period).sum()
    )


# ==========================================================
# EMA SET
# ==========================================================

def add_ema(df):

    for p in [5, 8, 13, 20, 34, 50, 89, 100, 144, 200]:

        df[f"EMA{p}"] = ema(df.Close, p)

    return df


# ==========================================================
# SMA SET
# ==========================================================

def add_sma(df):

    for p in [20, 50, 100, 200]:

        df[f"SMA{p}"] = sma(df.Close, p)

    return df


# ==========================================================
# HMA SET
# ==========================================================

def add_hma(df):

    for p in [20, 55]:

        df[f"HMA{p}"] = hma(df.Close, p)

    return df


# ==========================================================
# VWMA SET
# ==========================================================

def add_vwma(df):

    for p in [20, 50]:

        df[f"VWMA{p}"] = vwma(
            df.Close,
            df.Volume,
            p
        )

    return df
# ==========================================================
# RSI
# ==========================================================

def rsi(close, period=14):

    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ==========================================================
# MACD
# ==========================================================

def macd(close, fast=12, slow=26, signal=9):

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = ema_fast - ema_slow

    signal_line = ema(macd_line, signal)

    hist = macd_line - signal_line

    return macd_line, signal_line, hist


# ==========================================================
# ATR
# ==========================================================

def atr(df, period=14):

    high = df.High
    low = df.Low
    close = df.Close

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


# ==========================================================
# ADX
# ==========================================================

def adx(df, period=14):

    high = df.High
    low = df.Low
    close = df.Close

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_value = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_value
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_value
    )

    dx = (
        (plus_di - minus_di).abs() /
        (plus_di + minus_di)
    ) * 100

    adx_line = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_line, plus_di, minus_di


# ==========================================================
# BOLLINGER
# ==========================================================

def bollinger(close, period=20, std=2):

    mid = sma(close, period)

    sigma = close.rolling(period).std()

    upper = mid + sigma * std
    lower = mid - sigma * std

    return upper, mid, lower


# ==========================================================
# DONCHIAN CHANNEL
# ==========================================================

def donchian(df, period=20):

    upper = df.High.rolling(period).max()

    lower = df.Low.rolling(period).min()

    middle = (upper + lower) / 2

    return upper, middle, lower


# ==========================================================
# KELTNER CHANNEL
# ==========================================================

def keltner(df, period=20, multiplier=2):

    middle = ema(df.Close, period)

    atr_value = atr(df, period)

    upper = middle + atr_value * multiplier

    lower = middle - atr_value * multiplier

    return upper, middle, lower
# ==========================================================
# VWAP
# ==========================================================

def vwap(df):

    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    pv = tp * df["Volume"]

    return pv.cumsum() / df["Volume"].cumsum()


# ==========================================================
# OBV
# ==========================================================

def obv(df):

    direction = np.sign(df["Close"].diff()).fillna(0)

    return (direction * df["Volume"]).cumsum()


# ==========================================================
# MFI
# ==========================================================

def mfi(df, period=14):

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]

    direction = tp.diff()

    positive = mf.where(direction > 0, 0)
    negative = mf.where(direction < 0, 0)

    pos_sum = positive.rolling(period).sum()
    neg_sum = negative.abs().rolling(period).sum()

    ratio = pos_sum / neg_sum.replace(0, np.nan)

    return 100 - (100 / (1 + ratio))


# ==========================================================
# CCI
# ==========================================================

def cci(df, period=20):

    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    sma_tp = tp.rolling(period).mean()

    mad = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    return (tp - sma_tp) / (0.015 * mad)


# ==========================================================
# WILLIAMS %R
# ==========================================================

def williams_r(df, period=14):

    hh = df["High"].rolling(period).max()
    ll = df["Low"].rolling(period).min()

    return -100 * ((hh - df["Close"]) / (hh - ll))


# ==========================================================
# ROC
# ==========================================================

def roc(close, period=12):

    return ((close - close.shift(period))
            / close.shift(period)) * 100


# ==========================================================
# MOMENTUM
# ==========================================================

def momentum(close, period=10):

    return close - close.shift(period)


# ==========================================================
# STOCHASTIC
# ==========================================================

def stochastic(df, k_period=14, d_period=3):

    low = df["Low"].rolling(k_period).min()

    high = df["High"].rolling(k_period).max()

    k = 100 * ((df["Close"] - low) / (high - low))

    d = k.rolling(d_period).mean()

    return k, d


# ==========================================================
# STOCH RSI
# ==========================================================

def stochastic_rsi(close,
                   period=14,
                   smooth_k=3,
                   smooth_d=3):

    r = rsi(close, period)

    low = r.rolling(period).min()

    high = r.rolling(period).max()

    stoch = (r - low) / (high - low)

    k = stoch.rolling(smooth_k).mean() * 100

    d = k.rolling(smooth_d).mean()

    return k, d
# ==========================================================
# CALCULATE ALL
# ==========================================================

def calculate_indicators(df):

    df = df.copy()

    # Hareketli Ortalamalar
    df = add_ema(df)
    df = add_sma(df)
    df = add_hma(df)
    df = add_vwma(df)

    # RSI
    df["RSI"] = rsi(df.Close)

    # MACD
    (
        df["MACD"],
        df["MACD_SIGNAL"],
        df["MACD_HIST"]
    ) = macd(df.Close)

    # ATR
    df["ATR"] = atr(df)

    # ADX
    (
        df["ADX"],
        df["+DI"],
        df["-DI"]
    ) = adx(df)

    # Bollinger
    (
        df["BB_UPPER"],
        df["BB_MIDDLE"],
        df["BB_LOWER"]
    ) = bollinger(df.Close)

    # Donchian
    (
        df["DONCHIAN_UPPER"],
        df["DONCHIAN_MIDDLE"],
        df["DONCHIAN_LOWER"]
    ) = donchian(df)

    # Keltner
    (
        df["KC_UPPER"],
        df["KC_MIDDLE"],
        df["KC_LOWER"]
    ) = keltner(df)

    # VWAP
    df["VWAP"] = vwap(df)

    # OBV
    df["OBV"] = obv(df)

    # MFI
    df["MFI"] = mfi(df)

    # CCI
    df["CCI"] = cci(df)

    # Williams
    df["WILLIAMS_R"] = williams_r(df)

    # ROC
    df["ROC"] = roc(df.Close)

    # Momentum
    df["MOMENTUM"] = momentum(df.Close)

    # Stochastic
    (
        df["STOCH_K"],
        df["STOCH_D"]
    ) = stochastic(df)

    # Stochastic RSI
    (
        df["STOCH_RSI_K"],
        df["STOCH_RSI_D"]
    ) = stochastic_rsi(df.Close)

    # Temizlik
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df