"""
data_loader.py
V101 Quantum AI
Veri indirme ve ön işleme modülü
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# HTTP Session
# -------------------------------------------------------

def create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=config.DOWNLOAD_RETRY,
        connect=config.DOWNLOAD_RETRY,
        read=config.DOWNLOAD_RETRY,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        }
    )

    return session


SESSION = create_session()


# -------------------------------------------------------
# Yardımcı Fonksiyonlar
# -------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    """
    Borsa İstanbul sembollerini normalize eder.
    """

    symbol = symbol.strip().upper()

    if "." not in symbol:
        symbol += ".IS"

    return symbol


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri doğrulama
    """

    required = ["Open", "High", "Low", "Close", "Volume"]

    if df.empty:
        raise ValueError("Boş veri.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df = df[required]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    if len(df) < 50:
        raise ValueError("Yetersiz veri.")

    return df


# -------------------------------------------------------
# Veri İndirme
# -------------------------------------------------------

@st.cache_data(ttl=config.CACHE_TTL)
def load_data(
    symbol: str,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:

    symbol = normalize_symbol(symbol)

    period = period or config.DEFAULT_PERIOD
    interval = interval or config.DEFAULT_INTERVAL

    logger.info("Downloading %s", symbol)

    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=config.YF_AUTO_ADJUST,
        progress=config.YF_PROGRESS,
        threads=config.YF_THREADS,
        session=SESSION,
    )

    df = validate_dataframe(df)

    return df


# -------------------------------------------------------
# Bilgi
# -------------------------------------------------------

def last_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def last_volume(df: pd.DataFrame) -> int:
    return int(df["Volume"].iloc[-1])


def last_date(df: pd.DataFrame) -> datetime:
    return df.index[-1].to_pydatetime()