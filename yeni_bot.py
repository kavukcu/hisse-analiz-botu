# ==========================================
# KÜTÜPHANELER
# ==========================================
import os
import time
import logging
import sqlite3
import joblib
import optuna
import requests
import asyncio
import aiohttp
import pytz
from datetime import datetime, timedelta, timezone

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st

# --- STREAMLIT SAYFA AYARI (EN BAŞTA OLMALIDIR) ---
st.set_page_config(layout="wide", page_title="God Mode Terminal v102")

import matplotlib.pyplot as plt
import concurrent.futures

# Oturum ve HTTP Ayarları
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})
oturum = session  # İsim uyuşmazlığını önlemek için eşitlendi

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Yapay Zeka & Makine Öğrenmesi Kütüphaneleri
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, IsolationForest
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import shap
from tvDatafeed import TvDatafeed, Interval
import isyatirimhisse

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import gymnasium as gym
import gym_anytrading
from stable_baselines3 import A2C

from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ==========================================
# VERİTABANI İŞLEMLERİ
# ==========================================
DB_NAME = "godmode_ai.db"
LEGACY_DB_NAMES = ("hisse_hafiza.db", "ai_memory.db")


def db_connect(db_name=DB_NAME):
    """SQLite bağlantılarını tek noktadan ve güvenli ayarlarla açar."""
    conn = sqlite3.connect(db_name, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _kolonlari_tamamla(conn):
    """Eski tahminler tablosunu veri kaybetmeden birleşik şemaya yükseltir."""
    mevcut = {row[1] for row in conn.execute("PRAGMA table_info(tahminler)").fetchall()}
    kolonlar = {
        "id": "INTEGER",
        "fiyat": "REAL",
        "sinyal": "TEXT",
        "guven": "REAL",
        "beklenen_getiri": "REAL",
        "model": "TEXT",
        "sonuc5": "REAL",
        "sonuc10": "REAL",
        "sonuc20": "REAL",
    }
    for kolon, tip in kolonlar.items():
        if kolon not in mevcut:
            conn.execute(f"ALTER TABLE tahminler ADD COLUMN {kolon} {tip}")


def _eski_verileri_tasi(conn):
    """Eski DB dosyalarındaki kayıtları birleşik veritabanına bir kez taşır."""
    for eski_db in LEGACY_DB_NAMES:
        if not os.path.exists(eski_db) or os.path.abspath(eski_db) == os.path.abspath(DB_NAME):
            continue
        try:
            eski = sqlite3.connect(eski_db, timeout=5)
            tablolar = {r[0] for r in eski.execute("SELECT name FROM sqlite_master WHERE type='table'")}

            if "tahminler" in tablolar:
                for row in eski.execute(
                    "SELECT tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum FROM tahminler"
                ).fetchall():
                    conn.execute(
                        """INSERT INTO tahminler
                           (tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum)
                           SELECT ?, ?, ?, ?, ?
                           WHERE NOT EXISTS (
                               SELECT 1 FROM tahminler
                               WHERE tarih=? AND sembol=? AND hedef_fiyat=?
                           )""",
                        (*row, row[0], row[1], row[2]),
                    )

            if "ai_predictions" in tablolar:
                for row in eski.execute(
                    """SELECT tarih, sembol, fiyat, hedef, sinyal, guven,
                              beklenen_getiri, model, sonuc5, sonuc10, sonuc20
                       FROM ai_predictions"""
                ).fetchall():
                    conn.execute(
                        """INSERT INTO tahminler
                           (tarih, sembol, fiyat, hedef_fiyat, sinyal, guven,
                            beklenen_getiri, model, sonuc5, sonuc10, sonuc20, durum)
                           SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'BEKLİYOR'
                           WHERE NOT EXISTS (
                               SELECT 1 FROM tahminler
                               WHERE tarih=? AND sembol=? AND hedef_fiyat=? AND model IS NOT NULL
                           )""",
                        (*row, row[0], row[1], row[3]),
                    )
            eski.close()
        except Exception as exc:
            logging.warning(f"Eski veritabanı taşınamadı [{eski_db}]: {exc}")


def veritabani_baslat():
    """Tek tahmin tablosunu oluşturur ve eski verileri güvenle birleştirir."""
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tahminler (
                id INTEGER,
                tarih TEXT NOT NULL,
                sembol TEXT NOT NULL,
                fiyat REAL,
                hedef_fiyat REAL,
                gerceklesme_fiyati REAL,
                durum TEXT DEFAULT 'BEKLİYOR',
                sinyal TEXT,
                guven REAL,
                beklenen_getiri REAL,
                model TEXT,
                sonuc5 REAL,
                sonuc10 REAL,
                sonuc20 REAL
            )
        """)
        _kolonlari_tamamla(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tahminler_tarih_sembol ON tahminler(tarih, sembol)"
        )
        _eski_verileri_tasi(conn)


def tahmin_kaydet(sembol, hedef_fiyat):
    """Aynı gün ve sembol için yinelenmeyen temel tahmin kaydı oluşturur."""
    bugun = datetime.now().strftime("%Y-%m-%d")
    with db_connect() as conn:
        mevcut = conn.execute(
            "SELECT 1 FROM tahminler WHERE tarih=? AND sembol=? AND model IS NULL LIMIT 1",
            (bugun, sembol),
        ).fetchone()
        if not mevcut:
            conn.execute(
                """INSERT INTO tahminler
                   (tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum)
                   VALUES (?, ?, ?, NULL, 'BEKLİYOR')""",
                (bugun, sembol, hedef_fiyat),
            )


def tahminleri_degerlendir():
    """En az 5 günlük bekleyen tahminleri güncel fiyatla değerlendirir."""
    with db_connect() as conn:
        bekleyenler = conn.execute(
            """SELECT rowid, tarih, sembol, hedef_fiyat
               FROM tahminler
               WHERE durum='BEKLİYOR' AND hedef_fiyat IS NOT NULL"""
        ).fetchall()

        for rowid, tarih_str, sembol, hedef_fiyat in bekleyenler:
            try:
                kayit_tarihi = datetime.strptime(tarih_str[:10], "%Y-%m-%d")
            except (TypeError, ValueError):
                continue

            if (datetime.now() - kayit_tarihi).days < 5:
                continue

            try:
                df = veri_yukle(
                    sembol,
                    (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d"),
                )
                if df is None or df.empty:
                    continue
                gercek_fiyat = float(df["Close"].iloc[-1])
                sapma_orani = abs(gercek_fiyat - hedef_fiyat) / max(abs(gercek_fiyat), 1e-9)
                durum = "BAŞARILI ✅" if sapma_orani <= 0.05 else "BAŞARISIZ ❌"
                conn.execute(
                    """UPDATE tahminler
                       SET gerceklesme_fiyati=?, durum=?
                       WHERE rowid=?""",
                    (gercek_fiyat, durum, rowid),
                )
            except Exception as exc:
                logging.error(f"Tahmin değerlendirme hatası [{sembol}]: {exc}")


# Uygulama açıldığında birleşik veritabanını hazırla.
veritabani_baslat()

GECERLI_BIST_SEMBOLLERI = ["THYAO", "ASELS", "BIMAS", "TUPRS"]

def gecerli_bist_sembolu_mu(hisse_kodu):
    ana_sembol = hisse_kodu.replace('.IS', '').replace('BIST:', '').strip().upper()
    return ana_sembol in GECERLI_BIST_SEMBOLLERI

@st.cache_resource
def model_yukle(sembol):
    model_dosyasi = os.path.join("ai_modeller", f"{sembol}_ai_model.pkl")
    if os.path.exists(model_dosyasi):
        return joblib.load(model_dosyasi)
    return None

def sembol_formatla(hisse_kodu):
    ana_sembol = hisse_kodu.replace('.IS', '').replace('BIST:', '').strip().upper()
    return {
        'yfinance': f"{ana_sembol}.IS",
        'isyatirim': ana_sembol,
        'tradingview': f"BIST:{ana_sembol}",
        'saf_sembol': ana_sembol
    }

# ==========================================
# VERİ YÜKLEME VE ANALİZ FONKSİYONLARI
# ==========================================
@st.cache_resource(show_spinner=False)
def get_tv_datafeed():
    try:
        return TvDatafeed()
    except Exception as e:
        logging.error(f"TradingView Bağlantı Hatası: {e}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def veri_yukle(ticker, start, end, interval="1d", kaynak="Yahoo Finance (yfinance)"):
    ticker = ticker.replace("$", "").strip() 
    is_bist = ".IS" in ticker
    is_crypto = "-" in ticker
    
    tum_kaynaklar = ["Yahoo Finance (yfinance)", "TradingView (tvdatafeed)"]
    if is_bist:
        tum_kaynaklar.append("İş Yatırım (Sadece BIST)")
    
    if kaynak in tum_kaynaklar:
        tum_kaynaklar.remove(kaynak)
        tum_kaynaklar.insert(0, kaynak)

    for aktif_kaynak in tum_kaynaklar:
        if aktif_kaynak == "Yahoo Finance (yfinance)":
            for _ in range(2):
                try:
                    yf_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if end is not None else None
                    df = yf.download(
                        ticker, 
                        start=start, 
                        end=yf_end, 
                        interval=interval, 
                        progress=False, 
                        auto_adjust=True, 
                        threads=True
                    )
                    
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(1)
                        gerekli = ["Open", "High", "Low", "Close", "Volume"]
                        if all(c in df.columns for c in gerekli):
                            df = df.dropna(subset=['Close'])
                            df.index = df.index.tz_localize(None)
                            df.index = pd.to_datetime(df.index).normalize()
                            return df
                except Exception as e:
                    logging.debug(f"Yahoo deneme hatası ({ticker}): {e}")
                time.sleep(0.5)

        elif aktif_kaynak == "TradingView (tvdatafeed)":
            try:
                tv = get_tv_datafeed()
                if tv:
                    if is_bist:
                        exchange = 'BIST'
                        tv_symbol = ticker.replace(".IS", "")
                    elif is_crypto:
                        exchange = 'CRYPTO'
                        tv_symbol = ticker.replace("-", "")
                    else:
                        exchange = 'NASDAQ'
                        tv_symbol = ticker
                        
                    df = tv.get_hist(symbol=tv_symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)
                    if df is not None and not df.empty:
                        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                        df.index = df.index.tz_localize(None)
                        if end:
                            df = df[df.index.date <= pd.to_datetime(end).date()]
                        if start:
                            df = df[df.index.date >= pd.to_datetime(start).date()]
                        if not df.empty:
                            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"TradingView deneme hatası ({ticker}): {e}")

        elif aktif_kaynak == "İş Yatırım (Sadece BIST)" and is_bist:
            try:
                sembol = ticker.replace(".IS", "")
                start_str = pd.to_datetime(start).strftime('%d-%m-%Y') if start else None
                end_str = pd.to_datetime(end).strftime('%d-%m-%Y') if end else pd.Timestamp.today().strftime('%d-%m-%Y')
                
                df = isyatirimhisse.fetch_data(symbol=sembol, start_date=start_str, end_date=end_str)
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        'TARIH': 'Date', 'ACILIS_FIYATI': 'Open', 
                        'EN_YUKSEK_FIYAT': 'High', 'EN_DUSUK_FIYAT': 'Low', 
                        'KAPANIS_FIYATI': 'Close', 'ISLEM_ADEDI': 'Volume'
                    })
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    if not df.empty:
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"İş Yatırım deneme hatası ({ticker}): {e}")

    return pd.DataFrame()


def veri_4saatlik_getir(ticker, start, end, kaynak="Yahoo Finance (yfinance)"):
    kaynaklar = ["TradingView (tvdatafeed)", "Yahoo Finance (yfinance)"]
    if kaynak in kaynaklar:
        kaynaklar.remove(kaynak)
        kaynaklar.insert(0, kaynak)

    for aktif_kaynak in kaynaklar:
        if aktif_kaynak == "TradingView (tvdatafeed)":
            try:
                tv = get_tv_datafeed()
                if ".IS" in ticker:
                    exchange = 'BIST'
                    tv_symbol = ticker.replace(".IS", "")
                elif "-" in ticker:
                    exchange = 'CRYPTO'
                    tv_symbol = ticker.replace("-", "")
                else:
                    exchange = 'NASDAQ'
                    tv_symbol = ticker
                    
                df_4h = tv.get_hist(symbol=tv_symbol, exchange=exchange, interval=Interval.in_4_hour, n_bars=1000)
                if df_4h is not None and not df_4h.empty:
                    df_4h = df_4h.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df_4h.index = df_4h.index.tz_localize(None)
                    if end:
                        df_4h = df_4h[df_4h.index.date <= pd.to_datetime(end).date()]
                    if not df_4h.empty:
                        return df_4h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"TV 4H deneme hatası ({ticker}): {e}")

        elif aktif_kaynak == "Yahoo Finance (yfinance)":
            try:
                start_dt = pd.to_datetime(start)
                if (datetime.now() - start_dt).days > 729:
                    start_dt = datetime.now() - timedelta(days=729)
                    s_str = start_dt.strftime('%Y-%m-%d')
                else:
                    s_str = start

                for _ in range(2):
                    df_1h = yf.download(ticker, start=s_str, interval="1h", progress=False)
                    if not df_1h.empty:
                        if isinstance(df_1h.columns, pd.MultiIndex):
                            df_1h.columns = df_1h.columns.droplevel(1)
                        df_1h.index = df_1h.index.tz_localize(None)
                        if end:
                            df_1h = df_1h[df_1h.index.date <= pd.to_datetime(end).date()]
                        
                        df_4h = df_1h.resample('4h').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                        
                        if not df_4h.empty:
                            return df_4h
                    time.sleep(0.5)
            except Exception as e:
                logging.debug(f"Yahoo 4H resample deneme hatası ({ticker}): {e}")

    return pd.DataFrame()


def tilson_t3(close, period=5, vfactor=0.7):
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    ema4 = ema3.ewm(span=period, adjust=False).mean()
    ema5 = ema4.ewm(span=period, adjust=False).mean()
    ema6 = ema5.ewm(span=period, adjust=False).mean()
    
    c1 = -vfactor**3
    c2 = 3*vfactor**2 + 3*vfactor**3
    c3 = -6*vfactor**2 - 3*vfactor - 3*vfactor**3
    c4 = 1 + 3*vfactor + vfactor**3 + 3*vfactor**2
    
    return c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3


@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=session).info
    except: 
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def anomali_tespit_et(df):
    features = df[['Close', 'Volume']].pct_change().dropna()
    if len(features) < 20:
        return "Veri yetersiz"
        
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    features['Anomaly'] = iso_forest.fit_predict(features)
    son_durum = features['Anomaly'].iloc[-1]
    
    if son_durum == -1:
        return "⚠️ RİSKLİ: Fiyat/Hacim hareketlerinde anomali tespit edildi!"
    return "✅ Piyasaya uygun, normal hareket."


def sihirli_formul_skorla(sembol, df=None):
    try:
        info = sirket_bilgisi_getir(sembol)
        if not info:
            return {'Puan': 0}
            
        skor = 0
        fk = info.get('trailingPE', 999) or 999
        if 0 < fk <= 10: skor += 25
        elif 10 < fk <= 15: skor += 15
        elif 15 < fk <= 20: skor += 5
        
        pddd = info.get('priceToBook', 999) or 999
        if 0 < pddd <= 1.5: skor += 25
        elif 1.5 < pddd <= 3.0: skor += 15
        elif 3.0 < pddd <= 5.0: skor += 5
        
        roe = info.get('returnOnEquity', -1) or -1
        if roe > 0.20: skor += 25
        elif roe > 0.10: skor += 15
        elif roe > 0.05: skor += 5
        
        cari_oran = info.get('currentRatio', 0) or 0
        if cari_oran >= 1.5: skor += 25
        elif cari_oran >= 1.0: skor += 15
        
        if df is not None and not df.empty:
            son_mum = df.iloc[-1]
            if son_mum.get('Super_Sinyal', False):
                skor += 20
            elif son_mum.get('Pozitif_Uyusmazlik', False):
                skor += 10
                
            if son_mum.get('Wyckoff_Spring', False):
                skor += 15
            if son_mum.get('Hacim_Patlamasi', False):
                skor += 10

        return {'Puan': skor}
    except Exception as e:
        logging.warning(f"[{sembol}] Temel veri puanlama hatası: {str(e)}")
        return {'Puan': 0}


def stokastik_hesapla(df, k_periyot=14, d_periyot=3):
    try:
        low_min = df['Low'].rolling(window=k_periyot).min()
        high_max = df['High'].rolling(window=k_periyot).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-9))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_periyot).mean()
        return df
    except Exception:
        df['Stoch_K'] = 50.0
        df['Stoch_D'] = 50.0
        return df


def smc_hesapla(df):
    df_smc = df.copy()
    df_smc['FVG_Bullish'] = (df_smc['Low'] > df_smc['High'].shift(2)) & (df_smc['Close'].shift(1) > df_smc['Open'].shift(1))
    df_smc['FVG_Bearish'] = (df_smc['High'] < df_smc['Low'].shift(2)) & (df_smc['Close'].shift(1) < df_smc['Open'].shift(1))
    return df_smc


@st.cache_data(ttl=3600, show_spinner=False)
def ileri_teknik_gostergeler(df):
    df_ta = df.copy()
    high_9 = df_ta['High'].rolling(window=9).max()
    low_9 = df_ta['Low'].rolling(window=9).min()
    df_ta['Tenkan_Sen'] = (high_9 + low_9) / 2
    high_26 = df_ta['High'].rolling(window=26).max()
    low_26 = df_ta['Low'].rolling(window=26).min()
    df_ta['Kijun_Sen'] = (high_26 + low_26) / 2
    df_ta['Senkou_Span_A'] = ((df_ta['Tenkan_Sen'] + df_ta['Kijun_Sen']) / 2).shift(26)
    high_52 = df_ta['High'].rolling(window=52).max()
    low_52 = df_ta['Low'].rolling(window=52).min()
    df_ta['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    df_ta['Chikou_Span'] = df_ta['Close'].shift(-26)
    
    prev_high = df_ta['High'].shift(1)
    prev_low = df_ta['Low'].shift(1)
    prev_close = df_ta['Close'].shift(1)
    range_hl = prev_high - prev_low
    
    df_ta['Cam_H4'] = prev_close + (range_hl * 1.1 / 2)
    df_ta['Cam_H3'] = prev_close + (range_hl * 1.1 / 4)
    df_ta['Cam_L3'] = prev_close - (range_hl * 1.1 / 4)
    df_ta['Cam_L4'] = prev_close - (range_hl * 1.1 / 2)

    df_ta['Ichimoku_Trend'] = np.where(df_ta['Close'] > df_ta['Senkou_Span_A'], 
        np.where(df_ta['Close'] > df_ta['Senkou_Span_B'], "GÜÇLÜ BOĞA", "NÖTR"), 
        np.where(df_ta['Close'] < df_ta['Senkou_Span_B'], "GÜÇLÜ AYI", "NÖTR"))

    df_ta['SMA_5'] = df_ta['Close'].rolling(window=5).mean()
    df_ta['SMA_8'] = df_ta['Close'].rolling(window=8).mean()
    df_ta['SMA_13'] = df_ta['Close'].rolling(window=13).mean()

    df_ta['EMA_5'] = df_ta['Close'].ewm(span=5, adjust=False).mean()
    df_ta['EMA_8'] = df_ta['Close'].ewm(span=8, adjust=False).mean()
    df_ta['EMA_13'] = df_ta['Close'].ewm(span=13, adjust=False).mean()
    df_ta['EMA_52'] = df_ta['Close'].ewm(span=52, adjust=False).mean()
    df_ta['EMA_89'] = df_ta['Close'].ewm(span=89, adjust=False).mean()
    df_ta['EMA_144'] = df_ta['Close'].ewm(span=144, adjust=False).mean()
    
    df_ta['Fibo_MA_Trend'] = np.where(
        (df_ta['EMA_5'] > df_ta['EMA_8']) & (df_ta['EMA_8'] > df_ta['EMA_13']), "🚀 GÜÇLÜ YÜKSELİŞ",
        np.where((df_ta['EMA_5'] < df_ta['EMA_8']) & (df_ta['EMA_8'] < df_ta['EMA_13']), "🔻 GÜÇLÜ DÜŞÜŞ", "⚖️ YATAY NÖTR")
    )
    return df_ta


def yapay_zeka_icin_formasyon_bul(df):
    df_f = df.copy()
    govde = abs(df_f['Close'] - df_f['Open'])
    mum_boyu = df_f['High'] - df_f['Low']
    ust_golge = df_f['High'] - df_f[['Close', 'Open']].max(axis=1)
    alt_golge = df_f[['Close', 'Open']].min(axis=1) - df_f['Low']
    
    df_f['Doji'] = np.where(govde <= (mum_boyu * 0.1), 1, 0)
    
    bullish_engulfing = (df_f['Close'].shift(1) < df_f['Open'].shift(1)) & (df_f['Open'] < df_f['Close'].shift(1)) & (df_f['Close'] > df_f['Open'].shift(1))
    bearish_engulfing = (df_f['Close'].shift(1) > df_f['Open'].shift(1)) & (df_f['Open'] > df_f['Close'].shift(1)) & (df_f['Close'] < df_f['Open'].shift(1))
    
    hammer = (alt_golge > (2 * govde)) & (ust_golge < (govde * 0.2)) & (df_f['Close'] > df_f['Close'].rolling(10).mean())
    shooting_star = (ust_golge > (2 * govde)) & (alt_golge < (govde * 0.2)) & (df_f['Close'] < df_f['Close'].rolling(10).mean())
    
    df_f['P_Engulfing'] = np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))
    df_f['P_Pinbar'] = np.where(hammer, 1, np.where(shooting_star, -1, 0))
    df_f['AI_Formasyon_Skoru'] = df_f['P_Engulfing'] + df_f['P_Pinbar']
    
    return df_f


def makro_formasyonlari_bul(df, window=20):
    df_f = df.copy()
    df_f['Rolling_Max'] = df_f['High'].rolling(window=window).max()
    df_f['Rolling_Min'] = df_f['Low'].rolling(window=window).min()
    
    df_f['Ikili_Tepe'] = np.where((df_f['High'] >= df_f['Rolling_Max'] * 0.99) & (df_f['Close'] < df_f['Open']), -1, 0)
    df_f['Ikili_Dip'] = np.where((df_f['Low'] <= df_f['Rolling_Min'] * 1.01) & (df_f['Close'] > df_f['Open']), 1, 0)
    
    df_f['High_Slope'] = df_f['High'].diff(3).rolling(10).mean() 
    df_f['Low_Slope'] = df_f['Low'].diff(3).rolling(10).mean()   
    
    df_f['Simetrik_Ucgen'] = np.where((df_f['High_Slope'] < 0) & (df_f['Low_Slope'] > 0), 1, 0)
    df_f['Yukselen_Ucgen'] = np.where((abs(df_f['High_Slope']) < (df_f['Close'] * 0.002)) & (df_f['Low_Slope'] > 0), 1, 0)
    df_f['Alcalan_Ucgen'] = np.where((df_f['High_Slope'] < 0) & (abs(df_f['Low_Slope']) < (df_f['Close'] * 0.002)), -1, 0)
    
    df_f['Sert_Yukselis'] = df_f['Close'].pct_change(5) > 0.06
    df_f['Dar_Bant_Konsolidasyon'] = (df_f['High'].rolling(4).max() - df_f['Low'].rolling(4).min()) < (df_f['Close'] * 0.02)
    df_f['Bayrak_Formasyonu'] = np.where(df_f['Sert_Yukselis'].shift(4) & df_f['Dar_Bant_Konsolidasyon'], 1, 0)
    
    df_f['Tepe_Uzakligi_Z'] = (df_f['Rolling_Max'] - df_f['Close']) / df_f['Close'].rolling(window).std()
    df_f['Dip_Uzakligi_Z'] = (df_f['Close'] - df_f['Rolling_Min']) / df_f['Close'].rolling(window).std()
    
    df_f['Makro_Guc_Skoru'] = df_f['Ikili_Dip'] + df_f['Yukselen_Ucgen'] + df_f['Simetrik_Ucgen'] + df_f['Bayrak_Formasyonu'] - abs(df_f['Ikili_Tepe']) - abs(df_f['Alcalan_Ucgen'])
    df_f.fillna(0, inplace=True)
    return df_f


def trend_ve_harmonik_bul(df):
    df_f = df.copy()
    df_f['SMA_50'] = df_f['Close'].rolling(window=50).mean()
    df_f['SMA_200'] = df_f['Close'].rolling(window=200).mean()
    
    golden_cross = (df_f['SMA_50'] > df_f['SMA_200']) & (df_f['SMA_50'].shift(1) <= df_f['SMA_200'].shift(1))
    death_cross = (df_f['SMA_50'] < df_f['SMA_200']) & (df_f['SMA_50'].shift(1) >= df_f['SMA_200'].shift(1))
    
    df_f['Cross_Sinyali'] = np.where(golden_cross, 1, np.where(death_cross, -1, 0))
    df_f['SMA_50_200_Farki'] = (df_f['SMA_50'] - df_f['SMA_200']) / df_f['SMA_200']
    
    swing_ab = df_f['Close'].shift(10) - df_f['Close'].shift(15) 
    swing_bc = df_f['Close'].shift(5) - df_f['Close'].shift(10)  
    swing_cd = df_f['Close'] - df_f['Close'].shift(5)            
    
    ab_boyu = abs(swing_ab)
    cd_boyu = abs(swing_cd)
    
    bullish_abcd = (swing_ab < 0) & (swing_bc > 0) & (swing_cd < 0) & (cd_boyu > ab_boyu * 0.7) & (cd_boyu < ab_boyu * 1.3)
    bearish_abcd = (swing_ab > 0) & (swing_bc < 0) & (swing_cd > 0) & (cd_boyu > ab_boyu * 0.7) & (cd_boyu < ab_boyu * 1.3)
    
    df_f['ABCD_Formasyonu'] = np.where(bullish_abcd, 1, np.where(bearish_abcd, -1, 0))
    df_f.fillna(0, inplace=True)
    return df_f


def formasyon_tespit_et_ve_hedefle(df):
    if df is None or len(df) < 20:
        return "Yok", "% 0.00"

    df_f = df.copy()
    df_f = yapay_zeka_icin_formasyon_bul(df_f)
    df_f = makro_formasyonlari_bul(df_f, window=20)
    df_f = trend_ve_harmonik_bul(df_f)
    
    son = df_f.iloc[-1]
    fiyat = son['Close'] if son['Close'] > 0 else 1.0
    
    formasyon_adi = "Yok"
    hedef_yuzde = 0.0

    if son.get('Ikili_Dip', 0) == 1:
        formasyon_adi = "📐 İkili Dip"
        derinlik = (son['Rolling_Max'] - son['Rolling_Min']) / son['Rolling_Min'] * 100
        hedef_yuzde = round(derinlik, 2)
    elif son.get('Ikili_Tepe', 0) == -1:
        formasyon_adi = "📐 İkili Tepe"
        derinlik = (son['Rolling_Max'] - son['Rolling_Min']) / son['Rolling_Max'] * 100
        hedef_yuzde = round(-derinlik, 2)
    elif son.get('Bayrak_Formasyonu', 0) == 1:
        formasyon_adi = "🚩 Bayrak Formasyonu"
        direk_boyu = df_f['Close'].pct_change(5).iloc[-4] * 100 if len(df_f) > 5 else 7.5
        hedef_yuzde = round(abs(direk_boyu), 2)
    elif son.get('Yukselen_Ucgen', 0) == 1:
        formasyon_adi = "🔺 Yükselen Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(yukseklik, 2)
    elif son.get('Alcalan_Ucgen', 0) == -1:
        formasyon_adi = "🔻 Alçalan Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(-yukseklik, 2)
    elif son.get('Simetrik_Ucgen', 0) == 1:
        formasyon_adi = "📐 Simetrik Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(yukseklik / 2, 2)
    elif son.get('ABCD_Formasyonu', 0) == 1:
        formasyon_adi = "⚡ Bullish ABCD"
        hedef_yuzde = 6.5
    elif son.get('ABCD_Formasyonu', 0) == -1:
        formasyon_adi = "⚡ Bearish ABCD"
        hedef_yuzde = -6.5
    elif son.get('Cross_Sinyali', 0) == 1:
        formasyon_adi = "🌟 Golden Cross"
        hedef_yuzde = 12.0
    elif son.get('Cross_Sinyali', 0) == -1:
        formasyon_adi = "💀 Death Cross"
        hedef_yuzde = -12.0
    elif son.get('P_Engulfing', 0) == 1:
        formasyon_adi = "🕯️ Yutan Boğa"
        hedef_yuzde = 4.0
    elif son.get('P_Engulfing', 0) == -1:
        formasyon_adi = "🕯️ Yutan Ayı"
        hedef_yuzde = -4.0
    elif son.get('P_Pinbar', 0) == 1:
        formasyon_adi = "🔨 Çekiç (Pinbar)"
        hedef_yuzde = 3.5
    elif son.get('P_Pinbar', 0) == -1:
        formasyon_adi = "🏹 Kayan Yıldız"
        hedef_yuzde = -3.5
    elif son.get('Doji', 0) == 1:
        formasyon_adi = "⚖️ Doji (Kararsızlık)"
        hedef_yuzde = 0.0

    hedef_str = f"% {hedef_yuzde:+.2f}" if hedef_yuzde != 0 else "% 0.00"
    return formasyon_adi, hedef_str


def dipten_donus_analizi(df):
    if df is None or len(df) < 20:
        df_copy = df.copy() if df is not None else pd.DataFrame()
        df_copy['Hacim_Patlamasi'] = False
        df_copy['Wyckoff_Spring'] = False
        df_copy['RSI_Uyumsuzluk'] = False
        df_copy['MACD_Uyumsuzluk'] = False
        df_copy['Stokastik_Uyumsuzluk'] = False
        df_copy['Pozitif_Uyusmazlik'] = False
        df_copy['Super_Sinyal'] = False
        return df_copy

    df_dip = df.copy()

    df_dip['Vol_SMA_20'] = df_dip['Volume'].rolling(20).mean()
    df_dip['Hacim_Patlamasi'] = df_dip['Volume'] > (df_dip['Vol_SMA_20'] * 2)

    df_dip['SMA_20_Dip'] = df_dip['Close'].rolling(20).mean()
    df_dip['STD_20_Dip'] = df_dip['Close'].rolling(20).std()
    df_dip['Lower_Band'] = df_dip['SMA_20_Dip'] - (df_dip['STD_20_Dip'] * 2)

    df_dip['Wyckoff_Spring'] = (df_dip['Low'] < df_dip['Lower_Band']) & \
                               (df_dip['Close'] > df_dip['Lower_Band']) & \
                               (df_dip['Close'] > df_dip['Open'])

    if 'RSI' not in df_dip.columns:
        delta = df_dip['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, 1e-9))
        df_dip['RSI'] = 100 - (100 / (1 + rs))

    if 'MACD_Hist' not in df_dip.columns:
        ema12 = df_dip['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_dip['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df_dip['MACD_Hist'] = macd - macd_signal

    if 'Stoch_K' not in df_dip.columns:
        low_min = df_dip['Low'].rolling(window=14).min()
        high_max = df_dip['High'].rolling(window=14).max()
        df_dip['Stoch_K'] = 100 * ((df_dip['Close'] - low_min) / (high_max - low_min + 1e-9))

    df_dip['RSI_Uyumsuzluk'] = False
    df_dip['MACD_Uyumsuzluk'] = False
    df_dip['Stokastik_Uyumsuzluk'] = False
    df_dip['Pozitif_Uyusmazlik'] = False
    df_dip['Super_Sinyal'] = False

    dip_mask = (df_dip['Low'].shift(1) > df_dip['Low']) & (df_dip['Low'].shift(-1) > df_dip['Low'])
    dipler = df_dip[dip_mask]['Low'].tail(2)

    if len(dipler) == 2:
        eski_idx, yeni_idx = dipler.index[0], dipler.index[1]
        eski_fiyat = df_dip['Low'].loc[eski_idx]
        yeni_fiyat = df_dip['Low'].loc[yeni_idx]

        if yeni_fiyat < eski_fiyat:
            rsi_uyum = df_dip['RSI'].loc[yeni_idx] > df_dip['RSI'].loc[eski_idx]
            macd_uyum = df_dip['MACD_Hist'].loc[yeni_idx] > df_dip['MACD_Hist'].loc[eski_idx]
            stoch_uyum = df_dip['Stoch_K'].loc[yeni_idx] > df_dip['Stoch_K'].loc[eski_idx]

            super_sinyal = rsi_uyum and macd_uyum and stoch_uyum
            herhangi_uyum = rsi_uyum or macd_uyum or stoch_uyum

            df_dip.loc[yeni_idx:, 'RSI_Uyumsuzluk'] = rsi_uyum
            df_dip.loc[yeni_idx:, 'MACD_Uyumsuzluk'] = macd_uyum
            df_dip.loc[yeni_idx:, 'Stokastik_Uyumsuzluk'] = stoch_uyum
            df_dip.loc[yeni_idx:, 'Pozitif_Uyusmazlik'] = herhangi_uyum
            df_dip.loc[yeni_idx:, 'Super_Sinyal'] = super_sinyal

    return df_dip


def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=session).news
        if not news_data: return []
        olumlu = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma"]
        olumsuz = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza"]
        sonuclar = []
        for n in news_data[:5]:
            metin = (str(n.get('title', '')) + " " + str(n.get('summary', ''))).lower()
            ol_skor = sum(1 for k in olumlu if k in metin)
            sz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if ol_skor > sz_skor else ("🔴 OLUMSUZ" if sz_skor > ol_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: return []


@st.cache_data(ttl=86400)
def en_iyi_xgb_parametrelerini_bul(sembol, X_matrisi, y_vektoru):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        }
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X_matrisi):
            X_train, X_test = X_matrisi[train_index], X_matrisi[test_index]
            y_train, y_test = y_vektoru[train_index], y_vektoru[test_index]
        model = XGBRegressor(**param, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    return study.best_params


def ai_guven_skoru_hesapla(ensemble, son_veri, teknik_skor, formasyon_skoru, risk_odul, atr_orani):
    try:
        tahminler = []
        if hasattr(ensemble, "estimators_"):
            for est in ensemble.estimators_:
                try:
                    tahmin = float(est.predict(son_veri)[0])
                    tahminler.append(tahmin)
                except Exception:
                    pass

        if len(tahminler) >= 2:
            std = np.std(tahminler)
            ort = max(abs(np.mean(tahminler)), 0.001)
            oy_birligi = max(0, 100 - (std / ort) * 100)
        else:
            oy_birligi = 50

        teknik = np.clip(teknik_skor, 0, 100)
        formasyon = np.clip(formasyon_skoru, 0, 100)
        rr = np.clip(risk_odul * 25, 0, 100)
        volatilite = np.clip(100 - atr_orani * 100, 0, 100)

        guven = (
            oy_birligi * 0.35 +
            teknik * 0.25 +
            formasyon * 0.20 +
            rr * 0.10 +
            volatilite * 0.10
        )
        return round(np.clip(guven, 0, 99), 1)
    except Exception:
        return 50.0


def ai_feature_importance(model):
    try:
        tum_onemler = []
        if hasattr(model, "named_estimators_"):
            for isim, est in model.named_estimators_.items():
                if hasattr(est, "steps"):
                    est = est.steps[-1][1]
                if hasattr(est, "feature_importances_"):
                    tum_onemler.append(est.feature_importances_)
                elif hasattr(est, "coef_"):
                    tum_onemler.append(np.abs(est.coef_))
            if len(tum_onemler):
                importance = np.mean(tum_onemler, axis=0)
            else:
                return None
        elif hasattr(model, "estimators_"):
            for est in model.estimators_:
                if hasattr(est, "steps"):
                    est = est.steps[-1][1]
                imp = None
                if hasattr(est, "feature_importances_"):
                    imp = np.asarray(est.feature_importances_, dtype=float)
                elif hasattr(est, "coef_"):
                    imp = np.abs(np.asarray(est.coef_, dtype=float)).flatten()
                if imp is not None:
                    imp = np.nan_to_num(imp)
                    toplam = imp.sum()
                    if toplam > 0:
                        tum_onemler.append(imp / toplam)
        else:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_"):
                imp = np.abs(np.asarray(model.coef_, dtype=float)).flatten()
            else:
                return None
            imp = np.nan_to_num(imp)
            return imp / imp.sum() if imp.sum() > 0 else None

        if not tum_onemler:
            return None

        min_len = min(len(x) for x in tum_onemler)
        tum_onemler = [x[:min_len] for x in tum_onemler]
        importance = np.mean(tum_onemler, axis=0)
        return importance / importance.sum() if importance.sum() > 0 else None
    except Exception as e:
        logging.error(f"Feature Importance Hatası: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def ensemble_prediction(df, sembol="Genel"):
    try:
        t_df = df.copy()
        t_df = yapay_zeka_icin_formasyon_bul(t_df)
        t_df = makro_formasyonlari_bul(t_df, window=20)
        t_df = trend_ve_harmonik_bul(t_df)
        
        if 'Stoch_K' not in t_df.columns:
            low_min = t_df['Low'].rolling(window=14).min()
            high_max = t_df['High'].rolling(window=14).max()
            t_df['Stoch_K'] = 100 * ((t_df['Close'] - low_min) / (high_max - low_min + 1e-9))
            
        t_df['Stoch_D'] = t_df['Stoch_K'].rolling(window=3).mean()
        t_df['Stoch_Diff'] = t_df['Stoch_K'] - t_df['Stoch_D']
        
        t_df['Tilson_T3'] = tilson_t3(t_df['Close'])
        t_df['Tilson_Dist'] = (t_df['Close'] - t_df['Tilson_T3']) / t_df['Close'].replace(0, 0.0001)
        
        delta = t_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        t_df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 0.0001)))

        macd = t_df['Close'].ewm(span=12, adjust=False).mean() - t_df['Close'].ewm(span=26, adjust=False).mean()
        t_df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()

        bb_orta = t_df['Close'].rolling(window=20).mean()
        bb_std = t_df['Close'].rolling(window=20).std()
        bb_fark = (bb_std * 4).replace(0, 0.0001)
        t_df['BB_Pozisyon'] = (t_df['Close'] - (bb_orta - (bb_std * 2))) / bb_fark

        high_low = t_df['High'] - t_df['Low']
        high_close = (t_df['High'] - t_df['Close'].shift()).abs()
        low_close = (t_df['Low'] - t_df['Close'].shift()).abs()
        t_df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        t_df['Z_Score'] = (t_df['Close'] - t_df['Close'].rolling(20).mean()) / t_df['Close'].rolling(20).std().replace(0, 0.0001)
        t_df['Vol_Change'] = t_df['Volume'].pct_change()
        t_df['EMA_Trend'] = np.where(t_df['Close'] > t_df['Close'].ewm(span=20).mean(), 1, -1)

        t_df['Return_1d'] = t_df['Close'].pct_change(1)
        t_df['Return_2d'] = t_df['Close'].pct_change(2)
        t_df['Return_3d'] = t_df['Close'].pct_change(3)
        t_df['Vol_Lag1'] = t_df['Vol_Change'].shift(1)
        t_df['Vol_Lag2'] = t_df['Vol_Change'].shift(2)
        
        t_df['EMA_5'] = t_df['Close'].ewm(span=5, adjust=False).mean()
        t_df['EMA_8'] = t_df['Close'].ewm(span=8, adjust=False).mean()
        t_df['EMA_13'] = t_df['Close'].ewm(span=13, adjust=False).mean()
        
        t_df['EMA_5_Dist'] = (t_df['Close'] - t_df['EMA_5']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_8_Dist'] = (t_df['Close'] - t_df['EMA_8']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_13_Dist'] = (t_df['Close'] - t_df['EMA_13']) / t_df['Close'].replace(0, 0.0001)
        
        t_df['Trend_5_8'] = np.where(t_df['EMA_5'] > t_df['EMA_8'], 1, -1)
        t_df['Trend_8_13'] = np.where(t_df['EMA_8'] > t_df['EMA_13'], 1, -1)
        
        t_df['Target_Return'] = ((t_df['Close'].shift(-10) - t_df['Close']) / t_df['Close']) * 100
        
        for col in ['EMA_52', 'EMA_89', 'EMA_144']:
            if col not in t_df.columns:
                t_df[col] = t_df['Close'].ewm(span=int(col.split('_')[1]), adjust=False).mean()
                
        t_df['EMA_52_Dist'] = (t_df['Close'] - t_df['EMA_52']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_89_Dist'] = (t_df['Close'] - t_df['EMA_89']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_144_Dist'] = (t_df['Close'] - t_df['EMA_144']) / t_df['Close'].replace(0, 0.0001)

        features = [
            'RSI', 'MACD_Hist', 'BB_Pozisyon', 'ATR', 'Z_Score', 
            'Vol_Change', 'Stoch_K', 'Stoch_D', 'Stoch_Diff',
            'Tilson_Dist', 'Return_1d', 'Return_2d', 'Return_3d', 'Vol_Lag1', 'Vol_Lag2',
            'EMA_5_Dist', 'EMA_8_Dist', 'EMA_13_Dist', 'Trend_5_8', 'Trend_8_13',
            'EMA_52_Dist', 'EMA_89_Dist', 'EMA_144_Dist',
            'Doji', 'P_Engulfing', 'P_Pinbar', 'AI_Formasyon_Skoru', 
            'Ikili_Tepe', 'Ikili_Dip', 'Simetrik_Ucgen', 'Yukselen_Ucgen',
            'Alcalan_Ucgen', 'Bayrak_Formasyonu', 'Tepe_Uzakligi_Z', 'Dip_Uzakligi_Z', 
            'Cross_Sinyali', 'SMA_50_200_Farki', 'ABCD_Formasyonu'
        ]
       
        t_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        t_df[features] = t_df[features].ffill().bfill().fillna(0)
        ml_df = t_df.dropna(subset=['Target_Return'])
    
        if len(ml_df) < 50:
            return {
                "rf_prediction": float(t_df['Close'].iloc[-1]),
                "signal": "VERİ YETERSİZ",
                "confidence": 50.0,
                "expected_return_pct": 0.0,
                "feature_importances": None
            }
        
        X = ml_df[features].values
        y = ml_df['Target_Return'].values
        son_veri = t_df[features].iloc[-1].values.reshape(1, -1)

        best_xgb_params = en_iyi_xgb_parametrelerini_bul(sembol, X, y)

        model_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        model_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=1.5, epsilon=0.1, kernel='rbf'))])
        model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])

        base_models = [
            ("xgb", model_xgb),
            ("rf", model_rf),
            ("svr", model_svr),
            ("gb", model_gb),
            ("ridge", model_ridge),
        ]

        ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42),
            passthrough=True,
            cv=5,
            n_jobs=-1
        )

        model_klasoru = "ai_modeller"
        os.makedirs(model_klasoru, exist_ok=True)
        model_dosyasi = os.path.join(model_klasoru, f"{sembol}_ai_model.pkl")

        if os.path.exists(model_dosyasi):
            try:
                ensemble = joblib.load(model_dosyasi)
            except Exception:
                ensemble.fit(X, y)
                joblib.dump(ensemble, model_dosyasi)
        else:
            ensemble.fit(X, y)
            joblib.dump(ensemble, model_dosyasi)

        feature_importances = ai_feature_importance(ensemble)
        if feature_importances is None:
            feature_importances = np.ones(len(features), dtype=float)
            feature_importances /= feature_importances.sum()

        beklenen_getiri_pct = float(ensemble.predict(son_veri)[0])
        guncel_fiyat = float(t_df['Close'].iloc[-1])
        hedef_fiyat = guncel_fiyat * (1 + beklenen_getiri_pct / 100)
        
        atr_degeri = float(t_df['ATR'].iloc[-1]) if 'ATR' in t_df.columns and t_df['ATR'].iloc[-1] > 0 else guncel_fiyat * 0.02
        risk_mesafesi = atr_degeri * 2
        odul_mesafesi = abs(hedef_fiyat - guncel_fiyat)
        risk_odul_orani = odul_mesafesi / risk_mesafesi if risk_mesafesi > 0 else 1.0

        temp_dip_analiz = dipten_donus_analizi(t_df)
        dipten_donus_var = temp_dip_analiz['Wyckoff_Spring'].iloc[-1] or \
                           temp_dip_analiz['Super_Sinyal'].iloc[-1] or \
                           (t_df['Ikili_Dip'].iloc[-1] == 1) or \
                           (t_df['P_Engulfing'].iloc[-1] == 1)

        if beklenen_getiri_pct >= 10.0 and dipten_donus_var and risk_odul_orani >= 1.5:
            sinyal = "🎯 KESİN AL (Ödül >%10, Dip Onaylı)"
        elif beklenen_getiri_pct >= 5.0 and dipten_donus_var:
            sinyal = "🚀 POTANSİYEL AL (Dipten Dönüş)"
        elif beklenen_getiri_pct >= 10.0 and not dipten_donus_var:
            sinyal = "⚠️ RİSKLİ AL (Hedef Yüksek ama Dip Onayı Yok)"
        elif beklenen_getiri_pct < 0:
            sinyal = "🛑 SAT / UZAK DUR (Negatif Beklenti)"
        else:
            sinyal = "⚖️ NÖTR / BEKLE"
            
        teknik_skor = min(abs(beklenen_getiri_pct) * 10, 100)
        formasyon_skoru = float(t_df["AI_Formasyon_Skoru"].iloc[-1]) * 10 if "AI_Formasyon_Skoru" in t_df.columns else 50.0
        atr_orani = atr_degeri / guncel_fiyat if guncel_fiyat > 0 else 0

        guven_skoru = ai_guven_skoru_hesapla(
            ensemble=ensemble,
            son_veri=son_veri,
            teknik_skor=teknik_skor,
            formasyon_skoru=formasyon_skoru,
            risk_odul=risk_odul_orani,
            atr_orani=atr_orani
        )

        try:
            conn = db_connect()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO tahminler (tarih, sembol, fiyat, hedef_fiyat, sinyal, guven, beklenen_getiri, model, durum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'BEKLİYOR')
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sembol, guncel_fiyat, hedef_fiyat, sinyal, guven_skoru, beklenen_getiri_pct, "Stacking"
            ))
            conn.commit()
            conn.close()
        except Exception as db_hata:
            logging.error(f"AI kayıt hatası: {db_hata}")

        return {
            "rf_prediction": round(hedef_fiyat, 2),
            "signal": sinyal,
            "confidence": max(round(guven_skoru, 1), 0.0),
            "expected_return_pct": round(beklenen_getiri_pct, 2),
            "feature_importances": feature_importances,
            "feature_names": features
        }
    except Exception as e:
        logging.error(f"AI Ensemble Hatası: {e}")
        return {
            "rf_prediction": float(t_df["Close"].iloc[-1]) if "t_df" in locals() and not t_df.empty else 0.0,
            "signal": "AI HATASI",
            "confidence": 0.0,
            "expected_return_pct": 0.0,
            "feature_importances": None,
            "feature_names": []
        }

# ==========================================
# REINFORCEMENT LEARNING (RL) AJANI (DÜZELTİLDİ)
# ==========================================
def rl_ajani_egit(df):
    """
    Verilen hisse verisi üzerinde bir RL ajanı eğitir ve strateji üretir.
    Veride 'Open', 'High', 'Low', 'Close', 'Volume' sütunları olmalıdır.
    """
    try:
        window_size = 30
        if df is None or len(df) <= window_size:
            return None, "Yetersiz Veri"
            
        start_index = window_size
        end_index = len(df)
        
        env = gym.make('stocks-v0', 
                       df=df, 
                       frame_bound=(start_index, end_index), 
                       window_size=window_size)
        
        # Advantage Actor-Critic (A2C) Modeli Eğitimi
        model = A2C('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=1000)
        
        return model, env
    except Exception as e:
        logging.error(f"RL Eğitimi Hatası: {e}")
        return None, str(e)

# ==========================================
# STREAMLIT ANA UYGULAMA ARAYÜZÜ (MAIN)
# ==========================================
st.title("⚡ God Mode Terminal v102")
st.sidebar.header("Uygulama Kontrol Paneli")

hisse_kodu = st.sidebar.text_input("BIST Hisse Kodu (Örn: THYAO.IS):", "THYAO.IS")

if st.sidebar.button("Analiz Başlat"):
    with st.spinner("Piyasa verileri çekiliyor ve AI modelleri çalıştırılıyor..."):
        df_veriler = veri_yukle(hisse_kodu, "2023-01-01", datetime.now().strftime("%Y-%m-%d"))
        
        if not df_veriler.empty:
            st.success(f"**{hisse_kodu}** için veriler başarıyla yüklendi.")
            st.line_chart(df_veriler['Close'])
            
            ai_res = ensemble_prediction(df_veriler, sembol=hisse_kodu)
            st.metric("Tahmini Hedef Fiyat", f"{ai_res.get('rf_prediction', 0)} TL", delta=f"% {ai_res.get('expected_return_pct', 0)}")
            st.info(f"AI Sinyali: {ai_res.get('signal', 'NÖTR')} | Güven Skoru: %{ai_res.get('confidence', 0)}")
        else:
            st.error("Veri alınamadı, lütfen hisse sembolünü veya İnternet bağlantınızı kontrol edin.")