# ==========================================
# KÜTÜPHANELER (En üste taşındı ve hızlandırıldı)
# ==========================================
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import requests
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})
import concurrent.futures
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Yapay Zeka Kütüphaneleri
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import sqlite3
import optuna
from sklearn.metrics import mean_squared_error
from tvDatafeed import TvDatafeed, Interval
import isyatirimhisse
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
import shap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
import gym_anytrading
from stable_baselines3 import A2C
import asyncio
import aiohttp
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import time as tm

# --- TRADINGVIEW BAĞLANTISINI HAFIZADA TUTAN BLOK ---
st.set_page_config(layout="wide", page_title="God Mode Terminal v100")
@st.cache_resource(show_spinner=False)
def get_tv_datafeed():
    """TradingView bağlantısını bir kez kurar ve hafızada (cache) tutar."""
    try:
        tv = TvDatafeed() 
        return tv
    except Exception as e:
        logging.error(f"TradingView Bağlantı Hatası: {e}")
        return None

# ==========================================
# SAYFA AYARLARI VE OTURUM
# ==========================================
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

# ==========================================
# VERİTABANI VE HAFIZA YÖNETİMİ
# ==========================================
def veritabani_baslat():
    """Yapay zekanın tahminlerini tutacağı yerel veritabanını oluşturur."""
    conn = sqlite3.connect('hisse_hafiza.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tahminler
                 (tarih TEXT, sembol TEXT, hedef_fiyat REAL, gerceklesme_fiyati REAL, durum TEXT)''')
    conn.commit()
    conn.close()

def tahmin_kaydet(sembol, hedef_fiyat):
    """Bugünün tahminlerini hafızaya yazar."""
    conn = sqlite3.connect('hisse_hafiza.db', timeout=10)
    c = conn.cursor()
    bugun = datetime.now().strftime("%Y-%m-%d")
    
    c.execute("SELECT * FROM tahminler WHERE tarih=? AND sembol=?", (bugun, sembol))
    if not c.fetchone():
        c.execute("INSERT INTO tahminler (tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum) VALUES (?, ?, ?, NULL, 'BEKLİYOR')", 
                  (bugun, sembol, hedef_fiyat))
    conn.commit()
    conn.close()

def tahminleri_degerlendir():
    """5 gün öncesinin tahminlerini bugünün gerçek fiyatlarıyla kıyaslar."""
    conn = sqlite3.connect('hisse_hafiza.db', timeout=10)
    c = conn.cursor()
    c.execute("SELECT rowid, tarih, sembol, hedef_fiyat FROM tahminler WHERE durum = 'BEKLİYOR'")
    bekleyenler = c.fetchall()
    
    for row in bekleyenler:
        rowid, tarih_str, sembol, hedef_fiyat = row
        kayit_tarihi = datetime.strptime(tarih_str, "%Y-%m-%d")
        
        if (datetime.now() - kayit_tarihi).days >= 5:
            try:
                df = yf.download(sembol, period="1d", progress=False)
                if not df.empty:
                    gercek_fiyat = float(df['Close'].iloc[-1])
                    sapma_orani = abs(gercek_fiyat - hedef_fiyat) / gercek_fiyat
                    durum = "BAŞARILI ✅" if sapma_orani <= 0.05 else "BAŞARISIZ ❌"
                    c.execute("UPDATE tahminler SET gerceklesme_fiyati = ?, durum = ? WHERE rowid = ?", 
                              (gercek_fiyat, durum, rowid))
            except Exception as e:
                logging.error(f"Tahmin değerlendirme hatası [{sembol}]: {e}")
    conn.commit()
    conn.close()

veritabani_baslat()

def sembol_formatla(hisse_kodu):
    ana_sembol = hisse_kodu.replace('.IS', '').replace('BIST:', '').strip().upper()
    formatlar = {
        'yfinance': f"{ana_sembol}.IS",
        'isyatirim': ana_sembol,
        'tradingview': f"BIST:{ana_sembol}",
        'saf_sembol': ana_sembol
    }
    return formatlar

# ==========================================
# 1. TEMEL VE İLERİ TEKNİK FONKSİYONLAR
# ==========================================
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
                    if end is not None:
                        yf_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    else:
                        yf_end = None

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
                tm.sleep(0.5)

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
                    tm.sleep(0.5)
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
        return yf.Ticker(ticker, session=oturum).info
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

def sihirli_formul_skorla(sembol):
    try:
        info = sirket_bilgisi_getir(sembol)
        if not info:
            return {'Puan': 0}
            
        skor = 0
        fk = info.get('trailingPE', 999)
        if fk is None: fk = 999
        if 0 < fk <= 10: skor += 25
        elif 10 < fk <= 15: skor += 15
        elif 15 < fk <= 20: skor += 5
        
        pddd = info.get('priceToBook', 999)
        if pddd is None: pddd = 999
        if 0 < pddd <= 1.5: skor += 25
        elif 1.5 < pddd <= 3.0: skor += 15
        elif 3.0 < pddd <= 5.0: skor += 5
        
        roe = info.get('returnOnEquity', -1)
        if roe is None: roe = -1
        if roe > 0.20: skor += 25
        elif roe > 0.10: skor += 15
        elif roe > 0.05: skor += 5
        
        cari_oran = info.get('currentRatio', 0)
        if cari_oran is None: cari_oran = 0
        if cari_oran >= 1.5: skor += 25
        elif cari_oran >= 1.0: skor += 15
        
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

    df_ta['Fibo_MA_Trend'] = np.where(
        (df_ta['EMA_5'] > df_ta['EMA_8']) & (df_ta['EMA_8'] > df_ta['EMA_13']), "🚀 GÜÇLÜ YÜKSELİŞ",
        np.where((df_ta['EMA_5'] < df_ta['EMA_8']) & (df_ta['EMA_8'] < df_ta['EMA_13']), "🔻 GÜÇLÜ DÜŞÜŞ", "⚖️ YATAY NÖTR")
    )
    return df_ta

def grafik_formasyon_bul(df, window=10, tolerans=0.03):
    try:
        df_form = df.copy()
        df_form['Roll_Max'] = df_form['High'].rolling(window=window*2+1).max()
        df_form['Roll_Min'] = df_form['Low'].rolling(window=window*2+1).min()
        
        df_form['Local_Max'] = df_form['High'].shift(window) == df_form['Roll_Max']
        df_form['Local_Min'] = df_form['Low'].shift(window) == df_form['Roll_Min']
        
        ikili_tepeler, ikili_dipler = [], []
        max_idx = df_form[df_form['Local_Max']].index
        min_idx = df_form[df_form['Local_Min']].index
        
        for i in range(1, len(max_idx)):
            f1, f2 = df_form.loc[max_idx[i-1], 'High'], df_form.loc[max_idx[i], 'High']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (max_idx[i] - max_idx[i-1]).days
                if 5 < zaman_farki < 90:
                    ikili_tepeler.append((max_idx[i-1], max_idx[i], f1, f2))
                    
        for i in range(1, len(min_idx)):
            f1, f2 = df_form.loc[min_idx[i-1], 'Low'], df_form.loc[min_idx[i], 'Low']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (min_idx[i] - min_idx[i-1]).days
                if 5 < zaman_farki < 90:
                    ikili_dipler.append((min_idx[i-1], min_idx[i], f1, f2))
        return ikili_tepeler, ikili_dipler
    except:
        return [], []

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

# ==========================================
# YENİ EKLENEN: FORMASYON TESPİT VE HEDEF HESAPLAMA
# ==========================================
def formasyon_tespit_et_ve_hedefle(df):
    """
    Son güncel verileri analiz ederek tespit edilen formasyonu ve
    teknik hedef yüzde (%) değişimini döndürür.
    """
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

    # 1. Makro / Grafik Formasyonları
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

    # 2. Harmonik ve Trend Kesişimleri
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

    # 3. Mikro Mum Formasyonları
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
        df_copy['Pozitif_Uyusmazlik'] = False
        df_copy['Wyckoff_Spring'] = False
        return df_copy

    df_dip = df.copy()
    df_dip['Vol_SMA_20'] = df_dip['Volume'].rolling(20).mean()
    df_dip['Hacim_Patlamasi'] = df_dip['Volume'] > (df_dip['Vol_SMA_20'] * 2)
    
    df_dip['SMA_20_Dip'] = df_dip['Close'].rolling(20).mean()
    df_dip['STD_20_Dip'] = df_dip['Close'].rolling(20).std()
    df_dip['Lower_Band'] = df_dip['SMA_20_Dip'] - (df_dip['STD_20_Dip'] * 2)
    
    df_dip['Wyckoff_Spring'] = (df_dip['Low'] < df_dip['Lower_Band']) & (df_dip['Close'] > df_dip['Lower_Band']) & (df_dip['Close'] > df_dip['Open'])
    
    delta = df_dip['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df_dip['RSI_DIP'] = 100 - (100 / (1 + rs))
    
    if len(df_dip) >= 20:
        son_5_fiyat = df_dip['Close'].iloc[-5:].min()
        eski_15_fiyat = df_dip['Close'].iloc[-20:-5].min()
        son_5_rsi = df_dip['RSI_DIP'].iloc[-5:].min()
        eski_15_rsi = df_dip['RSI_DIP'].iloc[-20:-5].min()
        
        uyusmazlik = (son_5_fiyat < eski_15_fiyat) and (son_5_rsi > eski_15_rsi) and (son_5_rsi < 45)
        df_dip['Pozitif_Uyusmazlik'] = uyusmazlik
    else:
        df_dip['Pozitif_Uyusmazlik'] = False
        
    return df_dip

def backtest_motoru(df, kisa_periyot=20, uzun_periyot=50):
    bt_df = df[['Close']].copy()
    bt_df['Kisa_SMA'] = bt_df['Close'].rolling(window=kisa_periyot).mean()
    bt_df['Uzun_SMA'] = bt_df['Close'].rolling(window=uzun_periyot).mean()
    bt_df.dropna(inplace=True)
    bt_df['Sinyal'] = np.where(bt_df['Kisa_SMA'] > bt_df['Uzun_SMA'], 1, 0)
    bt_df['Günlük_Getiri'] = bt_df['Close'].pct_change()
    bt_df['Strateji_Getirisi'] = bt_df['Günlük_Getiri'] * bt_df['Sinyal'].shift(1)
    bt_df['Piyasa_Kumulatif'] = (1 + bt_df['Günlük_Getiri']).cumprod() * 100
    bt_df['Strateji_Kumulatif'] = (1 + bt_df['Strateji_Getirisi']).cumprod() * 100
    return bt_df

def hizli_backtest_yap(sembol, baslangic, bitis):
    try:
        df = veri_yukle(sembol, baslangic, bitis)
        if df is None or df.empty or len(df) < 50:
            return None
            
        b_df = df.copy()
        b_df['Tilson_T3'] = tilson_t3(b_df['Close'])
        
        low_min = b_df['Low'].rolling(window=14).min()
        high_max = b_df['High'].rolling(window=14).max()
        b_df['Stoch_K'] = 100 * ((b_df['Close'] - low_min) / (high_max - low_min + 1e-9))
        b_df['Stoch_D'] = b_df['Stoch_K'].rolling(window=3).mean()
        
        b_df['AL_Sinyali'] = (b_df['Stoch_K'] > b_df['Stoch_D']) & (b_df['Stoch_K'] < 30) & (b_df['Close'] > b_df['Tilson_T3'])
        b_df['5_Gunluk_Getiri'] = ((b_df['Close'].shift(-5) - b_df['Close']) / b_df['Close']) * 100
        
        islemler = b_df[b_df['AL_Sinyali']].dropna(subset=['5_Gunluk_Getiri'])
        toplam_islem = len(islemler)
        
        if toplam_islem == 0:
            return None
            
        basarili_islem = len(islemler[islemler['5_Gunluk_Getiri'] > 0])
        basari_orani = (basarili_islem / toplam_islem) * 100
        ortalama_getiri = islemler['5_Gunluk_Getiri'].mean()
        kümülatif_getiri = islemler['5_Gunluk_Getiri'].sum()
        
        return {
            "Hisse": sembol,
            "İşlem Sayısı": toplam_islem,
            "Başarı Oranı (%)": round(basari_orani, 2),
            "İşlem Başı Ort. Kâr (%)": round(ortalama_getiri, 2),
            "Kümülatif Kâr (%)": round(kümülatif_getiri, 2)
        }
    except Exception as e:
        logging.error(f"[{sembol}] Backtest Hatası: {str(e)}")
        return None

def stacking_model_olustur(xgb_model, rf_model, svr_model):
    estimators = [
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('svr', svr_model)
    ]
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    )
    return stack_model

def monte_carlo_simulasyonu(df, gun_sayisi=30, sim_sayisi=100):
    getiriler = df['Close'].pct_change().dropna()
    ortalama_getiri = getiriler.mean()
    volatilite = getiriler.std()
    son_fiyat = df['Close'].iloc[-1]
    simulasyonlar = np.zeros((gun_sayisi, sim_sayisi))
    for i in range(sim_sayisi):
        rastgele_getiriler = np.random.normal(ortalama_getiri, volatilite, gun_sayisi)
        simulasyonlar[:, i] = son_fiyat * (1 + rastgele_getiriler).cumprod()
    return simulasyonlar

def shap_aciklamasi_goster(model, X_train, hisse_adi):
    st.subheader(f"{hisse_adi} - Yapay Zeka Karar Gerekçeleri (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP grafiği oluşturulurken hata: {e}")

def python_istatistik_analizi(df):
    try:
        getiriler = df['Close'].pct_change().dropna()
        yillik_volatilite = getiriler.std() * np.sqrt(252)
        sharpe_orani = (getiriler.mean() * 252) / yillik_volatilite
        var_95 = getiriler.quantile(0.05)
        return {
            'Yıllık Volatilite': f"% {yillik_volatilite * 100:.2f}",
            'Sharpe Oranı': f"{sharpe_orani:.2f}",
            'Günlük VaR (%95)': f"% {var_95 * 100:.2f}"
        }
    except:
        return {'Yıllık Volatilite': "% 0.00", 'Sharpe Oranı': "0.00", 'Günlük VaR (%95)': "% 0.00"}

def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
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

def asenkron_analiz_yap(sembol, baslangic, bitis, analiz_tipi="radar", veri_kaynagi="Yahoo Finance (yfinance)"):
    try:
        df_gunluk = veri_yukle(sembol, baslangic, bitis, interval="1d", kaynak=veri_kaynagi)
        if df_gunluk is None or df_gunluk.empty or len(df_gunluk) < 20: 
            return None
            
        df_g = df_gunluk.copy()
        
        try:
            guncel_df = yf.download(sembol, period="1d", interval="1m", progress=False)
            guncel_fiyat = float(guncel_df['Close'].iloc[-1]) if not guncel_df.empty else float(df_g['Close'].iloc[-1])
        except Exception:
            guncel_fiyat = float(df_g['Close'].iloc[-1])

        df_g = stokastik_hesapla(df_g)
        df_g['Tilson_T3'] = tilson_t3(df_g['Close'])
        df_g = ileri_teknik_gostergeler(df_g)
        temp_g = dipten_donus_analizi(df_g)
        
        g_fiyat = df_g['Close'].iloc[-1]
        g_tilson = df_g['Tilson_T3'].iloc[-1]
        g_stoch_k = df_g['Stoch_K'].iloc[-1]
        g_stoch_d = df_g['Stoch_D'].iloc[-1]
        g_ema5 = df_g['EMA_5'].iloc[-1]
        g_ema8 = df_g['EMA_8'].iloc[-1]
        g_ema13 = df_g['EMA_13'].iloc[-1]
        
        g_boga = g_fiyat > g_tilson
        g_stoch_al = (g_stoch_k < 35) and (g_stoch_k > g_stoch_d)
        g_hacim = temp_g['Hacim_Patlamasi'].iloc[-1]
        g_uyusmazlik = temp_g['Pozitif_Uyusmazlik'].iloc[-1]
        g_spring = temp_g['Wyckoff_Spring'].iloc[-1]
        g_ma_kestimi = (g_ema5 > g_ema8) and (g_ema8 > g_ema13)
        
        umut_var_mi = g_boga or g_stoch_al or g_hacim or g_uyusmazlik or g_spring or g_ma_kestimi
        
        if not umut_var_mi and analiz_tipi == "radar":
            return {
                "Varlık": sembol,
                "Güncel Fiyat": f"{guncel_fiyat:.2f}",
                "Kapanış Fiyatı": f"{g_fiyat:.2f}",
                "🎯 AL/SAT Kararı": "🐻 PAS GEÇİLDİ (Ölü Trend)",
                "Günlük T3": "🐻 AYI",
                "4S T3": "-",
                "📊 Temel Skor": "-",
                "💥 Hacim Analizi": "Normal",
                "📈 Pozitif Uyuşmazlık": "-",
                "🪤 Spring (Tuzak)": "-",
                "🔍 Tespit Edilen Formasyon": "Yok",
                "🎯 Formasyon Hedefi (%)": "% 0.00",
                "🤖 AI Kararı": "Zaman Tasarrufu",
                "🎯 AI Hedef": "-"
            }

        df_4h = veri_4saatlik_getir(sembol, baslangic, bitis, kaynak=veri_kaynagi)
        
        h4_fiyat, h4_tilson = g_fiyat, g_tilson
        h4_stoch_k, h4_stoch_d = g_stoch_k, g_stoch_d
        h4_boga, h4_stoch_al = g_boga, g_stoch_al

        if not df_4h.empty and len(df_4h) >= 20:
            try:
                df_4h = stokastik_hesapla(df_4h)
                df_4h['Tilson_T3'] = tilson_t3(df_4h['Close'])
                
                h4_fiyat = df_4h['Close'].iloc[-1]
                h4_tilson = df_4h['Tilson_T3'].iloc[-1]
                h4_stoch_k = df_4h['Stoch_K'].iloc[-1]
                h4_stoch_d = df_4h['Stoch_D'].iloc[-1]
                
                h4_boga = h4_fiyat > h4_tilson
                h4_stoch_al = (h4_stoch_k < 35) and (h4_stoch_k > h4_stoch_d)
            except Exception as e:
                logging.error(f"[{sembol}] 4S Analiz Hatası: {e}")

        if g_boga and h4_boga:
            al_sat_karari = "🚀 GÜÇLÜ AL (4S + Günlük Onaylı)" if (g_stoch_al and h4_stoch_al) else "🟢 AL (Trend Onaylı)"
        elif g_boga and not h4_boga:
            al_sat_karari = "⚠️ DÜZELTME (Günlük Boğa / 4S Ayı)"
        elif not g_boga and h4_boga:
            al_sat_karari = "⚡ TEPKİ YÜKSELİŞİ (4S Boğa / Günlük Ayı)"
        else:
            al_sat_karari = "🐻 GÜÇLÜ SAT / AYI"

        if analiz_tipi == "radar":
            temp_4h = dipten_donus_analizi(df_4h) if not df_4h.empty else temp_g
            h4_hacim = temp_4h['Hacim_Patlamasi'].iloc[-1] if not temp_4h.empty else False
            
            hacim_durum = "🔥 GÜÇLÜ PATLAMA" if (g_hacim or h4_hacim) else "Normal"
            uyusmazlik_durum = "✅ POZİTİF UYUŞMAZLIK" if (g_uyusmazlik or temp_4h.get('Pozitif_Uyusmazlik', pd.Series([False])).iloc[-1]) else "-"
            spring_durum = "✅ VAR" if (g_spring or temp_4h.get('Wyckoff_Spring', pd.Series([False])).iloc[-1]) else "-"
            
            try:
                s_skor = sihirli_formul_skorla(sembol)['Puan']
            except Exception:
                s_skor = 0

            ai_veri = ensemble_prediction(df_g, sembol) if umut_var_mi else {'signal': "ZAYIF", 'rf_prediction': 0.0}
            
            # --- FORMASYON TESPİTİ VE YÜZDESEL HEDEF HESABI ---
            formasyon_adi, formasyon_hedef = formasyon_tespit_et_ve_hedefle(df_g)

            return {
                "Varlık": sembol,
                "Güncel Fiyat": f"{guncel_fiyat:.2f}",
                "Kapanış Fiyatı": f"{g_fiyat:.2f}",
                "🎯 AL/SAT Kararı": al_sat_karari,
                "Günlük T3": "🚀 BOĞA" if g_boga else "🐻 AYI",
                "4S T3": "🚀 BOĞA" if h4_boga else "🐻 AYI",
                "📊 Temel Skor": s_skor,
                "💥 Hacim Analizi": hacim_durum,
                "📈 Pozitif Uyuşmazlık": uyusmazlik_durum,
                "🪤 Spring (Tuzak)": spring_durum,
                "🔍 Tespit Edilen Formasyon": formasyon_adi,
                "🎯 Formasyon Hedefi (%)": formasyon_hedef,
                "🤖 AI Kararı": ai_veri.get('signal', 'NÖTR'),
                "🎯 AI Hedef": f"{ai_veri.get('rf_prediction', 0.0)} TL"
            }

        elif analiz_tipi == "stoch":
            return {
                "Varlık": sembol,
                "Son Fiyat": f"{g_fiyat:.2f}",
                "Günlük Stoch %K": round(g_stoch_k, 2),
                "4S Stoch %K": round(h4_stoch_k, 2),
                "Durum": "🟢 Çift Dip/Al" if (g_stoch_al and h4_stoch_al) else ("↗️ Pozitif" if h4_stoch_al else "⚪ Nötr")
            }

    except Exception as e:
        logging.error(f"[{sembol}] Analiz Hatası: {str(e)}")
        return None

# ==========================================
# 2. YAPAY ZEKA VE KURUMSAL MOTORLAR
# ==========================================
def institutional_decision(df):
    try:
        return {
            "decision": "BİRİKİM (ACCUMULATION)", 
            "regime": "Yükseliş Trendi" if df['Close'].iloc[-1] > df['Close'].rolling(50).mean().iloc[-1] else "Düşüş / Range", 
            "score": 8.5, 
            "risk": 30
        }
    except:
        return {"decision": "BEKLE", "regime": "Belirsiz", "score": 5.0, "risk": 50}

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
        mse = mean_squared_error(y_test, preds)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    return study.best_params

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

        t_df['Target_Return'] = ((t_df['Close'].shift(-5) - t_df['Close']) / t_df['Close']) * 100

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

        features = [
            'RSI', 'MACD_Hist', 'BB_Pozisyon', 'ATR', 'Z_Score', 
            'Vol_Change', 'EMA_Trend', 'Stoch_K', 'Stoch_D', 'Stoch_Diff',
            'Tilson_Dist', 'Return_1d', 'Return_2d', 'Return_3d', 
            'Vol_Lag1', 'Vol_Lag2',
            'EMA_5_Dist', 'EMA_8_Dist', 'EMA_13_Dist', 'Trend_5_8', 'Trend_8_13',
            'Doji', 'P_Engulfing', 'P_Pinbar', 'AI_Formasyon_Skoru', 
            'Ikili_Tepe', 'Ikili_Dip', 'Simetrik_Ucgen', 'Yukselen_Ucgen',
            'Alcalan_Ucgen', 'Bayrak_Formasyonu', 'Tepe_Uzakligi_Z', 
            'Dip_Uzakligi_Z', 'High_Slope', 'Low_Slope', 'Makro_Guc_Skoru',
            'Cross_Sinyali', 'SMA_50_200_Farki', 'ABCD_Formasyonu'
        ]
        
        t_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        t_df[features] = t_df[features].ffill().bfill().fillna(0)
        ml_df = t_df.dropna(subset=['Target_Return'])

        if len(ml_df) < 50:
            return {"rf_prediction": float(t_df['Close'].iloc[-1]), "signal": "VERİ YETERSİZ", "confidence": 50.0, "expected_return_pct": 0.0, "feature_importances": {}}

        X = ml_df[features].values
        y = ml_df['Target_Return'].values
        son_veri = t_df[features].iloc[-1].values.reshape(1, -1)

        best_xgb_params = en_iyi_xgb_parametrelerini_bul(sembol, X, y)

        model_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        model_svr = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=1.5, epsilon=0.1, kernel='rbf'))
        ])
        model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model_ridge = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])

        ensemble = VotingRegressor(estimators=[
            ('xgb', model_xgb),
            ('rf', model_rf),
            ('svr', model_svr),
            ('gb', model_gb),
            ('ridge', model_ridge)
        ])

        ensemble.fit(X, y)

        beklenen_getiri_pct = float(ensemble.predict(son_veri)[0])
        anlik_fiyat = float(t_df['Close'].iloc[-1])
        hedef_fiyat = anlik_fiyat * (1 + (beklenen_getiri_pct / 100))
        
        sinyal = "🚀 GÜÇLÜ AL" if beklenen_getiri_pct > 2.0 else ("⚠️ SAT" if beklenen_getiri_pct < -1.0 else "NÖTR")
        guven_skoru = min(abs(beklenen_getiri_pct) * 8 + 50, 99.0)

        try:
            f_importances = ensemble.named_estimators_['xgb'].feature_importances_
            oznitelik_agirliklari = {f: float(imp) for f, imp in zip(features, f_importances)}
        except Exception:
            oznitelik_agirliklari = {}

        return {
            "rf_prediction": round(hedef_fiyat, 2),
            "signal": sinyal,
            "confidence": max(round(guven_skoru, 1), 0.0),
            "expected_return_pct": round(beklenen_getiri_pct, 2),
            "feature_importances": oznitelik_agirliklari
        }
        
    except Exception as e:
        logging.error(f"AI Ensemble Hatası: {e}")
        return {"rf_prediction": 0.0, "signal": "Hata", "confidence": 0.0, "expected_return_pct": 0.0, "feature_importances": {}}

@st.cache_data(ttl=3600, show_spinner=False)
def gelismis_ai_tahmin(df, gelecek_gun=10):
    try:
        df_ml = df.copy()
        df_ml['Return'] = df_ml['Close'].pct_change()
        df_ml['Log_Return'] = np.log(df_ml['Close'] / df_ml['Close'].shift(1))
        df_ml['SMA_10_Dist'] = df_ml['Close'] / df_ml['Close'].rolling(10).mean() - 1
        df_ml['Volatilite_14'] = df_ml['Return'].rolling(14).std()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        
        df_ml.dropna(inplace=True)
        if len(df_ml) < 50:
            son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun

        features = ['Close', 'Volume', 'Log_Return', 'SMA_10_Dist', 'Volatilite_14']
        X = df_ml[features].values
        y = df_ml['Target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, objective='reg:squarederror', n_jobs=-1)
        model.fit(X_scaled, y)

        tahminler = []
        son_veri = X_scaled[-1].reshape(1, -1)
        
        gecmis_kapanislar = df_ml['Close'].tail(20).tolist()
        
        for _ in range(gelecek_gun):
            pred = float(model.predict(son_veri)[0])
            tahminler.append(pred)
            
            gecmis_kapanislar.append(pred)
            gecmis_kapanislar = gecmis_kapanislar[-20:]
            
            yeni_log_ret = np.log(gecmis_kapanislar[-1] / gecmis_kapanislar[-2])
            yeni_sma_10 = np.mean(gecmis_kapanislar[-10:])
            yeni_sma_10_dist = (gecmis_kapanislar[-1] / yeni_sma_10) - 1
            
            getiriler = [np.log(gecmis_kapanislar[i] / gecmis_kapanislar[i-1]) for i in range(1, len(gecmis_kapanislar))]
            yeni_vol = np.std(getiriler[-14:]) if len(getiriler) >= 14 else np.std(getiriler)
            
            yeni_ham_veri = np.array([[pred, son_veri[0, 1], yeni_log_ret, yeni_sma_10_dist, yeni_vol]])
            son_veri = scaler.transform(yeni_ham_veri)            
            
        tarihler = [df.index[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
        return tarihler, tahminler

    except Exception:
        son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
        return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun

def rl_ajani_egit(df):
    window_size = 30
    start_index = window_size
    end_index = len(df)
    
    env = gym.make('stocks-v0', 
                   df=df, 
                   frame_bound=(start_index, end_index), 
                   window_size=window_size)
    
    model = A2C('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=10000)
    
    obs = env.reset()[0]
    action, _states = model.predict(obs, deterministic=True)
    
    aksiyon_metni = "AL" if action == 1 else "SAT / BEKLE"
    return aksiyon_metni

def lstm_tahmin_yap(df, lookback_days=60):
    df = df.copy()
    df.dropna(inplace=True)
    
    if len(df) <= lookback_days:
        return None 

    yapay_zeka_ozellikleri = [
        'Open', 'High', 'Low', 'Volume', 
        'Tilson_T3', 'Stoch_K', 'Stoch_D',
        'SMA_5', 'SMA_8', 'SMA_13',   
        'EMA_5', 'EMA_8', 'EMA_13'    
    ]
    
    kullanilacak_ozellikler = [col for col in yapay_zeka_ozellikleri if col in df.columns]
    
    X = df[kullanilacak_ozellikler].values
    y = df['Close'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaled_X = scaler_X.fit_transform(X)
    scaled_y = scaler_y.fit_transform(y)
    
    X_train, y_train = [], []
    for i in range(lookback_days, len(scaled_X)):
        X_train.append(scaled_X[i-lookback_days:i, :])
        y_train.append(scaled_y[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    son_veri = scaled_X[-lookback_days:]
    X_test = np.reshape(son_veri, (1, son_veri.shape[0], son_veri.shape[1]))
    
    tahmin_olcekli = model.predict(X_test, verbose=0)
    gercek_tahmin = scaler_y.inverse_transform(tahmin_olcekli)
    
    return gercek_tahmin[0][0]

# ==========================================
# 4. ASENKRON PİYASA TARAMA
# ==========================================
async def tek_hisse_getir(session, sem, hisse_kodu):
    async with sem:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{hisse_kodu}.IS?interval=1d&range=1y"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    timestamps = data['chart']['result'][0]['timestamp']
                    quotes = data['chart']['result'][0]['indicators']['quote'][0]
                    
                    df = pd.DataFrame({
                        'Date': pd.to_datetime(timestamps, unit='s'),
                        'Close': quotes['close'],
                        'Volume': quotes['volume']
                    })
                    df.set_index('Date', inplace=True)
                    return hisse_kodu, df
                else:
                    return hisse_kodu, None
        except Exception:
            return hisse_kodu, None

async def tum_piyasayi_tara_async(hisse_listesi):
    sem = asyncio.Semaphore(50) 
    async with aiohttp.ClientSession() as session:
        gorevler = [tek_hisse_getir(session, sem, hisse) for hisse in hisse_listesi]
        sonuclar = await asyncio.gather(*gorevler)
        basarili_veriler = {hisse: df for hisse, df in sonuclar if df is not None}
        return basarili_veriler

@st.cache_data(ttl=86400, show_spinner=False)
def tum_bist_hisselerini_getir():
    try:
        url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/HisseOzet"
        res = requests.get(url, timeout=10)
        data = res.json()
        return [f"{row['kod']}.IS" for row in data['value']]
    except Exception as e:
        logging.error(f"BIST Hisseleri çekilemedi: {e}")
        return ["XU100.IS", "ACSEL.IS", "ADEL.IS", "ADESE.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS", "AGHOL.IS", "AHGAZ.IS", 
"AKBNK.IS", "AKCNS.IS", "AKENR.IS", "AKFGY.IS", "AKFYE.IS", "AKGRT.IS", "AKMGY.IS", "AKSA.IS", 
"AKSEN.IS", "AKSUE.IS", "AKYHO.IS", "ALARK.IS", "ALBRK.IS", "ALCAR.IS", "ALCTL.IS", "ALFAS.IS", 
"ALGYO.IS", "ALKA.IS", "ALKIM.IS", "ALTNY.IS", "ALVES.IS", "ANELE.IS", "ANGEN.IS", 
"ANHYT.IS", "ANSGR.IS", "ARASE.IS", "ARCLK.IS", "ARDYZ.IS", "ARENA.IS", "ARSAN.IS", 
"ARTMS.IS", "ARZUM.IS", "ASELS.IS", "ASGYO.IS", "ASTOR.IS", "ASUZU.IS", "ATAGY.IS", 
"ATATP.IS", "ATEKS.IS", "ATLAS.IS", "AVGYO.IS", "AVHOL.IS", "AVOD.IS", "AVTUR.IS", "AYCES.IS", 
"AYDEM.IS", "AYEN.IS", "AYGAZ.IS", "AZTEK.IS", "BAGFS.IS", "BAKAB.IS", "BALAT.IS", "BANVT.IS", 
"BARMA.IS", "BASCM.IS", "BASGZ.IS", "BAYRK.IS", "BEYAZ.IS", "BFREN.IS", "BIENY.IS", "BIGCH.IS", 
"BIMAS.IS", "BINHO.IS", "BIOEN.IS", "BIZIM.IS", "BJKAS.IS", "BLCYT.IS", "BMSCH.IS", "BMSTL.IS", 
"BNTAS.IS", "BOBET.IS", "BORSK.IS", "BOSSA.IS", "BRISA.IS", "BRKO.IS", "BRKSN.IS", 
"BRKVY.IS", "BRLSM.IS", "BRMEN.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "BUCIM.IS", 
"BURCE.IS", "BURVA.IS", "BVSAN.IS", "BYDNR.IS", "CANTE.IS", "CASA.IS", "CATES.IS", "CCOLA.IS", 
"CELHA.IS", "CEMAS.IS", "CEMTS.IS", "CEOEM.IS", "CIMSA.IS", "CLEBI.IS", "CMBTN.IS", "CMENT.IS", 
"CONSE.IS", "COSMO.IS", "CRDFA.IS", "CRFSA.IS", "CUSAN.IS", "CVKMD.IS", "CWENE.IS", 
"DAGI.IS", "DAPGM.IS", "DARDL.IS", "DERHL.IS", "DERIM.IS", "DESA.IS", "DESPC.IS", "DEVA.IS", "DITAS.IS", "DMRGD.IS", "DOAS.IS", "DOCO.IS", "DOFER.IS", "DOGUB.IS", "DOHOL.IS", 
"DOKTA.IS", "DURDO.IS", "DYOBY.IS", "DZGYO.IS", "EBEBK.IS", "ECILC.IS", "ECZYT.IS", "EDATA.IS", 
"EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EKGYO.IS", "DIRIT.IS",
"EKIZ.IS", "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "ENSRI.IS", 
"EPLAS.IS", "ERBOS.IS", "EREGL.IS", "ERSU.IS", "ESCAR.IS", "ESCOM.IS", "ESEN.IS", 
"ETILR.IS", "ETYAT.IS", "EUHOL.IS", "EUPWR.IS", "EUREN.IS", "EUYO.IS", "EYGYO.IS", "FADE.IS", 
"FENER.IS", "FLAP.IS", "FMIZP.IS", "FONET.IS", "FORMT.IS", "FORTE.IS", "FRIGO.IS", "FROTO.IS", 
"FZLGY.IS", "GARAN.IS", "GARFA.IS", "GEDIK.IS", "GEDZA.IS", "GENIL.IS", "GENTS.IS", "GEREL.IS", 
"GESAN.IS", "GIPTA.IS", "GLBMD.IS", "GLCVY.IS", "GLRYH.IS", "GLYHO.IS", "GMTAS.IS", "GOKNR.IS", 
"GOLTS.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS", "GRSEL.IS", "GSDDE.IS", "GSDHO.IS", 
"GSRAY.IS", "GUBRF.IS", "GWIND.IS", "GZNMI.IS", "HALKB.IS", "HATEK.IS", "HATSN.IS", "HDFGS.IS", 
"HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "HRKET.IS", "HTTBT.IS", "HUBVC.IS", "HUNER.IS", 
"HURGZ.IS", "ICBCT.IS", "IDGYO.IS", "IEYHO.IS", "IHAAS.IS", "IHEVA.IS", "IHGZT.IS", 
"IHLAS.IS", "IHLGM.IS", "IHYAY.IS", "IMASM.IS", "INDES.IS", "INFO.IS", "INGRM.IS", "INTEM.IS", 
"INVEO.IS", "INVES.IS", "ISBTR.IS", "ISCTR.IS", "ISDMR.IS", "ISFIN.IS", "ISGSY.IS", 
"ISGYO.IS", "ISKPL.IS", "ISKUR.IS", "ISMEN.IS", "ISSEN.IS", "ISYAT.IS", "IZENR.IS", "IZFAS.IS", "IZINV.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS", "KAREL.IS", "KARSN.IS", 
"KARTN.IS", "KATMR.IS", "KAYSE.IS", "KCAER.IS", "KCHOL.IS", "KENT.IS", "KERVN.IS", "KFEIN.IS", "KGYO.IS", "KIMMR.IS", "KLGYO.IS", "KLKIM.IS", "KLMSN.IS", "KLNMA.IS", 
"KLRHO.IS", "KLSYN.IS", "KMPUR.IS", "KNFRT.IS", "KONKA.IS", "KONTR.IS", "KONYA.IS", "KOPOL.IS", 
"KORDS.IS", "KRDMA.IS", "KRDMB.IS", "KRDMD.IS", "KRGYO.IS", "KRONT.IS", 
"KRPLS.IS", "KRSTL.IS", "KRTEK.IS", "KRVGD.IS", "KSTUR.IS", "KTLEV.IS", "KTSKR.IS", "KUTPO.IS", 
"KUVVA.IS", "KUYAS.IS", "KZBGY.IS", "KZGYO.IS", "LIDER.IS", "LIDFA.IS", "LINK.IS", "LKMNH.IS", "LOGO.IS", "LRSHO.IS", "LUKSK.IS", "MAALT.IS", "MACKO.IS", "MAGEN.IS", 
"MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MARBL.IS", "MARKA.IS", "MARTI.IS", "MAVI.IS", "MEDTR.IS", 
"MEGAP.IS", "MEKAG.IS", "MEPET.IS", "MERCN.IS", "MERIT.IS", "MERKO.IS", "METRO.IS", 
"MGROS.IS", "MHRGY.IS", "MIATK.IS", "MMCAS.IS", "MNDRS.IS", "MNDTR.IS", "MOBTL.IS", 
"MOGAN.IS", "MPARK.IS", "MRGYO.IS", "MRSHL.IS", "MSGYO.IS", "MTRKS.IS", "MTRYO.IS", "MZHLD.IS", 
"NATEN.IS", "NETAS.IS", "NIBAS.IS", "NTGAZ.IS", "NTHOL.IS", "NUGYO.IS", "NUHCM.IS", "OBASE.IS", 
"OBAMS.IS", "ODAS.IS", "OFSYM.IS", "ONCSM.IS", "ORCAY.IS", "ORGE.IS", "ORMA.IS", "OSMEN.IS", 
"OSTIM.IS", "OTKAR.IS", "OYAKC.IS", "OYAYO.IS", "OYLUM.IS", "OYYAT.IS", "OZGYO.IS", 
"OZKGY.IS", "OZRDN.IS", "OZSUB.IS", "PAGYO.IS", "PAMEL.IS", "PAPIL.IS", "PARSN.IS", "PASEU.IS", 
"PATEK.IS", "PCILT.IS", "PEKGY.IS", "PENGD.IS", "PENTA.IS", "PETKM.IS", "PETUN.IS", 
"PGSUS.IS", "PINSU.IS", "PKART.IS", "PKENT.IS", "PLTUR.IS", "PNLSN.IS", "PNSUT.IS", "POLHO.IS", 
"POLTK.IS", "PRDGS.IS", "PRKAB.IS", "PRKME.IS", "PRZMA.IS", "PSDTC.IS", "PSGYO.IS", "QUAGR.IS", "RALYH.IS", "RAYSG.IS", "REEDR.IS", "RNPOL.IS", "RODRG.IS", "RTALB.IS", 
"RUBNS.IS", "RYGYO.IS", "RYSAS.IS", "SAHOL.IS", "SAMAT.IS", "SANEL.IS", "SANFM.IS", "SANKO.IS", 
"SARKY.IS", "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SEGYO.IS", "SEKFK.IS", "SEKUR.IS", "SELEC.IS", "SELVA.IS", "SEYKM.IS", "SILVR.IS", "SISE.IS", "SKBNK.IS", "SKTAS.IS", "SMART.IS", 
"SMRTG.IS", "SNGYO.IS", "SNICA.IS", "SNPAM.IS", "SOKE.IS", "SOKM.IS", "SONME.IS", 
"SRVGY.IS", "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "SUWEN.IS", "TABGD.IS", "TARKM.IS", "TATEN.IS", 
"TATGD.IS", "TAVHL.IS", "TBORG.IS", "TCELL.IS", "TDGYO.IS", "TEKTU.IS", "TERA.IS", 
"TEZOL.IS", "TGSAS.IS", "THYAO.IS", "TKFEN.IS", "TKNSA.IS", "TLMAN.IS", "TMPOL.IS", "TMSN.IS", 
"TOASO.IS", "TRCAS.IS", "TRGYO.IS", "TRILC.IS", "TSGYO.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS", 
"TTRAK.IS", "TUCLK.IS", "TUKAS.IS", "TUPRS.IS", "TUREX.IS"]