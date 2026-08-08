"""
indicators.py — Saf teknik indikatör hesaplamaları (Tilson T3, Stokastik,
SMC, RSI/MACD/Bollinger/ADX, Fibonacci, haftalık türetme, anomali tespiti).
Hiçbiri ağ isteği atmaz; sadece OHLCV DataFrame'i üzerinde çalışır.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


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
    # ==========================================
    # YENİ EKLENEN: 5, 8, 13 HAREKETLİ ORTALAMALAR
    # ==========================================
    # Basit Hareketli Ortalamalar (SMA)
    df_ta['SMA_5'] = df_ta['Close'].rolling(window=5).mean()
    df_ta['SMA_8'] = df_ta['Close'].rolling(window=8).mean()
    df_ta['SMA_13'] = df_ta['Close'].rolling(window=13).mean()
    df_ta['EMA_21'] = df_ta['Close'].ewm(span=21, adjust=False).mean()
    df_ta['EMA_34'] = df_ta['Close'].ewm(span=34, adjust=False).mean()
    df_ta['EMA_52'] = df_ta['Close'].ewm(span=52, adjust=False).mean()
    df_ta['EMA_89'] = df_ta['Close'].ewm(span=89, adjust=False).mean()
    df_ta['EMA_144'] = df_ta['Close'].ewm(span=144, adjust=False).mean()
    # Üstel Hareketli Ortalamalar (EMA) - (Yapay zeka için daha etkilidir)
    df_ta['EMA_5'] = df_ta['Close'].ewm(span=5, adjust=False).mean()
    df_ta['EMA_8'] = df_ta['Close'].ewm(span=8, adjust=False).mean()
    df_ta['EMA_13'] = df_ta['Close'].ewm(span=13, adjust=False).mean()
    df_ta['Destek'] = df_ta['Low'].rolling(window=20).min()
    df_ta['Direnc'] = df_ta['High'].rolling(window=20).max()
    df_ta['Destege_Uzaklik'] = (df_ta['Close'] - df_ta['Destek']) / df_ta['Close'].replace(0, 1e-9)
    df_ta['Dirence_Uzaklik'] = (df_ta['Direnc'] - df_ta['Close']) / df_ta['Close'].replace(0, 1e-9)
    # Kısa Vadeli Fibonacci Kesişim Trend Sinyali (5 > 8 > 13)
    df_ta['Fibo_MA_Trend'] = np.where(
        (df_ta['EMA_5'] > df_ta['EMA_8']) & (df_ta['EMA_8'] > df_ta['EMA_13']), "🚀 GÜÇLÜ YÜKSELİŞ",
        np.where((df_ta['EMA_5'] < df_ta['EMA_8']) & (df_ta['EMA_8'] < df_ta['EMA_13']), "🔻 GÜÇLÜ DÜŞÜŞ", "⚖️ YATAY NÖTR")
    )
    return df_ta


def genisletilmis_indikatorler_hesapla(df, period=14):
    """
    RSI(14), MACD(12,26,9), Bollinger Bantları(20,2), ADX/+DI/-DI(14) ve
    Wilder ATR(14) hesaplayıp DataFrame'e ekler. Tüm hesaplamalar sadece
    OHLCV verisi üzerinden yapılır, ek ağ isteği gerektirmez.
    """
    d = df.copy()
    if d is None or len(d) < period + 1:
        for kol in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Orta', 'BB_Ust',
                    'BB_Alt', 'BB_Yuzde', 'ATR', 'Plus_DI', 'Minus_DI', 'ADX']:
            d[kol] = np.nan
        return d

    # --- RSI (Wilder) ---
    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    d['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['MACD_Signal']

    # --- Bollinger Bantları (20, 2) ---
    d['BB_Orta'] = d['Close'].rolling(window=20).mean()
    bb_std = d['Close'].rolling(window=20).std()
    d['BB_Ust'] = d['BB_Orta'] + (bb_std * 2)
    d['BB_Alt'] = d['BB_Orta'] - (bb_std * 2)
    bb_genislik = (d['BB_Ust'] - d['BB_Alt']).replace(0, 1e-9)
    # %B: 0 = alt bant, 0.5 = orta bant, 1 = üst bant
    d['BB_Yuzde'] = (d['Close'] - d['BB_Alt']) / bb_genislik

    # --- ATR (Wilder, True Range üzerinden) ---
    prev_close = d['Close'].shift(1)
    tr = pd.concat([
        d['High'] - d['Low'],
        (d['High'] - prev_close).abs(),
        (d['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    d['ATR'] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # --- ADX / +DI / -DI (Wilder) ---
    up_move = d['High'].diff()
    down_move = -d['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_di = d['ATR'].replace(0, 1e-9)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_di
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_di
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    d['Plus_DI'] = plus_di
    d['Minus_DI'] = minus_di
    d['ADX'] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return d


def fibonacci_seviyeleri_hesapla(df, lookback=60):
    """
    Son `lookback` bar içindeki en yüksek/en düşük noktalar arasında
    Fibonacci geri çekilme seviyelerini (0, %23.6, %38.2, %50, %61.8, %78.6,
    %100) hesaplar ve fiyata en yakın seviyeyi döner.
    """
    if df is None or df.empty:
        return None
    pencere = df.tail(lookback) if len(df) >= 10 else df
    if len(pencere) < 10:
        return None

    tepe = float(pencere['High'].max())
    dip = float(pencere['Low'].min())
    fark = tepe - dip
    if fark <= 0:
        return None

    fiyat = float(df['Close'].iloc[-1])
    tepe_idx = pencere['High'].idxmax()
    dip_idx = pencere['Low'].idxmin()
    # Dip, tepeden önce oluştuysa bu bir YÜKSELİŞ bacağıdır -> tepeden geri çekilme
    yukselis_bacagi = dip_idx < tepe_idx

    oranlar = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    seviyeler = {}
    for o in oranlar:
        seviyeler[o] = (tepe - fark * o) if yukselis_bacagi else (dip + fark * o)

    en_yakin_oran = min(seviyeler, key=lambda o: abs(seviyeler[o] - fiyat))
    en_yakin_fiyat = seviyeler[en_yakin_oran]
    uzaklik_pct = (abs(fiyat - en_yakin_fiyat) / fiyat * 100) if fiyat else 0.0

    return {
        "tepe": tepe, "dip": dip, "yukselis_bacagi": yukselis_bacagi,
        "en_yakin_oran": en_yakin_oran, "en_yakin_fiyat": en_yakin_fiyat,
        "uzaklik_pct": uzaklik_pct, "seviyeler": seviyeler,
    }


def haftalik_veri_turet(df_gunluk):
    """
    Ayrı bir ağ isteği ATMADAN, zaten elde olan günlük veriyi haftalığa
    (Pazartesi-Cuma) resample ederek türetir. Haftalık teyit için ekstra
    API çağrısı gerekmediğinden tarama hızını etkilemez.
    """
    if df_gunluk is None or df_gunluk.empty:
        return pd.DataFrame()
    try:
        gerekli = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df_gunluk.columns for c in gerekli):
            return pd.DataFrame()
        df_w = df_gunluk[gerekli].resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        return df_w
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False) # Veriyi 24 saatte bir günceller, API'yi yormaz
def anomali_tespit_et(df):
    """
    Son 100 günlük veriye bakarak son gündeki hareketin anomali olup olmadığını test eder.
    """
    # İlgili özellikleri seç (Örn: Hacim değişimi, Kapanış değişimi, RSI vb.)
    features = df[['Close', 'Volume']].pct_change().dropna()
    
    if len(features) < 20:
        return "Veri yetersiz"
        
    # contamination=0.02: Verinin %2'sini "anormal" kabul edecek şekilde eğit
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    features['Anomaly'] = iso_forest.fit_predict(features)
    
    # Son günün anomali durumu (-1 = Anomali, 1 = Normal)
    son_durum = features['Anomaly'].iloc[-1]
    
    if son_durum == -1:
        return "⚠️ RİSKLİ: Fiyat/Hacim hareketlerinde anomali tespit edildi!"
    return "✅ Piyasaya uygun, normal hareket."
