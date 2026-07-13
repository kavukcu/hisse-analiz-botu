import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import logging
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor

# ==========================================================
# 1. AYARLAR VE OTURUM
# ==========================================================
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v101")

# ==========================================================
# 2. TÜM HESAPLAMA VE İNDİKATÖR FONKSİYONLARI 
# (Hataları önlemek için hepsi en üste taşındı)
# ==========================================================

def sma(series, period):
    return series.rolling(period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def hma(series, period):
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma1 = wma(series, half)
    wma2 = wma(series, period)
    diff = 2 * wma1 - wma2
    return wma(diff, sqrt)

def kama(series, period=10, fast=2, slow=30):
    change = abs(series - series.shift(period))
    volatility = abs(series.diff()).rolling(period).sum()
    er = change / volatility
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
    return kama

def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    data = df.copy()
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_value = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = (100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_value)
    minus_di = (100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_value)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx_value = dx.ewm(alpha=1/period, adjust=False).mean()
    data["PLUS_DI"] = plus_di
    data["MINUS_DI"] = minus_di
    data["ADX"] = adx_value

    # ======== V101 FEATURES (Kopuk blok buraya entegre edildi) ========
    data["ADX_Strong"] = data["ADX"] > 25
    data["ADX_VeryStrong"] = data["ADX"] > 40
    data["ADX_Rising"] = (data["ADX"] > data["ADX"].shift(1))
    data["ADX_Falling"] = (data["ADX"] < data["ADX"].shift(1))
    data["ADX_Bull"] = (data["PLUS_DI"] > data["MINUS_DI"])
    data["ADX_Bear"] = (data["PLUS_DI"] < data["MINUS_DI"])
    data["ADX_Slope"] = data["ADX"].diff()
    data["Trend_Acceleration"] = (data["ADX_Slope"] > 0)
    
    return data

def supertrend(df, period=10, multiplier=3):
    data = df.copy()
    atr_values = atr(data, period)
    hl2 = (data["High"] + data["Low"]) / 2
    upperband = hl2 + multiplier * atr_values
    lowerband = hl2 - multiplier * atr_values
    st = pd.Series(index=data.index, dtype=float)
    direction = pd.Series(index=data.index, dtype=int)
    st.iloc[0] = lowerband.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(data)):
        if data["Close"].iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif data["Close"].iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1:
                lowerband.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[i-1])
            else:
                upperband.iloc[i] = min(upperband.iloc[i], upperband.iloc[i-1])
        st.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]
    data["Supertrend"] = st
    data["ST_Direction"] = direction
    return data

def calculate_ichimoku(df):
    high9 = df["High"].rolling(9).max()
    low9 = df["Low"].rolling(9).min()
    df["Tenkan"] = (high9 + low9) / 2
    high26 = df["High"].rolling(26).max()
    low26 = df["Low"].rolling(26).min()
    df["Kijun"] = (high26 + low26) / 2
    df["SenkouA"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    high52 = df["High"].rolling(52).max()
    low52 = df["Low"].rolling(52).min()
    df["SenkouB"] = ((high52 + low52) / 2).shift(26)
    df["Chikou"] = df["Close"].shift(-26)
    df["CloudTop"] = df[["SenkouA", "SenkouB"]].max(axis=1)
    df["CloudBottom"] = df[["SenkouA", "SenkouB"]].min(axis=1)
    df["PriceAboveCloud"] = df["Close"] > df["CloudTop"]
    df["PriceBelowCloud"] = df["Close"] < df["CloudBottom"]
    df["TenkanBull"] = df["Tenkan"] > df["Kijun"]
    df["CloudBull"] = df["SenkouA"] > df["SenkouB"]
    return df

def calculate_obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df

def calculate_mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    df["MFI"] = 100 - (100 / (1 + ratio))
    return df

def calculate_cci(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["CCI"] = (tp - sma) / (0.015 * mad)
    return df

def calculate_williams_r(df, period=14):
    hh = df["High"].rolling(period).max()
    ll = df["Low"].rolling(period).min()
    df["WilliamsR"] = -100 * (hh - df["Close"]) / (hh - ll)
    return df

def trend_score(df):
    score = 0
    if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]: score += 15
    if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]: score += 20
    if df["ST_Direction"].iloc[-1] == 1: score += 15
    if df["Close"].iloc[-1] > df["Supertrend"].iloc[-1]: score += 10
    if df["ADX_Strong"].iloc[-1]: score += 10
    if df["ADX_Bull"].iloc[-1]: score += 10
    if df["Trend_Acceleration"].iloc[-1]: score += 5
    if "CloudTop" in df.columns and df["Close"].iloc[-1] > df["CloudTop"].iloc[-1]: score += 10
    if "OBV" in df.columns and df["OBV"].iloc[-1] > df["OBV"].rolling(20).mean().iloc[-1]: score += 5
    return min(score, 100)

def ai_signal(df):
    score = trend_score(df)
    if score >= 85: return "🟢 STRONG BUY"
    elif score >= 70: return "🟢 BUY"
    elif score >= 55: return "🟡 HOLD"
    elif score >= 40: return "🟠 WEAK"
    else: return "🔴 SELL"

def trend_strength(adx_value):
    if adx_value < 20: return "Weak"
    elif adx_value < 25: return "Developing"
    elif adx_value < 40: return "Strong"
    elif adx_value < 50: return "Very Strong"
    else: return "Extreme"

@st.cache_data(ttl=300, show_spinner=False)
def veri_yukle(ticker, start, end):
    for _ in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, session=oturum, progress=False, auto_adjust=True, threads=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            gerekli=["Open","High","Low","Close","Volume"]
            if df.empty or any(c not in df.columns for c in gerekli): raise ValueError("Eksik veya boş veri")
            return df.dropna()
        except Exception as e:
            logging.warning(f"Veri indirilemedi: {e}")
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: return yf.Ticker(ticker, session=oturum).info
    except: return {}

def mum_formasyonlarini_bul(df):
    df_f = df.copy()
    govde = abs(df_f['Close'] - df_f['Open'])
    mum_boyu = df_f['High'] - df_f['Low']
    df_f['Doji'] = govde <= (mum_boyu * 0.1)
    df_f['Bullish_Engulfing'] = (df_f['Close'].shift(1) < df_f['Open'].shift(1)) & (df_f['Open'] < df_f['Close'].shift(1)) & (df_f['Close'] > df_f['Open'].shift(1))
    df_f['Bearish_Engulfing'] = (df_f['Close'].shift(1) > df_f['Open'].shift(1)) & (df_f['Open'] > df_f['Close'].shift(1)) & (df_f['Close'] < df_f['Open'].shift(1))
    alt_golge = df_f[['Close', 'Open']].min(axis=1) - df_f['Low']
    ust_golge = df_f['High'] - df_f[['Close', 'Open']].max(axis=1)
    df_f['Hammer'] = (alt_golge > (govde * 2)) & (ust_golge < (govde * 0.2)) & (govde > 0)
    return df_f

def grafik_formasyon_bul(df, window=10, tolerans=0.03):
    try:
        df_form = df.copy()
        df_form['Local_Max'] = df_form['High'] == df_form['High'].rolling(window=window*2+1, center=True).max()
        df_form['Local_Min'] = df_form['Low'] == df_form['Low'].rolling(window=window*2+1, center=True).min()
        ikili_tepeler = []
        ikili_dipler = []
        max_idx = df_form[df_form['Local_Max']].index
        min_idx = df_form[df_form['Local_Min']].index
        
        for i in range(1, len(max_idx)):
            f1, f2 = df_form.loc[max_idx[i-1], 'High'], df_form.loc[max_idx[i], 'High']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (max_idx[i] - max_idx[i-1]).days
                if 5 < zaman_farki < 90: ikili_tepeler.append((max_idx[i-1], max_idx[i], f1, f2))
                    
        for i in range(1, len(min_idx)):
            f1, f2 = df_form.loc[min_idx[i-1], 'Low'], df_form.loc[min_idx[i], 'Low']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (min_idx[i] - min_idx[i-1]).days
                if 5 < zaman_farki < 90: ikili_dipler.append((min_idx[i-1], min_idx[i], f1, f2))
                    
        return ikili_tepeler, ikili_dipler
    except:
        return [], []

def smc_hesapla(df):
    df_smc = df.copy()
    df_smc['FVG_Bullish'] = (df_smc['Low'] > df_smc['High'].shift(2)) & (df_smc['Close'].shift(1) > df_smc['Open'].shift(1))
    df_smc['FVG_Bearish'] = (df_smc['High'] < df_smc['Low'].shift(2)) & (df_smc['Close'].shift(1) < df_smc['Open'].shift(1))
    return df_smc

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

def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
        if not news_data: return []
        olumlu = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma", "kâr", "temettü", "bullish", "breakout"]
        olumsuz = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza", "risk", "zayıf", "bearish"]
        sonuclar = []
        for n in news_data[:5]:
            metin = (str(n.get('title', '')) + " " + str(n.get('summary', ''))).lower()
            olumlu_skor = sum(1 for k in olumlu if k in metin)
            olumsuz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if olumlu_skor > olumsuz_skor else ("🔴 OLUMSUZ" if olumsuz_skor > olumlu_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: return []

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

def makine_ogrenmesi_tahmin(df, gelecek_gun=30):
    df_ml = df[['Close']].copy()
    df_ml['Lag1'] = df_ml['Close'].shift(1)
    df_ml['Lag2'] = df_ml['Close'].shift(2)
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml.dropna(inplace=True)
    
    X = df_ml[['Lag1', 'Lag2', 'SMA_10']]
    y = df_ml['Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    tahminler = []
    son_satir = df_ml.iloc[-1]
    lag1 = son_satir['Close']
    lag2 = son_satir['Lag1']
    sma_10 = son_satir['SMA_10']
    
    for _ in range(gelecek_gun):
        pred = model.predict([[lag1, lag2, sma_10]])[0]
        tahminler.append(pred)
        lag2 = lag1
        lag1 = pred
        sma_10 = (sma_10 * 9 + pred) / 10
        
    tarihler = [df.index[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
    return tarihler, tahminler

def python_istatistik_analizi(df):
    df_stats = df.copy()
    df_stats['Log_Return'] = np.log(df_stats['Close'] / df_stats['Close'].shift(1))
    volatilite = df_stats['Log_Return'].std() * np.sqrt(252)
    getiri = df_stats['Log_Return'].mean() * 252
    sharpe = (getiri - 0.40) / volatilite
    var_95 = np.percentile(df_stats['Log_Return'].dropna(), 5)
    return {
        "Yıllık Volatilite": f"%{round(volatilite * 100, 2)}",
        "Sharpe Oranı": round(sharpe, 2),
        "Günlük VaR (%95)": f"%{round(var_95 * 100, 2)}"
    }

# ==========================================================
# 3. UYGULAMA ARAYÜZÜ VE ANA MOTOR (MAIN)
# ==========================================================

st.title("👁️ Pro Küresel Yatırım Terminali v101 (SMC, Fibo & Grafik Formasyonları)")

st.sidebar.header("🌍 Küresel Piyasa Ayarları")
piyasa_tipi = st.sidebar.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "Amerikan Borsası (ABD)", "Kripto Para"])

if piyasa_tipi == "Borsa İstanbul (BIST)":
    varsayilan_hisse = "THYAO.IS"
    tarama_listesi = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "GARAN.IS", "SASA.IS"]
elif piyasa_tipi == "Amerikan Borsası (ABD)":
    varsayilan_hisse = "AAPL"
    tarama_listesi = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
else:
    varsayilan_hisse = "BTC-USD"
    tarama_listesi = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

hisse_kodu = st.sidebar.text_input("Varlık Kodu:", value=varsayilan_hisse).upper()
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=730)) 
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Kurumsal teknik analiz verileri hesaplanıyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis).copy()
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # --- TEMEL İNDİKATÖRLER ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['EMA200'] = ema(df['Close'], 200)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = ema(df['Close'],12)
    df['EMA_26'] = ema(df['Close'],26)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    min_val = df['RSI'].rolling(window=14).min()
    max_val = df['RSI'].rolling(window=14).max()
    df['Stoch_RSI'] = (df['RSI'] - min_val) / (max_val - min_val)
    df['Stoch_RSI_K'] = df['Stoch_RSI'].rolling(window=3).mean() * 100
    df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(window=3).mean()

    df['True_Range'] = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
    df['ATR_14'] = df['True_Range'].rolling(14).mean()
    df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    df = smc_hesapla(df)

    # --- V101 PROFESSIONAL INDICATORS ---
    df = adx(df)
    df = supertrend(df)
    df = calculate_ichimoku(df)
    df = calculate_obv(df)
    df = calculate_mfi(df)
    df = calculate_cci(df)
    df = calculate_williams_r(df)
    trend = trend_score(df)
    signal = ai_signal(df)
    
    # --- ÜST METRİKLER ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Trend Score", f"{trend}/100")
    with col2: st.metric("AI Signal", signal)
    with col3: st.metric("ADX", f"{df['ADX'].iloc[-1]:.1f}")
    with col4: st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    # --- SEKMELER ---
    tabs = st.tabs([
        "📈 SMC & Quant Fiyat Hareketi", "🔍 Akıllı Asenkron Radar", "💼 Cüzdan & Akıllı Stop", 
        "🏢 Temel & Temettü", "📰 Haber", "📊 Isı Haritası", 
        "⚙️ Backtest", "🎲 Risk Simülasyonu", "🛠️ Sistem Durumu", "🧬 Python İstatistik"
    ])

    with tabs[0]:
        st.subheader("📈 Kurumsal Quant Grafiği & Likidite Analizi")
        c_ayar1, c_ayar2, c_ayar3 = st.columns(3)
        with c_ayar1:
            goster_vpvr = st.checkbox("📊 Hacim Profili (VPVR)", value=True)
            goster_smc = st.checkbox("🏦 FVG & Likidite Boşlukları (SMC)", value=True)
            goster_fibo = st.checkbox("📐 Altın Oran (Fibonacci)", value=True)
        with c_ayar2:
            goster_grafik_formasyon = st.checkbox("📉 İkili Tepe/Dip (Makro)", value=True)
            goster_formasyon = st.checkbox("🕯️ Mum Formasyonları (Mikro)", value=False)
        with c_ayar3:
            goster_vwap = st.checkbox("⚖️ VWAP (Kurumsal Maliyet)", value=False)
            goster_ai = st.checkbox("🤖 Random Forest AI Tahmini", value=True)
            
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Fiyat", increasing_line_color="#00ff66", decreasing_line_color="#ff4444", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Supertrend"], name="Supertrend", line=dict(color="lime", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Tenkan"], name="Tenkan", line=dict(color="cyan", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Kijun"], name="Kijun", line=dict(color="orange", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["CloudTop"], line=dict(color="rgba(0,255,0,0)"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["CloudBottom"], fill="tonexty", fillcolor="rgba(0,255,0,0.15)", line=dict(color="rgba(0,255,0,0)"), name="Ichimoku Cloud"), row=1, col=1)

        if goster_vpvr:
            hacim_bölümleri, fiyat_araliklari = np.histogram(df['Close'].dropna(), bins=40, weights=df['Volume'].dropna())
            bölüm_merkezleri = (fiyat_araliklari[:-1] + fiyat_araliklari[1:]) / 2
            max_hacim = hacim_bölümleri.max()
            sure_uzunlugu = df.index[-1] - df.index[0]
            x_koordinatlari = [df.index[0] + sure_uzunlugu * 0.3 * (v / max_hacim) for v in hacim_bölümleri]
            for i in range(len(bölüm_merkezleri)):
                fig.add_shape(type="line", x0=df.index[0], y0=bölüm_merkezleri[i], x1=x_koordinatlari[i], y1=bölüm_merkezleri[i], line=dict(color="rgba(100, 150, 255, 0.4)", width=4), row=1, col=1)
            poc_index = np.argmax(hacim_bölümleri)
            poc_fiyat = bölüm_merkezleri[poc_index]
            fig.add_hline(y=poc_fiyat, line_dash="solid", line_color="red", annotation_text="POC", row=1, col=1)

        if goster_smc:
            son = len(df) - 1
            for i in range(2, son + 1):
                x0 = max(0, i - 2)
                x1 = min(i + 5, son)
                if bool(df["FVG_Bullish"].iloc[i]):
                    fig.add_shape(type="rect", x0=df.index[x0], y0=float(df["High"].iloc[i-2]), x1=df.index[x1], y1=float(df["Low"].iloc[i]), fillcolor="rgba(0,255,0,0.20)", line=dict(width=0), layer="below", row=1, col=1)
                elif bool(df["FVG_Bearish"].iloc[i]):
                    fig.add_shape(type="rect", x0=df.index[x0], y0=float(df["Low"].iloc[i-2]), x1=df.index[x1], y1=float(df["High"].iloc[i]), fillcolor="rgba(255,0,0,0.20)", line=dict(width=0), layer="below", row=1, col=1)

        if goster_fibo:
            max_fiyat = df['High'].max()
            min_fiyat = df['Low'].min()
            fark = max_fiyat - min_fiyat
            seviyeler = {0: "100%", 0.236: "76.4%", 0.382: "61.8%", 0.5: "50%", 0.618: "38.2%", 0.786: "21.4%", 1: "0%"}
            renkler = ['#ff0000', '#ff9900', '#ffff00', '#33cc33', '#00ffcc', '#cc33ff', '#999999']
            for i, (level, oran) in enumerate(seviyeler.items()):
                fiyat_seviyesi = max_fiyat - (fark * level)
                if level == 0.618:
                    fig.add_hline(y=fiyat_seviyesi, line_dash="solid", line_width=2, line_color="#00ffcc", annotation_text=f"⭐ {oran}", row=1, col=1)
                else:
                    fig.add_hline(y=fiyat_seviyesi, line_dash="dash", line_width=1, line_color=renkler[i], annotation_text=f"Fibo {oran}", row=1, col=1)

        if goster_grafik_formasyon:
            ikili_tepeler, ikili_dipler = grafik_formasyon_bul(df)
            for tepe in ikili_tepeler:
                fig.add_shape(type="line", x0=tepe[0], y0=tepe[2], x1=tepe[1], y1=tepe[3], line=dict(color="red", width=3, dash="dot"), row=1, col=1)
            for dip in ikili_dipler:
                fig.add_shape(type="line", x0=dip[0], y0=dip[2], x1=dip[1], y1=dip[3], line=dict(color="green", width=3, dash="dot"), row=1, col=1)

        if goster_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)

        if goster_formasyon:
            df_form = mum_formasyonlarini_bul(df)
            yutan_boga = df_form[df_form['Bullish_Engulfing']]
            fig.add_trace(go.Scatter(x=yutan_boga.index, y=yutan_boga['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00ff00', size=12), name='Yutan Boğa'), row=1, col=1)

        if goster_ai:
            tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
            fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="RF Tahmini", line=dict(color='magenta', width=3, dash='dot')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#2962FF')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Sinyal", line=dict(color='#FF6D00')), row=2, col=1)
        hist_colors = np.where(df['MACD_Hist'] < 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram", marker_color=hist_colors), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_K'], name="%K", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_D'], name="%D", line=dict(color='orange')), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=1000, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader(f"⚡ {piyasa_tipi} Multi-Threading Hızlı Radar")
        if piyasa_tipi == "Borsa İstanbul (BIST)":
            tarama_modu = st.radio("Tarama Modu Seçin:", [
                "🟢 Aşırı Satım Radarı (RSI < 35)", "🔥 Hacim Patlaması Radarı", "💼 Temel Analiz Radarı (Düşük F/K)", "☁️ Ichimoku Kumo Kırılımı", "⚔️ Golden Cross"
            ])
            def tek_hisse_tara(hisse, mod):
                try:
                    temiz_ad = hisse.replace(".IS", "")
                    if "Temel Analiz" in mod:
                        s_info = yf.Ticker(hisse).info
                        fk = s_info.get('trailingPE', 999)
                        pddd = s_info.get('priceToBook', 999)
                        if isinstance(fk, (int, float)) and 0 < fk < 10 and 0 < pddd < 3:
                            return {"Hisse Kodu": temiz_ad, "Fiyat": s_info.get('currentPrice', 0), "Değer": f"F/K: {round(fk, 2)}", "Durum": "💼 Ucuz Çarpanlar"}
                    else:
                        t_df = yf.download(hisse, start=datetime.today() - timedelta(days=365), end=datetime.today(), progress=False)
                        if t_df.empty: return None
                        if isinstance(t_df.columns, pd.MultiIndex): t_df.columns = t_df.columns.droplevel(1)
                        if len(t_df) > 52:
                            son_kap = float(t_df['Close'].iloc[-1])
                            if "Aşırı Satım" in mod:
                                delta_h = t_df['Close'].diff()
                                gain_h = delta_h.where(delta_h > 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                loss_h = -delta_h.where(delta_h < 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                rs_h = gain_h / (loss_h + 1e-9)
                                rsi_son = (100 - (100 / (1 + rs_h))).iloc[-1]
                                if rsi_son < 35: return {"Hisse Kodu": temiz_ad, "Fiyat": round(son_kap,2), "Değer": f"RSI: {round(rsi_son, 1)}", "Durum": "🟢 Aşırı Satım"}
                            elif "Hacim" in mod:
                                if t_df['Volume'].iloc[-1] > (t_df['Volume'].iloc[-21:-1].mean() * 1.8):
                                    return {"Hisse Kodu": temiz_ad, "Fiyat": round(son_kap,2), "Değer": "Hacim Patlaması", "Durum": "🔥 Yüksek Hacim"}
                            elif "Golden Cross" in mod and len(t_df) > 200:
                                if t_df['Close'].rolling(50).mean().iloc[-1] > t_df['Close'].rolling(200).mean().iloc[-1] and t_df['Close'].rolling(50).mean().iloc[-2] <= t_df['Close'].rolling(200).mean().iloc[-2]:
                                    return {"Hisse Kodu": temiz_ad, "Fiyat": round(son_kap,2), "Değer": "Kesişim", "Durum": "⚔️ Golden Cross"}
                except: return None
                return None

            if st.button("🚀 Hızlı Asenkron Radarı Çalıştır"):
                with st.spinner("🚀 BİST Hisseleri Taranıyor..."):
                    bist30_hisseler = ["AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS", "ENKAI.IS", "KRDMD.IS", "PETKM.IS"] 
                    firsatlar = []
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        sonuclar = executor.map(lambda h: tek_hisse_tara(h, tarama_modu), bist30_hisseler)
                        for sonuc in sonuclar:
                            if sonuc is not None: firsatlar.append(sonuc)
                    if firsatlar:
                        st.success(f"✅ Tarama tamamlandı! {len(firsatlar)} adet hisse bulundu:")
                        st.dataframe(pd.DataFrame(firsatlar).set_index("Hisse Kodu"), use_container_width=True)
                    else: st.warning("📉 Seçilen kritere uyan varlık tespit edilemedi.")

    with tabs[2]:
        st.subheader("📊 Canlı Varlık Portföyüm ve ATR Destekli Fiyat Alarmları")
        c1, c2 = st.columns([2, 1])
        with c1:
            if 'portfoy_verisi' not in st.session_state:
                st.session_state.portfoy_verisi = pd.DataFrame([{"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0}])
            guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
            st.session_state.portfoy_verisi = guncel_portfoy
            if st.button("Portföyü Hesapla"):
                top_mal = 0.0; top_deg = 0.0
                for index, row in guncel_portfoy.iterrows():
                    kod = str(row["Varlık"]).upper(); mal = float(row["Maliyet"]); lot = float(row["Lot"])
                    if kod and lot > 0:
                        try:
                            c_veri = yf.download(kod, period="1d", progress=False, session=oturum)
                            if isinstance(c_veri.columns, pd.MultiIndex): c_veri.columns = c_veri.columns.droplevel(1)
                            g_fiyat = float(c_veri['Close'].iloc[-1])
                            top_mal += (mal * lot); top_deg += (g_fiyat * lot)
                        except: pass
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Toplam Maliyet", f"{round(top_mal, 2)}")
                cc2.metric("Güncel Değer", f"{round(top_deg, 2)}")
                cc3.metric("Net Kâr", f"{round(top_deg - top_mal, 2)}", f"%{round(((top_deg - top_mal) / top_mal) * 100, 2) if top_mal > 0 else 0}")
        with c2:
            tavsiye_stop = round(float(df['Close'].iloc[-1]) - (float(df['ATR_14'].iloc[-1]) * 2), 2)
            st.info(f"💡 Tavsiye edilen teknik Stop-Loss: **{tavsiye_stop}**")
            alarm_fiyat = st.number_input(f"{hisse_kodu} Tetikleme Fiyatı:", value=tavsiye_stop)
            if st.button("Alarmı Kur"): st.success("Alarm kuruldu!")

    with tabs[3]:
        st.subheader(f"🏢 {info.get('longName', hisse_kodu)} Temel Veriler & Temettü")
        c1, c2, c3 = st.columns(3)
        c1.metric("F/K Oranı", info.get('trailingPE', '-'))
        c2.metric("PD/DD", info.get('priceToBook', '-'))
        c3.metric("Piyasa Değeri", info.get('marketCap', '-'))

    with tabs[4]:
        st.subheader("📰 Küresel Haber Duygu Analizi")
        for h in haber_duygu_analizi(hisse_kodu):
            with st.expander(f"{h['duygu']} | {h['baslik']} ({h['kaynak']})"):
                st.markdown(f"[Habere Git]({h['link']})")

    with tabs[5]:
        st.subheader(f"📊 {piyasa_tipi} Korelasyon Matrisi")
        if st.button("Isı Haritasını Oluştur"):
            korelasyon_df = pd.DataFrame()
            for ticker in tarama_listesi[:6]:
                tmp_df = yf.download(ticker, period="6mo", progress=False, session=oturum)
                if isinstance(tmp_df.columns, pd.MultiIndex): tmp_df.columns = tmp_df.columns.droplevel(1)
                if not tmp_df.empty: korelasyon_df[ticker] = tmp_df['Close']
            st.plotly_chart(px.imshow(korelasyon_df.corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

    with tabs[6]:
        st.subheader("⚙️ Strateji Testi (Backtest): SMA 20 vs SMA 50")
        bt_sonuc = backtest_motoru(df)
        if not bt_sonuc.empty:
            son_piyasa = bt_sonuc['Piyasa_Kumulatif'].iloc[-1] - 100
            son_strateji = bt_sonuc['Strateji_Kumulatif'].iloc[-1] - 100
            c1, c2 = st.columns(2)
            c1.metric("Alıp Bekleseydin", f"%{round(son_piyasa, 2)}")
            c2.metric("Strateji ile Alsatsaydın", f"%{round(son_strateji, 2)}", delta=round(son_strateji - son_piyasa, 2))
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Piyasa_Kumulatif'], name="Piyasa Getirisi", line=dict(color='white')))
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Strateji_Kumulatif'], name="Strateji Getirisi", line=dict(color='green', width=3)))
            fig_bt.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bt, use_container_width=True)

    with tabs[7]:
        st.subheader("🎲 Monte Carlo Risk Simülasyonu (Gelecek 30 Gün)")
        if st.button("Simülasyonu Başlat"):
            sim_verisi = monte_carlo_simulasyonu(df)
            fig_sim = go.Figure()
            for i in range(sim_verisi.shape[1]):
                fig_sim.add_trace(go.Scatter(y=sim_verisi[:, i], mode='lines', line=dict(width=1), showlegend=False))
            fig_sim.update_layout(template="plotly_dark")
            st.plotly_chart(fig_sim, use_container_width=True)

    with tabs[8]:
        st.subheader("🛠️ Terminal Entegrasyon Durumu")
        st.success("🤖 Multi-Threading Radar: Aktif")
        st.success("🧠 Random Forest Engine: Yüklendi")
        st.success("📉 İkili Tepe / Dip Algoritması: Aktif")
        st.info("Terminal v101 (God Mode Master Sürümü) Kararlı Modda Çalışıyor.")

    with tabs[9]:
        st.subheader("🧬 Python İleri İstatistik Analizi")
        if st.button("İstatistikleri Hesapla"):
            stats = python_istatistik_analizi(df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Yıllık Volatilite", stats["Yıllık Volatilite"])
            col2.metric("Sharpe Oranı", stats["Sharpe Oranı"])
            col3.metric("VaR (%95)", stats["Günlük VaR (%95)"])

else:
    st.error("Veri çekilemedi. Bağlantınızı veya hisse kodunu kontrol edin.")