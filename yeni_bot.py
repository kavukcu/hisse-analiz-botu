import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LinearRegression

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v17.0")
st.title("👁️ Pro Küresel Yatırım Terminali v17.0 (Fully Fixed)")

# --- VERİ GÜVENLİĞİ VE SÜTUN DÜZLEŞTİRİCİ MOTOR ---
def sutunlari_duzlestir(df_downloaded):
    if df_downloaded is None or df_downloaded.empty:
        return df_downloaded
    if isinstance(df_downloaded.columns, pd.MultiIndex):
        if any(col in ['Close', 'Open', 'High', 'Low', 'Volume'] for col in df_downloaded.columns.get_level_values(0)):
            df_downloaded.columns = df_downloaded.columns.get_level_values(0)
        else:
            df_downloaded.columns = df_downloaded.columns.get_level_values(1)
    df_downloaded.columns = [str(c).strip() for c in df_downloaded.columns]
    return df_downloaded

# --- TELEGRAM VE OTOMASYON ---
def telegram_gonder(mesaj):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mesaj}"
        requests.get(url)
    except:
        pass

# --- VERİ MOTORLARI VE FONKSİYONLAR ---
@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    try:
        df_dl = yf.download(ticker, start=start, end=end, session=oturum)
        return sutunlari_duzlestir(df_dl)
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}

def monte_carlo_simulasyonu(df, gun_sayisi=30, sim_sayisi=100):
    getiriler = df['Close'].pct_change().dropna()
    ortalama_getiri = getiriler.mean()
    volatilite = getiriler.std()
    son_fiyat = float(df['Close'].iloc[-1])
    
    simulasyonlar = np.zeros((gun_sayisi, sim_sayisi))
    for i in range(sim_sayisi):
        rastgele_getiriler = np.random.normal(ortalama_getiri, volatilite, gun_sayisi)
        simulasyonlar[:, i] = son_fiyat * (1 + rastgele_getiriler).cumprod()
    return simulasyonlar

def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
        if not news_data: 
            return []
        olumlu = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma", "kâr", "temettü", "bullish", "breakout"]
        olumsuz = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza", "risk", "zayıf", "bearish"]
        sonuclar = []
        for n in news_data[:5]:
            metin = (n.get('title', '') + " " + (n.get('summary') or '')).lower()
            olumlu_skor = sum(1 for k in olumlu if k in metin)
            olumsuz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if olumlu_skor > olumsuz_skor else ("🔴 OLUMSUZ" if olumsuz_skor > olumlu_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: 
        return []

def backtest_motoru(df, kisa_periyot=20, uzun_periyot=50):
    bt_df = df[['Close']].copy()
    bt_df['Kisa_SMA'] = bt_df['Close'].rolling(window=kisa_periyot).mean()
    bt_df['Uzun_SMA'] = bt_df['Close'].rolling(window=uzun_periyot).mean()
    bt_df.dropna(inplace=True)
    
    bt_df['Sinyal'] = np.where(bt_df['Kisa_SMA'] > bt_df['Uzun_SMA'], 1, 0)
    bt_df['Günlük_Getiri'] = bt_df['Close'].pct_change()
    bt_df['Strateji_Getirisi'] = bt_df['Günlük_Getiri'] * bt_df['Sinyal'].shift(1)
    bt_df['Piyasa_Kumulatif'] = (1 + bt_df['Günlük_Getiri'].fillna(0)).cumprod() * 100
    bt_df['Strateji_Kumulatif'] = (1 + bt_df['Strateji_Getirisi'].fillna(0)).cumprod() * 100
    
    bt_df['Zirve'] = bt_df['Strateji_Kumulatif'].cummax()
    bt_df['Drawdown'] = (bt_df['Strateji_Kumulatif'] - bt_df['Zirve']) / bt_df['Zirve'] * 100
    return bt_df

def makine_ogrenmesi_tahmin(df, gelecek_gun=30):
    df_ml = df.copy()
    df_ml['Gun'] = np.arange(len(df_ml))
    
    X = df_ml[['Gun']]
    y = df_ml['Close']
    model = LinearRegression()
    model.fit(X, y)
    
    son_gun = df_ml['Gun'].iloc[-1]
    gelecek_X = np.array([[son_gun + i] for i in range(1, gelecek_gun + 1)])
    tahminler = model.predict(gelecek_X)
    
    son_tarih = df.index[-1]
    tarihler = [son_tarih + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
    return tarihler, tahminler

def atr_hesapla(df, periyot=14):
    df_atr = df.copy()
    df_atr['H-L'] = abs(df_atr['High'] - df_atr['Low'])
    df_atr['H-PC'] = abs(df_atr['High'] - df_atr['Close'].shift(1))
    df_atr['L-PC'] = abs(df_atr['Low'] - df_atr['Close'].shift(1))
    df_atr['TR'] = df_atr[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return df_atr['TR'].rolling(window=periyot).mean()

def mum_formasyonu_bul(df):
    df_mum = df.copy()
    df_mum['Govde'] = abs(df_mum['Close'] - df_mum['Open'])
    df_mum['Ust_Golge'] = df_mum['High'] - df_mum[['Open', 'Close']].max(axis=1)
    df_mum['Alt_Golge'] = df_mum[['Open', 'Close']].min(axis=1) - df_mum['Low']
    
    formasyonlar = []
    for i in range(1, len(df_mum)):
        if (df_mum['Close'].iloc[i-1] < df_mum['Open'].iloc[i-1]) and \
           (df_mum['Close'].iloc[i] > df_mum['Open'].iloc[i]) and \
           (df_mum['Open'].iloc[i] <= df_mum['Close'].iloc[i-1]) and \
           (df_mum['Close'].iloc[i] >= df_mum['Open'].iloc[i-1]):
            formasyonlar.append((df_mum.index[i], float(df_mum['Low'].iloc[i]), "Yutan Boğa 🐂"))
            
        elif (df_mum['Alt_Golge'].iloc[i] > 2 * df_mum['Govde'].iloc[i]) and \
             (df_mum['Ust_Golge'].iloc[i] < 0.2 * df_mum['Govde'].iloc[i]) and \
             (df_mum['Govde'].iloc[i] > 0): 
            formasyonlar.append((df_mum.index[i], float(df_mum['Low'].iloc[i]), "Çekiç 🔨"))
    return formasyonlar

# --- SİDEBAR VE PİYASA SEÇİMİ ---
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
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=365))
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Yapay zeka verileri analiz ediyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if df is not None and not df.empty:
    df['SMA_20'] = df