import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
from sklearn.ensemble import RandomForestRegressor
import time

# --- OTURUM ---
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

st.set_page_config(layout="wide", page_title="Otonom Bot v13.0")
st.title("🧠 Pro Küresel Yatırım Terminali v13.0 (Deep Forest Edition)")

# --- FONKSİYONLAR ---
@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

def makine_ogrenmesi_sinyal(df_subset):
    df_ml = df_subset.copy().reset_index()
    df_ml['Gun'] = np.arange(len(df_ml))
    
    # Özellikleri oluştur
    df_ml['SMA_5'] = df_ml['Close'].rolling(window=5).mean().bfill()
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean().bfill()
    df_ml['Volatilite'] = df_ml['Close'].rolling(window=5).std().bfill()
    
    X = df_ml[['Gun', 'SMA_5', 'SMA_10', 'Volatilite']]
    y = df_ml['Close']
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Yarın tahmini
    son_gun = df_ml['Gun'].iloc[-1]
    son_sma5 = df_ml['SMA_5'].iloc[-1]
    son_sma10 = df_ml['SMA_10'].iloc[-1]
    son_volatilite = df_ml['Volatilite'].iloc[-1]
    son_fiyat = df_ml['Close'].iloc[-1]
    
    gelecek_X = pd.DataFrame({
        'Gun': [son_gun + 1],
        'SMA_5': [son_sma5],
        'SMA_10': [son_sma10],
        'Volatilite': [son_volatilite]
    })
    
    yarin_tahmin = model.predict(gelecek_X)[0]
    beklenen_degisim = ((yarin_tahmin - son_fiyat) / son_fiyat) * 100
    
    if beklenen_degisim > 1.0:
        return "AL", yarin_tahmin
    elif beklenen_degisim < -1.0:
        return "SAT", yarin_tahmin
    else:
        return "BEKLE", yarin_tahmin

# --- SİDEBAR ---
with st.sidebar:
    st.header("🌍 Otonom Ayarlar")
    hisse_kodu = st.text_input("Varlık Kodu:", value="THYAO.IS").upper()
    baslangic = st.date_input("Analiz Başlangıcı:", value=datetime.today() - pd.Timedelta(days=365))
    bitis = st.date_input("Analiz Bitişi:", value=datetime.today())
    
    if 'sanal_bakiye' not in st.session_state:
        st.session_state.sanal_bakiye = 100000.0 

with st.spinner('Model eğitiliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    t1, t2 = st.tabs(["🚀 Otonom Simülatör", "📈 Canlı Grafik"])
    
    with t1:
        if st.button("🚀 Akıllı Botu Başlat"):
            sinyal, tahmin = makine_ogrenmesi_sinyal(df)
            st.metric("Yarınki Tahmin", round(tahmin, 2))
            st.subheader(f"Yapay Zeka Kararı: {sinyal}")

    with t2:
        st.subheader(f"Canlı Grafik - {hisse_kodu}")
        st.line_chart(df['Close'])