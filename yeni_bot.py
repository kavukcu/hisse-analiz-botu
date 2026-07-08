import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from sklearn.linear_model import LinearRegression

# 1. OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="Sandbox Terminal v11.0")
st.title("🧪 Pro Küresel Yatırım Terminali v11.0 (Sandbox Edition)")
st.markdown("*" + "Canlı Veri Akışı, Makine Öğrenmesi Sinyalleri ve Sanal İşlem (Paper Trading) Motoru" + "*")

# --- FONKSİYONLAR ---
@st.cache_data(show_spinner=False, ttl=60) # Veri ömrünü 1 dakikaya düşürdük (Canlı akış için)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

def makine_ogrenmesi_sinyal(df, gelecek_gun=5):
    df_ml = df.copy().reset_index()
    df_ml['Gun'] = np.arange(len(df_ml))
    X = df_ml[['Gun']]
    y = df_ml['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    son_gun = df_ml['Gun'].iloc[-1]
    son_fiyat = df_ml['Close'].iloc[-1]
    
    # Sadece yarınki tahmini al
    gelecek_X = np.array([[son_gun + 1]])
    yarin_tahmin = model.predict(gelecek_X)[0]
    
    # %1'den fazla artış bekleniyorsa AL, %1'den fazla düşüş bekleniyorsa SAT
    beklenen_degisim = ((yarin_tahmin - son_fiyat) / son_fiyat) * 100
    
    if beklenen_degisim > 1.0:
        return "🟢 GÜÇLÜ AL", yarin_tahmin, beklenen_degisim
    elif beklenen_degisim < -1.0:
        return "🔴 GÜÇLÜ SAT", yarin_tahmin, beklenen_degisim
    else:
        return "🟡 BEKLE", yarin_tahmin, beklenen_degisim

# --- PAPER TRADING MOTORU ---
def sanal_islem_yap(varlik, islem_tipi, fiyat, lot, bakiye):
    toplam_tutar = fiyat * lot
    if islem_tipi == "AL":
        if bakiye >= toplam_tutar:
            return True, bakiye - toplam_tutar
        else:
            return False, bakiye
    elif islem_tipi == "SAT":
        return True, bakiye + toplam_tutar
    return False, bakiye

# --- SİDEBAR ---
with st.sidebar:
    st.header("🌍 Sandbox Ayarları")
    piyasa_tipi = st.selectbox("Piyasa Türü:", ["Kripto Para", "Borsa İstanbul (BIST)", "Amerikan Borsası (ABD)"])
    
    if piyasa_tipi == "Kripto Para": hisse_kodu = st.text_input("Varlık Kodu:", value="BTC-USD").upper()
    elif piyasa_tipi == "Borsa İstanbul (BIST)": hisse_kodu = st.text_input("Varlık Kodu:", value="THYAO.IS").upper()
    else: hisse_kodu = st.text_input("Varlık Kodu:", value="AAPL").upper()

    baslangic = st.date_input("Analiz Başlangıcı:", value=datetime.today() - pd.Timedelta(days=365))
    bitis = st.date_input("Analiz Bitişi:", value=datetime.today())
    
    # Sanal Bakiye State
    if 'sanal_bakiye' not in st.session_state:
        st.session_state.sanal_bakiye = 100000.0 # 100.000 TL/USD başlangıç bakiyesi
    if 'sanal_portfoy' not in st.session_state:
        st.session_state.sanal_portfoy = {} # {'BTC-USD': {'lot': 0.5, 'maliyet': 60000}}

    st.divider()
    st.markdown(f"**💰 Sanal Bakiye: {round(st.session_state.sanal_bakiye, 2)}**")

with st.spinner('Canlı piyasa verileri ve makine öğrenmesi modelleri yükleniyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    guncel_fiyat = float(df['Close'].iloc[-1])
    
    t1, t2, t3 = st.tabs(["🧪 Paper Trading (Sanal İşlem)", "🤖 ML Sinyal Motoru", "📈 Canlı Grafik"])

    # --- SEKME 1: PAPER TRADING ---
    with t1:
        st.subheader("💵 Sanal İşlem Laboratuvarı")
        st.write("Sistemin ürettiği sinyalleri risksiz bir şekilde gerçek piyasa verileriyle test et.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("İşlem Gören Varlık", hisse_kodu)
        c2.metric("Güncel Fiyat (Canlı)", round(guncel_fiyat, 2))
        c3.metric("Kullanılabilir Sanal Bakiye", round(st.session_state.sanal_bakiye, 2))
        
        st.divider()
        col_islem1, col_islem2 = st.columns(2)
        
        with col_islem1:
            islem_lot = st.number_input("İşlem Yapılacak Miktar (Lot/Adet):", min_value=0.01, value=1.0)
            toplam_islem_tutari = islem_lot * guncel_fiyat
            st.write(f"Tahmini Tutar: **{round(toplam_islem_tutari, 2)}**")
            
            c_al, c_sat = st.columns(2)
            if c_al.button("🟢 Piyasa Fiyatından AL", use_container_width=True):
                basarili, yeni_bakiye = sanal_islem_yap(hisse_kodu, "AL", guncel_fiyat, islem_lot, st.session_state.sanal_bakiye)
                if basarili:
                    st.session_state.sanal_bakiye = yeni_bakiye
                    if hisse_kodu in st.session_state.sanal_portfoy:
                        eski_lot = st.session_state.sanal_portfoy[hisse_kodu]['lot']
                        eski_maliyet = st.session_state.sanal_portfoy[hisse_kodu]['maliyet']
                        yeni_maliyet = ((eski_lot * eski_maliyet) + (islem_lot * guncel_fiyat)) / (eski_lot + islem_lot)
                        st.session_state.sanal_portfoy[hisse_kodu]['lot'] += islem_lot
                        st.session_state.sanal_portfoy[hisse_kodu]['maliyet'] = yeni_maliyet
                    else:
                        st.session_state.sanal_portfoy[hisse_kodu] = {'lot': islem_lot, 'maliyet': guncel_fiyat}
                    st.success(f"{islem_lot} adet {hisse_kodu} başarıyla alındı.")
                    st.rerun() # Bakiyeyi anında güncellemek için
                else:
                    st.error("Yetersiz Bakiye!")
                    
            if c_sat.button("🔴 Piyasa Fiyatından SAT", use_container_width=True):
                if hisse_kodu in st.session_state.sanal_portfoy and st.session_state.sanal_portfoy[hisse_kodu]['lot'] >= islem_lot:
                    _, yeni_bakiye = sanal_islem_yap(hisse_kodu, "SAT", guncel_fiyat, islem_lot, st.session_state.sanal_bakiye)
                    st.session_state.sanal_bakiye = yeni_bakiye
                    st.session_state.sanal_portfoy[hisse_kodu]['lot'] -= islem_lot
                    
                    if st.session_state.sanal_portfoy[hisse_kodu]['lot'] == 0:
                        del st.session_state.sanal_portfoy[hisse_kodu]
                        
                    st.success(f"{islem_lot} adet {hisse_kodu} başarıyla satıldı.")
                    st.rerun()
                else:
                    st.error(f"Elinizde yeterli {hisse_kodu} bulunmuyor!")

        with col_islem2:
            st.markdown("#### 💼 Açık Sanal Pozisyonlarım")
            if st.session_state.sanal_portfoy:
                portfoy_liste = []
                for kod, data in st.session_state.sanal_portfoy.items():
                    # Gerçekçi pnl için anlık fiyatı çek (basitçe kendi fiyatını alıyoruz test için)
                    anlik_fiyat = guncel_fiyat if kod == hisse_kodu else data['maliyet'] # Farklı hisseler için api isteği gerekir normalde
                    pnl = (anlik_fiyat - data['maliyet']) * data['lot']
                    portfoy_liste.append({
                        "Varlık": kod, "Maliyet": round(data['maliyet'], 2), 
                        "Lot": data['lot'], "Güncel Değer": round(anlik_fiyat * data['lot'], 2),
                        "Kâr/Zarar": round(pnl, 2)
                    })
                st.dataframe(pd.DataFrame(portfoy_liste), use_container_width=True)
            else:
                st.info("Şu an açık pozisyonunuz bulunmamaktadır.")

    # --- SEKME 2: ML SİNYAL MOTORU ---
    with t2:
        st.subheader("🤖 Otonom Makine Öğrenmesi (ML) Karar Motoru")
        st.write("Sistem geçmiş veriyi analiz ederek yarınki fiyat hareketini tahmin eder ve buna göre aksiyon önerir.")
        
        sinyal, yarin_tahmin, beklenen_yuzde = makine_ogrenmesi_sinyal(df)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Güncel Fiyat", round(guncel_fiyat, 2))
        c2.metric("AI Yarınki Tahmin", round(yarin_tahmin, 2), f"%{round(beklenen_yuzde, 2)}")
        
        if "AL" in sinyal:
            c3.success(f"Yapay Zeka Kararı: {sinyal}")
        elif "SAT" in sinyal:
            c3.error(f"Yapay Zeka Kararı: {sinyal}")
        else:
            c3.warning(f"Yapay Zeka Kararı: {sinyal}")

    # --- SEKME 3: GRAFİK ---
    with t3:
        st.subheader(f"Canlı Grafik - {hisse_kodu}")
        df['SMA_20'] = df['Close'].rolling(20).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='cyan')))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Veri çekilemedi. Bağlantınızı veya varlık kodunu kontrol edin.")