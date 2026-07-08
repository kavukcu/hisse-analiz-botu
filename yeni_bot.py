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
st.markdown("*Random Forest Algoritması ile Güçlendirilmiş Yapay Zeka Karar Motoru*")

# --- FONKSİYONLAR ---
@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

def makine_ogrenmesi_sinyal(df_subset):
    # Veri hazırlığı
    df_ml = df_subset.copy().reset_index()
    df_ml['Gun'] = np.arange(len(df_ml))
    
    # Basit özellikleri (features) zenginleştiriyoruz ki Random Forest daha iyi öğrensin
    df_ml['SMA_5'] = df_ml['Close'].rolling(window=5).mean().fillna(method='bfill')
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean().fillna(method='bfill')
    df_ml['Volatilite'] = df_ml['Close'].rolling(window=5).std().fillna(method='bfill')
    
    X = df_ml[['Gun', 'SMA_5', 'SMA_10', 'Volatilite']]
    y = df_ml['Close']
    
    # Yıldız Oyuncu Sahada: Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Yarın için verileri hazırla
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
    
    # Karar mekanizması
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
    if 'sanal_portfoy' not in st.session_state:
        st.session_state.sanal_portfoy = {} 

    st.divider()
    st.markdown(f"**💰 Güncel Bakiye: {round(st.session_state.sanal_bakiye, 2)}**")

with st.spinner('Random Forest Modeli eğitiliyor... (Bu işlem Linear regresyondan biraz daha uzun sürebilir)'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    guncel_fiyat = float(df['Close'].iloc[-1])
    
    t1, t2, t3 = st.tabs(["🔁 Otonom Trade Simülatörü", "🧪 Manuel Sandbox", "📈 Canlı Grafik"])

    # --- SEKME 1: OTONOM SİMÜLATÖR ---
    with t1:
        st.subheader("🔁 Otonom Zaman Makinesi (Deep Forest)")
        st.write("Yapay zeka botunu geçmiş 30 güne göndererek Random Forest zekasıyla al-sat yapmasına izin ver.")
        
        c1, c2 = st.columns(2)
        baslangic_bakiyesi = c1.number_input("Bota Verilecek Başlangıç Bakiyesi:", value=100000)
        islem_gunu = c2.number_input("Kaç Günlük Simülasyon Yapılsın?", value=30, max_value=100)
        
        if st.button("🚀 Akıllı Botu Başlat", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_kutusu = st.empty()
            
            bakiye = baslangic_bakiyesi
            elindeki_lot = 0
            islem_gecmisi = []
            
            toplam_veri = len(df)
            test_baslangic_idx = toplam_veri - islem_gunu
            
            for i in range(islem_gunu):
                guncel_idx = test_baslangic_idx + i
                if guncel_idx >= toplam_veri: break
                
                df_gecmis = df.iloc[:guncel_idx]
                o_gunku_fiyat = float(df_gecmis['Close'].iloc[-1])
                o_gunku_tarih = df.index[guncel_idx-1].strftime('%Y-%m-%d')
                
                # Minimum veri kontrolü (Hareketli ortalamalar için en az 10 gün lazım)
                if len(df_gecmis) < 15:
                    continue
                    
                sinyal, tahmin = makine_ogrenmesi_sinyal(df_gecmis)
                
                islem_notu = f"{o_gunku_tarih} | Fiyat: {round(o_gunku_fiyat,2)} | Sinyal: {sinyal}"
                
                if sinyal == "AL" and bakiye > o_gunku_fiyat:
                    alinacak_lot = bakiye // o_gunku_fiyat
                    bakiye -= alinacak_lot * o_gunku_fiyat
                    elindeki_lot += alinacak_lot
                    islem_notu += f" 🟢 {alinacak_lot} lot ALINDI."
                
                elif sinyal == "SAT" and elindeki_lot > 0:
                    bakiye += elindeki_lot * o_gunku_fiyat
                    islem_notu += f" 🔴 {elindeki_lot} lot SATILDI."
                    elindeki_lot = 0
                else:
                    islem_notu += " 🟡 İŞLEM YOK."
                    
                islem_gecmisi.append(islem_notu)
                
                status_text.markdown(f"**Random Forest Karar Veriyor:** {o_gunku_tarih}... (Kalan Gün: {islem_gunu - i})")
                log_kutusu.code("\n".join(islem_gecmisi[-5:]))
                progress_bar.progress((i + 1) / islem_gunu)
                time.sleep(0.1)
            
            son_fiyat = float(df['Close'].iloc[-1])
            nihai_deger = bakiye + (elindeki_lot * son_fiyat)
            net_kar = nihai_deger - baslangic_bakiyesi
            kar_yuzdesi = (net_kar / baslangic_bakiyesi) * 100
            
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ Simülasyon Tamamlandı!")
            st.divider()
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Başlangıç Bakiyesi", f"{round(baslangic_bakiyesi, 2)}")
            res_c2.metric("Nihai Portföy Değeri", f"{round(nihai_deger, 2)}")
            res_c3.metric("Otonom Botun Kârı", f"{round(net_kar, 2)}", f"%{round(kar_yuzdesi, 2)}")
            
            with st.expander("Detaylı İşlem Loglarını Gör"):
                for log in islem_gecmisi:
                    st.write(log)

    # --- SEKME 2: MANUEL SANDBOX ---
    with t2:
        st.subheader("💵 Manuel İşlem (Sandbox)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Güncel Fiyat", round(guncel_fiyat, 2))
        c2.metric("Sanal Bakiye", round(st.session_state.sanal_bakiye, 2))
        c3.metric("ML Kararı (Yarın İçin)", makine_ogrenmesi_sinyal(df)[0])

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
    st.error("Veri çekilemedi.")
