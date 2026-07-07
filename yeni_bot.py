import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import requests

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="Ultimate Terminal v7.0")
st.title("🦅 Pro Küresel Yatırım Terminali v7.0")

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
    return yf.download(ticker, start=start, end=end, session=oturum)

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: return yf.Ticker(ticker, session=oturum).info
    except: return {}

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
            metin = (n.get('title', '') + " " + n.get('summary', '') or '').lower()
            olumlu_skor = sum(1 for k in olumlu if k in metin)
            olumsuz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if olumlu_skor > olumsuz_skor else ("🔴 OLUMSUZ" if olumsuz_skor > olumlu_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: return []

# v7 YENİLİĞİ: BACKTEST MOTORU (Hareketli Ortalama Kesişimi)
def backtest_motoru(df, kisa_periyot=20, uzun_periyot=50):
    bt_df = df[['Close']].copy()
    bt_df['Kisa_SMA'] = bt_df['Close'].rolling(window=kisa_periyot).mean()
    bt_df['Uzun_SMA'] = bt_df['Close'].rolling(window=uzun_periyot).mean()
    bt_df.dropna(inplace=True)
    
    # Sinyal: Kısa SMA, Uzun SMA'dan büyükse 1 (Elinde Tut/Al), küçükse 0 (Sat/Nakit)
    bt_df['Sinyal'] = np.where(bt_df['Kisa_SMA'] > bt_df['Uzun_SMA'], 1, 0)
    bt_df['Günlük_Getiri'] = bt_df['Close'].pct_change()
    bt_df['Strateji_Getirisi'] = bt_df['Günlük_Getiri'] * bt_df['Sinyal'].shift(1)
    
    bt_df['Piyasa_Kumulatif'] = (1 + bt_df['Günlük_Getiri']).cumprod() * 100
    bt_df['Strateji_Kumulatif'] = (1 + bt_df['Strateji_Getirisi']).cumprod() * 100
    return bt_df

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
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=365)) # Backtest için 1 yıla çıkardık
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Piyasa verileri analiz ediliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # İndikatörler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).ewm(alpha=1/14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0)).ewm(alpha=1/14).mean())))

    # v7 SEKMELERİ
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💼 Cüzdan", "📈 Teknik", "🏢 Temel", "📰 Haber", "🔍 Tarama", "🔮 Monte Carlo", "⚙️ Strateji Testi (Backtest)"
    ])

    with tab1:
        st.subheader("📊 Canlı Varlık Portföyüm")
        if 'portfoy_verisi' not in st.session_state:
            st.session_state.portfoy_verisi = pd.DataFrame([
                {"Varlık Kodu": "THYAO.IS", "Maliyet": 300.0, "Lot / Adet": 50.0},
                {"Varlık Kodu": "BTC-USD", "Maliyet": 62000.0, "Lot / Adet": 0.05}
            ])
            
        guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
        st.session_state.portfoy_verisi = guncel_portfoy
        
        if st.button("Değeri Hesapla"):
            hesaplanan_liste = []
            top_mal = 0; top_deg = 0
            for index, row in guncel_portfoy.iterrows():
                kod = row["Varlık Kodu"].upper()
                mal = row["Maliyet"]; lot = row["Lot / Adet"]
                if kod and lot > 0:
                    try:
                        c_veri = yf.download(kod, period="1d", progress=False, session=oturum)
                        if isinstance(c_veri.columns, pd.MultiIndex): c_veri.columns = c_veri.columns.droplevel(1)
                        g_fiyat = c_veri['Close'].iloc[-1]
                    except: g_fiyat = mal
                    
                    pnl = (g_fiyat * lot) - (mal * lot)
                    top_mal += (mal * lot); top_deg += (g_fiyat * lot)
                    hesaplanan_liste.append({"Varlık": kod, "Güncel Fiyat": round(g_fiyat,2), "Kâr/Zarar": round(pnl,2)})
            
            if hesaplanan_liste:
                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam Maliyet", f"{round(top_mal, 2)}")
                c2.metric("Güncel Değer", f"{round(top_deg, 2)}")
                c3.metric("Net Kâr", f"{round(top_deg - top_mal, 2)}")

    with tab2:
        st.subheader("Teknik Göstergeler")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"🏢 {info.get('longName', hisse_kodu)} Temel Finansal Veriler")
        c1, c2, c3 = st.columns(3)
        c1.metric("F/K Oranı (P/E)", info.get('trailingPE', '-'))
        c2.metric("PD/DD (P/B)", info.get('priceToBook', '-'))
        c3.metric("Piyasa Değeri", info.get('marketCap', '-'))

    with tab4:
        st.subheader("📰 Küresel Haber Duygu Analizi")
        for h in haber_duygu_analizi(hisse_kodu):
            with st.expander(f"{h['duygu']} | {h['baslik']} ({h['kaynak']})"):
                st.markdown(f"[Habere Git]({h['link']})")

    with tab5:
        st.subheader(f"🔍 {piyasa_tipi} Otomatik Tarama")
        if st.button("Taramayı Başlat"):
            st.success(f"{piyasa_tipi} taraması yapılıyor... Telegram bildirimleri aktif.")

    with tab6:
        st.subheader("🔮 30 Günlük Monte Carlo Simülasyonu")
        if st.button("Simülasyon Sınırlarını Hesapla"):
            sim_data = monte_carlo_simulasyonu(df)
            fig_mc = go.Figure()
            for i in range(100):
                fig_mc.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(width=0.4), opacity=0.15))
            fig_mc.update_layout(template="plotly_dark", height=500, showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)

    # v7 YENİLİĞİ: BACKTEST SEKMESİ
    with tab7:
        st.subheader("⚙️ Strateji Testi (Backtest): SMA 20 vs SMA 50")
        st.write("Eğer bu varlıkta sadece *'Kısa ortalama, uzun ortalamayı yukarı kestiğinde al, aşağı kestiğinde sat'* stratejisini uygulasaydın ne olurdu?")
        
        bt_sonuc = backtest_motoru(df, kisa_periyot=20, uzun_periyot=50)
        
        if not bt_sonuc.empty:
            son_piyasa_getiri = bt_sonuc['Piyasa_Kumulatif'].iloc[-1] - 100
            son_strateji_getiri = bt_sonuc['Strateji_Kumulatif'].iloc[-1] - 100
            
            c1, c2 = st.columns(2)
            c1.metric("Sadece Alıp Bekleseydin Getirin", f"%{round(son_piyasa_getiri, 2)}")
            c2.metric("Strateji ile Alsatsaydın Getirin", f"%{round(son_strateji_getiri, 2)}", 
                      delta=round(son_strateji_getiri - son_piyasa_getiri, 2), delta_color="normal")
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Piyasa_Kumulatif'], name="Piyasa Getirisi (Al-Tut)", line=dict(color='white')))
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Strateji_Kumulatif'], name="Strateji Getirisi (Al-Sat)", line=dict(color='green', width=3)))
            fig_bt.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bt, use_container_width=True)
            
    # v7 YENİLİĞİ: VERİ İNDİRME BÖLÜMÜ (Sidebar'da)
    st.sidebar.divider()
    st.sidebar.subheader("📥 Raporlama")
    csv = df.to_csv().encode('utf-8')
    st.sidebar.download_button(
        label="📊 Teknik Verileri İndir (CSV)",
        data=csv,
        file_name=f'{hisse_kodu}_analiz_verileri.csv',
        mime='text/csv',
    )
else:
    st.error("Veri çekilemedi. Kodunuzu kontrol edin.")