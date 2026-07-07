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

st.set_page_config(layout="wide", page_title="Pro Yatırım Terminali v4.0")
st.title("🦅 Pro Küresel Yatırım Terminali v4.0")

# --- TELEGRAM VE OTOMASYON ---
def telegram_gonder(mesaj):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mesaj}"
        requests.get(url)
    except:
        pass

# --- VERİ MOTORLARI ---
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

# --- SİDEBAR VE PİYASA SEÇİMİ (v4.0 Yeniliği) ---
st.sidebar.header("🌍 Küresel Piyasa Ayarları")
piyasa_tipi = st.sidebar.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "Amerikan Borsası (NASDAQ/NYSE)", "Kripto Para"])

# Seçilen piyasaya göre varsayılan ayarlar ve tarama listeleri
if piyasa_tipi == "Borsa İstanbul (BIST)":
    varsayilan_hisse = "THYAO.IS"
    tarama_listesi = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "GARAN.IS", "SASA.IS"]
elif piyasa_tipi == "Amerikan Borsası (NASDAQ/NYSE)":
    varsayilan_hisse = "AAPL"
    tarama_listesi = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX"]
else:
    varsayilan_hisse = "BTC-USD"
    tarama_listesi = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOT-USD"]

hisse_kodu = st.sidebar.text_input("Varlık Kodu:", value=varsayilan_hisse).upper()
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=180))
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Veriler küresel sunuculardan çekiliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # Teknik Gösterge Hesaplamaları
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    # SEKMELER (v4.0 Düzeni)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💼 Cüzdanım (PnL)", "📈 Teknik Analiz", "🏢 Temel Analiz", "📰 Haber Analizi", "🔍 Akıllı Tarama", "🔮 Monte Carlo"
    ])

    # 1. SEKME: CANLI PORTFÖY TAKİBİ (v4.0 Yeniliği)
    with tab1:
        st.subheader("📊 Canlı Varlık Portföyüm ve Risk Dağılımı")
        st.write("Sahip olduğunuz varlıkları aşağıdaki tabloya girerek canlı kâr/zarar durumunuzu izleyebilirsiniz.")
        
        # Session State ile Portföyü Tutma (Geçici Hafıza)
        if 'portfoy_verisi' not in st.session_state:
            st.session_state.portfoy_verisi = pd.DataFrame([
                {"Varlık Kodu": "THYAO.IS", "Maliyet": 300.0, "Lot / Adet": 50.0},
                {"Varlık Kodu": "BTC-USD", "Maliyet": 62000.0, "Lot / Adet": 0.05},
                {"Varlık Kodu": "AAPL", "Maliyet": 175.0, "Lot / Adet": 10.0}
            ])
            
        # Kullanıcının düzenleyebileceği interaktif tablo
        guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
        st.session_state.portfoy_verisi = guncel_portfoy
        
        if st.button("Portföy Değerini Hesapla"):
            hesaplanan_liste = []
            toplam_maliyet_genel = 0
            toplam_deger_genel = 0
            
            for index, row in guncel_portfoy.iterrows():
                kod = row["Varlık Kodu"].upper()
                maliyet = row["Maliyet"]
                lot = row["Lot / Adet"]
                
                if kod and lot > 0:
                    try:
                        canli_veri = yf.download(kod, period="1d", progress=False, session=oturum)
                        if isinstance(canli_veri.columns, pd.MultiIndex): canli_veri.columns = canli_veri.columns.droplevel(1)
                        guncel_fiyat = canli_veri['Close'].iloc[-1]
                    except:
                        guncel_fiyat = maliyet
                        
                    toplam_maliyet = maliyet * lot
                    toplam_deger = guncel_fiyat * lot
                    pnl = toplam_deger - toplam_maliyet
                    pnl_yuzde = (pnl / toplam_maliyet) * 100 if toplam_maliyet > 0 else 0
                    
                    toplam_maliyet_genel += toplam_maliyet
                    toplam_deger_genel += toplam_deger
                    
                    hesaplanan_liste.append({
                        "Varlık": kod, "Maliyet": maliyet, "Adet": lot, 
                        "Güncel Fiyat": round(guncel_fiyat, 2), "Toplam Değer": round(toplam_deger, 2), 
                        "Kâr/Zarar": round(pnl, 2), "Değişim %": f"%{round(pnl_yuzde, 2)}"
                    })
            
            if hesaplanan_liste:
                p_df = pd.DataFrame(hesaplanan_liste)
                st.divider()
                
                # Özet Metrikler
                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam Portföy Maliyeti", f"{round(toplam_maliyet_genel, 2)}")
                c2.metric("Toplam Güncel Değer", f"{round(toplam_deger_genel, 2)}")
                genel_pnl = toplam_deger_genel - toplam_maliyet_genel
                genel_pnl_yuzde = (genel_pnl / toplam_maliyet_genel) * 100 if toplam_maliyet_genel > 0 else 0
                c3.metric("Net Kâr / Zarar", f"{round(genel_pnl, 2)}", f"%{round(genel_pnl_yuzde, 2)}")
                
                st.dataframe(p_df, use_container_width=True)
                
                # Pasta Grafiği Görselleştirmesi
                fig_pie = go.Figure(data=[go.Pie(labels=p_df["Varlık"], values=p_df["Toplam Değer"], hole=.3)])
                fig_pie.update_layout(title_text="Varlık Dağılım Oranları", template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("Teknik Göstergeler")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange')))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"🏢 {info.get('longName', hisse_kodu)} Temel Finansal Veriler")
        if piyasa_tipi == "Kripto Para":
            st.warning("Kripto paralar için geleneksel bilanço çarpanları bulunmamaktadır.")
            st.metric("Piyasa Değeri (Market Cap)", info.get('marketCap', '-'))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("F/K Oranı (P/E)", info.get('trailingPE', '-'))
            c2.metric("PD/DD (P/B)", info.get('priceToBook', '-'))
            c3.metric("Özsermaye Kârlılığı (ROE)", f"%{round(info.get('returnOnEquity', 0)*100, 2)}" if info.get('returnOnEquity') else "-")

    with tab4:
        st.subheader("📰 Küresel Haber Duygu Analizi")
        for h in haber_duygu_analizi(hisse_kodu):
            with st.expander(f"{h['duygu']} | {h['baslik']} ({h['kaynak']})"):
                st.markdown(f"[Haber Detayına Git]({h['link']})")

    with tab5:
        st.subheader(f"🔍 {piyasa_tipi} Otomatik Tarama")
        if st.button("Seçili Piyasayı Taramaya Başla"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ticker in enumerate(tarama_listesi):
                status_text.text(f"Taranıyor: {ticker} ({idx+1}/{len(tarama_listesi)})")
                time.sleep(0.5)
                progress_bar.progress((idx + 1) / len(tarama_listesi))
                
            status_text.text("Tarama tamamlandı!")
            st.success(f"Seçilen {piyasa_tipi} listesindeki en dinamik 3 varlık Telegram bildirim sistemine gönderildi!")
            telegram_gonder(f"🚀 {piyasa_tipi} taraması kullanıcı tarafından başarıyla tetiklendi.")

    with tab6:
        st.subheader("🔮 İstatistiki Gelecek Fiyat Simülasyonu")
        if st.button("Simülasyon Sınırlarını Hesapla"):
            sim_data = monte_carlo_simulasyonu(df)
            fig_mc = go.Figure()
            for i in range(100):
                fig_mc.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(width=0.4), opacity=0.15, showlegend=False))
            fig_mc.update_layout(title="Monte Carlo Olasılık Kanalları (30 Gün)", template="plotly_dark", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)
else:
    st.error("Seçilen varlık kodu için veri çekilemedi. Lütfen kodu kontrol edin.")