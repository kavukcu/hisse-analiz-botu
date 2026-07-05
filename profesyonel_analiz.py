import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import requests

# Sayfa Genişlik Ayarı
st.set_page_config(layout="wide", page_title="Pro Hisse Analiz Paneli")
st.title("🚀 Tam Kapsamlı Hisse Analiz Merkezi")

# ==========================================
# TELEGRAM BİLDİRİM AYARLARI
# ==========================================
TELEGRAM_TOKEN = "8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk"
TELEGRAM_CHAT_ID = "1634044181"

def telegram_gonder(mesaj):
    # Eğer token girilmemişse hata vermesin, sadece geçsin
    if TELEGRAM_TOKEN == "8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk":
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={mesaj}"
    try:
        requests.get(url)
    except Exception as e:
        print(f"Telegram mesajı gönderilemedi: {e}")

# ==========================================
# VERİ ÇEKME VE ANALİZ FONKSİYONLARI
# ==========================================
@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    return yf.Ticker(ticker).info

@st.cache_data(show_spinner=False)
def tarama_yap(tickers):
    bulunanlar = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if len(df) < 200: continue
            
            sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
            onceki_sma50 = df['Close'].rolling(window=50).mean().iloc[-2]
            onceki_sma200 = df['Close'].rolling(window=200).mean().iloc[-2]

            # Golden Cross (Altın Kesişme) Şartı
            if onceki_sma50 <= onceki_sma200 and sma50 > sma200:
                bulunanlar.append(ticker)
                
                # Sinyal Bulunduğunda Telegram'a Mesaj At!
                mesaj = f"🚀 SİNYAL GELDİ!\n\n📈 Hisse: {ticker}\n🔔 Formasyon: Golden Cross (50 Günlük SMA, 200 Günlüğü Yukarı Kesti)\n⏱ Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                telegram_gonder(mesaj)
        except:
            continue
    return bulunanlar

# ==========================================
# SOL MENÜ (SIDEBAR)
# ==========================================
st.sidebar.header("🔧 Analiz Parametreleri")
hisse_kodu = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS):", value="THYAO.IS").upper()

bugun = datetime.today()
baslangic_tarihi = st.sidebar.date_input("Başlangıç Tarihi:", value=pd.to_datetime("2025-01-01"))
bitis_tarihi = st.sidebar.date_input("Bitiş Tarihi:", value=bugun)

# ==========================================
# ANA UYGULAMA GÖVDESİ (SEKMELER)
# ==========================================
try:
    with st.spinner('Veriler çekiliyor...'):
        df = veri_yukle(hisse_kodu, baslangic_tarihi, bitis_tarihi)
        info = sirket_bilgisi_getir(hisse_kodu)
        
    if df.empty:
        st.error("Veri çekilemedi. BIST hisseleri için sonuna '.IS' eklediğinizden emin olun.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        tab1, tab2, tab3 = st.tabs(["📈 Teknik Analiz", "🏢 Temel Analiz", "🔍 Otomatik Tarama (Screener)"])

        with tab1:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Mum Grafiği"))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="SMA 200", line=dict(color='red')))
            
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"🏢 {info.get('longName', hisse_kodu)}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sektör", info.get('sector', 'Bilinmiyor'))
            col2.metric("F/K Oranı", info.get('trailingPE', 'Veri Yok'))
            col3.metric("PD/DD", info.get('priceToBook', 'Veri Yok'))

        with tab3:
            st.subheader("🔍 Piyasayı Tara ve Bildirim Gönder")
            st.write("Aşağıdaki butona bastığınızda sistem BIST listesini tarar. Eğer 'Golden Cross' yakalarsa telefonunuza anında Telegram mesajı gelir.")
            
            if st.button("Taramayı Başlat"):
                # Örnek BIST Tarama Listesi (Genişletebilirsin)
                bist_liste = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", "GARAN.IS", "AKBNK.IS", "TUPRS.IS"]
                
                with st.spinner("Piyasa taranıyor, analiz yapılıyor..."):
                    sonuclar = tarama_yap(bist_liste)
                    
                    if sonuclar:
                        st.success(f"✅ Sinyal Yakalandı ve Telegram'a Gönderildi: {sonuclar}")
                    else:
                        st.warning("Şu an analiz edilen hisselerde yeni bir alım sinyali bulunamadı.")

except Exception as e:
    st.error(f"Beklenmeyen bir hata oluştu: {e}")