import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from sklearn.linear_model import LinearRegression

oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="Omniscient Terminal v30.0", initial_sidebar_state="expanded")
st.title("👁️‍🗨️ Omniscient Fon Yönetim Terminali v30.0 (Algorithmic Edition)")
st.markdown("*" + "Opsiyon Fiyatlama, Çoklu Zaman Dilimi (Fraktal) Analiz ve Webhook Trade Altyapısı" + "*")

try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    TELEGRAM_TOKEN = "TEST_MODU"
    TELEGRAM_CHAT_ID = "TEST_MODU"

def telegram_gonder(mesaj):
    if TELEGRAM_TOKEN == "TEST_MODU": return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={mesaj}"
        requests.get(url, timeout=5)
    except: pass

@st.cache_data(show_spinner=False, ttl=300)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

@st.cache_data(show_spinner=False, ttl=3600)
def sirket_bilgisi_getir(ticker):
    try: return yf.Ticker(ticker, session=oturum).info
    except: return {}

@st.cache_data(show_spinner=False, ttl=3600)
def bilanco_getir(ticker):
    try: 
        sirket = yf.Ticker(ticker, session=oturum)
        return sirket.financials, sirket.balance_sheet, sirket.cashflow
    except: return None, None, None

def risk_metrikleri_hesapla(df):
    getiriler = df['Close'].pct_change().dropna()
    yillik_getiri = getiriler.mean() * 252
    yillik_volatilite = getiriler.std() * np.sqrt(252)
    sharpe_orani = (yillik_getiri - 0.05) / (yillik_volatilite + 1e-9)
    kumulatif = (1 + getiriler).cumprod()
    zirve = kumulatif.cummax()
    dusus = (kumulatif - zirve) / zirve
    max_dusus = dusus.min()
    return sharpe_orani, max_dusus, yillik_volatilite

def makine_ogrenmesi_tahmin(df, gelecek_gun=30):
    df_ml = df.copy().reset_index()
    df_ml['Gun'] = np.arange(len(df_ml))
    X = df_ml[['Gun']]
    y = df_ml['Close']
    model = LinearRegression()
    model.fit(X, y)
    son_gun = df_ml['Gun'].iloc[-1]
    gelecek_X = np.array([[son_gun + i] for i in range(1, gelecek_gun + 1)])
    tahminler = model.predict(gelecek_X)
    tarihler = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
    return tarihler, tahminler

# Black-Scholes Matematiksel Modeli (Standart Normal Dağılım Kümülatif Yoğunluk Fonksiyonu)
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

with st.sidebar:
    st.header("🎛️ Merkez Komuta")
    piyasa_tipi = st.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "ABD Borsaları", "Kripto Para", "Emtia & Endeks"])
    
    if piyasa_tipi == "Borsa İstanbul (BIST)": hisse_kodu = st.text_input("Varlık Kodu:", value="THYAO.IS").upper()
    elif piyasa_tipi == "ABD Borsaları": hisse_kodu = st.text_input("Varlık Kodu:", value="NVDA").upper()
    elif piyasa_tipi == "Kripto Para": hisse_kodu = st.text_input("Varlık Kodu:", value="BTC-USD").upper()
    else: hisse_kodu = st.text_input("Varlık Kodu:", value="GC=F").upper()

    baslangic = st.date_input("Analiz Başlangıcı:", value=datetime.today() - pd.Timedelta(days=730))
    bitis = st.date_input("Analiz Bitişi:", value=datetime.today())
    
    st.divider()
    oto_alarm = st.checkbox("Otonom Hacim Alarmları", value=True)
    webhook_aktif = st.checkbox("API/Webhook Trade Sinyalleri (Mock)", value=False)

with st.spinner('Kuantum algoritmaları, opsiyon zincirleri ve fraktal veriler senkronize ediliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)
    gelir_tablosu, bilanco, nakit_akisi = bilanco_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    if oto_alarm and len(df) > 20:
        son_hacim = df['Volume'].iloc[-1]
        ort_hacim = df['Volume'].rolling(20).mean().iloc[-2]
        if son_hacim > (ort_hacim * 2.5):
            telegram_gonder(f"🚨 OTONOM ALARM
{hisse_kodu} Hacim Patlaması! (x2.5)")

    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Bollinger_Ust'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
    df['Bollinger_Alt'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🔭 Çoklu Zaman Dilimi (Fraktal)", "🧮 Black-Scholes Opsiyon Modeli", "🛡️ Kantitatif Risk", 
        "🏦 Bilanço", "💼 Cüzdan", "⚙️ Algo-Backtest", "🔌 Algoritmik Trade (API)"
    ])

    with t1:
        st.subheader(f"Fraktal Piyasa Analizi - {hisse_kodu}")
        st.write("Tek grafikte Fiyat, Ortalamalar, Volatilite Bantları ve ML Projeksiyonu.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='#f39c12', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="SMA 200", line=dict(color='#e74c3c', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Ust'], name="BB Üst", line=dict(color='rgba(255,255,255,0.1)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Alt'], name="BB Alt", line=dict(color='rgba(255,255,255,0.1)', dash='dot')), row=1, col=1)
        tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="AI Trend", line=dict(color='#9b59b6', width=3, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#3498db')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🧮 Black-Scholes Türev & Opsiyon Fiyatlama Motoru")
        st.write("Varlığın güncel fiyatına ve geçmiş volatilitesine dayanarak Avrupa tipi Call (Alım) opsiyonlarının teorik adil değerini hesaplar.")
        
        S_guncel = df['Close'].iloc[-1]
        _, _, yillik_vol = risk_metrikleri_hesapla(df)
        
        c1, c2, c3 = st.columns(3)
        hedef_fiyat = c1.number_input("Hedef Kullanım Fiyatı (Strike - K):", value=float(round(S_guncel * 1.05, 2)))
        vade_gun = c2.number_input("Vadeye Kalan Gün (T):", value=30, min_value=1)
        faiz_orani = c3.number_input("Risksiz Faiz Oranı (r) %:", value=5.0) / 100.0

        if st.button("Teorik Opsiyon Primini Hesapla"):
            T_yil = vade_gun / 365.0
            call_fiyat = black_scholes_call(S_guncel, hedef_fiyat, T_yil, faiz_orani, yillik_vol)
            
            st.divider()
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Güncel Dayanak Varlık Fiyatı (S)", round(S_guncel, 2))
            cc2.metric("Hesaplanan Volatilite (σ)", f"%{round(yillik_vol*100, 2)}")
            cc3.metric("Adil Alım (Call) Opsiyon Primi (C)", round(call_fiyat, 4))
            st.info("Bu model, akademik Black-Scholes denklemini kullanarak arbitrajsız teorik fiyatı bulur.")

    with t3:
        st.subheader("🛡️ Kantitatif Risk Analizi")
        sharpe, max_dd, volatilite = risk_metrikleri_hesapla(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Oranı", f"{round(sharpe, 2)}", "1.0 Üzeri İyidir" if sharpe > 1 else "Riskli", delta_color="normal" if sharpe>1 else "inverse")
        c2.metric("Max Drawdown", f"%{round(max_dd*100, 2)}")
        c3.metric("Yıllık Volatilite", f"%{round(volatilite*100, 2)}")
        c4.metric("Beta", info.get("beta", "-"))

    with t4:
        st.subheader(f"🏦 Finansal Tablolar")
        if gelir_tablosu is not None and not gelir_tablosu.empty:
            st.dataframe(gelir_tablosu.head(10), use_container_width=True)
        else: st.warning("Bilanço verisi yok.")

    with t5:
        st.subheader("💼 Portföy PnL (Kâr/Zarar)")
        if 'portfoy_verisi' not in st.session_state:
            st.session_state.portfoy_verisi = pd.DataFrame([{"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0}])
        st.session_state.portfoy_verisi = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)

    with t6:
        st.subheader("⚙️ Kantitatif Backtest")
        st.write("Strateji: KISA Ort > UZUN Ort = AL")
        bt_df = df[['Close']].copy()
        bt_df['SMA_20'] = bt_df['Close'].rolling(20).mean()
        bt_df['SMA_50'] = bt_df['Close'].rolling(50).mean()
        bt_df.dropna(inplace=True)
        bt_df['Sinyal'] = np.where(bt_df['SMA_20'] > bt_df['SMA_50'], 1, 0)
        bt_df['Getiri'] = bt_df['Close'].pct_change() * bt_df['Sinyal'].shift(1)
        st.line_chart((1 + bt_df['Getiri']).cumprod() * 100)

    with t7:
        st.subheader("🔌 Algoritmik Trade & Webhook Bağlantısı")
        st.write("Uygulamanın ürettiği al/sat sinyallerini aracı kurumlara (Binance, Midas, Interactive Brokers) API ile iletme merkezi.")
        if webhook_aktif:
            st.success("🟢 Webhook Dinleyicisi Aktif. Sinyaller JSON formatında iletime hazır.")
            st.code('{
  "ticker": "' + hisse_kodu + '",
  "action": "BUY",
  "price": ' + str(round(df['Close'].iloc[-1],2)) + ',
  "confidence_score": 0.87
}', language='json')
        else:
            st.warning("🔴 Webhook devre dışı. Sol menüden aktifleştirin.")

    st.sidebar.divider()
    st.sidebar.download_button(label="📥 Kuantum Veriyi İndir", data=df.to_csv().encode('utf-8'), file_name=f'{hisse_kodu}_v30.csv', mime='text/csv')

else: st.error("Sinyal alınamadı.")
