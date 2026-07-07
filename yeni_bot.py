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

# 1. OTURUM VE GÜVENLİK
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="Galactic Terminal v20.0", initial_sidebar_state="expanded")
st.title("🌌 Kurumsal Fon Yönetim Terminali v20.0 (Galactic Edition)")
st.markdown("*" + "Entegre Makine Öğrenmesi, Risk Analitiği ve Otonom Telegram Bildirim Sistemi" + "*")

# --- KİMLİK BİLGİLERİ ---
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    TELEGRAM_TOKEN = "TEST_MODU"
    TELEGRAM_CHAT_ID = "TEST_MODU"

# --- OTONOM SİSTEMLER ---
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

# --- GELİŞMİŞ MATEMATİK VE RİSK MODÜLLERİ ---
def risk_metrikleri_hesapla(df):
    getiriler = df['Close'].pct_change().dropna()
    yillik_getiri = getiriler.mean() * 252
    yillik_volatilite = getiriler.std() * np.sqrt(252)
    
    # Risksiz getiri oranı (temsili %5)
    sharpe_orani = (yillik_getiri - 0.05) / (yillik_volatilite + 1e-9)
    
    # Max Drawdown (Maksimum Düşüş)
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

# --- YAN MENÜ (SİDEBAR) ---
with st.sidebar:
    st.header("🎛️ Komuta Merkezi")
    piyasa_tipi = st.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "ABD Borsaları", "Kripto Para", "Emtia & Endeks"])
    
    if piyasa_tipi == "Borsa İstanbul (BIST)":
        hisse_kodu = st.text_input("Varlık Kodu:", value="THYAO.IS").upper()
    elif piyasa_tipi == "ABD Borsaları":
        hisse_kodu = st.text_input("Varlık Kodu:", value="NVDA").upper()
    elif piyasa_tipi == "Kripto Para":
        hisse_kodu = st.text_input("Varlık Kodu:", value="BTC-USD").upper()
    else:
        hisse_kodu = st.text_input("Varlık Kodu (Örn: GC=F Altın):", value="GC=F").upper()

    baslangic = st.date_input("Analiz Başlangıcı:", value=datetime.today() - pd.Timedelta(days=730)) # 2 yıllık default
    bitis = st.date_input("Analiz Bitişi:", value=datetime.today())
    
    st.divider()
    st.markdown("### 📡 Canlı Bildirim Sistemi")
    oto_alarm = st.checkbox("Olağandışı Hacimde Telefona Bildir", value=True)

with st.spinner('Kuantum algoritmaları ve veri akışları senkronize ediliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)
    gelir_tablosu, bilanco, nakit_akisi = bilanco_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # Dinamik Hacim Alarmı Kontrolü
    if oto_alarm and len(df) > 20:
        son_hacim = df['Volume'].iloc[-1]
        ort_hacim = df['Volume'].rolling(20).mean().iloc[-2] # Bugünü katmadan önceki 20 gün
        if son_hacim > (ort_hacim * 2.5):
            telegram_gonder(f"🚨 HACİM PATLAMASI ALARMI
Varlık: {hisse_kodu}
Güncel Hacim ortalamanın 2.5 katına ulaştı!")

    # İndikatörler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Bollinger_Ust'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
    df['Bollinger_Alt'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)

    # v20 SEKMELERİ
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🔬 Süper Grafik & AI", "🛡️ Kurumsal Risk (Quant)", "🏦 Bilanço Röntkeni", 
        "💼 Cüzdan & PnL", "🌐 Global Isı Haritası", "⚙️ Gelişmiş Backtest", "🤖 Finansal AI Asistan"
    ])

    with t1:
        st.subheader(f"Gelişmiş Fiyat Hareketi ve Makine Öğrenmesi Projeksiyonu - {hisse_kodu}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='#f39c12', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="SMA 200", line=dict(color='#e74c3c', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Ust'], name="BB Üst", line=dict(color='rgba(255,255,255,0.1)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Alt'], name="BB Alt", line=dict(color='rgba(255,255,255,0.1)', dash='dot')), row=1, col=1)
        
        # ML Tahmin Çizgisi
        tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="AI Trend Tahmini (30 Gün)", line=dict(color='#9b59b6', width=3, dash='dot')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#3498db')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🛡️ Kantitatif Risk ve Performans Analizi")
        st.write("Fon yöneticilerinin kullandığı ileri düzey matematiksel metrikler.")
        
        sharpe, max_dd, volatilite = risk_metrikleri_hesapla(df)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Oranı", f"{round(sharpe, 2)}", "1.0 Üzeri İyidir" if sharpe > 1 else "Riskli", delta_color="normal" if sharpe>1 else "inverse")
        c2.metric("Maksimum Düşüş (Max DD)", f"%{round(max_dd*100, 2)}", "Geçmişteki en büyük kriz")
        c3.metric("Yıllık Volatilite", f"%{round(volatilite*100, 2)}", "Fiyatın oynaklık derecesi")
        c4.metric("Beta (Piyasa Hassasiyeti)", info.get("beta", "Hesaplanamadı"))

    with t3:
        st.subheader(f"🏦 {info.get('longName', hisse_kodu)} Finansal Tablo Röntkeni")
        if gelir_tablosu is not None and not gelir_tablosu.empty:
            st.write("Yıllık Gelir Tablosu Özet (Milyon / Milyar)")
            st.dataframe(gelir_tablosu.head(10), use_container_width=True)
        else:
            st.warning("Bu varlık için (muhtemelen kripto/endeks) detaylı bilanço verisi bulunmuyor.")

    with t4:
        st.subheader("💼 Portföy Yönetimi ve Performans İzleme")
        if 'portfoy_verisi' not in st.session_state:
            st.session_state.portfoy_verisi = pd.DataFrame([
                {"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0},
                {"Varlık": "NVDA", "Maliyet": 80.0, "Lot": 10.0}
            ])
            
        guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
        st.session_state.portfoy_verisi = guncel_portfoy
        
        if st.button("Kâr/Zarar Senkronizasyonu Başlat"):
            top_mal = 0; top_deg = 0
            for index, row in guncel_portfoy.iterrows():
                kod = row["Varlık"].upper()
                mal = row["Maliyet"]; lot = row["Lot"]
                if kod and lot > 0:
                    try:
                        c_veri = yf.download(kod, period="1d", progress=False, session=oturum)
                        if isinstance(c_veri.columns, pd.MultiIndex): c_veri.columns = c_veri.columns.droplevel(1)
                        g_fiyat = c_veri['Close'].iloc[-1]
                    except: g_fiyat = mal
                    top_mal += (mal * lot); top_deg += (g_fiyat * lot)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Yatırım (Maliyet)", f"{round(top_mal, 2)}")
            c2.metric("Güncel Cüzdan Büyüklüğü", f"{round(top_deg, 2)}")
            c3.metric("Net Kâr / Zarar", f"{round(top_deg - top_mal, 2)}", f"%{round(((top_deg - top_mal)/top_mal)*100,2) if top_mal>0 else 0}")

    with t5:
        st.subheader("🌐 Global Piyasa Korelasyon Isı Haritası")
        st.write("Belirlediğiniz piyasadaki majör varlıkların birbirleriyle olan matematiksel ilişkisi.")
        tarama_listeleri = {
            "Borsa İstanbul (BIST)": ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS"],
            "ABD Borsaları": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
            "Kripto Para": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
            "Emtia & Endeks": ["GC=F", "SI=F", "CL=F", "^GSPC", "^IXIC"]
        }
        if st.button("Isı Haritasını Çıkar"):
            with st.spinner("Matris hesaplanıyor..."):
                korelasyon_df = pd.DataFrame()
                for ticker in tarama_listeleri.get(piyasa_tipi, ["BTC-USD"]):
                    tmp_df = yf.download(ticker, period="1y", progress=False, session=oturum)
                    if isinstance(tmp_df.columns, pd.MultiIndex): tmp_df.columns = tmp_df.columns.droplevel(1)
                    if not tmp_df.empty: korelasyon_df[ticker] = tmp_df['Close']
                
                corr_matrix = korelasyon_df.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

    with t6:
        st.subheader("⚙️ Kantitatif Backtest (SMA Cross Stratejisi)")
        bt_df = df[['Close']].copy()
        bt_df['Kisa_SMA'] = bt_df['Close'].rolling(window=20).mean()
        bt_df['Uzun_SMA'] = bt_df['Close'].rolling(window=50).mean()
        bt_df.dropna(inplace=True)
        bt_df['Sinyal'] = np.where(bt_df['Kisa_SMA'] > bt_df['Uzun_SMA'], 1, 0)
        bt_df['Günlük_Getiri'] = bt_df['Close'].pct_change()
        bt_df['Strateji_Getirisi'] = bt_df['Günlük_Getiri'] * bt_df['Sinyal'].shift(1)
        bt_df['Piyasa_Kumulatif'] = (1 + bt_df['Günlük_Getiri']).cumprod() * 100
        bt_df['Strateji_Kumulatif'] = (1 + bt_df['Strateji_Getirisi']).cumprod() * 100
        
        son_piyasa = bt_df['Piyasa_Kumulatif'].iloc[-1] - 100
        son_strateji = bt_df['Strateji_Kumulatif'].iloc[-1] - 100
        
        c1, c2 = st.columns(2)
        c1.metric("Al-Tut Getirisi (Piyasa)", f"%{round(son_piyasa, 2)}")
        c2.metric("Strateji Getirisi (Al-Sat)", f"%{round(son_strateji, 2)}", delta=round(son_strateji - son_piyasa, 2))

    with t7:
        st.subheader("🤖 Finansal AI Asistan (LLM Hazırlığı)")
        st.write("Bu sekme, hisse hakkında doğal dille soru sorabileceğin OpenAI / Gemini entegrasyonu için ayrılmıştır. (API Key eklendiğinde aktif olur)")
        soru = st.text_input("Asistana Sor (Örn: Bu hissenin borç durumu nasıl?):")
        if st.button("Sor"):
            st.info("API bağlantısı bekleniyor... v21'de canlıya alınacak. Ancak tüm indikatör verileri asistan için hazırlandı!")

    st.sidebar.divider()
    st.sidebar.download_button(label="📥 Kuantum Veri Setini İndir", data=df.to_csv().encode('utf-8'), file_name=f'{hisse_kodu}_v20.csv', mime='text/csv')

else:
    st.error("Sinyal alınamadı. Kodunuzu kontrol edin.")
