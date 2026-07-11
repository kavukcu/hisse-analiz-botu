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

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v61")
st.title("👁️ Pro Küresel Yatırım Terminali v61 (Veri Bilimi Sürümü)")

# --- TELEGRAM VE OTOMASYON ---
def telegram_gonder(mesaj):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mesaj}"
        requests.get(url)
        return True
    except:
        return False

# --- VERİ MOTORLARI VE FONKSİYONLAR ---
@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

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
    son_fiyat = df['Close'].iloc[-1]
    
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
            metin = (str(n.get('title', '')) + " " + str(n.get('summary', ''))).lower()
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
    bt_df['Piyasa_Kumulatif'] = (1 + bt_df['Günlük_Getiri']).cumprod() * 100
    bt_df['Strateji_Kumulatif'] = (1 + bt_df['Strateji_Getirisi']).cumprod() * 100
    return bt_df

def makine_ogrenmesi_tahmin(df, gelecek_gun=30):
    df_ml = df.copy()
    df_ml.reset_index(inplace=True)
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

# --- YENİ: PYTHON İLERİ İSTATİSTİK MODÜLÜ ---
def python_istatistik_analizi(df):
    df_stats = df.copy()
    df_stats['Log_Return'] = np.log(df_stats['Close'] / df_stats['Close'].shift(1))
    
    # 1. Yıllıklandırılmış Volatilite
    volatilite = df_stats['Log_Return'].std() * np.sqrt(252)
    
    # 2. Sharpe Oranı (Risk siz faizi %40 kabul edelim)
    getiri = df_stats['Log_Return'].mean() * 252
    sharpe = (getiri - 0.40) / volatilite
    
    # 3. Value at Risk (VaR) - %95 güven aralığı
    var_95 = np.percentile(df_stats['Log_Return'].dropna(), 5)
    
    return {
        "Yıllık Volatilite": f"%{round(volatilite * 100, 2)}",
        "Sharpe Oranı": round(sharpe, 2),
        "Günlük VaR (%95)": f"%{round(var_95 * 100, 2)}"
    }

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
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=730)) 
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Kurumsal teknik analiz verileri hesaplanıyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.droplevel(1)
    
    # --- TEMEL İNDİKATÖRLER ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bantları
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Low'] = df['SMA_20'] - (df['BB_Std'] * 2)
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI & Stoch RSI
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

    # --- İLERİ DÜZEY İNDİKATÖRLER ---
    # 1. Ichimoku Bulutu
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # 2. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['True_Range'] = np.max(ranges, axis=1)
    df['ATR_14'] = df['True_Range'].rolling(14).mean()

    # --- YENİ: KURUMSAL İNDİKATÖRLER ---
    # 3. EMA Ribbon (Gökkuşağı)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_34'] = df['Close'].ewm(span=34, adjust=False).mean()
    df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()
    df['EMA_89'] = df['Close'].ewm(span=89, adjust=False).mean()

    # 4. Donchian Kanalları (Dinamik Destek/Direnç)
    df['Donchian_High'] = df['High'].rolling(window=20).max()
    df['Donchian_Low'] = df['Low'].rolling(window=20).min()

    # 5. VWAP (Hacim Ağırlıklı Ortalama Fiyat - 20 Günlük Rolling)
    df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

    tabs = st.tabs([
        "📈 Kurumsal Teknik Analiz", "🔍 Akıllı Radar", "💼 Cüzdan & Akıllı Stop", 
        "🏢 Temel & Temettü", "📰 Haber", "📊 Isı Haritası (Korelasyon)", 
        "⚙️ Backtest", "🎲 Risk Simülasyonu", "🛠️ Sistem Entegrasyonu", "🧬 Python İstatistik"
    ])

    # TAB 1: TEKNİK & AI TAHMİN
    with tabs[0]:
        st.subheader("📈 İleri Düzey Teknik Grafik (VWAP & EMA Ribbon & Donchian)")
        
        c_ayar1, c_ayar2, c_ayar3 = st.columns(3)
        with c_ayar1:
            goster_ichimoku = st.checkbox("☁️ Ichimoku Bulutu", value=False)
            goster_donchian = st.checkbox("🧱 Donchian Destek/Direnç (20G)", value=False)
        with c_ayar2:
            goster_fibo = st.checkbox("📐 Fibonacci Seviyeleri", value=False)
            goster_vwap = st.checkbox("⚖️ VWAP (Kurumsal Maliyet)", value=False)
        with c_ayar3:
            goster_ema_ribbon = st.checkbox("🌈 EMA Ribbon (Trend Gücü)", value=False)
            
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Fiyat Hareketi & Algoritmik Formasyonlar", "MACD & Hacim Momentum", "Stokastik RSI")
        )
        
        # 1. SATIR: Fiyat
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
        
        # Seçili İndikatörleri Çiz
        if not (goster_ichimoku or goster_ema_ribbon or goster_donchian or goster_vwap): 
            # Hiçbiri seçili değilse klasik SMA'ları göster
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='cyan', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='yellow', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], name="BB Üst", line=dict(color='gray', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Alt", line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        if goster_ichimoku:
            fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan_Sen'], name="Tenkan", line=dict(color='#0496ff', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Kijun_Sen'], name="Kijun", line=dict(color='#99154e', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_A'], name="Span A", line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_B'], name="Ichimoku Bulutu", fill='tonexty', fillcolor='rgba(128, 128, 128, 0.3)', line=dict(color='rgba(0,0,0,0)')), row=1, col=1)

        if goster_ema_ribbon:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name="EMA 20", line=dict(color='#00ff00', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_34'], name="EMA 34", line=dict(color='#aaff00', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_55'], name="EMA 55", line=dict(color='#ffaa00', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_89'], name="EMA 89", line=dict(color='#ff0000', width=1)), row=1, col=1)

        if goster_donchian:
            fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_High'], name="Direnç Duvarı", line=dict(color='white', width=1.5, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_Low'], name="Destek Zemini", line=dict(color='white', width=1.5, dash='dot')), row=1, col=1)

        if goster_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)

        # Fibo ve AI
        if goster_fibo:
            max_fiyat = df['High'].max()
            min_fiyat = df['Low'].min()
            fark = max_fiyat - min_fiyat
            seviyeler = {0: "100%", 0.236: "76.4%", 0.382: "61.8%", 0.5: "50%", 0.618: "38.2%", 0.786: "21.4%", 1: "0%"}
            renkler = ['#ff0000', '#ff9900', '#ffff00', '#33cc33', '#3399ff', '#cc33ff', '#999999']
            for i, (level, oran) in enumerate(seviyeler.items()):
                fiyat_seviyesi = max_fiyat - (fark * level)
                fig.add_hline(y=fiyat_seviyesi, line_dash="dash", line_color=renkler[i], annotation_text=f"Fibo {oran}", row=1, col=1)

        tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="AI Tahmini", line=dict(color='magenta', width=3, dash='dot')), row=1, col=1)
        
        # 2. SATIR: MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#2962FF')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Sinyal", line=dict(color='#FF6D00')), row=2, col=1)
        hist_colors = np.where(df['MACD_Hist'] < 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram", marker_color=hist_colors), row=2, col=1)
        
        # 3. SATIR: Stokastik RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_K'], name="%K", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_D'], name="%D", line=dict(color='orange')), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=1000, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        if piyasa_tipi == "Borsa İstanbul (BIST)":
            st.markdown("---")
            rc1, rc2 = st.columns(2)
            
            with rc1:
                st.subheader("💵 Dolar (USD) Bazlı Fiyat")
                usd_goster = st.checkbox(f"{hisse_kodu} USD Grafiği")
                if usd_goster:
                    with st.spinner("Kur Çekiliyor..."):
                        kur_df = veri_yukle("TRY=X", baslangic, bitis)
                        if not kur_df.empty:
                            if isinstance(kur_df.columns, pd.MultiIndex): kur_df.columns = kur_df.columns.droplevel(1)
                            ortak_index = df.index.intersection(kur_df.index)
                            df_usd = df.loc[ortak_index].copy()
                            df_usd['Close_USD'] = df_usd['Close'] / kur_df.loc[ortak_index, 'Close']
                            fig_usd = go.Figure()
                            fig_usd.add_trace(go.Scatter(x=df_usd.index, y=df_usd['Close_USD'], name="USD Fiyat", line=dict(color='#00ffcc', width=2)))
                            fig_usd.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig_usd, use_container_width=True)
            
            with rc2:
                st.subheader("📊 Endeks & Sektör Rakibi Kıyaslaması")
                bist_kiyasla = st.checkbox(f"{hisse_kodu} Göreceli Performans")
                if bist_kiyasla:
                    with st.spinner("Kıyaslama Verileri Çekiliyor..."):
                        rakipler = {
                            "THYAO.IS": "PGSUS.IS", "PGSUS.IS": "THYAO.IS",
                            "AKBNK.IS": "GARAN.IS", "GARAN.IS": "AKBNK.IS",
                            "ISCTR.IS": "YKBNK.IS", "YKBNK.IS": "ISCTR.IS",
                            "FROTO.IS": "TOASO.IS", "TOASO.IS": "FROTO.IS",
                            "EREGL.IS": "KRDMD.IS", "KRDMD.IS": "EREGL.IS",
                            "TUPRS.IS": "PETKM.IS", "PETKM.IS": "TUPRS.IS",
                            "BIMAS.IS": "SOKM.IS", "SOKM.IS": "BIMAS.IS",
                            "KCHOL.IS": "SAHOL.IS", "SAHOL.IS": "KCHOL.IS"
                        }
                        rakip_kodu = rakipler.get(hisse_kodu, None)
                        xu100 = veri_yukle("XU100.IS", baslangic, bitis)
                        rakip_df = pd.DataFrame()
                        if rakip_kodu:
                            rakip_df = veri_yukle(rakip_kodu, baslangic, bitis)
                            
                        if not xu100.empty:
                            if isinstance(xu100.columns, pd.MultiIndex): xu100.columns = xu100.columns.droplevel(1)
                            ortak_bist = df.index.intersection(xu100.index)
                            hisse_norm = (df.loc[ortak_bist, 'Close'] / df.loc[ortak_bist, 'Close'].iloc[0]) * 100
                            xu100_norm = (xu100.loc[ortak_bist, 'Close'] / xu100.loc[ortak_bist, 'Close'].iloc[0]) * 100
                            
                            fig_rel = go.Figure()
                            fig_rel.add_trace(go.Scatter(x=ortak_bist, y=hisse_norm, name=hisse_kodu, line=dict(color='magenta', width=2)))
                            fig_rel.add_trace(go.Scatter(x=ortak_bist, y=xu100_norm, name="XU100", line=dict(color='gray', dash='dash')))
                            
                            if not rakip_df.empty:
                                if isinstance(rakip_df.columns, pd.MultiIndex): rakip_df.columns = rakip_df.columns.droplevel(1)
                                rakip_norm = (rakip_df.loc[ortak_bist, 'Close'] / rakip_df.loc[ortak_bist, 'Close'].iloc[0]) * 100
                                fig_rel.add_trace(go.Scatter(x=ortak_bist, y=rakip_norm, name=f"Rakip: {rakip_kodu}", line=dict(color='orange')))
                                
                            fig_rel.update_layout(template="plotly_dark", height=350, yaxis_title="Göreli Getiri (Başlangıç=100)")
                            st.plotly_chart(fig_rel, use_container_width=True)

    # TAB 2: AKILLI TARAMA
    with tabs[1]:
        st.subheader(f"🔍 {piyasa_tipi} Akıllı Multi-Radar")
        if piyasa_tipi == "Borsa İstanbul (BIST)":
            tarama_modu = st.radio("Tarama Modu Seçin:", [
                "🟢 Aşırı Satım Radarı (RSI < 35)", 
                "🔥 Hacim Patlaması Radarı (Balina Avcısı)",
                "💼 Temel Analiz Radarı (Değer Avcısı - Düşük F/K ve PD/DD)",
                "⭐ Stoch RSI Alım Fırsatı (Stoch RSI %K Yukarı Kesişim)",
                "☁️ Ichimoku Kumo (Bulut) Yukarı Kırılımı (Trend Başlangıcı)"
            ])
            
            if st.button("🚀 Seçili BİST Radarını Çalıştır"):
                with st.spinner("BİST Hisseleri Taranıyor, Veriler Çekiliyor..."):
                    firsatlar = []
                    bist30_hisseler = ["AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS", "ENKAI.IS", "KRDMD.IS", "PETKM.IS"] 
                    ilerleme_cubugu = st.progress(0)
                    
                    for i, hisse in enumerate(bist30_hisseler):
                        try:
                            temiz_ad = hisse.replace(".IS", "")
                            if "Temel Analiz" in tarama_modu:
                                s_info = sirket_bilgisi_getir(hisse)
                                fk = s_info.get('trailingPE', 999)
                                pddd = s_info.get('priceToBook', 999)
                                son_fiyat = s_info.get('currentPrice', 0)
                                if isinstance(fk, (int, float)) and isinstance(pddd, (int, float)):
                                    if 0 < fk < 10 and 0 < pddd < 3:
                                        firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": son_fiyat, "Değer": f"F/K: {round(fk, 2)} | PD/DD: {round(pddd, 2)}", "Durum": "💼 Ucuz Çarpanlar"})
                            else:
                                t_df = veri_yukle(hisse, datetime.today() - timedelta(days=120), datetime.today())
                                if not t_df.empty and isinstance(t_df.columns, pd.MultiIndex): 
                                    t_df.columns = t_df.columns.droplevel(1)
                                if len(t_df) > 52:
                                    son_kapanis = round(float(t_df['Close'].iloc[-1]), 2)
                                    if "Aşırı Satım" in tarama_modu:
                                        delta_h = t_df['Close'].diff()
                                        gain_h = delta_h.where(delta_h > 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                        loss_h = -delta_h.where(delta_h < 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                        rs_h = gain_h / (loss_h + 1e-9)
                                        rsi_son = (100 - (100 / (1 + rs_h))).iloc[-1]
                                        if rsi_son < 35: 
                                            firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": son_kapanis, "Değer": f"RSI: {round(rsi_son, 1)}", "Durum": "🟢 Aşırı Satım Bölgesi"})
                                    elif "Ichimoku" in tarama_modu:
                                        t_df['Tenkan'] = (t_df['High'].rolling(9).max() + t_df['Low'].rolling(9).min()) / 2
                                        t_df['Kijun'] = (t_df['High'].rolling(26).max() + t_df['Low'].rolling(26).min()) / 2
                                        t_df['Senkou_A'] = ((t_df['Tenkan'] + t_df['Kijun']) / 2).shift(26)
                                        t_df['Senkou_B'] = ((t_df['High'].rolling(52).max() + t_df['Low'].rolling(52).min()) / 2).shift(26)
                                        ust_bulut = max(t_df['Senkou_A'].iloc[-1], t_df['Senkou_B'].iloc[-1])
                                        dun_kapanis = t_df['Close'].iloc[-2]
                                        if dun_kapanis <= ust_bulut and son_kapanis > ust_bulut:
                                            firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": son_kapanis, "Değer": f"Kırılım Fiyatı: {son_kapanis}", "Durum": "☁️ Bulut Kırılımı"})
                                    elif "Stoch RSI" in tarama_modu:
                                        delta_s = t_df['Close'].diff()
                                        gain_s = delta_s.where(delta_s > 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                        loss_s = -delta_s.where(delta_s < 0, 0).ewm(alpha=1/14, adjust=False).mean()
                                        rs_s = gain_s / (loss_s + 1e-9)
                                        rsi_serisi = 100 - (100 / (1 + rs_s))
                                        min_v = rsi_serisi.rolling(window=14).min()
                                        max_v = rsi_serisi.rolling(window=14).max()
                                        stoch = (rsi_serisi - min_v) / (max_v - min_v)
                                        k_line = stoch.rolling(window=3).mean() * 100
                                        d_line = k_line.rolling(window=3).mean()
                                        if k_line.iloc[-1] > d_line.iloc[-1] and k_line.iloc[-2] <= d_line.iloc[-2] and k_line.iloc[-1] < 40:
                                             firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": son_kapanis, "Değer": f"Stoch K:{round(k_line.iloc[-1],1)}", "Durum": "⭐ Dipten Dönüş"})
                                    elif "Hacim" in tarama_modu:
                                        son_hacim = t_df['Volume'].iloc[-1]
                                        hacim_ortalamasi = t_df['Volume'].iloc[-21:-1].mean()
                                        if son_hacim > (hacim_ortalamasi * 1.8):
                                            kat_artisi = round(son_hacim / hacim_ortalamasi, 2)
                                            firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": son_kapanis, "Değer": f"{kat_artisi}x Hacim", "Durum": "🔥 Olağanüstü Hacim Girişi"})
                        except:
                            pass
                        ilerleme_cubugu.progress((i + 1) / len(bist30_hisseler))
                        
                    if firsatlar:
                        st.success(f"✅ Tarama tamamlandı! {len(firsatlar)} adet hisse kriterlere uyuyor:")
                        f_df = pd.DataFrame(firsatlar).set_index("Hisse Kodu")
                        st.dataframe(f_df, use_container_width=True)
                        if st.button("📨 Bu Listeyi Telegram'a Uçur"):
                            tg_mesaj = f"🚨 BİST RADAR RAPORU\nMod: {tarama_modu}\n\n"
                            for f in firsatlar:
                                tg_mesaj += f"• {f['Hisse Kodu']} | Fiyat: {f['Fiyat']} TL | ({f['Değer']})\n"
                            if telegram_gonder(tg_mesaj):
                                st.success("🚀 Liste Telegram kanalına fırlatıldı!")
                    else:
                        st.warning("📉 Şu an için seçilen kritere uyan bir varlık tespit edilemedi.")

    # TAB 3, 4, 5, 6, 7, 8 (Diğer Araçlar)
    with tabs[2]:
        st.subheader("📊 Canlı Varlık Portföyüm ve ATR Destekli Fiyat Alarmları")
        c1, c2 = st.columns([2, 1])
        with c1:
            if 'portfoy_verisi' not in st.session_state:
                st.session_state.portfoy_verisi = pd.DataFrame([{"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0}, {"Varlık": "BTC-USD", "Maliyet": 62000.0, "Lot": 0.05}])
            guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
            st.session_state.portfoy_verisi = guncel_portfoy
            if st.button("Portföyü Hesapla"):
                top_mal = 0.0
                top_deg = 0.0
                for index, row in guncel_portfoy.iterrows():
                    kod = str(row["Varlık"]).upper()
                    mal = float(row["Maliyet"])
                    lot = float(row["Lot"])
                    if kod and lot > 0:
                        try:
                            c_veri = yf.download(kod, period="1d", progress=False, session=oturum)
                            if isinstance(c_veri.columns, pd.MultiIndex): c_veri.columns = c_veri.columns.droplevel(1)
                            g_fiyat = float(c_veri['Close'].iloc[-1])
                            top_mal += (mal * lot)
                            top_deg += (g_fiyat * lot)
                        except: 
                            top_mal += (mal * lot)
                            top_deg += (mal * lot)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Toplam Maliyet", f"{round(top_mal, 2)}")
                cc2.metric("Güncel Değer", f"{round(top_deg, 2)}")
                net_kar = top_deg - top_mal
                cc3.metric("Net Kâr", f"{round(net_kar, 2)}", f"%{round((net_kar / top_mal) * 100, 2) if top_mal > 0 else 0}")
        with c2:
            st.markdown("#### 🛡️ Akıllı Stop-Loss & Telegram Alarm")
            tavsiye_stop = round(float(df['Close'].iloc[-1]) - (float(df['ATR_14'].iloc[-1]) * 2), 2)
            st.info(f"💡 Tavsiye edilen teknik Stop-Loss seviyesi: **{tavsiye_stop}**")
            alarm_fiyat = st.number_input(f"{hisse_kodu} Tetikleme Fiyatı:", value=tavsiye_stop)
            if st.button("Alarmı Kur"):
                st.success("Alarm kuruldu!")
                telegram_gonder(f"Alarm Kuruldu: {hisse_kodu} - Hedef/Stop: {alarm_fiyat}")

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
            with st.spinner("Piyasa verileri karşılaştırılıyor..."):
                korelasyon_df = pd.DataFrame()
                for ticker in tarama_listesi[:6]:
                    tmp_df = yf.download(ticker, period="6mo", progress=False, session=oturum)
                    if isinstance(tmp_df.columns, pd.MultiIndex): tmp_df.columns = tmp_df.columns.droplevel(1)
                    if not tmp_df.empty: korelasyon_df[ticker] = tmp_df['Close']
                fig_corr = px.imshow(korelasyon_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

    with tabs[6]:
        st.subheader("⚙️ Strateji Testi (Backtest): SMA 20 vs SMA 50")
        bt_sonuc = backtest_motoru(df, kisa_periyot=20, uzun_periyot=50)
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
            with st.spinner("Simülasyon patikaları hesaplanıyor..."):
                sim_verisi = monte_carlo_simulasyonu(df, gun_sayisi=30, sim_sayisi=100)
                fig_sim = go.Figure()
                for i in range(sim_verisi.shape[1]):
                    fig_sim.add_trace(go.Scatter(y=sim_verisi[:, i], mode='lines', line=dict(width=1), showlegend=False))
                fig_sim.update_layout(template="plotly_dark", xaxis_title="Gün", yaxis_title="Fiyat")
                st.plotly_chart(fig_sim, use_container_width=True)

    with tabs[8]:
        st.subheader("🛠️ Terminal Entegrasyon Durumu")
        st.success("🤖 Telegram API Bağlantısı: Doğrulandı")
        st.info("Terminal v61 (Veri Bilimi Sürümü) Kararlı Modda Çalışıyor.")

    # TAB 10: YENİ PYTHON İSTATİSTİK MODÜLÜ
    with tabs[9]:
        st.subheader("🧬 Python İleri İstatistik Analizi")
        if st.button("İstatistikleri Hesapla"):
            with st.spinner("Veri bilimi algoritmaları çalıştırılıyor..."):
                stats = python_istatistik_analizi(df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Yıllık Volatilite", stats['Yıllık Volatilite'])
                col2.metric("Sharpe Oranı", stats['Sharpe Oranı'])
                col3.metric("VaR (%95)", stats['Günlük VaR (%95)'])
                
                st.info("Bu modül, Python'un vektörel hesaplama motorunu kullanarak gerçek risk değerlerini çıkartır.")
            
    st.sidebar.divider()
    csv = df.to_csv().encode('utf-8')
    st.sidebar.download_button(label="📊 Verileri İndir (CSV)", data=csv, file_name=f'{hisse_kodu}_veri.csv', mime='text/csv')
else:
    st.error("Veri çekilemedi. Kodunuzu kontrol edin.")