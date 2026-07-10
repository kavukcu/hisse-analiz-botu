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

st.set_page_config(layout="wide", page_title="God Mode Terminal v22.0")
st.title("👁️ Pro Küresel Yatırım Terminali v22.0 (Algoritmik Pro Sürüm)")

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
def haftalik_veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, interval='1wk', session=oturum)

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}

def supertrend_hesapla(df, period=10, multiplier=3):
    df_st = df.copy()
    tr1 = df_st['High'] - df_st['Low']
    tr2 = abs(df_st['High'] - df_st['Close'].shift(1))
    tr3 = abs(df_st['Low'] - df_st['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_st['ATR'] = tr.rolling(period).mean()
    
    hl2 = (df_st['High'] + df_st['Low']) / 2
    df_st['Upperband'] = hl2 + (multiplier * df_st['ATR'])
    df_st['Lowerband'] = hl2 - (multiplier * df_st['ATR'])
    df_st['InUptrend'] = True
    df_st['Supertrend'] = np.nan
    
    for i in range(1, len(df_st.index)):
        if df_st['Close'].iloc[i] > df_st['Upperband'].iloc[i-1]:
            df_st.loc[df_st.index[i], 'InUptrend'] = True
        elif df_st['Close'].iloc[i] < df_st['Lowerband'].iloc[i-1]:
            df_st.loc[df_st.index[i], 'InUptrend'] = False
        else:
            df_st.loc[df_st.index[i], 'InUptrend'] = df_st['InUptrend'].iloc[i-1]
            if df_st['InUptrend'].iloc[i] and df_st['Lowerband'].iloc[i] < df_st['Lowerband'].iloc[i-1]:
                df_st.loc[df_st.index[i], 'Lowerband'] = df_st['Lowerband'].iloc[i-1]
            if not df_st['InUptrend'].iloc[i] and df_st['Upperband'].iloc[i] > df_st['Upperband'].iloc[i-1]:
                df_st.loc[df_st.index[i], 'Upperband'] = df_st['Upperband'].iloc[i-1]
                
        df_st.loc[df_st.index[i], 'Supertrend'] = df_st['Lowerband'].iloc[i] if df_st['InUptrend'].iloc[i] else df_st['Upperband'].iloc[i]
    return df_st

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

with st.spinner('Piyasa verileri analiz ediliyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    df_haftalik = haftalik_veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.droplevel(1)
    if not df_haftalik.empty and isinstance(df_haftalik.columns, pd.MultiIndex):
        df_haftalik.columns = df_haftalik.columns.droplevel(1)
    
    # --- İNDİKATÖR HESAPLAMALARI ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df_haftalik['SMA_20'] = df_haftalik['Close'].rolling(window=20).mean()
    
    # SuperTrend
    df = supertrend_hesapla(df)
    
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

    tabs = st.tabs([
        "📈 Teknik & AI Tahmin", "💼 Cüzdan & Alarm", "🏢 Temel Analiz", 
        "🔍 Akıllı Radar (Price Action)", "📊 Isı Haritası", 
        "⚙️ Backtest", "🎲 Risk Simülasyonu"
    ])

    # TAB 1: TEKNİK & AI TAHMİN (SUPERTREND VE MTF EKLENDİ)
    with tabs[0]:
        c_mtf1, c_mtf2, c_mtf3 = st.columns(3)
        with c_mtf1:
            gunluk_trend = "🟢 YÜKSELİŞ" if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else "🔴 DÜŞÜŞ"
            st.metric("Günlük Ana Trend (SMA 20)", gunluk_trend)
        with c_mtf2:
            haftalik_trend = "🟢 YÜKSELİŞ" if df_haftalik['Close'].iloc[-1] > df_haftalik['SMA_20'].iloc[-1] else "🔴 DÜŞÜŞ"
            st.metric("Haftalık MTF Trend (SMA 20)", haftalik_trend)
        with c_mtf3:
            if gunluk_trend == "🟢 YÜKSELİŞ" and haftalik_trend == "🟢 YÜKSELİŞ":
                mtf_karar = "✅ GÜÇLÜ AL"
            elif gunluk_trend == "🔴 DÜŞÜŞ" and haftalik_trend == "🔴 DÜŞÜŞ":
                mtf_karar = "❌ GÜÇLÜ SAT"
            else:
                mtf_karar = "⏳ NÖTR / BEKLE"
            st.metric("Çoklu Zaman Dilimi (MTF) Sinyali", mtf_karar)
            
        st.markdown("---")
        st.subheader("📈 Profesyonel Teknik Göstergeler (SuperTrend & MACD)")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
        
        # SuperTrend Çizimi (Yeşil/Kırmızı koşullu renklendirme)
        st_colors = ['#00E676' if val else '#FF1744' for val in df['InUptrend']]
        fig.add_trace(go.Scatter(x=df.index, y=df['Supertrend'], mode='lines', name="SuperTrend", line=dict(color='yellow', width=2), marker=dict(color=st_colors)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='cyan', dash='dot')), row=1, col=1)
        
        tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="AI Tahmini", line=dict(color='magenta', width=3, dash='dot')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#2962FF')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Sinyal", line=dict(color='#FF6D00')), row=2, col=1)
        hist_colors = np.where(df['MACD_Hist'] < 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram", marker_color=hist_colors), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: CÜZDAN & ALARM
    with tabs[1]:
        st.subheader("📊 Canlı Varlık Portföyüm ve Fiyat Alarmları")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            if 'portfoy_verisi' not in st.session_state:
                st.session_state.portfoy_verisi = pd.DataFrame([{"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0}])
                
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
                            top_mal += (mal * lot); top_deg += (g_fiyat * lot)
                        except: pass
                
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Toplam Maliyet", f"{round(top_mal, 2)}")
                cc2.metric("Güncel Değer", f"{round(top_deg, 2)}")
                net_kar = top_deg - top_mal
                cc3.metric("Net Kâr", f"{round(net_kar, 2)}", f"%{round((net_kar / top_mal) * 100, 2) if top_mal > 0 else 0}")

        with c2:
            st.markdown("#### 🔔 Telegram Alarm Kur")
            guncel_son_fiyat = float(df['Close'].iloc[-1])
            alarm_fiyat = st.number_input(f"{hisse_kodu} Hedef Fiyatı:", min_value=0.0, value=guncel_son_fiyat * 1.05)
            if st.button("Alarmı Kur"):
                st.success("Alarm kuruldu!")
                telegram_gonder(f"Alarm Kuruldu: {hisse_kodu} - Hedef: {alarm_fiyat}")

    # TAB 3: TEMEL ANALİZ
    with tabs[2]:
        st.subheader(f"🏢 {info.get('longName', hisse_kodu)} Temel Veriler")
        c1, c2, c3 = st.columns(3)
        c1.metric("F/K Oranı (P/E)", info.get('trailingPE', '-'))
        c2.metric("PD/DD (P/B)", info.get('priceToBook', '-'))
        c3.metric("Piyasa Değeri", info.get('marketCap', '-'))

    # TAB 4: AKILLI RADAR (MUM FORMASYONLARI EKLENDİ)
    with tabs[3]:
        st.subheader(f"🔍 {piyasa_tipi} Akıllı Multi-Radar")
        
        tarama_modu = st.radio("Tarama Modu Seçin:", [
            "🟢 Aşırı Satım Radarı (RSI < 35)", 
            "🔥 Hacim Patlaması Radarı",
            "💼 Değer Avcısı (Düşük F/K ve PD/DD)",
            "🕯️ Mum Formasyonları (Price Action)"
        ])
        
        if st.button("🚀 Radarı Çalıştır"):
            with st.spinner("Piyasa Taranıyor, Veriler Çekiliyor..."):
                firsatlar = []
                bist30_hisseler = ["AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS"] 
                ilerleme_cubugu = st.progress(0)
                
                for i, hisse in enumerate(bist30_hisseler):
                    try:
                        temiz_ad = hisse.replace(".IS", "")
                        if "Değer Avcısı" in tarama_modu:
                            s_info = sirket_bilgisi_getir(hisse)
                            fk = s_info.get('trailingPE', 999)
                            pddd = s_info.get('priceToBook', 999)
                            if isinstance(fk, (int, float)) and isinstance(pddd, (int, float)) and 0 < fk < 10 and 0 < pddd < 3:
                                firsatlar.append({"Hisse Kodu": temiz_ad, "Değer": f"F/K: {round(fk, 2)}", "Durum": "Ucuz Hisseler"})
                        else:
                            t_df = veri_yukle(hisse, datetime.today() - timedelta(days=60), datetime.today())
                            if not t_df.empty and isinstance(t_df.columns, pd.MultiIndex): t_df.columns = t_df.columns.droplevel(1)
                            
                            if len(t_df) > 5:
                                O, H, L, C = t_df['Open'].iloc[-1], t_df['High'].iloc[-1], t_df['Low'].iloc[-1], t_df['Close'].iloc[-1]
                                p_O, p_H, p_L, p_C = t_df['Open'].iloc[-2], t_df['High'].iloc[-2], t_df['Low'].iloc[-2], t_df['Close'].iloc[-2]
                                
                                body = abs(C - O)
                                range_hl = H - L
                                
                                if "Mum Formasyonları" in tarama_modu:
                                    if body < (range_hl * 0.1): 
                                        firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": round(C, 2), "Durum": "⚖️ Doji (Kararsızlık)"})
                                    elif (C > O) and ((O - L) > 2 * body) and ((H - C) < 0.2 * body):
                                        firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": round(C, 2), "Durum": "🔨 Çekiç (Dönüş)"})
                                    elif (p_C < p_O) and (C > O) and (C > p_O) and (O < p_C):
                                        firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": round(C, 2), "Durum": "🐂 Yutan Boğa"})
                                elif "Aşırı Satım" in tarama_modu:
                                    delta_h = t_df['Close'].diff()
                                    rs_h = delta_h.where(delta_h > 0, 0).ewm(alpha=1/14, adjust=False).mean() / (-delta_h.where(delta_h < 0, 0).ewm(alpha=1/14, adjust=False).mean() + 1e-9)
                                    rsi_son = (100 - (100 / (1 + rs_h))).iloc[-1]
                                    if rsi_son < 35: firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": round(C, 2), "Durum": f"🟢 RSI: {round(rsi_son, 1)}"})
                                elif "Hacim" in tarama_modu:
                                    if t_df['Volume'].iloc[-1] > (t_df['Volume'].iloc[-21:-1].mean() * 1.8):
                                        firsatlar.append({"Hisse Kodu": temiz_ad, "Fiyat": round(C, 2), "Durum": "🔥 Hacim Artışı"})
                    except: pass
                    ilerleme_cubugu.progress((i + 1) / len(bist30_hisseler))
                    
                if firsatlar:
                    st.success(f"✅ {len(firsatlar)} adet sonuç bulundu:")
                    st.dataframe(pd.DataFrame(firsatlar).set_index("Hisse Kodu"), use_container_width=True)
                else:
                    st.warning("📉 Kriterlere uyan varlık tespit edilemedi.")

    # DİĞER SEKMELER (Isı Haritası, Backtest, Simülasyon)
    with tabs[4]:
        st.subheader("📊 Korelasyon Matrisi")
        if st.button("Isı Haritasını Oluştur"):
            korelasyon_df = pd.DataFrame()
            for ticker in tarama_listesi[:6]:
                tmp_df = yf.download(ticker, period="6mo", progress=False, session=oturum)
                if isinstance(tmp_df.columns, pd.MultiIndex): tmp_df.columns = tmp_df.columns.droplevel(1)
                if not tmp_df.empty: korelasyon_df[ticker] = tmp_df['Close']
            st.plotly_chart(px.imshow(korelasyon_df.corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

    with tabs[5]:
        st.subheader("⚙️ SMA Kesişim Stratejisi Backtest")
        bt_df = df[['Close']].copy()
        bt_df['Sinyal'] = np.where(bt_df['Close'].rolling(20).mean() > bt_df['Close'].rolling(50).mean(), 1, 0)
        bt_df['Getiri'] = bt_df['Close'].pct_change() * bt_df['Sinyal'].shift(1)
        st.line_chart((1 + bt_df[['Getiri']]).cumprod() * 100)

    with tabs[6]:
        st.subheader("🎲 Monte Carlo Risk Simülasyonu")
        if st.button("Simülasyonu Başlat"):
            sim_verisi = monte_carlo_simulasyonu(df)
            fig_sim = go.Figure()
            for i in range(sim_verisi.shape[1]):
                fig_sim.add_trace(go.Scatter(y=sim_verisi[:, i], mode='lines', line=dict(width=1), showlegend=False))
            fig_sim.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_sim, use_container_width=True)
else:
    st.error("Veri çekilemedi. Kodunuzu kontrol edin.")