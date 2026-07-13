import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v100")
st.title("👁️ Pro Küresel Yatırım Terminali (SMC, Fibo & YZ Motoru)")

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
@st.cache_data(ttl=300, show_spinner=False)
def veri_yukle(ticker, start, end):
    import time, logging
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                session=oturum,
                progress=False,
                auto_adjust=True,
                threads=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            gerekli=["Open","High","Low","Close","Volume"]
            if df.empty or any(c not in df.columns for c in gerekli):
                raise ValueError("Eksik veya boş veri")
            return df.dropna()
        except Exception as e:
            logging.warning(f"Veri indirilemedi: {e}")
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}

def mum_formasyonlarini_bul(df):
    df_f = df.copy()
    govde = abs(df_f['Close'] - df_f['Open'])
    mum_boyu = df_f['High'] - df_f['Low']
    df_f['Doji'] = govde <= (mum_boyu * 0.1)
    df_f['Bullish_Engulfing'] = (df_f['Close'].shift(1) < df_f['Open'].shift(1)) & (df_f['Open'] < df_f['Close'].shift(1)) & (df_f['Close'] > df_f['Open'].shift(1))
    df_f['Bearish_Engulfing'] = (df_f['Close'].shift(1) > df_f['Open'].shift(1)) & (df_f['Open'] > df_f['Close'].shift(1)) & (df_f['Close'] < df_f['Open'].shift(1))
    alt_golge = df_f[['Close', 'Open']].min(axis=1) - df_f['Low']
    ust_golge = df_f['High'] - df_f[['Close', 'Open']].max(axis=1)
    df_f['Hammer'] = (alt_golge > (govde * 2)) & (ust_golge < (govde * 0.2)) & (govde > 0)
    return df_f

def grafik_formasyon_bul(df, window=10, tolerans=0.03):
    try:
        df_form = df.copy()
        df_form['Local_Max'] = df_form['High'] == df_form['High'].rolling(window=window*2+1, center=True).max()
        df_form['Local_Min'] = df_form['Low'] == df_form['Low'].rolling(window=window*2+1, center=True).min()
        
        ikili_tepeler = []
        ikili_dipler = []
        
        max_idx = df_form[df_form['Local_Max']].index
        min_idx = df_form[df_form['Local_Min']].index
        
        for i in range(1, len(max_idx)):
            f1, f2 = df_form.loc[max_idx[i-1], 'High'], df_form.loc[max_idx[i], 'High']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (max_idx[i] - max_idx[i-1]).days
                if 5 < zaman_farki < 90:
                    ikili_tepeler.append((max_idx[i-1], max_idx[i], f1, f2))
                    
        for i in range(1, len(min_idx)):
            f1, f2 = df_form.loc[min_idx[i-1], 'Low'], df_form.loc[min_idx[i], 'Low']
            if abs(f1 - f2) / f1 <= tolerans:
                zaman_farki = (min_idx[i] - min_idx[i-1]).days
                if 5 < zaman_farki < 90:
                    ikili_dipler.append((min_idx[i-1], min_idx[i], f1, f2))
                    
        return ikili_tepeler, ikili_dipler
    except:
        return [], []

def smc_hesapla(df):
    df_smc = df.copy()
    df_smc['FVG_Bullish'] = (df_smc['Low'] > df_smc['High'].shift(2)) & (df_smc['Close'].shift(1) > df_smc['Open'].shift(1))
    df_smc['FVG_Bearish'] = (df_smc['High'] < df_smc['Low'].shift(2)) & (df_smc['Close'].shift(1) < df_smc['Open'].shift(1))
    return df_smc

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
            metin = (str(n.get('title', '')) + " " + str(n.get('summary', ''))).lower()
            olumlu_skor = sum(1 for k in olumlu if k in metin)
            olumsuz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if olumlu_skor > olumsuz_skor else ("🔴 OLUMSUZ" if olumsuz_skor > olumlu_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: return []

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
    df_ml = df[['Close']].copy()
    df_ml['Lag1'] = df_ml['Close'].shift(1)
    df_ml['Lag2'] = df_ml['Close'].shift(2)
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml.dropna(inplace=True)
    
    X = df_ml[['Lag1', 'Lag2', 'SMA_10']]
    y = df_ml['Close']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.values, y) 
    
    tahminler = []
    son_satir = df_ml.iloc[-1]
    lag1 = son_satir['Close']
    lag2 = son_satir['Lag1']
    sma_10 = son_satir['SMA_10']
    
    for _ in range(gelecek_gun):
        pred = model.predict([[lag1, lag2, sma_10]])[0]
        tahminler.append(pred)
        lag2 = lag1
        lag1 = pred
        sma_10 = (sma_10 * 9 + pred) / 10
        
    tarihler = [df.index[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
    return tarihler, tahminler

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def python_istatistik_analizi(df):
    df_stats = df.copy()
    df_stats['Log_Return'] = np.log(df_stats['Close'] / df_stats['Close'].shift(1))
    volatilite = df_stats['Log_Return'].std() * np.sqrt(252)
    getiri = df_stats['Log_Return'].mean() * 252
    sharpe = (getiri - 0.40) / volatilite
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
    varsayilan_hisse = "MIATK.IS"
    tarama_listesi = ["THYAO.IS", "AKBNK.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "GARAN.IS", "SASA.IS"]
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
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = ema(df['Close'],12)
    df['EMA_26'] = ema(df['Close'],26)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
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

    df['True_Range'] = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
    df['ATR_14'] = df['True_Range'].rolling(14).mean()
    df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

    df = smc_hesapla(df)

    # ---> YENİ SEKMEYİ BURAYA EKLİYORUZ <---
    tabs = st.tabs([
        "📈 SMC & Quant Fiyat Hareketi", "🔍 Akıllı Asenkron Radar", "💼 Cüzdan & Akıllı Stop", 
        "🏢 Temel & Temettü", "📰 Haber", "📊 Isı Haritası", 
        "⚙️ Backtest", "🎲 Risk Simülasyonu", "🛠️ Sistem Durumu", "🧬 Python İstatistik",
        "🤖 YZ Öneri Motoru"
    ])

    with tabs[0]:
        st.subheader("📈 Kurumsal Quant Grafiği & Likidite Analizi")
        
        c_ayar1, c_ayar2, c_ayar3 = st.columns(3)
        with c_ayar1:
            goster_vpvr = st.checkbox("📊 Hacim Profili (VPVR)", value=True)
            goster_smc = st.checkbox("🏦 FVG & Likidite Boşlukları (SMC)", value=True)
            goster_fibo = st.checkbox("📐 Altın Oran (Fibonacci)", value=True)
        with c_ayar2:
            goster_grafik_formasyon = st.checkbox("📉 İkili Tepe/Dip (Makro)", value=True)
            goster_formasyon = st.checkbox("🕯️ Mum Formasyonları (Mikro)", value=False)
        with c_ayar3:
            goster_vwap = st.checkbox("⚖️ VWAP (Kurumsal Maliyet)", value=False)
            goster_ai = st.checkbox("🤖 Random Forest AI Tahmini", value=True)
            
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
        
        if goster_vpvr:
            hacim_bölümleri, fiyat_araliklari = np.histogram(df['Close'].dropna(), bins=40, weights=df['Volume'].dropna())
            bölüm_merkezleri = (fiyat_araliklari[:-1] + fiyat_araliklari[1:]) / 2
            max_hacim = hacim_bölümleri.max()
            sure_uzunlugu = df.index[-1] - df.index[0]
            x_koordinatlari = [df.index[0] + sure_uzunlugu * 0.3 * (v / max_hacim) for v in hacim_bölümleri]
            
            for i in range(len(bölüm_merkezleri)):
                fig.add_shape(type="line", x0=df.index[0], y0=bölüm_merkezleri[i], x1=x_koordinatlari[i], y1=bölüm_merkezleri[i], line=dict(color="rgba(100, 150, 255, 0.4)", width=4), row=1, col=1)
            
            poc_index = np.argmax(hacim_bölümleri)
            poc_fiyat = bölüm_merkezleri[poc_index]
            fig.add_hline(y=poc_fiyat, line_dash="solid", line_color="red", annotation_text="POC (En Yoğun Maliyet)", row=1, col=1)
        
        if goster_smc:
            for i in range(2, len(df)):
                bitis_idx = i+5 if i+5 < len(df) else len(df)-1 
                if df['FVG_Bullish'].iloc[i]:
                    fig.add_shape(type="rect", x0=df.index[i-2], y0=df['High'].iloc[i-2], x1=df.index[bitis_idx], y1=df['Low'].iloc[i], fillcolor="rgba(0, 255, 0, 0.2)", line=dict(width=0), layer="below", row=1, col=1)
                elif df['FVG_Bearish'].iloc[i]:
                    fig.add_shape(type="rect", x0=df.index[i-2], y0=df['Low'].iloc[i-2], x1=df.index[bitis_idx], y1=df['High'].iloc[i], fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0), layer="below", row=1, col=1)
            
        if goster_fibo:
            max_fiyat = df['High'].max()
            min_fiyat = df['Low'].min()
            fark = max_fiyat - min_fiyat
            seviyeler = {0: "100% (Tepe)", 0.236: "76.4%", 0.382: "61.8%", 0.5: "50%", 0.618: "38.2% (Altın Oran)", 0.786: "21.4%", 1: "0% (Dip)"}
            renkler = ['#ff0000', '#ff9900', '#ffff00', '#33cc33', '#00ffcc', '#cc33ff', '#999999']
            for i, (level, oran) in enumerate(seviyeler.items()):
                fiyat_seviyesi = max_fiyat - (fark * level)
                if level == 0.618:
                    fig.add_hline(y=fiyat_seviyesi, line_dash="solid", line_width=2, line_color="#00ffcc", annotation_text=f"⭐ {oran}", row=1, col=1)
                else:
                    fig.add_hline(y=fiyat_seviyesi, line_dash="dash", line_width=1, line_color=renkler[i], annotation_text=f"Fibo {oran}", row=1, col=1)
        
        if goster_grafik_formasyon:
            ikili_tepeler, ikili_dipler = grafik_formasyon_bul(df)
            for tepe in ikili_tepeler:
                fig.add_shape(type="line", x0=tepe[0], y0=tepe[2], x1=tepe[1], y1=tepe[3], line=dict(color="red", width=3, dash="dot"), row=1, col=1)
                fig.add_annotation(x=tepe[1], y=tepe[3], text="📉 İkili Tepe", showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="red"), row=1, col=1)
            for dip in ikili_dipler:
                fig.add_shape(type="line", x0=dip[0], y0=dip[2], x1=dip[1], y1=dip[3], line=dict(color="green", width=3, dash="dot"), row=1, col=1)
                fig.add_annotation(x=dip[1], y=dip[3], text="📈 İkili Dip", showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color="green"), row=1, col=1)
                
        if goster_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)
        
        if goster_formasyon:
            df_form = mum_formasyonlarini_bul(df)
            yutan_boga = df_form[df_form['Bullish_Engulfing']]
            fig.add_trace(go.Scatter(x=yutan_boga.index, y=yutan_boga['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00ff00', size=12), name='Yutan Boğa'), row=1, col=1)

        if goster_ai:
            tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30)
            fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="RF Tahmini", line=dict(color='magenta', width=3, dash='dot')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#2962FF')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Sinyal", line=dict(color='#FF6D00')), row=2, col=1)
        hist_colors = np.where(df['MACD_Hist'] < 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram", marker_color=hist_colors), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_K'], name="%K", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_D'], name="%D", line=dict(color='orange')), row=3, col=1)

        fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=1000, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.subheader(f"⚡ {piyasa_tipi} Multi-Threading Hızlı Radar")
        st.info("Bu bölümde seçili piyasaya ait tarama listesi asenkron (çoklu iş parçacığı) olarak taranır ve YZ/Kurumsal Karar motorundan geçirilir.")
        
        if st.button("🚀 Akıllı Radarı Başlat"):
            with st.spinner(f"Sistem {len(tarama_listesi)} varlığı aynı anda tarıyor, lütfen bekleyin..."):
                
                # Arka plan iş parçacıkları için güvenli, önbelleksiz tarama fonksiyonu
                def tekli_tara(sembol):
                    try:
                        # Önbellek hatasını engellemek için doğrudan yf.download çağırıyoruz
                        df_radar = yf.download(
                            sembol,
                            start=(datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d'),
                            end=datetime.today().strftime('%Y-%m-%d'),
                            session=oturum,
                            progress=False,
                            auto_adjust=True
                        )
                        
                        if isinstance(df_radar.columns, pd.MultiIndex):
                            df_radar.columns = df_radar.columns.droplevel(1)
                            
                        if df_radar.empty or len(df_radar) < 20: 
                            return None
                            
                        df_radar = df_radar.dropna()

                        # Risk motorunun doğru çalışması için ATR hesaplamasını ekliyoruz
                        df_radar['True_Range'] = np.max(pd.concat([
                            df_radar['High'] - df_radar['Low'], 
                            (df_radar['High'] - df_radar['Close'].shift()).abs(), 
                            (df_radar['Low'] - df_radar['Close'].shift()).abs()
                        ], axis=1), axis=1)
                        df_radar['ATR_14'] = df_radar['True_Range'].rolling(14).mean()

                        # İndikatör zincirini radar verisine işlet
                        df_radar = calculate_adx(df_radar)
                        df_radar = calculate_supertrend(df_radar)
                        df_radar = calculate_mfi(df_radar)
                        df_radar = calculate_cci(df_radar)
                        df_radar = detect_bos_choch(df_radar)
                        
                        # Kurumsal karar mekanizmasını çalıştır
                        karar = institutional_decision(df_radar)
                        kapanis = round(float(df_radar['Close'].iloc[-1]), 2)
                        
                        return {
                            "Sembol": sembol,
                            "Fiyat": kapanis,
                            "Sinyal": karar["decision"],
                            "Güç Skoru": karar["score"],
                            "Risk Skoru": karar["risk"],
                            "Piyasa Rejimi": karar["regime"]
                        }
                    except Exception as e:
                        return None

                sonuclar = []
                # 10 iş parçacığı ile asenkron paralel sorgu
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for sonuc in executor.map(tekli_tara, tarama_listesi):
                        if sonuc is not None:
                            sonuclar.append(sonuc)
                
                # Sonuç ekranını bas
                if sonuclar:
                    df_sonuc = pd.DataFrame(sonuclar)
                    df_sonuc = df_sonuc.sort_values(by="Güç Skoru", ascending=False).reset_index(drop=True)
                    
                    def renk_ver(val):
                        if val in ["GÜÇLÜ AL", "AL"]:
                            return 'color: #00ff00; font-weight: bold;'
                        elif val in ["SAT", "GÜÇLÜ SAT", "AZALT"]:
                            return 'color: #ff3333; font-weight: bold;'
                        elif val == "TUT":
                            return 'color: #ffff00; font-weight: bold;'
                        return ''
                        
                    st.success("Tüm taramalar başarıyla tamamlandı!")
                    st.dataframe(
                        df_sonuc.style.applymap(renk_ver, subset=['Sinyal']), 
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.warning("Tarama listesindeki varlıklardan veri çekilemedi. Lütfen tarih aralığını veya sembol formatlarını kontrol edin.")
    with tabs[2]:
        st.subheader("📊 Canlı Varlık Portföyüm ve ATR Destekli Fiyat Alarmları")
        c1, c2 = st.columns([2, 1])
        with c2:
            tavsiye_stop = round(float(df['Close'].iloc[-1]) - (float(df['ATR_14'].iloc[-1]) * 2), 2)
            st.info(f"💡 Tavsiye edilen teknik Stop-Loss: **{tavsiye_stop}**")

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
        st.info("Korelasyon analizi için butona basarak işlemi başlatabilirsiniz.")

    with tabs[6]:
        st.subheader("⚙️ Strateji Testi (Backtest): SMA 20 vs SMA 50")
        bt_sonuc = backtest_motoru(df)
        if not bt_sonuc.empty:
            son_piyasa = bt_sonuc['Piyasa_Kumulatif'].iloc[-1] - 100
            son_strateji = bt_sonuc['Strateji_Kumulatif'].iloc[-1] - 100
            c1, c2 = st.columns(2)
            c1.metric("Alıp Bekleseydin", f"%{round(son_piyasa, 2)}")
            c2.metric("Strateji ile Alsatsaydın", f"%{round(son_strateji, 2)}", delta=round(son_strateji - son_piyasa, 2))

    with tabs[7]:
        st.subheader("🎲 Monte Carlo Risk Simülasyonu (Gelecek 30 Gün)")

    with tabs[8]:
        st.subheader("🛠️ Terminal Entegrasyon Durumu")
        st.success("🤖 YZ Öneri Motoru: Aktif")
        st.info("Terminal v100 Kararlı Modda Çalışıyor.")

    with tabs[9]:
        st.subheader("🧬 Python İleri İstatistik Analizi")
        if st.button("İstatistikleri Hesapla"):
            stats = python_istatistik_analizi(df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Yıllık Volatilite", stats['Yıllık Volatilite'])
            col2.metric("Sharpe Oranı", stats['Sharpe Oranı'])
            col3.metric("VaR (%95)", stats['Günlük VaR (%95)'])

# ==============================================================================
# ALTYAPI FONKSİYONLARI (v66'dan v100'e kadar) 
# YZ Motorunun beslendiği temel kod blokları 
# ==============================================================================

def calculate_adx(df, period=14):
    high=df["High"]; low=df["Low"]; close=df["Close"]
    plus_dm=high.diff()
    minus_dm=-low.diff()
    plus_dm=np.where((plus_dm>minus_dm)&(plus_dm>0),plus_dm,0.0)
    minus_dm=np.where((minus_dm>plus_dm)&(minus_dm>0),minus_dm,0.0)
    tr=pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(period).mean()
    plus_di=100*(pd.Series(plus_dm,index=df.index).rolling(period).sum()/atr)
    minus_di=100*(pd.Series(minus_dm,index=df.index).rolling(period).sum()/atr)
    dx=((plus_di-minus_di).abs()/(plus_di+minus_di))*100
    adx=dx.rolling(period).mean()
    df["PLUS_DI"]=plus_di
    df["MINUS_DI"]=minus_di
    df["ADX"]=adx
    return df

def calculate_supertrend(df, period=10, multiplier=3):
    hl2=(df["High"]+df["Low"])/2
    tr=pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(period).mean()
    upper=hl2+multiplier*atr
    lower=hl2-multiplier*atr
    st=[lower.iloc[0] if len(lower) else 0]
    trend=[True]
    for i in range(1,len(df)):
        if df["Close"].iloc[i]>upper.iloc[i-1]:
            trend.append(True)
        elif df["Close"].iloc[i]<lower.iloc[i-1]:
            trend.append(False)
        else:
            trend.append(trend[-1])
        st.append(lower.iloc[i] if trend[-1] else upper.iloc[i])
    df["SuperTrend"]=st
    df["ST_Trend"]=trend
    return df

def calculate_mfi(df, period=14):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    mf=tp*df["Volume"]
    pos=[0]; neg=[0]
    for i in range(1,len(df)):
        if tp.iloc[i]>tp.iloc[i-1]:
            pos.append(mf.iloc[i]); neg.append(0)
        else:
            pos.append(0); neg.append(mf.iloc[i])
    pos=pd.Series(pos,index=df.index).rolling(period).sum()
    neg=pd.Series(neg,index=df.index).rolling(period).sum()
    ratio=pos/neg.replace(0,np.nan)
    df["MFI"]=100-(100/(1+ratio))
    return df

def calculate_cci(df, period=20):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    sma=tp.rolling(period).mean()
    mad=(tp-sma).abs().rolling(period).mean()
    df["CCI"]=(tp-sma)/(0.015*mad)
    return df

def detect_bos_choch(df, swing=5):
    df=df.copy()
    df["SwingHigh"]=df["High"].rolling(swing,center=True).max()==df["High"]
    df["SwingLow"]=df["Low"].rolling(swing,center=True).min()==df["Low"]
    bos=[]; choch=[]
    last_high=None; last_low=None; trend=None
    for i,row in df.iterrows():
        b=False; c=False
        if row["SwingHigh"]:
            if last_high is not None and row["High"]>last_high and trend=="up": b=True
            last_high=row["High"]; trend="up"
        if row["SwingLow"]:
            if last_low is not None and row["Low"]<last_low and trend=="down": b=True
            last_low=row["Low"]; trend="down"
        if last_high is not None and row["Close"]>last_high: c=True
        if last_low is not None and row["Close"]<last_low: c=True
        bos.append(b); choch.append(c)
    df["BOS"]=bos
    df["CHOCH"]=choch
    return df

def generate_signal_score(df):
    score=0
    reasons=[]
    last=df.iloc[-1]
    if "ST_Trend" in df.columns and bool(last.get("ST_Trend",False)): score+=2; reasons.append("SuperTrend")
    if "ADX" in df.columns and last.get("ADX",0)>25: score+=1; reasons.append("ADX>25")
    if "PLUS_DI" in df.columns and "MINUS_DI" in df.columns:
        if last["PLUS_DI"]>last["MINUS_DI"]: score+=1; reasons.append("+DI")
    if "MFI" in df.columns:
        if last["MFI"]<20: score+=1; reasons.append("MFI Oversold")
        elif last["MFI"]>80: score-=1; reasons.append("MFI Overbought")
    if "CCI" in df.columns:
        if last["CCI"]>100: score+=1; reasons.append("CCI Strong")
        elif last["CCI"]<-100: score-=1; reasons.append("CCI Weak")
        
    if score>=5: signal="GÜÇLÜ AL"
    elif score>=3: signal="AL"
    elif score<=-2: signal="SAT"
    else: signal="NÖTR"
    return {"score":score, "signal":signal, "reasons":reasons}

def calculate_confidence_score(df):
    score = 50
    last = df.iloc[-1]
    try:
        if bool(last.get("ST_Trend", False)): score += 10
        if last.get("ADX", 0) > 25: score += 10
        mfi = last.get("MFI", 50)
        if mfi < 20: score += 10
        elif mfi > 80: score -= 10
        if bool(last.get("BOS", False)): score += 10
        if bool(last.get("CHOCH", False)): score += 5
    except: pass
    return max(0, min(100, score))

def calculate_volatility(df, period=14):
    return float(df["Close"].pct_change().rolling(period).std().iloc[-1]*100)

def calculate_risk_score(df):
    risk=50
    try:
        vol=calculate_volatility(df); risk+=min(vol*2,20)
        adx=float(df["ADX"].iloc[-1])
        risk += -10 if adx>35 else (10 if adx<15 else 0)
        atr=float(df["ATR_14"].iloc[-1]); close=float(df["Close"].iloc[-1])
        r=atr/close
        risk += 15 if r>0.05 else (-10 if r<0.02 else 0)
    except: pass
    return round(max(0,min(100,risk)),2)

def detect_market_regime(df):
    try: adx=float(df["ADX"].iloc[-1])
    except: adx=20
    try:
        atr=float(df["ATR_14"].iloc[-1]); close=float(df["Close"].iloc[-1]); rr=atr/close
    except: rr=0.02
    if adx>=25 and rr<0.04: return "TREND"
    if adx<20: return "YATAY (RANGE)"
    if rr>=0.04: return "VOLATİL"
    return "KARIŞIK"

def calculate_total_score(df):
    technical = max(0, min(10, generate_signal_score(df).get("score", 0)))
    confidence = calculate_confidence_score(df)
    total = max(0, min(100, (technical * 5) + confidence))
    if total >= 85: rating = "A+"
    elif total >= 70: rating = "A"
    elif total >= 55: rating = "B"
    elif total >= 40: rating = "C"
    else: rating = "D"
    return {"technical": technical, "confidence": confidence, "total": total, "rating": rating}

def calculate_trend_strength(df):
    last=df.iloc[-1]
    adx=float(last["ADX"]) if "ADX" in df.columns and not hasattr(last.get("ADX"),"isna") else float(last.get("ADX",0) or 0)
    conf=calculate_confidence_score(df)
    return {"adx":adx,"confidence":conf,"strength":round((min(adx,50)/50)*50+conf*0.5,2)}

def institutional_decision(df):
    total = calculate_total_score(df)
    risk = calculate_risk_score(df)
    regime = detect_market_regime(df)
    score = total["total"]
    if score >= 90 and risk < 35: d = "GÜÇLÜ AL"
    elif score >= 75: d = "AL"
    elif score >= 55: d = "TUT"
    elif score >= 40: d = "AZALT"
    else: d = "SAT"
    return {"decision": d, "score": score, "risk": risk, "regime": regime}

def advanced_technical_analysis(df):
    result = {}
    close = float(df["Close"].iloc[-1])
    if "RSI" in df.columns:
        rsi = float(df["RSI"].iloc[-1])
        if rsi >= 70: result["RSI"] = "AŞIRI ALIM"
        elif rsi <= 30: result["RSI"] = "AŞIRI SATIM"
        else: result["RSI"] = "NÖTR"
    if {"MACD", "MACD_Signal"}.issubset(df.columns):
        macd = float(df["MACD"].iloc[-1]); sig = float(df["MACD_Signal"].iloc[-1])
        result["MACD"] = "BOĞA (YÜKSELİŞ)" if macd > sig else "AYI (DÜŞÜŞ)"
    if {"EMA_12", "EMA_26"}.issubset(df.columns):
        e12 = float(df["EMA_12"].iloc[-1]); e26 = float(df["EMA_26"].iloc[-1])
        result["EMA"] = "YÜKSELİŞ TRENDİ" if e12 > e26 else "DÜŞÜŞ TRENDİ"
    return result

def generate_alerts(symbol, df):
    alerts = []
    try:
        decision = institutional_decision(df)
        if decision["decision"] in ("GÜÇLÜ AL", "AL"):
            alerts.append({"level": "INFO", "title": "Güçlü Boğa Sinyali", "message": f"{symbol} için kurumsal karar: {decision['decision']}"})
        elif decision["decision"] == "SAT":
            alerts.append({"level": "WARNING", "title": "Ayı (Düşüş) Sinyali", "message": f"{symbol} elden çıkarılması öneriliyor."})
        risk = calculate_risk_score(df)
        if risk >= 75:
            alerts.append({"level": "RISK", "title": "Yüksek Risk Uyarısı", "message": f"{symbol} için volatilite tehlikeli boyutta (Risk: {risk})"})
    except: pass
    return alerts

def build_master_terminal(symbol, df):
    ai_karar = {}
    try: _, preds = makine_ogrenmesi_tahmin(df, gelecek_gun=5); ai_karar["rf_prediction"] = preds[-1]
    except: ai_karar["rf_prediction"] = float(df["Close"].iloc[-1])
    ai_karar["last_close"] = float(df["Close"].iloc[-1])
    
    return {
        "symbol": symbol,
        "dashboard": {
            "trend_strength": calculate_trend_strength(df)["strength"],
            "market_regime": detect_market_regime(df),
            "quality": generate_signal_score(df)["signal"]
        },
        "technical": {
            "action": institutional_decision(df)["decision"],
            "technical_score": calculate_total_score(df)["total"],
            "signals": advanced_technical_analysis(df)
        },
        "ai": {"ensemble": ai_karar},
        "alerts": generate_alerts(symbol, df)
    }

def terminal_summary(master):
    rf = master["ai"]["ensemble"]["rf_prediction"]
    lc = master["ai"]["ensemble"]["last_close"]
    return {
        "decision": master["technical"]["action"],
        "technical_score": master["technical"]["technical_score"],
        "technical_bias": master["technical"]["signals"].get("MACD", "NÖTR"),
        "ai_signal": "YÜKSELİŞ" if rf > lc else "DÜŞÜŞ",
        "dashboard_rating": calculate_total_score(df)["rating"] if not df.empty else "N/A"
    }

# ============================================================
# YAPAY ZEKA ÖNERİ MOTORU ARAYÜZÜ (TAB 10)
# ============================================================
if not df.empty:
    with tabs[10]:
        st.subheader("🤖 Yapay Zeka ve Kurumsal Karar Motoru Önerileri")
        st.markdown("Bu modül, arka planda çalışan **v100 Master Terminal Engine**, **Random Forest** ve **Smart Money** algoritmalarını harmanlayarak nihai alım/satım tavsiyesi üretir.")

        with st.spinner("Yapay Zeka Analizleri Tamamlıyor..."):
            try:
                # Motorların gereksinim duyduğu indikatörleri DataFrame'e ekle
                df_ai = df.copy()
                df_ai = calculate_adx(df_ai)
                df_ai = calculate_supertrend(df_ai)
                df_ai = calculate_mfi(df_ai)
                df_ai = calculate_cci(df_ai)
                df_ai = detect_bos_choch(df_ai)

                # Master Terminal raporunu çek
                master_rapor = build_master_terminal(hisse_kodu, df_ai)
                ozet = terminal_summary(master_rapor)

                st.markdown("### 🎯 Yapay Zeka Nihai Sinyali")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ortak YZ Kararı", ozet["decision"])
                c2.metric("Sinyal Gücü Skoru", f"%{ozet['technical_score']}")
                c3.metric("ML (Makine Öğrenmesi) Yönü", ozet["ai_signal"])
                c4.metric("Kurumsal Kalite", ozet["dashboard_rating"])

                st.divider()

                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    st.markdown("### 🧠 Random Forest Fiyat Tahmini")
                    rf_fiyat = round(master_rapor['ai']['ensemble']['rf_prediction'], 2)
                    mevcut_fiyat = round(master_rapor['ai']['ensemble']['last_close'], 2)
                    st.write(f"- **Mevcut Kapanış:** {mevcut_fiyat}")
                    st.write(f"- **AI Hedef Fiyat (Kısa Vade):** {rf_fiyat}")
                    
                    if rf_fiyat > mevcut_fiyat:
                        st.success(f"Yapay zeka fiyatın **%{round(((rf_fiyat-mevcut_fiyat)/mevcut_fiyat)*100, 2)}** artacağını öngörüyor.")
                    else:
                        st.error(f"Yapay zeka fiyatın **%{round(((mevcut_fiyat-rf_fiyat)/mevcut_fiyat)*100, 2)}** düşeceğini öngörüyor.")

                    st.markdown("### 📊 Teknik İndikatör Sinerjisi (Confluence)")
                    st.json(master_rapor["technical"]["signals"])

                with col_ai2:
                    st.markdown("### ⚠️ Sistem ve Risk Uyarıları")
                    uyarilar = master_rapor.get("alerts", [])
                    if uyarilar:
                        for u in uyarilar:
                            if u["level"] in ["WARNING", "RISK"]:
                                st.warning(f"**{u['title']}**: {u['message']}")
                            else:
                                st.info(f"**{u['title']}**: {u['message']}")
                    else:
                        st.success("Mevcut durumda tespit edilen kritik bir risk bulunmuyor.")

                    st.markdown("### 🏢 Kurumsal Çıkarımlar")
                    st.write(f"- **Trend Gücü Skoru:** {master_rapor['dashboard']['trend_strength']}")
                    st.write(f"- **Piyasa Rejimi:** {master_rapor['dashboard']['market_regime']}")
                    st.write(f"- **Sinyal Kalitesi (Teknik):** {master_rapor['dashboard']['quality']}")

            except Exception as e:
                st.error(f"Yapay Zeka Karar Motoru hesaplanırken bir hata oluştu: {str(e)}")
                st.warning("İpucu: AI indikatörlerinin (ADX, MFI vb.) hesaplanabilmesi için tarih aralığını biraz daha genişletmeyi (en az 60 gün geriden başlatmayı) deneyin.")