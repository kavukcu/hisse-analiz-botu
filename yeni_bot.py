import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v11.0")
st.title("👁️ Pro Küresel Yatırım Terminali v11.0 (AI Edition)")

# --- TELEGRAM VE OTOMASYON ---
def telegram_gonder(mesaj):
    try:
        token = st.secrets.get("TELEGRAM_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
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

def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
        if not news_data: return []
        olumlu = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma", "kâr", "temettü", "bullish", "breakout"]
        olumsuz = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza", "risk", "zayıf", "bearish"]
        sonuclar = []
        for n in news_data[:5]:
            metin = (n.get('title', '') + " " + (n.get('summary') or '')).lower()
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
# v12 YENİLİĞİ: ATR ve Dinamik Risk Motoru
def atr_hesapla(df, periyot=14):
    df_atr = df.copy()
    df_atr['H-L'] = abs(df_atr['High'] - df_atr['Low'])
    df_atr['H-PC'] = abs(df_atr['High'] - df_atr['Close'].shift(1))
    df_atr['L-PC'] = abs(df_atr['Low'] - df_atr['Close'].shift(1))
    df_atr['TR'] = df_atr[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_atr['ATR'] = df_atr['TR'].rolling(window=periyot).mean()
    return df_atr['ATR']
# v11 YENİLİĞİ: POLİNOMSAL REGRESYON (Daha Gerçekçi AI Eğrisi)
def makine_ogrenmesi_tahmin(df, gelecek_gun=30, derece=3):
    df_ml = df.copy()
    df_ml.reset_index(inplace=True)
    df_ml['Gun'] = np.arange(len(df_ml))
    
    X = df_ml[['Gun']]
    y = df_ml['Close']
    
    # Doğrusal yerine Polinomsal (kıvrımlı) model kullanıyoruz
    model = make_pipeline(PolynomialFeatures(derece), LinearRegression())
    model.fit(X, y)
    
    son_gun = df_ml['Gun'].iloc[-1]
    gelecek_X = pd.DataFrame({'Gun': [son_gun + i for i in range(1, gelecek_gun + 1)]})
    tahminler = model.predict(gelecek_X)
    
    tarihler = [df_ml['Date'].iloc[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
    return tarihler, tahminler

# --- SİDEBAR VE PİYASA SEÇİMİ ---
st.sidebar.header("🌍 Küresel Piyasa Ayarları")
piyasa_tipi = st.sidebar.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "Amerikan Borsası (ABD)", "Kripto Para"])

if piyasa_tipi == "Borsa İstanbul (BIST)":
    varsayilan_hisse = "THYAO.IS"
    tarama_listesi = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "GARAN.IS", "SASA.IS", "FROTO.IS", "BIMAS.IS"]
elif piyasa_tipi == "Amerikan Borsası (ABD)":
    varsayilan_hisse = "AAPL"
    tarama_listesi = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX"]
else:
    varsayilan_hisse = "BTC-USD"
    tarama_listesi = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD"]

hisse_kodu = st.sidebar.text_input("Varlık Kodu:", value=varsayilan_hisse).upper()
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=365))
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Yapay zeka verileri analiz ediyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    # v11 YENİLİĞİ: Zenginleştirilmiş İndikatörler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).ewm(alpha=1/14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0)).ewm(alpha=1/14).mean())))
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    df['BB_Ust'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Alt'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
# ATR İndikatörünü veri setine ekle
    df['ATR'] = atr_hesapla(df)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💼 Cüzdan & Alarm", "📈 Teknik & AI Tahmin", "🏢 Temel Analiz", "📰 Haber", "🔍 Akıllı Tarama", "📊 Isı Haritası", "⚙️ Backtest"
    ])

    with tab1:
        st.subheader("📊 Canlı Varlık Portföyüm ve Fiyat Alarmları")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            if 'portfoy_verisi' not in st.session_state:
                st.session_state.portfoy_verisi = pd.DataFrame([
                    {"Varlık": "THYAO.IS", "Maliyet": 300.0, "Lot": 50.0},
                    {"Varlık": "BTC-USD", "Maliyet": 62000.0, "Lot": 0.05}
                ])
                
            guncel_portfoy = st.data_editor(st.session_state.portfoy_verisi, num_rows="dynamic", use_container_width=True)
            st.session_state.portfoy_verisi = guncel_portfoy
            
            if st.button("Portföyü Hesapla"):
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
                
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Toplam Maliyet", f"{round(top_mal, 2)}")
                cc2.metric("Güncel Değer", f"{round(top_deg, 2)}")
                cc3.metric("Net Kâr", f"{round(top_deg - top_mal, 2)}", f"%{round(((top_deg - top_mal)/top_mal)*100,2) if top_mal>0 else 0}")

# v12 YENİLİĞİ: DİNAMİK RİSK YÖNETİMİ (ATR)
        with c2:
            st.markdown(f"#### 🛡️ AI Risk & Stop-Loss Motoru ({hisse_kodu})")
            
            son_kapanis = df['Close'].iloc[-1]
            son_atr = df['ATR'].iloc[-1]
            
            st.info(f"Güncel Fiyat: **{son_kapanis:.2f}** | Günlük Dalgalanma (ATR): **{son_atr:.2f}**")
            
            # Risk Profilini Seç
            risk_profili = st.selectbox("Risk Profilinizi Seçin:", ["Muhafazakar (Dar Stop)", "Dengeli (Normal Stop)", "Agresif (Geniş Stop)"], index=1)
            
            # ATR Çarpanları
            if risk_profili == "Muhafazakar (Dar Stop)":
                sl_carpan = 1.5; tp_carpan = 2.0
            elif risk_profili == "Dengeli (Normal Stop)":
                sl_carpan = 2.0; tp_carpan = 3.0
            else:
                sl_carpan = 3.0; tp_carpan = 5.0
                
            dinamik_stop = son_kapanis - (son_atr * sl_carpan)
            dinamik_hedef = son_kapanis + (son_atr * tp_carpan)
            
            st.metric("🔴 Otomatik Stop-Loss (Zarar Kes)", f"{dinamik_stop:.2f}", f"-%{((son_kapanis - dinamik_stop)/son_kapanis)*100:.2f}")
            st.metric("🟢 Otomatik Take-Profit (Kâr Al)", f"{dinamik_hedef:.2f}", f"+%{((dinamik_hedef - son_kapanis)/son_kapanis)*100:.2f}")
            
            if st.button("Risk Kurallarını Telegrama Gönder"):
                mesaj = f"🛡️ GÜVENLİ TRADE KURULUMU ({hisse_kodu})\nFiyat: {son_kapanis:.2f}\n\nHedef (TP): {dinamik_hedef:.2f}\nStop (SL): {dinamik_stop:.2f}\nProfil: {risk_profili}"
                st.success("Kurulum Telegram'a iletildi!")
                telegram_gonder(mesaj)
    with tab2:
        st.subheader("📈 Gelişmiş Teknik Göstergeler ve AI Trend Tahmini")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='cyan', width=1)))
        
        # Bollinger Bantları
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Ust'], name="BB Üst", line=dict(color='gray', dash='dot', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Alt'], name="BB Alt", line=dict(color='gray', dash='dot', width=1), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'))
        
        # Polinomsal ML Tahmini
        tarihler, tahminler = makine_ogrenmesi_tahmin(df, gelecek_gun=30, derece=4)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="AI Eğri Tahmini (30 Gün)", line=dict(color='magenta', width=3, dash='dash')))
        
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

    # v11 YENİLİĞİ: GERÇEKTEN ÇALIŞAN TARAMA MOTORU
    with tab5:
        st.subheader(f"🔍 {piyasa_tipi} Otomatik Tarama Motoru")
        st.write("Seçili piyasadaki takip listesi taranır. **Kriter:** RSI < 45 (Aşırı satıma yakın) ve Fiyat > SMA20 (Kısa vadeli dönüş sinyali)")
        
        if st.button("Taramayı Başlat"):
            with st.spinner(f"{piyasa_tipi} varlıkları taranıyor, bu biraz sürebilir..."):
                firsatlar = []
                for t in tarama_listesi:
                    try:
                        veri = yf.download(t, period="2mo", progress=False, session=oturum)
                        if isinstance(veri.columns, pd.MultiIndex): veri.columns = veri.columns.droplevel(1)
                        if len(veri) > 20:
                            son_fiyat = veri['Close'].iloc[-1]
                            sma20 = veri['Close'].rolling(20).mean().iloc[-1]
                            
                            delta = veri['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs)).iloc[-1]
                            
                            if rsi < 45 and son_fiyat > sma20:
                                firsatlar.append({"Varlık": t, "Fiyat": round(son_fiyat, 2), "RSI": round(rsi, 2)})
                    except:
                        continue
                
                if firsatlar:
                    st.success("🎯 Kriterlere Uyan Potansiyel Fırsatlar Bulundu!")
                    st.dataframe(pd.DataFrame(firsatlar), use_container_width=True)
                    mesaj = "🔍 AI TARAMA SONUCU:\n" + "\n".join([f"{f['Varlık']} - Fiyat: {f['Fiyat']} - RSI: {f['RSI']}" for f in firsatlar])
                    telegram_gonder(mesaj)
                else:
                    st.warning("Şu an için belirlediğimiz kriterlere uyan bir varlık bulunamadı.")

    with tab6:
        st.subheader(f"📊 {piyasa_tipi} Korelasyon Matrisi")
        if st.button("Isı Haritasını Oluştur"):
            with st.spinner("Piyasa verileri karşılaştırılıyor..."):
                korelasyon_df = pd.DataFrame()
                for ticker in tarama_listesi[:6]:
                    tmp_df = yf.download(ticker, period="6mo", progress=False, session=oturum)
                    if isinstance(tmp_df.columns, pd.MultiIndex): tmp_df.columns = tmp_df.columns.droplevel(1)
                    if not tmp_df.empty:
                        korelasyon_df[ticker] = tmp_df['Close']
                
                corr_matrix = korelasyon_df.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

    with tab7:
        st.subheader("⚙️ Strateji Testi (Backtest): SMA 20 vs SMA 50")
        bt_sonuc = backtest_motoru(df, kisa_periyot=20, uzun_periyot=50)
        
        if not bt_sonuc.empty:
            son_piyasa = bt_sonuc['Piyasa_Kumulatif'].iloc[-1] - 100
            son_strateji = bt_sonuc['Strateji_Kumulatif'].iloc[-1] - 100
            
            c1, c2 = st.columns(2)
            c1.metric("Alıp Bekleseydin", f"%{round(son_piyasa, 2)}")
            c2.metric("Strateji ile Alsatsaydın", f"%{round(son_strateji, 2)}", delta=round(son_strateji - son_piyasa, 2), delta_color="normal")
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Piyasa_Kumulatif'], name="Piyasa Getirisi", line=dict(color='white')))
            fig_bt.add_trace(go.Scatter(x=bt_sonuc.index, y=bt_sonuc['Strateji_Kumulatif'], name="Strateji Getirisi", line=dict(color='green', width=3)))
            fig_bt.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bt, use_container_width=True)
            
    st.sidebar.divider()
    csv = df.to_csv().encode('utf-8')
    st.sidebar.download_button(label="📊 Verileri İndir (CSV)", data=csv, file_name=f'{hisse_kodu}_veri.csv', mime='text/csv')
else:
    st.error("Veri çekilemedi. Kodunuzu veya internet bağlantınızı kontrol edin.")