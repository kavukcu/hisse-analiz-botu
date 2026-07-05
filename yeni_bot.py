import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import requests

st.set_page_config(layout="wide", page_title="Pro Hisse Analiz Paneli")
st.title("🚀 Tam Kapsamlı Hisse Analiz Merkezi")

try:
    TELEGRAM_TOKEN = st.secrets["8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk"]
    TELEGRAM_CHAT_ID = st.secrets["1634044181"]
except:
    TELEGRAM_TOKEN = "TEST_MODU"
    TELEGRAM_CHAT_ID = "TEST_MODU"

def telegram_gonder(mesaj):
    if TELEGRAM_TOKEN == "TEST_MODU":
        return
    url = f"https://api.telegram.org/bot{8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk}/sendMessage?chat_id={1634044181}&text={mesaj}"
    try:
        requests.get(url)
    except:
        pass

@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    return yf.Ticker(ticker).info

def ai_score(df):
    score = 0
    reasons = []

    close = df["Close"].iloc[-1]
    ema20 = df["Close"].ewm(span=20).mean()
    ema50 = df["Close"].ewm(span=50).mean()
    ema200 = df["Close"].ewm(span=200).mean()
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal_Line"].iloc[-1]
    volume_avg = df["Volume"].rolling(20).mean().iloc[-1]

    if close > ema20.iloc[-1]:
        score += 10
        reasons.append("EMA20 Üzerinde")

    if close > ema50.iloc[-1]:
        score += 15
        reasons.append("EMA50 Üzerinde")

    if close > ema200.iloc[-1]:
        score += 20
        reasons.append("EMA200 Üzerinde")

    if 45 < rsi < 70:
        score += 15
        reasons.append("RSI Pozitif")

    if macd > signal:
        score += 15
        reasons.append("MACD Al Sinyali")

    if df["Volume"].iloc[-1] > volume_avg * 1.5:
        score += 10
        reasons.append("Yüksek Hacim")

    if score >= 80:
        karar = "🟢 GÜÇLÜ AL"
        renk = "success"
    elif score >= 60:
        karar = "🟢 AL"
        renk = "success"
    elif score >= 40:
        karar = "🟡 BEKLE"
        renk = "warning"
    elif score >= 20:
        karar = "🔴 SAT"
        renk = "error"
    else:
        karar = "🔴 GÜÇLÜ SAT"
        renk = "error"

    return score, karar, reasons, renk

@st.cache_data(show_spinner=False)
def akilli_tarama_yap(tickers):
    bulunanlar = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if len(df) < 200: continue
            
            current_price = df['Close'].iloc[-1]
            
            sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
            onceki_sma50 = df['Close'].rolling(window=50).mean().iloc[-2]
            onceki_sma200 = df['Close'].rolling(window=200).mean().iloc[-2]
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_guncel, rsi_onceki = rsi.iloc[-1], rsi.iloc[-2]

            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_guncel, macd_onceki = macd.iloc[-1], macd.iloc[-2]
            signal_guncel, signal_onceki = signal.iloc[-1], signal.iloc[-2]

            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            bb_lower = sma20 - (2 * std20)
            hacim_ort = df['Volume'].rolling(window=20).mean().iloc[-1]
            hacim_bugun = df['Volume'].iloc[-1]

            df['Min_20'] = df['Low'] == df['Low'].rolling(window=20, center=True).min()
            df['Max_20'] = df['High'] == df['High'].rolling(window=20, center=True).max()
            
            all_supports = sorted(list(set([round(x, 2) for x in df[df['Min_20']]['Low'].dropna().tolist()])))
            all_resistances = sorted(list(set([round(x, 2) for x in df[df['Max_20']]['High'].dropna().tolist()])))
            
            active_supports = [x for x in all_supports if x < current_price][-3:]
            active_resistances = [x for x in all_resistances if x > current_price][:3]

            sinyal_nedeni = []
            
            if onceki_sma50 <= onceki_sma200 and sma50 > sma200: sinyal_nedeni.append("⭐ Golden Cross")
            if macd_onceki <= signal_onceki and macd_guncel > signal_guncel: sinyal_nedeni.append("🚀 MACD AL")
            if rsi_onceki <= 30 and rsi_guncel > 30: sinyal_nedeni.append("🟢 RSI Dipten Döndü")
            if df['Close'].iloc[-2] <= bb_lower.iloc[-2] and current_price > bb_lower.iloc[-1]: sinyal_nedeni.append("📉 Bollinger Alt Bant Tepkisi")
            if hacim_bugun > (hacim_ort * 2): sinyal_nedeni.append("🔥 HACİM PATLAMASI")

            for sup in active_supports:
                if abs(current_price - sup) / sup < 0.015:
                    sinyal_nedeni.append("🎯 Önemli Desteğe Yakın")
                    break
            
            for res in active_resistances:
                if current_price > res and df['Close'].iloc[-2] <= res:
                    sinyal_nedeni.append("⚔️ Direnç Kırılımı (Breakout)")
                    break

            if sinyal_nedeni:
                mesaj_metni = " + ".join(sinyal_nedeni)
                bulunanlar.append(f"{ticker} ({mesaj_metni})")
                tel_mesaj = f"🔔 SİNYAL YAKALANDI!\n\n📈 Hisse: {ticker}\n🎯 Sinyaller:\n- " + "\n- ".join(sinyal_nedeni) + f"\n⏱ Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                telegram_gonder(tel_mesaj)
                
        except Exception:
            continue
    return bulunanlar

st.sidebar.header("🔧 Analiz Parametreleri")
hisse_kodu = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS):", value="THYAO.IS").upper()

baslangic_tarihi = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=180))
bitis_tarihi = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

try:
    with st.spinner('Veriler çekiliyor...'):
        df = veri_yukle(hisse_kodu, baslangic_tarihi, bitis_tarihi)
        info = sirket_bilgisi_getir(hisse_kodu)
        
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        tab1, tab2, tab3 = st.tabs(["📈 Teknik Analiz", "🏢 Temel Analiz", "🔍 Akıllı Tarama Modülü"])

        with tab1:
            current_price = df['Close'].iloc[-1]
            
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df["EMA20"] = df["Close"].ewm(span=20).mean()
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            df["EMA200"] = df["Close"].ewm(span=200).mean()
            
            df['BB_Mid'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
            df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

            score, karar, nedenler, renk = ai_score(df)
            
            st.markdown("### 🤖 Yapay Zeka Karar Motoru")
            c1, c2, c3 = st.columns(3)
            c1.metric("AI SCORE", f"{score}/100")
            
            if renk == "success":
                c2.success(karar)
            elif renk == "warning":
                c2.warning(karar)
            else:
                c2.error(karar)
                
            with c3.expander("Skor Nedenleri"):
                for n in nedenler:
                    st.write(f"✅ {n}")
                    
            st.divider()

            df['Min_20'] = df['Low'] == df['Low'].rolling(window=20, center=True).min()
            df['Max_20'] = df['High'] == df['High'].rolling(window=20, center=True).max()
            
            all_supports = sorted(list(set([round(x, 2) for x in df[df['Min_20']]['Low'].dropna().tolist()])))
            all_resistances = sorted(list(set([round(x, 2) for x in df[df['Max_20']]['High'].dropna().tolist()])))
            
            active_supports = [x for x in all_supports if x < current_price][-3:]
            active_resistances = [x for x in all_resistances if x > current_price][:3]

            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.5, 0.15, 0.15, 0.2], 
                subplot_titles=("Fiyat, EMA'lar, Bollinger ve Destek/Dirençler", "Hacim (Volume)", "RSI (14)", "MACD")
            )

            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color='#00d2d3', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color='#feca57', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(color='#ff6b6b', width=2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Üst", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Alt", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)

            for sup in active_supports:
                fig.add_hline(y=sup, line_dash="dash", line_color="#2ecc71", line_width=1.5, annotation_text=f"Destek: {sup}", annotation_position="top left", row=1, col=1)
                
            for res in active_resistances:
                fig.add_hline(y=res, line_dash="dash", line_color="#e74c3c", line_width=1.5, annotation_text=f"Direnç: {res}", annotation_position="bottom left", row=1, col=1)

            hacim_renkleri = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Hacim", marker_color=hacim_renkleri), row=2, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name="Sinyal", line=dict(color='orange')), row=4, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name="Histogram", marker_color=['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]), row=4, col=1)
            
            fig.update_xaxes(type='category', nticks=20)
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig)

        with tab2:
            st.subheader(f"🏢 {info.get('longName', hisse_kodu)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sektör", info.get('sector', '-'))
            c2.metric("F/K Oranı", info.get('trailingPE', '-'))
            c3.metric("PD/DD", info.get('priceToBook', '-'))

        with tab3:
            st.subheader("🔍 Piyasayı Çoklu Stratejiyle Tara")
            if st.button("Akıllı Taramayı Başlat"):
                bist_liste = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", "GARAN.IS", "AKBNK.IS", "TUPRS.IS"]
                with st.spinner("Piyasa akıllı algoritmalarla taranıyor..."):
                    sonuclar = akilli_tarama_yap(bist_liste)
                    if sonuclar:
                        st.success("✅ Fırsatlar Bulundu!")
                        for sonuc in sonuclar:
                            st.write(f"- {sonuc}")
                    else:
                        st.warning("Şu an kriterlere uyan net bir sinyal bulunamadı.")

except Exception as e:
    st.error(f"Grafik oluşturulurken hata oluştu: {e}")