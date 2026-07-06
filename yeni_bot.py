import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import requests

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM (USER-AGENT)
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="Pro Hisse Analiz Paneli v2.0")
st.title("🚀 Pro Hisse Analiz ve Tarama Merkezi v2.0")

# Telegram Yapılandırması
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    TELEGRAM_TOKEN = "TEST_MODU"
    TELEGRAM_CHAT_ID = "TEST_MODU"

def telegram_gonder(mesaj):
    if TELEGRAM_TOKEN == "TEST_MODU":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={mesaj}"
    try:
        requests.get(url)
    except:
        pass

@st.cache_data(show_spinner=False)
def veri_yukle(ticker, start, end):
    return yf.download(ticker, start=start, end=end, session=oturum)

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try:
        return yf.Ticker(ticker, session=oturum).info
    except:
        return {}

# 2. SEÇENEK: HABERLERİ ÇEKME VE KELİME TABANLI YAPAY ZEKA DUYGU ANALİZİ MOTORU
def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
        if not news_data:
            return []
            
        olumlu_kelimeler = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma", "ortaklık", "kâr", "alım", "proje", "güçlü", "temettü"]
        olumsuz_kelimeler = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza", "iptal", "satış", "risk", "kriz", "zayıf", "geriledi"]
        
        analiz_sonuclari = []
        for n in news_data[:5]:  # En son 5 haberi analiz et
            baslik = n.get('title', '')
            ozet = n.get('summary', '') or ''
            metin = (baslik + " " + ozet).lower()
            
            olumlu_skor = sum(1 for k in olumlu_kelimeler if k in metin)
            olumsuz_skor = sum(1 for k in olumsuz_kelimeler if k in metin)
            
            if olumlu_skor > olumsuz_skor:
                duygu = "🟢 OLUMLU"
            elif olumsuz_skor > olumlu_skor:
                duygu = "🔴 OLUMSUZ"
            else:
                duygu = "🟡 NÖTR"
                
            analiz_sonuclari.append({
                "baslik": baslik,
                "kaynak": n.get('publisher', 'Bilinmiyor'),
                "link": n.get('link', '#'),
                "duygu": duygu
            })
        return analiz_sonuclari
    except:
        return []

def ai_score(df):
    score = 0
    reasons = []

    close = df["Close"].iloc[-1]
    ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["Close"].ewm(span=200).mean().iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal_Line"].iloc[-1]
    volume_avg = df["Volume"].rolling(20).mean().iloc[-1]

    if close > ema20:
        score += 10
        reasons.append("EMA20 Üzerinde Tutunma")
    if close > ema50:
        score += 15
        reasons.append("EMA50 Güçlü Trend")
    if close > ema200:
        score += 20
        reasons.append("EMA200 Uzun Vade Yükseliş Trendi")
    if 45 < rsi < 70:
        score += 15
        reasons.append("RSI Pozitif Bölgede")
    if macd > signal:
        score += 15
        reasons.append("MACD Al Sinyali")
    if df["Volume"].iloc[-1] > volume_avg * 1.5:
        score += 10
        reasons.append("Ortalama Üstü Hacim Patlaması")

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

# 2. SEÇENEK: GENİŞLETİLMİŞ AMİRAL GEMİSİ BIST TARAMA LİSTESİ
@st.cache_data(show_spinner=False)
def akilli_tarama_yap(tickers):
    bulunanlar = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", progress=False, session=oturum)
            time.sleep(1.0) # Rate limit koruması
            
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

            sinyal_nedeni = []
            
            if onceki_sma50 <= onceki_sma200 and sma50 > sma200: sinyal_nedeni.append("⭐ Golden Cross")
            if macd_onceki <= signal_onceki and macd_guncel > signal_guncel: sinyal_nedeni.append("🚀 MACD AL")
            if rsi_onceki <= 30 and rsi_guncel > 30: sinyal_nedeni.append("🟢 RSI Dipten Döndü")
            if df['Close'].iloc[-2] <= bb_lower.iloc[-2] and current_price > bb_lower.iloc[-1]: sinyal_nedeni.append("📉 Bollinger Alt Bant Tepkisi")
            if hacim_bugun > (hacim_ort * 2): sinyal_nedeni.append("🔥 HACİM PATLAMASI")

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
    with st.spinner('Veriler analiz ediliyor...'):
        df = veri_yukle(hisse_kodu, baslangic_tarihi, bitis_tarihi)
        info = sirket_bilgisi_getir(hisse_kodu)
        
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Temel İndikatörler
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
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
        
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()

        # 2. SEÇENEK: FIBONACCI RETRACEMENT HESAPLAMA MOTORU
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        fib_levels = {
            '0.0%': max_price,
            '23.6%': max_price - 0.236 * diff,
            '38.2%': max_price - 0.382 * diff,
            '50.0%': max_price - 0.5 * diff,
            '61.8%': max_price - 0.618 * diff,
            '100.0%': min_price
        }

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Gelişmiş Teknik Analiz", "🏢 Derin Temel Analiz", "📰 Yapay Zeka Haber Analizi", "🔍 Akıllı Tarama Modülü"])

        with tab1:
            current_price = df['Close'].iloc[-1]
            
            df['Min_20'] = df['Low'] == df['Low'].rolling(window=20, center=True).min()
            df['Max_20'] = df['High'] == df['High'].rolling(window=20, center=True).max()
            
            all_supports = sorted(list(set([round(x, 2) for x in df[df['Min_20']]['Low'].dropna().tolist()])))
            all_resistances = sorted(list(set([round(x, 2) for x in df[df['Max_20']]['High'].dropna().tolist()])))
            
            active_supports = [x for x in all_supports if x < current_price][-3:]
            active_resistances = [x for x in all_resistances if x > current_price][:3]

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
                
            with c3.expander("Skor Analiz Detayları"):
                for n in nedenler:
                    st.write(f"✅ {n}")
                    
            st.divider()

            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.5, 0.15, 0.15, 0.2], 
                subplot_titles=("Grafik & İndikatörler & Fibonacci Seviyeleri", "Hacim (Volume)", "RSI (14)", "MACD")
            )

            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name="SMA 200", line=dict(color='red', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Üst", line=dict(color='rgba(255,255,255,0.15)', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Alt", line=dict(color='rgba(255,255,255,0.15)', dash='dot')), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color='cyan', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color='magenta', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(color='white', width=1.5)), row=1, col=1)

            # Fibonacci Seviyelerini Çizgi Olarak Ekleme
            fib_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
            for (level, val), color in zip(fib_levels.items(), fib_colors):
                fig.add_hline(y=val, line_dash="dashdot", line_color=color, line_width=1, annotation_text=f"Fib {level}: {round(val,2)}", annotation_position="top right", row=1, col=1)

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
            fig.update_layout(height=850, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # 3. SEÇENEK: DERİN TEMEL ANALİZ VE BÜYÜME VERİLERİ
            st.subheader(f"🏢 {info.get('longName', hisse_kodu)} Şirket Profili")
            st.write(info.get('longBusinessSummary', 'Şirket açıklaması bulunamadı.'))
            st.divider()
            
            st.subheader("📊 Temel Değerleme Çarpanları")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sektör", info.get('sector', '-'))
            c2.metric("F/K Oranı (P/E)", info.get('trailingPE', '-'))
            c3.metric("PD/DD (P/B)", info.get('priceToBook', '-'))
            c4.metric("Firma Değeri / FAVÖK", info.get('enterpriseToEbitda', '-'))
            
            st.subheader("📈 Finansal Büyüme ve Kârlılık Performansı")
            cc1, cc2, cc3, cc4 = st.columns(4)
            
            # Yüzdesel formatlama fonksiyonu
            def yuzde_format(val):
                return f"%{round(val * 100, 2)}" if val and val != '-' else '-'
                
            cc1.metric("Çeyreklik Gelir Büyümesi", yuzde_format(info.get('revenueGrowth', '-')))
            cc2.metric("Çeyreklik Kâr Büyümesi", yuzde_format(info.get('earningsGrowth', '-')))
            cc3.metric("Brüt Kâr Marjı", yuzde_format(info.get('grossMargins', '-')))
            cc4.metric("Özsermaye Kârlılığı (ROE)", yuzde_format(info.get('returnOnEquity', '-')))

        with tab3:
            # 3. SEÇENEK: HABER DUYGU ANALİZİ ARAYÜZÜ
            st.subheader("📰 Güncel Haber Başlıkları ve Yapay Zeka Duygu Analizi")
            st.write("Sistem, şirket hakkındaki son haber akışlarını tarayarak piyasa duyarlılığını ölçer.")
            
            haberler = haber_duygu_analizi(hisse_kodu)
            if haberler:
                for h in haberler:
                    with st.expander(f"{h['duygu']} | {h['baslik']}"):
                        st.write(f"**Kaynak:** {h['kaynak']}")
                        st.markdown(f"[Habere Gitmek İçin Tıkla]({h['link']})")
            else:
                st.warning("Bu hisse kodu için yakın zamanda yayınlanmış Türkçe/İngilizce haber akışı bulunamadı.")

        with tab4:
            st.subheader("🔍 Piyasayı Çoklu Stratejiyle Tara")
            st.write("Aşağıdaki amiral gemisi hisseler teknik indikatör kombinasyonları, hacim patlamaları ve osilatör sinyalleriyle derinlemesine taranır.")
            
            if st.button("Gelişmiş Taramayı Başlat"):
                # Genişletilmiş BIST Amiral Gemisi Listesi (24 Hisse)
                bist_liste = [
                    "THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", 
                    "GARAN.IS", "AKBNK.IS", "TUPRS.IS", "SAHOL.IS", "BIMAS.IS", 
                    "ISCTR.IS", "YKBNK.IS", "FROTO.IS", "TOASO.IS", "HEKTS.IS", 
                    "KONTR.IS", "SASA.IS", "ODAS.IS", "EKGYO.IS", "PETKM.IS",
                    "ENKAI.IS", "ARCLK.IS", "PGSUS.IS", "TCELL.IS"
                ]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                sonuclar = []
                toplam = len(bist_liste)
                
                for idx, ticker in enumerate(bist_liste):
                    status_text.text(f"Taranıyor: {ticker} ({idx+1}/{toplam})")
                    res = akilli_tarama_yap([ticker])
                    if res:
                        sonuclar.extend(res)
                    progress_bar.progress((idx + 1) / toplam)
                    
                status_text.text("Tarama tamamlandı!")
                
                if sonuclar:
                    st.success(f"✅ Toplam {len(sonuclar)} Hisse İçin Formasyon Sinyali Yakalandı!")
                    for sonuc in sonuclar:
                        st.write(f"- {sonuc}")
                else:
                    st.warning("Şu an tarama listesindeki kriterlere uyan agresif bir formasyon sinyali bulunamadı.")
    else:
        st.error("Seçilen hisse için veri bulunamadı. Lütfen kodu kontrol edin (Örn: THYAO.IS).")

except Exception as e:
    st.error(f"Sistem geçici olarak meşgul veya veri çekilemedi. Birkaç dakika sonra tekrar deneyin. Hata: {e}")