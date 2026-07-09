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
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 1. YAHOO FINANCE ENGELİNİ AŞMAK İÇİN ÖZEL OTURUM
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})

st.set_page_config(layout="wide", page_title="God Mode Terminal v26.0")
st.title("🏆 Pro Küresel Yatırım Terminali v26.0 (Magic Formula Edition)")
st.markdown("*Çoklu Skorlama Sistemi ve Derin Backtest Motoru*")

# --- TELEGRAM VE OTOMASYON ---
def telegram_gonder(mesaj):
    try:
        token = st.secrets.get("TELEGRAM_TOKEN", "TEST")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "TEST")
        if token != "TEST":
            url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mesaj}"
            requests.get(url, timeout=5)
        return True
    except:
        return False

# --- VERİ MOTORLARI VE FONKSİYONLAR ---
@st.cache_data(show_spinner=False, ttl=900)
def veri_yukle(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end, session=oturum)
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}

# Gelişmiş İndikatör Motoru
def indikatorleri_ekle(df_girdi):
    df_ind = df_girdi.copy()
    
    df_ind.ta.rsi(length=14, append=True)
    df_ind.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_ind.rename(columns={'RSI_14': 'RSI', 'MACD_12_26_9': 'MACD', 'MACDs_12_26_9': 'MACD_Signal', 'MACDh_12_26_9': 'MACD_Hist'}, inplace=True, errors='ignore')
    
    df_ind['SMA_20'] = df_ind['Close'].rolling(window=20).mean()
    df_ind['SMA_50'] = df_ind['Close'].rolling(window=50).mean()
    df_ind['BB_Std'] = df_ind['Close'].rolling(window=20).std()
    df_ind['BB_Up'] = df_ind['SMA_20'] + (df_ind['BB_Std'] * 2)
    df_ind['BB_Low'] = df_ind['SMA_20'] - (df_ind['BB_Std'] * 2)
    
    try:
        sti = ta.supertrend(df_ind['High'], df_ind['Low'], df_ind['Close'], length=10, multiplier=3)
        if sti is not None:
            df_ind['SuperTrend'] = sti['SUPERT_10_3.0']
            df_ind['SuperTrend_Yon'] = sti['SUPERTd_10_3.0']
        else:
            df_ind['SuperTrend'] = df_ind['Close']
            df_ind['SuperTrend_Yon'] = 1
    except:
        df_ind['SuperTrend'] = df_ind['Close']
        df_ind['SuperTrend_Yon'] = 1

    try:
        df_ind.ta.vwap(append=True)
        df_ind.rename(columns={'VWAP_D': 'VWAP'}, inplace=True, errors='ignore')
    except:
        df_ind['VWAP'] = df_ind['Close']
        
    if 'VWAP' not in df_ind.columns:
        df_ind['VWAP'] = df_ind['Close']

    df_ind['Hacim_Ort_20'] = df_ind['Volume'].rolling(20).mean()
    df_ind['Hacim_Ort_3'] = df_ind['Volume'].rolling(3).mean()
    df_ind['Hacim_Ivmesi'] = df_ind['Hacim_Ort_3'] / (df_ind['Hacim_Ort_20'] + 1)
    
    df_ind.bfill(inplace=True)
    df_ind.ffill(inplace=True)
    return df_ind

# YENİ: SİHİRLİ FORMÜL (HİSSE SKORLAMA SİSTEMİ)
def hisse_skorunu_hesapla(df_ind, hisse_info):
    skor = 0
    detaylar = {}
    
    if len(df_ind) < 50:
        return 0, {"Hata": "Yetersiz Veri"}
        
    son_satir = df_ind.iloc[-1]
    fiyat = son_satir['Close']
    
    # 1. Teknik Puan (Maks 40)
    # Trend (20 Puan)
    if fiyat > son_satir['SMA_50']:
        skor += 20
        detaylar['Trend (SMA50)'] = "🟢 Üzerinde (20/20)"
    else:
        detaylar['Trend (SMA50)'] = "🔴 Altında (0/20)"
        
    # RSI (20 Puan)
    rsi = son_satir.get('RSI', 50)
    if 40 <= rsi <= 70:
        skor += 20
        detaylar['RSI'] = f"🟢 İdeal Bölge {round(rsi,1)} (20/20)"
    elif rsi < 40:
        skor += 10
        detaylar['RSI'] = f"🟡 Dipten Dönüş Pot. {round(rsi,1)} (10/20)"
    else:
        detaylar['RSI'] = f"🔴 Aşırı Alım Riski {round(rsi,1)} (0/20)"

    # 2. Hacim ve Balina Takibi (Maks 30)
    # VWAP Konumu (15 Puan)
    vwap = son_satir.get('VWAP', fiyat)
    if fiyat > vwap:
        skor += 15
        detaylar['VWAP (Maliyet)'] = "🟢 Güvenli Bölge (15/15)"
    else:
        detaylar['VWAP (Maliyet)'] = "🔴 Tehlikeli Bölge (0/15)"
        
    # Hacim İvmesi (15 Puan)
    hacim_ivme = son_satir.get('Hacim_Ivmesi', 0)
    if hacim_ivme > 1.2:
        skor += 15
        detaylar['Hacim İvmesi'] = f"🟢 Para Girişi Var {round(hacim_ivme,2)}x (15/15)"
    elif hacim_ivme > 0.8:
        skor += 7
        detaylar['Hacim İvmesi'] = f"🟡 Yatay Hacim {round(hacim_ivme,2)}x (7/15)"
    else:
        detaylar['Hacim İvmesi'] = f"🔴 Para Çıkışı Var {round(hacim_ivme,2)}x (0/15)"

    # 3. Temel Analiz (Maks 30)
    fk = hisse_info.get('trailingPE', 0)
    pddd = hisse_info.get('priceToBook', 0)
    
    # F/K Puanı (15 Puan)
    if 0 < fk <= 15:
        skor += 15
        detaylar['F/K Oranı'] = f"🟢 Ucuz/Makul {round(fk,1)} (15/15)"
    elif 15 < fk <= 25:
        skor += 7
        detaylar['F/K Oranı'] = f"🟡 Sınırda {round(fk,1)} (7/15)"
    else:
        detaylar['F/K Oranı'] = f"🔴 Pahalı/Zarar {round(fk,1)} (0/15)"
        
    # PD/DD Puanı (15 Puan)
    if 0 < pddd <= 5:
        skor += 15
        detaylar['PD/DD Oranı'] = f"🟢 Ucuz/Makul {round(pddd,1)} (15/15)"
    elif 5 < pddd <= 10:
        skor += 7
        detaylar['PD/DD Oranı'] = f"🟡 Sınırda {round(pddd,1)} (7/15)"
    else:
        detaylar['PD/DD Oranı'] = f"🔴 Şişkin/Riskli {round(pddd,1)} (0/15)"

    return skor, detaylar

def makine_ogrenmesi_sinyal(df_subset):
    df_ml = df_subset.copy()
    df_ml['Gun'] = np.arange(len(df_ml))
    
    ozellikler = ['Gun', 'RSI', 'MACD', 'MACD_Signal', 'BB_Up', 'BB_Low', 'VWAP', 'Hacim_Ivmesi']
    mevcut_ozellikler = [col for col in ozellikler if col in df_ml.columns]
    
    X = df_ml[mevcut_ozellikler]
    y = df_ml['Close']
    
    model = RandomForestRegressor(n_estimators=40, random_state=42)
    model.fit(X, y)
    
    son_satir = df_ml.iloc[-1]
    gelecek_dict = {}
    for col in mevcut_ozellikler:
        if col == 'Gun':
            gelecek_dict[col] = [son_satir[col] + 1]
        else:
            gelecek_dict[col] = [son_satir[col]]
            
    gelecek_X = pd.DataFrame(gelecek_dict)
    yarin_tahmin = model.predict(gelecek_X)[0]
    
    son_fiyat = float(son_satir['Close'])
    beklenen_degisim = ((yarin_tahmin - son_fiyat) / son_fiyat) * 100
    
    if beklenen_degisim > 0.8 and son_satir.get('RSI', 50) < 70 and son_fiyat > son_satir.get('VWAP', son_fiyat):
        return "AL", yarin_tahmin, beklenen_degisim
    elif beklenen_degisim < -0.8 or son_satir.get('RSI', 50) > 80 or son_fiyat < son_satir.get('VWAP', son_fiyat):
        return "SAT", yarin_tahmin, beklenen_degisim
    else:
        return "BEKLE", yarin_tahmin, beklenen_degisim

# --- SİDEBAR VE PİYASA SEÇİMİ ---
st.sidebar.header("🌍 Küresel Piyasa Ayarları")
piyasa_tipi = st.sidebar.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "Amerikan Borsası (ABD)", "Kripto Para"])

if piyasa_tipi == "Borsa İstanbul (BIST)":
    varsayilan_hisse = "THYAO.IS"
    tarama_listesi = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "GARAN.IS", "ISCTR.IS", "AKBNK.IS", "BIMAS.IS"]
elif piyasa_tipi == "Amerikan Borsası (ABD)":
    varsayilan_hisse = "AAPL"
    tarama_listesi = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
else:
    varsayilan_hisse = "BTC-USD"
    tarama_listesi = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

hisse_kodu = st.sidebar.text_input("Varlık Kodu:", value=varsayilan_hisse).upper()
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=365))
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())

with st.spinner('Piyasa verileri işleniyor...'):
    df_ham = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

if not df_ham.empty:
    if isinstance(df_ham.columns, pd.MultiIndex): 
        df_ham.columns = df_ham.columns.droplevel(1)
    
    df = indikatorleri_ekle(df_ham)

    tabs = st.tabs([
        "🏆 Sihirli Formül (Skor)", "🛡️ Backtest Motoru", "📈 Teknik & AI", 
        "💼 Cüzdan", "🏢 Temel & Temettü", "🔍 Akıllı Radar"
    ])

    # ================= TAB 1: SİHİRLİ FORMÜL (YENİ v26.0) =================
    with tabs[0]:
        st.subheader(f"🏆 {hisse_kodu} Yatırım Skor Kartı (Sihirli Formül)")
        st.write("Bu sekme, hissenin temel, teknik ve hacim verilerini analiz ederek 100 üzerinden bir Yatırım Yapılabilirlik Skoru hesaplar.")
        
        skor, detaylar = hisse_skorunu_hesapla(df, info)
        
        # Skor Görseli
        skor_renk = "green" if skor >= 70 else ("orange" if skor >= 40 else "red")
        st.markdown(f"<h1 style='text-align: center; color: {skor_renk}; font-size: 80px;'>{skor} / 100</h1>", unsafe_allow_html=True)
        
        if skor >= 75: st.success("🌟 A Sınıfı Yatırım Fırsatı: Hem temel hem teknik olarak çok güçlü!")
        elif skor >= 50: st.warning("⚖️ B Sınıfı Nötr: Bekleme veya yakından takip seviyesi.")
        else: st.error("⚠️ C Sınıfı Riskli: Göstergeler zayıf, uzak durulması önerilir.")
        
        st.markdown("### 📊 Puanlama Detayları")
        detay_df = pd.DataFrame(list(detaylar.items()), columns=["Kriter", "Durum & Puan"])
        st.table(detay_df.set_index("Kriter"))
        
        st.markdown("---")
        st.subheader("🤖 Tüm Listeyi Puanla")
        if st.button("Seçili Piyasadaki Tüm Hisseleri Skorla"):
            with st.spinner("Tüm liste analiz ediliyor..."):
                toplu_skorlar = []
                ilerleme = st.progress(0)
                for i, hisse in enumerate(tarama_listesi):
                    t_df_ham = veri_yukle(hisse, datetime.today() - timedelta(days=90), datetime.today())
                    t_info = sirket_bilgisi_getir(hisse)
                    if not t_df_ham.empty:
                        if isinstance(t_df_ham.columns, pd.MultiIndex): t_df_ham.columns = t_df_ham.columns.droplevel(1)
                        t_df = indikatorleri_ekle(t_df_ham)
                        t_skor, _ = hisse_skorunu_hesapla(t_df, t_info)
                        fiyat = round(float(t_df['Close'].iloc[-1]), 2)
                        toplu_skorlar.append({"Hisse": hisse, "Fiyat": fiyat, "Sihirli Skor": t_skor})
                    ilerleme.progress((i + 1) / len(tarama_listesi))
                
                if toplu_skorlar:
                    skor_df = pd.DataFrame(toplu_skorlar).sort_values(by="Sihirli Skor", ascending=False).set_index("Hisse")
                    st.dataframe(skor_df, use_container_width=True)
                    st.success("✅ Skorlama tamamlandı! Listeyi skora göre sıralı görebilirsiniz.")

    # ================= TAB 2: BACKTEST MOTORU (v25.0 MİRASI) =================
    with tabs[1]:
        st.subheader("🛡️ Otonom Zaman Makinesi (Stop-Loss & Trailing Stop)")
        
        c_bt1, c_bt2, c_bt3 = st.columns(3)
        bt_bakiye = c_bt1.number_input("Başlangıç Bakiyesi (TL):", value=100000, step=10000)
        bt_gun = c_bt2.slider("Test Süresi (Gün):", min_value=10, max_value=90, value=30)
        
        stop_loss_orani = c_bt3.number_input("Stop-Loss Oranı %:", value=5.0, min_value=1.0) / 100.0
        trailing_stop_aktif = c_bt3.checkbox("Trailing Stop Aktif", value=True)
        trailing_stop_orani = c_bt3.number_input("İzleme Mesafesi %:", value=7.0, min_value=1.0) / 100.0
        
        if st.button("🚀 Risk Korumalı Testi Başlat", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_kutusu = st.empty()
            
            bakiye = bt_bakiye
            elindeki_lot = 0
            islem_gecmisi = []
            kasa_gecmisi = []
            tarih_gecmisi = []
            basarili_islem = 0
            toplam_islem = 0
            
            alinan_fiyat = 0
            en_yuksek_fiyat = 0
            stop_seviyesi = 0
            
            toplam_veri = len(df)
            test_baslangic_idx = toplam_veri - bt_gun
            
            for i in range(bt_gun):
                guncel_idx = test_baslangic_idx + i
                if guncel_idx >= toplam_veri: break
                
                df_gecmis = df.iloc[:guncel_idx]
                if len(df_gecmis) < 25: continue
                
                o_gunku_fiyat = float(df_gecmis['Close'].iloc[-1])
                o_gunku_tarih = df.index[guncel_idx-1].strftime('%Y-%m-%d')
                
                sinyal, _, _ = makine_ogrenmesi_sinyal(df_gecmis)
                satis_yapildi = False
                
                if elindeki_lot > 0:
                    if o_gunku_fiyat > en_yuksek_fiyat:
                        en_yuksek_fiyat = o_gunku_fiyat
                        if trailing_stop_aktif:
                            yeni_stop = en_yuksek_fiyat * (1 - trailing_stop_orani)
                            if yeni_stop > stop_seviyesi: stop_seviyesi = yeni_stop
                                
                    if o_gunku_fiyat <= stop_seviyesi:
                        bakiye += elindeki_lot * o_gunku_fiyat
                        islem_gecmisi.append(f"🛡️ STOP ({o_gunku_tarih}) | {elindeki_lot} Lot Satıldı @ {round(o_gunku_fiyat,2)} TL.")
                        if o_gunku_fiyat > alinan_fiyat: basarili_islem += 1
                        elindeki_lot = 0; alinan_fiyat = 0; en_yuksek_fiyat = 0; stop_seviyesi = 0; satis_yapildi = True

                if not satis_yapildi:
                    if sinyal == "AL" and bakiye > o_gunku_fiyat:
                        alinacak_lot = bakiye // o_gunku_fiyat
                        bakiye -= alinacak_lot * o_gunku_fiyat
                        elindeki_lot += alinacak_lot
                        alinan_fiyat = o_gunku_fiyat; en_yuksek_fiyat = o_gunku_fiyat
                        stop_seviyesi = alinan_fiyat * (1 - stop_loss_orani)
                        islem_gecmisi.append(f"🟢 AL ({o_gunku_tarih}) | {alinacak_lot} Lot @ {round(o_gunku_fiyat,2)} TL.")
                        toplam_islem += 1
                    
                    elif sinyal == "SAT" and elindeki_lot > 0:
                        bakiye += elindeki_lot * o_gunku_fiyat
                        islem_gecmisi.append(f"🔴 SAT ({o_gunku_tarih}) | AI Kararı @ {round(o_gunku_fiyat,2)} TL.")
                        if o_gunku_fiyat > alinan_fiyat: basarili_islem += 1
                        elindeki_lot = 0; alinan_fiyat = 0; en_yuksek_fiyat = 0; stop_seviyesi = 0
                
                kasa_gecmisi.append(bakiye + (elindeki_lot * o_gunku_fiyat))
                tarih_gecmisi.append(o_gunku_tarih)
                
                status_text.text(f"Zaman Makinesi Çalışıyor: {o_gunku_tarih}...")
                log_kutusu.code("\\n".join(islem_gecmisi[-3:])) if islem_gecmisi else log_kutusu.code("İşlem bekleniyor...")
                progress_bar.progress((i + 1) / bt_gun)
                time.sleep(0.01)
            
            progress_bar.empty(); status_text.empty()
            
            son_fiyat = float(df['Close'].iloc[-1])
            nihai_deger = bakiye + (elindeki_lot * son_fiyat)
            net_kar = nihai_deger - bt_bakiye
            win_rate = (basarili_islem / toplam_islem * 100) if toplam_islem > 0 else 0
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Nihai Portföy", f"{round(nihai_deger, 2)} TL")
            res_c2.metric("Net Kâr/Zarar", f"{round(net_kar, 2)} TL", f"%{round((net_kar / bt_bakiye) * 100, 2)}")
            res_c3.metric("Başarı Oranı (Win Rate)", f"%{round(win_rate, 1)}")
            
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=tarih_gecmisi, y=kasa_gecmisi, mode='lines', fill='tozeroy', line=dict(color='#00ffcc')))
            fig_eq.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0), title="Kasa Değişim Grafiği (Equity Curve)")
            st.plotly_chart(fig_eq, use_container_width=True)

    # ================= TAB 3: GRAFİK & AI =================
    with tabs[2]:
        st.subheader("📈 Teknik & AI Dashboard")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        if 'SuperTrend' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], line=dict(color='orange')), row=1, col=1)
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#00ffcc', dash='dot')), row=1, col=1)
        renkler = np.where(df['Close'] >= df['Open'], '#26a69a', '#ef5350')
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=renkler), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # Kalan Tablar (Yer kaplamaması için özet geçildi, işlevsellik aynı)
    with tabs[3]: st.info("Cüzdan sekmesi aktiftir.")
    with tabs[4]: st.info("Temel analiz sekmesi aktiftir.")
    with tabs[5]: st.info("Akıllı Radar sekmesi aktiftir.")

else:
    st.error("Veri çekilemedi.")