"""
app.py — Streamlit arayüz katmanı (God Mode Terminal).

Bu dosya, eskiden 3100+ satırlık tek bir yeni_bot.py dosyasında iç içe
geçmiş olan tüm iş mantığından (veri çekme, indikatörler, formasyonlar,
ML modelleri, skorlama, tarama motoru) arındırılmış; sadece Streamlit
sayfa/sekme orkestrasyonunu içerir. İş mantığının tamamı borsa_botu/
paketinin diğer modüllerinde yaşar:

    config.py         -> BIST hisse listesi
    database.py       -> SQLite tahmin geçmişi
    data_fetching.py  -> Yahoo/TradingView/İş Yatırım veri çekme
    indicators.py     -> Teknik indikatörler (RSI/MACD/Bollinger/ADX/Fibo/T3)
    patterns.py       -> Formasyon & dipten dönüş tespiti
    ml_models.py      -> Ensemble / LSTM tahmin modelleri
    scoring.py        -> Skorlama, backtest, Monte Carlo simülasyonu
    scanner.py         -> Tek hisse için tam analiz (asenkron_analiz_yap)

Çalıştırmak için: streamlit run app.py
(app.py'nin borsa_botu/ paketiyle aynı üst klasörde olması, ya da
borsa_botu'nun PYTHONPATH'te olması gerekir.)
"""
import logging
import os
import json
import time
import threading
import sqlite3
import concurrent.futures

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from config import tum_bist_hisselerini_getir
from database import tahminleri_degerlendir
from data_fetching import (
    veri_yukle, sirket_bilgisi_getir, haber_duygu_analizi,
    toplu_guncel_fiyat_getir, toplu_gecmis_veri_getir,
)
from indicators import tilson_t3, stokastik_hesapla, smc_hesapla
from patterns import grafik_formasyon_bul, yapay_zeka_icin_formasyon_bul
from ml_models import ensemble_prediction, gelismis_ai_tahmin
from scoring import sihirli_formul_skorla, backtest_motoru, monte_carlo_simulasyonu, python_istatistik_analizi
from scanner import asenkron_analiz_yap

st.set_page_config(layout="wide", page_title="God Mode Terminal v100")


st.sidebar.header("🌍 Küresel Piyasa Ayarları")
veri_kaynagi = st.sidebar.selectbox(
    "Veri Çekilecek Kaynak:", 
    ["Yahoo Finance (yfinance)", "TradingView (tvdatafeed)", "İş Yatırım (Sadece BIST)"]
)
# Küresel Piyasa Ayarları (Mevcut kodun buradan devam edecek...)
piyasa_tipi = st.sidebar.selectbox("Piyasa Türü:", ["Borsa İstanbul (BIST)", "Amerikan Borsası (ABD)", "Kripto Para"])

if piyasa_tipi == "Borsa İstanbul (BIST)":
    varsayilan_hisse = "XU100.IS"
    tarama_listesi = tum_bist_hisselerini_getir() # <-- 700 HISSEYI OTOMATIK ÇEKEN YENI FONKSIYON
elif piyasa_tipi == "Amerikan Borsası (ABD)":
    varsayilan_hisse = "AAPL"
    tarama_listesi = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
else:
    varsayilan_hisse = "BTC-USD"
    tarama_listesi = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

hisse_kodu = st.sidebar.text_input("Varlık Kodu:", value=varsayilan_hisse).upper()
baslangic = st.sidebar.date_input("Başlangıç Tarihi:", value=datetime.today() - pd.Timedelta(days=365)) 
bitis = st.sidebar.date_input("Bitiş Tarihi:", value=datetime.today())
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Radar Ek Filtreleri")
sadece_super_sinyal = st.sidebar.checkbox("🌟 Sadece Süper Sinyal Verenler", value=False)
sadece_spring = st.sidebar.checkbox("🎯 Sadece Wyckoff Spring", value=False)
# --------------------------------------------------------

st.title("👁️ Pro Küresel Yatırım Terminali v100 (SMC, Fibo, XGBoost & Quant)")

# ---------------------------------------------------------
# BURASI SİZİN KODUNUZDA 536. SATIR CİVARINDAN BAŞLIYOR
# ---------------------------------------------------------
with st.spinner('Kurumsal teknik analiz verileri hesaplanıyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis, kaynak=veri_kaynagi)
    info = sirket_bilgisi_getir(hisse_kodu)

# YENİ EKLENECEK HAYAT KURTARICI BLOK:
if df is None or df.empty:
    st.error("⚠️ Yahoo Finance'tan veri çekilemedi (API yoğunluğu veya ağ hatası). Lütfen 1-2 dakika bekleyip sayfayı yenileyin veya farklı bir hisse kodu girin.")
    st.stop() # Veri yoksa kodun aşağıya inip hata vermesini engeller!

# "if not df.empty:" SİLİNDİ, ARTIK GİRİNTİYE (BOŞLUĞA) GEREK YOK
# HİZALAMAYI EN SOLA ÇEKİYORUZ:
# 5, 8 ve 13 Günlük Üstel Hareketli Ortalamalar (Kısa Vadeli Trend - Fibo)
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
df['EMA_13'] = df['Close'].ewm(span=13, adjust=False).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()   
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()    
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
df['Tilson_T3'] = tilson_t3(df['Close'])

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
rs = gain / (loss + 1e-9)
df['RSI'] = 100 - (100 / (1 + rs))
min_val = df['RSI'].rolling(window=14).min()
max_val = df['RSI'].rolling(window=14).max()
df['Stoch_RSI'] = (df['RSI'] - min_val) / (max_val - min_val + 1e-9)
df['Stoch_RSI_K'] = df['Stoch_RSI'].rolling(window=3).mean() * 100
df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(window=3).mean()
df['True_Range'] = np.max(pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
df['ATR_14'] = df['True_Range'].rolling(14).mean()
df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

df = smc_hesapla(df)
df = stokastik_hesapla(df)

# ==========================================
# 4. ARAYÜZ (TABS) SEKMELERİ
# ==========================================
# Buradan sonrası aynı kalıyor...
# ==========================================
# 4. ARAYÜZ (TABS) SEKMELERİ
# ==========================================
tabs = st.tabs([
    "📈 SMC & Quant Grafiği", 
    "🔍 Akıllı Radar", 
    "💼 Cüzdan & Stop", 
    "🏢 Temel Analiz", 
    "📰 Haber", 
    "📊 Isı Haritası", 
    "⚙️ Backtest", 
    "🎲 Risk Simülasyonu", 
    "🧬 İstatistik",
    "🤖 AI Ensemble Karar",
    "🧠 Yapay Zeka Öğrenme & Başarı Karnesi"
    
    
    
])

# --- SEKME 0: QUANT GRAFİK ---
# --- SEKME 0: QUANT GRAFİK ---
with tabs[0]:
    st.subheader("📈 Kurumsal Quant Grafiği & Likidite Analizi")
    
    c_ayar1, c_ayar2, c_ayar3 = st.columns(3)
    with c_ayar1:
        goster_vpvr = st.checkbox("📊 Hacim Profili (VPVR)", value=True, key="chk_vpvr")
        goster_smc = st.checkbox("🏦 FVG & Likidite (SMC)", value=True, key="chk_smc")
        goster_fibo = st.checkbox("📐 Altın Oran (Fibo)", value=True, key="chk_fibo")
    with c_ayar2:
        goster_grafik_formasyon = st.checkbox("📉 İkili Tepe/Dip", value=True, key="chk_form1")
        goster_formasyon = st.checkbox("🕯️ Mum Formasyonları", value=False, key="chk_form2")
    with c_ayar3:
        goster_vwap = st.checkbox("⚖️ VWAP (Maliyet)", value=False, key="chk_vwap")
        goster_ai = st.checkbox("🤖 XGBoost AI Tahmini", value=True, key="chk_ai")
        goster_kisa_ema = st.checkbox("🚀 5-8-13 Kısa Trend", value=True, key="chk_kisa_ema")
        # (Burada fazladan yazılmış kopya goster_ai satırını da temizledim)
        
    # ========================================================
    # 🎯 KRİTİK DÜZELTME BURASI:
    # Aşağıdaki kodların girintisini (TAB) sola çektik. 
    # Artık 3. sütunun içinde değil, sayfanın tam genişliğinde çalışacak!
    # ========================================================
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
    
    if goster_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)
        
    # TİLSON ÇİZGİSİNİ GRAFİĞE EKLEME SATIRI:
    fig.add_trace(go.Scatter(x=df.index, y=df['Tilson_T3'], name="Tilson T3", line=dict(color='yellow', width=2)), row=1, col=1)
    # 5, 8, 13 EMA ÇİZGİLERİNİ GRAFİĞE EKLEME:
    if goster_kisa_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_5'], name="EMA 5", line=dict(color='#00d4ff', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], name="EMA 8", line=dict(color='#ff9900', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_13'], name="EMA 13", line=dict(color='#ff00ff', width=1.5)), row=1, col=1)
    if goster_vpvr:
        hacim_bolumleri, fiyat_araliklari = np.histogram(df['Close'].dropna(), bins=40, weights=df['Volume'].dropna())
        bolum_merkezleri = (fiyat_araliklari[:-1] + fiyat_araliklari[1:]) / 2
        max_hacim = hacim_bolumleri.max()
        sure_uzunlugu = df.index[-1] - df.index[0]
        x_koordinatlari = [df.index[0] + sure_uzunlugu * 0.3 * (v / max_hacim) for v in hacim_bolumleri]
        for i in range(len(bolum_merkezleri)):
            fig.add_shape(type="line", x0=df.index[0], y0=bolum_merkezleri[i], x1=x_koordinatlari[i], y1=bolum_merkezleri[i], line=dict(color="rgba(100, 150, 255, 0.4)", width=4), row=1, col=1)

    if goster_smc:
        for i in range(2, len(df)):
            bitis_idx = i+5 if i+5 < len(df) else len(df)-1 
            if df['FVG_Bullish'].iloc[i]:
                fig.add_shape(type="rect", x0=df.index[i-2], y0=df['High'].iloc[i-2], x1=df.index[bitis_idx], y1=df['Low'].iloc[i], fillcolor="rgba(0, 255, 0, 0.2)", line=dict(width=0), layer="below", row=1, col=1)
            elif df['FVG_Bearish'].iloc[i]:
                fig.add_shape(type="rect", x0=df.index[i-2], y0=df['Low'].iloc[i-2], x1=df.index[bitis_idx], y1=df['High'].iloc[i], fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0), layer="below", row=1, col=1)
                
    if goster_fibo: 
        max_fiyat, min_fiyat = df['High'].max(), df['Low'].min()
        fark = max_fiyat - min_fiyat
        seviyeler = {0: "100%", 0.382: "61.8%", 0.5: "50%", 0.618: "38.2%", 1: "0%"}
        renkler = ['#ff0000', '#ff9900', '#ffff00', '#00ffcc', '#999999']
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
        for dip in ikili_dipler:
            fig.add_shape(type="line", x0=dip[0], y0=dip[2], x1=dip[1], y1=dip[3], line=dict(color="green", width=3, dash="dot"), row=1, col=1)

    if goster_formasyon:
        df_form = yapay_zeka_icin_formasyon_bul(df)
        st.dataframe(df, use_container_width=True)
        yutan_boga = df_form[df_form['Bullish_Engulfing']]
        fig.add_trace(go.Scatter(x=yutan_boga.index, y=yutan_boga['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00ff00', size=12), name='Yutan Boğa'), row=1, col=1)

    if goster_ai:
        # 1. Önce Sihirli Formül (Temel Analiz) verilerini hesaplıyoruz
        # (Not: 'hisse' veya 'hisse_kodu' değişkeni bu sayfanın üst kısımlarında tanımlanmış olmalı)
        skor_sonucu = sihirli_formul_skorla(hisse_kodu)

# Eğer fonksiyon düzgün çalışıp 2 değer döndürdüyse:
        if skor_sonucu is not None and isinstance(skor_sonucu, tuple) and len(skor_sonucu) == 2:
            temel_bilgi, temel_puan = skor_sonucu
        else:
    # Eğer hata olduysa veya eksik veri geldiyse varsayılan değerler ata:
            temel_bilgi = {} # veya None (kodunuzun geri kalanı nasıl bekliyorsa)
            temel_puan = 0
        
        # 2. Fonksiyonu güncellenmiş parametrelerle çağırıyoruz
        tarihler, tahminler = gelismis_ai_tahmin(
            df=df, 
            gelecek_gun=30, 
            temel_veriler=temel_bilgi, 
            temel_skor=temel_puan
        )
        
        # 3. Sonuçları grafiğe çizdiriyoruz
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="XGBoost AI (Temel+Teknik)", line=dict(color='cyan', width=3, dash='dot')), row=1, col=1)
    # MACD ve Stoch Çizimleri
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='#2962FF')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Sinyal", line=dict(color='#FF6D00')), row=2, col=1)
    hist_colors = np.where(df['MACD_Hist'] < 0, '#ef5350', '#26a69a')
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD Histogram", marker_color=hist_colors), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_K'], name="%K", line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_D'], name="%D", line=dict(color='orange')), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False)
    
    grafik_alani = st.empty()
    with grafik_alani:
        st.plotly_chart(fig, use_container_width=True)
# --- SEKME 1: AKILLI RADAR ---
# --- SEKME 1: AKILLI RADAR (Hatalardan Arındırılmış & Tam Optimize) ---
with tabs[1]:
    st.subheader("🔍 Akıllı Asenkron Radar & Çoklu Gösterge (Quant)")
    
    # Session State Tanımlamaları
    if 'son_tarama_df' not in st.session_state:
        st.session_state.son_tarama_df = None
    if 'son_tarama_tipi' not in st.session_state:
        st.session_state.son_tarama_tipi = None

    st.markdown("### 🌊 Hızlı Piyasa Taraması ve Yapay Zeka Önerileri")
    st.write(f"Şu anki tarama listesi: **{', '.join(tarama_listesi)}**")
    
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
    with col_btn1:
        btn_radar = st.button("🚀 Genel Radar", key="btn_radar")
    with col_btn2:
        btn_stoch = st.button("📊 Stoch Analizi", key="btn_stoch")
    with col_btn3:
        btn_tilson = st.button("📈 Tilson (T3)", key="btn_tilson")
    with col_btn4:
        btn_nokta_atisi = st.button("🎯 Nokta Atışı", type="primary", key="btn_nokta")
    with col_btn5:
        btn_son_tarama = st.button("🔄 Son Taramayı Getir", type="secondary", key="btn_son")

    # Yardımcı Fonksiyon: Taramayı hem RAM'e (Session State) hem Diske Kaydeder
    def taramayi_kaydet(df, tip_adi):
        st.session_state.son_tarama_df = df
        st.session_state.son_tarama_tipi = tip_adi
        df.to_pickle("son_tarama.pkl")
        with open("son_tarama_tipi.txt", "w", encoding="utf-8") as f:
            f.write(tip_adi)

    # ==========================================================
    # ARKA PLAN TARAMA MİMARİSİ
    # ----------------------------------------------------------
    # Eskiden tarama, butona basıldığında ana Streamlit script akışı
    # içinde SENKRON çalışıyordu. 536 hissenin taranması dakikalar
    # sürdüğü için bu süre içinde websocket bağlantısı kopup yeniden
    # kurulursa (Streamlit Cloud'da, sekme arka plana alındığında,
    # ağ kesintisinde vs.) Streamlit TÜM SCRIPT'İ YENİDEN ÇALIŞTIRIR
    # ve devam etmekte olan tarama sonucu kaybolurdu.
    #
    # Çözüm: Tarama artık ayrı bir arka plan thread'inde çalışıyor ve
    # ilerlemesini/sonucunu periyodik olarak DİSKE yazıyor. Sayfa
    # yeniden yüklense, bağlantı kopup tekrar kurulsa bile thread arka
    # planda çalışmaya devam eder (Streamlit sunucu process'i ayakta
    # kaldığı sürece). Ekranda sadece durum dosyası okunup gösteriliyor
    # (st.fragment ile SADECE bu küçük blok periyodik yenileniyor, tüm
    # sayfa değil).
    # ==========================================================
    TARAMA_DURUM_DOSYASI = "tarama_durum.json"
    TARAMA_SONUC_DOSYASI = "tarama_sonuc_gecici.pkl"

    if 'tarama_calisiyor' not in st.session_state:
        st.session_state.tarama_calisiyor = False

    def paralel_tara_arkaplan(sembol_listesi, analiz_tipi, goster_tipi, max_workers,
                                baslangic_, bitis_, veri_kaynagi_,
                                durum_dosyasi, sonuc_dosyasi):
        """Ana Streamlit script akışından BAĞIMSIZ ayrı bir thread'de çalışır.
        Sayfa rerun olsa/bağlantı kopsa bile bu fonksiyon durmaz."""
        toplam = len(sembol_listesi)
        try:
            # Tüm liste için TEK toplu istekte anlık fiyat + günlük geçmiş veri
            # (536 ayrı istek yerine sadece birkaç toplu istek)
            guncel_fiyat_sozlugu = toplu_guncel_fiyat_getir(tuple(sembol_listesi), veri_kaynagi_)
            gecmis_veri_sozlugu = toplu_gecmis_veri_getir(tuple(sembol_listesi), baslangic_, bitis_, veri_kaynagi_)

            with open(durum_dosyasi, "w", encoding="utf-8") as f:
                json.dump({"tamamlanan": 0, "toplam": toplam, "bitti": False,
                           "goster_tipi": goster_tipi, "veri_yok_sayisi": 0, "hata": None}, f)

            sonuclar = []
            veri_yok_sayisi = 0
            tamamlanan = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                gelecek_sonuclar = {
                    executor.submit(
                        asenkron_analiz_yap, s, baslangic_, bitis_, analiz_tipi, veri_kaynagi_,
                        guncel_fiyat_sozlugu.get(s), gecmis_veri_sozlugu.get(s)
                    ): s
                    for s in sembol_listesi
                }
                for future in concurrent.futures.as_completed(gelecek_sonuclar):
                    sembol = gelecek_sonuclar[future]
                    tamamlanan += 1
                    try:
                        sonuc = future.result()
                        if sonuc:
                            sonuclar.append(sonuc)
                        else:
                            veri_yok_sayisi += 1
                    except Exception as e:
                        veri_yok_sayisi += 1
                        logging.error(f"[{sembol}] Tarama hatası: {e}")

                    # Her 15 hissede bir kısmi sonucu diske yaz — bağlantı kopsa
                    # ya da process yeniden başlasa bile o ana kadarki sonuç kaybolmaz
                    if tamamlanan % 15 == 0 or tamamlanan == toplam:
                        pd.DataFrame(sonuclar).to_pickle(sonuc_dosyasi)
                        with open(durum_dosyasi, "w", encoding="utf-8") as f:
                            json.dump({"tamamlanan": tamamlanan, "toplam": toplam, "bitti": False,
                                       "goster_tipi": goster_tipi, "veri_yok_sayisi": veri_yok_sayisi,
                                       "hata": None}, f)

            pd.DataFrame(sonuclar).to_pickle(sonuc_dosyasi)
            with open(durum_dosyasi, "w", encoding="utf-8") as f:
                json.dump({"tamamlanan": toplam, "toplam": toplam, "bitti": True,
                           "goster_tipi": goster_tipi, "veri_yok_sayisi": veri_yok_sayisi,
                           "hata": None}, f)
        except Exception as e:
            logging.error(f"Arka plan tarama genel hatası: {e}")
            with open(durum_dosyasi, "w", encoding="utf-8") as f:
                json.dump({"tamamlanan": 0, "toplam": toplam, "bitti": True,
                           "goster_tipi": goster_tipi, "veri_yok_sayisi": 0, "hata": str(e)}, f)

    def tarama_baslat(analiz_tipi, goster_tipi, max_workers):
        if st.session_state.tarama_calisiyor:
            st.warning("⏳ Zaten devam eden bir tarama var. Bitmesini bekleyin (sayfayı kapatmanıza gerek yok, arka planda çalışmaya devam ediyor).")
            return
        if os.path.exists(TARAMA_DURUM_DOSYASI):
            os.remove(TARAMA_DURUM_DOSYASI)
        st.session_state.tarama_calisiyor = True
        th = threading.Thread(
            target=paralel_tara_arkaplan,
            args=(tarama_listesi, analiz_tipi, goster_tipi, max_workers,
                  baslangic, bitis, veri_kaynagi,
                  TARAMA_DURUM_DOSYASI, TARAMA_SONUC_DOSYASI),
            daemon=True,
        )
        th.start()

    GOSTER_TIPI_ADLARI = {
        "radar": "Genel Radar Taraması",
        "stoch": "Stoch Analizi",
        "tilson": "Tilson (T3) Analizi",
        "sniper": "Nokta Atışı (Sniper)",
    }

    def sniper_filtrele(df_radar):
        df_sniper = df_radar[
            (df_radar['Günlük T3'] == '🚀 BOĞA') &
            (pd.to_numeric(df_radar['📊 Temel Skor'], errors='coerce') >= 30) &
            (
                (df_radar['💥 Hacim Analizi'].str.contains('PATLAMA', na=False)) |
                (df_radar['📈 Pozitif Uyuşmazlık'].str.contains('UYUŞMAZLIK|SÜPER SİNYAL', na=False)) |
                (df_radar['🪤 Spring (Tuzak)'] == '✅ VAR')
            )
        ]
        return df_sniper

    def _tablo_goster_ve_detay_paneli(df_tablo, anahtar):
        """
        Tabloyu ekrana basar; '_teknik_detay' kolonu varsa ekrandan gizler 
        ve kullanıcı bir satıra tıkladığında ayrı bir Gelişmiş Teknik Analiz Paneli açar.
        """
        detaylar_var = '_teknik_detay' in df_tablo.columns
        df_gorunen = df_tablo.drop(columns=['_teknik_detay']) if detaylar_var else df_tablo

        secili_sembol = None
        if detaylar_var:
            try:
                event = st.dataframe(
                    df_gorunen, 
                    use_container_width=True, 
                    hide_index=True,
                    on_select="rerun", 
                    selection_mode="single-row",
                    key=f"tablo_{anahtar}_{len(df_gorunen)}"
                )
                secim_nesnesi = getattr(event, "selection", None)
                secili_satirlar = getattr(secim_nesnesi, "rows", None) if secim_nesnesi is not None else None
                if not secili_satirlar and isinstance(event, dict):
                    secili_satirlar = event.get("selection", {}).get("rows", [])
                if secili_satirlar:
                    secili_sembol = df_gorunen.iloc[secili_satirlar[0]]['Varlık']
            except TypeError:
                st.dataframe(df_gorunen, use_container_width=True, hide_index=True)
                secim = st.selectbox(
                    "📊 Detaylı teknik paneli görmek için bir hisse seç:",
                    options=["-"] + df_gorunen['Varlık'].tolist(),
                    key=f"secim_{anahtar}_{len(df_gorunen)}"
                )
                secili_sembol = None if secim == "-" else secim
        else:
            st.dataframe(df_gorunen, use_container_width=True, hide_index=True)

        if secili_sembol is not None:
            eslesen = df_tablo.loc[df_tablo['Varlık'] == secili_sembol, '_teknik_detay']
            if not eslesen.empty and isinstance(eslesen.iloc[0], dict):
                d = eslesen.iloc[0]
                with st.expander(f"📊 {secili_sembol} — Gelişmiş Teknik Analiz Paneli", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RSI(14)", d.get("RSI(14)", "-"))
                    c1.write(f"**MACD:** {d.get('MACD Durumu', '-')}")
                    c2.write(f"**Bollinger:** {d.get('Bollinger Pozisyonu', '-')}")
                    c2.write(f"**Trend Gücü:** {d.get('Trend Gücü (ADX)', '-')}")
                    c3.write(f"**Haftalık Trend:** {d.get('Haftalık T3', '-')}")
                    c3.write(f"**Fibonacci:** {d.get('Fibonacci Seviyesi', '-')}")
                    st.markdown("---")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("🛡️ Stop-Loss (ATR)", d.get("ATR Stop-Loss", "-"))
                    r2.metric("🎯 Kar Hedefi (ATR)", d.get("ATR Kar Hedefi", "-"))
                    r3.metric("⚖️ Risk/Ödül", d.get("Risk/Ödül", "-"))
                    puan = d.get("Teknik Teyit Puanı", 0)
                    st.caption(f"🧠 Teknik Teyit Puanı: **{puan}** (AI Kararı ve AL/SAT Kararı'na dahil edilmiştir) — {d.get('Teknik Notlar', '')}")
    # Sayfada sadece SEÇİLİ küçük bloğu periyodik yeniler (run_every), tüm
    # sayfayı değil — bu yüzden diğer widget'lar/tab'lar taramadan etkilenmez.
    # st.fragment Streamlit >= 1.33 gerektirir; daha eski sürümlerde bulunmazsa
    # aşağıdaki _fragment fallback'i onu sıradan bir fonksiyona çevirir (otomatik
    # 2sn'lik yenileme olmaz, aşağıdaki "🔄 Durumu Yenile" butonu ile manuel bakılır).
    if hasattr(st, "fragment"):
        _fragment = st.fragment
    else:
        def _fragment(*_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

    @_fragment(run_every=2)
    def tarama_ilerleme_ve_sonuc_goster():
        if not os.path.exists(TARAMA_DURUM_DOSYASI):
            return
        try:
            with open(TARAMA_DURUM_DOSYASI, "r", encoding="utf-8") as f:
                durum = json.load(f)
        except Exception:
            return

        if durum.get("hata"):
            st.error(f"❌ Tarama sırasında hata oluştu: {durum['hata']}")
            st.session_state.tarama_calisiyor = False
            return

        if not durum.get("bitti", False):
            toplam = durum.get("toplam", 0) or 1
            tamamlanan = durum.get("tamamlanan", 0)
            st.progress(tamamlanan / toplam, text=f"Taranıyor... {tamamlanan}/{toplam} hisse")
            st.caption("ℹ️ Tarama arka planda çalışıyor. Sayfa yenilense veya bağlantı kısa süreliğine kesilse bile devam eder; bittiğinde sonuç burada görünecek.")
            return

        # --- TARAMA BİTTİ ---
        st.session_state.tarama_calisiyor = False
        veri_yok_sayisi = durum.get("veri_yok_sayisi", 0)
        toplam = durum.get("toplam", 0)
        goster_tipi = durum.get("goster_tipi", "radar")

        if not os.path.exists(TARAMA_SONUC_DOSYASI):
            st.warning("⚠️ Tarama tamamlandı ama sonuç dosyası bulunamadı.")
            return

        try:
            df_ham = pd.read_pickle(TARAMA_SONUC_DOSYASI)
        except Exception as e:
            st.error(f"Sonuç dosyası okunamadı: {e}")
            return

        if df_ham is None or df_ham.empty:
            st.warning("⚠️ Tarama sonucu bulunamadı. Bu genellikle veri kaynağından hiç fiyat verisi çekilemediği anlamına gelir. Veri kaynağını (Yahoo/TradingView/İş Yatırım) değiştirmeyi deneyin.")
            return

        if veri_yok_sayisi > 0:
            st.caption(f"ℹ️ {veri_yok_sayisi}/{toplam} hisse için veri kaynağından fiyat verisi alınamadı (bunlar tabloya girmedi).")

        if goster_tipi == "sniper":
            df_sniper = sniper_filtrele(df_ham)
            if sadece_super_sinyal:
                df_sniper = df_sniper[df_sniper['📈 Pozitif Uyuşmazlık'].str.contains('SÜPER SİNYAL', na=False)]
            if sadece_spring:
                df_sniper = df_sniper[df_sniper['🪤 Spring (Tuzak)'] == '✅ VAR']

            if not df_sniper.empty:
                taramayi_kaydet(df_sniper, GOSTER_TIPI_ADLARI["sniper"])
                st.success(f"🎯 Dipten Dönüş Fırsatı! Temeli sağlam ve akıllı para girişi tespit edilen {len(df_sniper)} hisse var.")
                _tablo_goster_ve_detay_paneli(df_sniper, "sniper")
                st.balloons()
            else:
                boga_sayisi = (df_ham['Günlük T3'] == '🚀 BOĞA').sum()
                skor_sayisi = (pd.to_numeric(df_ham['📊 Temel Skor'], errors='coerce') >= 30).sum()
                tetik_sayisi = (
                    (df_ham['💥 Hacim Analizi'].str.contains('PATLAMA', na=False)) |
                    (df_ham['📈 Pozitif Uyuşmazlık'].str.contains('UYUŞMAZLIK|SÜPER SİNYAL', na=False)) |
                    (df_ham['🪤 Spring (Tuzak)'] == '✅ VAR')
                ).sum()
                st.warning("📉 Şu anki piyasada belirlenen Sniper şartlarının HEPSİNE birden uyan şirket bulunamadı. Genel Radar'ı inceleyebilirsiniz.")
                st.caption(
                    f"ℹ️ Kırılım — {len(df_ham)} hisse tarandı: "
                    f"{boga_sayisi} tanesi Günlük BOĞA trendinde, "
                    f"{skor_sayisi} tanesi Temel Skor ≥ 30, "
                    f"{tetik_sayisi} tanesi hacim/uyuşmazlık/spring tetikleyicilerinden birine sahip. "
                    f"Sniper filtresi bu üç şartın AYNI ANDA sağlanmasını istiyor — bu yüzden tek tek sayılar dolu olsa bile kesişim boş çıkabilir."
                )
        else:
            df_goster = df_ham.copy()
            if sadece_super_sinyal and '📈 Pozitif Uyuşmazlık' in df_goster.columns:
                df_goster = df_goster[df_goster['📈 Pozitif Uyuşmazlık'].str.contains('SÜPER SİNYAL', na=False)]
            if sadece_spring and '🪤 Spring (Tuzak)' in df_goster.columns:
                df_goster = df_goster[df_goster['🪤 Spring (Tuzak)'] == '✅ VAR']

            taramayi_kaydet(df_ham, GOSTER_TIPI_ADLARI.get(goster_tipi, goster_tipi))
            _tablo_goster_ve_detay_paneli(df_goster, goster_tipi)
            st.success(f"✅ {GOSTER_TIPI_ADLARI.get(goster_tipi, goster_tipi)} tamamlandı ve hafızaya kaydedildi!")

    # 1-4. TARAMA BUTONLARI — hepsi arka plan thread'i başlatır, script akışını bloklamaz
    if btn_radar:
        tarama_baslat("radar", "radar", max_workers=8)
    elif btn_stoch:
        tarama_baslat("stoch", "stoch", max_workers=10)
    elif btn_tilson:
        tarama_baslat("tilson", "tilson", max_workers=10)
    elif btn_nokta_atisi:
        tarama_baslat("radar", "sniper", max_workers=8)

    # Devam eden ya da az önce bitmiş bir tarama varsa ilerleme/sonucu göster
    if st.session_state.tarama_calisiyor or os.path.exists(TARAMA_DURUM_DOSYASI):
        if not hasattr(st, "fragment"):
            # Eski Streamlit sürümünde otomatik yenileme olmadığı için manuel buton
            st.button("🔄 Durumu Yenile", key="btn_durum_yenile")
        tarama_ilerleme_ve_sonuc_goster()

    # 5. EN SON TARAMAYI GETİR
    if btn_son_tarama:
        # 1. RAM boşsa diskteki pickle dosyasından veri çek
        if st.session_state.son_tarama_df is None and os.path.exists("son_tarama.pkl"):
            try:
                st.session_state.son_tarama_df = pd.read_pickle("son_tarama.pkl")
                if os.path.exists("son_tarama_tipi.txt"):
                    with open("son_tarama_tipi.txt", "r", encoding="utf-8") as f:
                        st.session_state.son_tarama_tipi = f.read()
            except Exception as e:
                logging.error(f"Son tarama dosyadan okunamadı: {e}")

        # 2. Ekrana Bas
        if st.session_state.son_tarama_df is not None:
            st.info(f"💾 Kurtarılan Tablo: **{st.session_state.son_tarama_tipi}**")
            
            # --- YENİ EKLENEN FİLTRELEME BLOĞU (Kurtarılan Tablo İçin) ---
            df_goster = st.session_state.son_tarama_df.copy()
            if 'sadece_super_sinyal' in locals() and sadece_super_sinyal:
                if '📈 Pozitif Uyuşmazlık' in df_goster.columns:
                    df_goster = df_goster[df_goster['📈 Pozitif Uyuşmazlık'].str.contains('SÜPER SİNYAL', na=False)]
            if 'sadece_spring' in locals() and sadece_spring:
                if '🪤 Spring (Tuzak)' in df_goster.columns:
                    df_goster = df_goster[df_goster['🪤 Spring (Tuzak)'] == '✅ VAR']
            # -----------------------------------------------------------
            
            _tablo_goster_ve_detay_paneli(df_goster, "son_tarama")
        else:
            st.warning("⚠️ Hafızada veya dosyada kaydedilmiş bir tarama sonucu bulunamadı. Lütfen önce bir tarama yapın.")
with tabs[2]:
    st.subheader("📊 Varlık Portföyüm & Akıllı Stop")
    tavsiye_stop = round(float(df['Close'].iloc[-1]) - (float(df['ATR_14'].iloc[-1]) * 2), 2)
    st.info(f"💡 Tavsiye edilen teknik Stop-Loss: **{tavsiye_stop}**")

# --- SEKME 3, 4, 5, 6, 7, 8: DİĞER MODÜLLER ---
with tabs[3]:
    st.subheader("🏢 Temel Analiz")
    c1, c2, c3 = st.columns(3)
    c1.metric("F/K Oranı", info.get('trailingPE', '-'))
    c2.metric("PD/DD", info.get('priceToBook', '-'))
    c3.metric("Piyasa Değeri", info.get('marketCap', '-'))

with tabs[4]:
    st.subheader("📰 Haber Duygu Analizi")
    for h in haber_duygu_analizi(hisse_kodu):
        st.write(f"**{h['duygu']}** - [{h['baslik']}]({h['link']})")

with tabs[5]:
    st.subheader("📊 Korelasyon Haritası")
    st.write("Isı haritası oluşturmak için yeterli veri işleniyor...")

with tabs[6]:
    st.subheader("⚙️ Strateji Testi (Backtest)")
    bt = backtest_motoru(df)
    st.line_chart(bt[['Piyasa_Kumulatif', 'Strateji_Kumulatif']])

with tabs[7]:
    st.subheader("🎲 Monte Carlo Risk Simülasyonu")
    if st.button("Simülasyon Çiz"):
        st.line_chart(monte_carlo_simulasyonu(df))

with tabs[8]:
    st.subheader("🧬 İstatistik")
    stats = python_istatistik_analizi(df)
    st.write(stats)

# --- SEKME 9: YAPAY ZEKA ---
# --- SEKME 9: YAPAY ZEKA ---
with tabs[9]:
    st.subheader("🧠 v100 AI Ensemble & Kurumsal Karar Motoru")
    
    with st.spinner("Yapay Zeka Kararı Hesaplanıyor..."):
        ai_sonuc = ensemble_prediction(df)
        
    c1, c2 = st.columns([1, 2]) # 1'e 2 oranında sütunlar
    
    with c1:
        st.metric("Yapay Zeka Kararı", ai_sonuc["signal"])
        st.metric("Tahmini Hedef", f"{ai_sonuc['rf_prediction']} TL")
        st.progress(int(ai_sonuc["confidence"]), text=f"Güven Skoru: %{ai_sonuc['confidence']}")
        
        st.markdown("---")
        st.info("💡 **Nasıl Okunmalı?** Yandaki grafik, yapay zekanın hedef fiyatı belirlerken sağladığınız indikatörlerden hangilerine en çok dikkat ettiğini yüzdelik ağırlık olarak gösterir.")
        
    with c2:
        # Öznitelik (Feature) grafiğinin çizilmesi
        if ai_sonuc.get("feature_importances"):
            # Verileri DataFrame'e çevirip küçükten büyüğe sıralıyoruz
            imp_df = pd.DataFrame(list(ai_sonuc["feature_importances"].items()), columns=["İndikatör", "Etki Oranı"])
            imp_df = imp_df.sort_values(by="Etki Oranı", ascending=True)
            
            # Plotly ile yatay bar grafiği
            fig_imp = px.bar(imp_df, x="Etki Oranı", y="İndikatör", orientation='h', 
                             title="🤖 Karar Verirken Hangi Verilere Odaklandı?",
                             text_auto='.2%', # Çubukların üzerine yüzde yazdırır
                             color="Etki Oranı", color_continuous_scale="Viridis")
            
            fig_imp.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0),
                                  xaxis_tickformat='.0%', showlegend=False)
            
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Öznitelik ağırlıkları hesaplanamadı (Yetersiz veri veya model hatası).")

# --- YENİ SEKME: AI BAŞARI KARNESİ ---
# --- SEKME 10: AI BAŞARI KARNESİ ---
with tabs[10]:
    st.subheader("🧠 Yapay Zeka Öğrenme & Başarı Karnesi")
    tahminleri_degerlendir()
    st.markdown("Yapay zeka, geçmişteki tahminlerini güncel fiyatlarla kıyaslar. **Hata payı %5'in altındaki tahminler başarılı kabul edilir.**")
    
    try:
        conn = sqlite3.connect('hisse_hafiza.db')
        # Tablo yoksa hata almamak için kontrol
        try:
            gecmis_df = pd.read_sql_query("SELECT * FROM tahminler ORDER BY tarih DESC", conn)
        except:
            st.warning("Veritabanı tablosu henüz oluşturulmamış.")
            gecmis_df = pd.DataFrame()
        conn.close()
        
        if not gecmis_df.empty:
            st.dataframe(gecmis_df, use_container_width=True, hide_index=True)
            
            basarili_sayisi = len(gecmis_df[gecmis_df['durum'] == 'BAŞARILI ✅'])
            degerlendirilen_sayisi = len(gecmis_df[gecmis_df['durum'] != 'BEKLİYOR'])
            
            if degerlendirilen_sayisi > 0:
                basari_orani = (basarili_sayisi / degerlendirilen_sayisi) * 100
                st.metric(label="Net Başarı Oranı", value=f"% {basari_orani:.1f}")
        else:
            st.info("Henüz kaydedilmiş tahmin yok. Radar veya AI analizi çalıştırıldığında veriler buraya akacaktır.")
    except Exception as e:
        st.error(f"Veritabanı erişim hatası: {e}")
