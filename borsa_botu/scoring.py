"""
scoring.py — Skorlama, backtest ve risk simülasyonu fonksiyonları.
"""
import logging

import numpy as np
import pandas as pd

from data_fetching import sirket_bilgisi_getir, veri_yukle
from indicators import tilson_t3


def sihirli_formul_skorla(sembol, df=None):
    """
    Şirketin temel çarpanlarını ve teknik dip dönüş sinyallerini harmanlayarak
    hisseye kapsamlı bir Hibrit Skor verir.
    """
    try:
        info = sirket_bilgisi_getir(sembol)
        if not info:
            # 1. DÜZELTİLEN YER: Sadece puan değil, iki değer (boş bilgi ve 0 puan) dönmeli
            return {}, 0 
            
        skor = 0
        
        # ==========================================
        # A. TEMEL ANALİZ SKORLAMASI (Maksimum 100 Puan)
        # ==========================================
        
        # 1. F/K Oranı (Maksimum 25 Puan)
        fk = info.get('trailingPE', 999)
        if fk is None: fk = 999
        if 0 < fk <= 10: skor += 25
        elif 10 < fk <= 15: skor += 15
        elif 15 < fk <= 20: skor += 5
        
        # 2. PD/DD Oranı (Maksimum 25 Puan)
        pddd = info.get('priceToBook', 999)
        if pddd is None: pddd = 999
        if 0 < pddd <= 1.5: skor += 25
        elif 1.5 < pddd <= 3.0: skor += 15
        elif 3.0 < pddd <= 5.0: skor += 5
        
        # 3. ROE - Özsermaye Kârlılığı (Maksimum 25 Puan)
        roe = info.get('returnOnEquity', -1)
        if roe is None: roe = -1
        if roe > 0.20: skor += 25
        elif roe > 0.10: skor += 15
        elif roe > 0.05: skor += 5
        
        # 4. Cari Oran - Borç Risk Durumu (Maksimum 25 Puan)
        cari_oran = info.get('currentRatio', 0)
        if cari_oran is None: cari_oran = 0
        if cari_oran >= 1.5: skor += 25
        elif cari_oran >= 1.0: skor += 15
        
        # ==========================================
        # B. TEKNİK DİP DÖNÜŞ BONUSLARI (Ekstra Puanlar)
        # ==========================================
        if df is not None and not df.empty:
            son_mum = df.iloc[-1]
            
            # 🌟 Süper Sinyal (RSI + MACD + Stoch 3'lü Uyumsuzluk): +20 Puan
            if son_mum.get('Super_Sinyal', False):
                skor += 20
            # Normal Pozitif Uyumsuzluk varsa: +10 Puan
            elif son_mum.get('Pozitif_Uyusmazlik', False):
                skor += 10
                
            # 🪤 Wyckoff Spring (Ayı Tuzağı): +15 Puan
            if son_mum.get('Wyckoff_Spring', False):
                skor += 15
                
            # 💥 Hacim Patlamasi: +10 Puan
            if son_mum.get('Hacim_Patlamasi', False):
                skor += 10

        # 2. DÜZELTİLEN YER: Hem 'info' (temel_bilgi değişkeni için) hem de 'skor' (temel_puan değişkeni için) dönmeli
        return info, skor
        
    except Exception as e:
        import logging
        logging.warning(f"[{sembol}] Temel veri puanlama hatası: {str(e)}")
        # 3. DÜZELTİLEN YER: Hata durumunda da çökmemesi için iki değer dönmeli
        return {}, 0


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


def hizli_backtest_yap(sembol, baslangic, bitis):
    """Geçmişe dönük (Backtest) strateji simülatörü."""
    try:
        # Geçmiş veriyi çek
        df = veri_yukle(sembol, baslangic, bitis)
        if df is None or df.empty or len(df) < 50:
            return None
            
        b_df = df.copy()
        
        # 1. İndikatörleri Hesapla
        b_df['Tilson_T3'] = tilson_t3(b_df['Close'])
        
        low_min = b_df['Low'].rolling(window=14).min()
        high_max = b_df['High'].rolling(window=14).max()
        b_df['Stoch_K'] = 100 * ((b_df['Close'] - low_min) / (high_max - low_min + 1e-9))
        b_df['Stoch_D'] = b_df['Stoch_K'].rolling(window=3).mean()
        
        # 2. Geçmişteki "AL" Şartlarının (Sniper) Tespiti
        # Şart: Stoch dipten (30 altı) yukarı kesmiş VE Fiyat Tilson trendinin üzerine çıkmış
        b_df['AL_Sinyali'] = (b_df['Stoch_K'] > b_df['Stoch_D']) & (b_df['Stoch_K'] < 30) & (b_df['Close'] > b_df['Tilson_T3'])
        
        # 3. Kâr/Zarar Hesaplama (Pozisyon 5 gün tutulursa)
        b_df['5_Gunluk_Getiri'] = ((b_df['Close'].shift(-5) - b_df['Close']) / b_df['Close']) * 100
        
        # Sadece "AL" sinyali üretilen günleri filtrele
        islemler = b_df[b_df['AL_Sinyali']].dropna(subset=['5_Gunluk_Getiri'])
        toplam_islem = len(islemler)
        
        if toplam_islem == 0:
            return None
            
        # 4. Performans Metriklerini Çıkar
        basarili_islem = len(islemler[islemler['5_Gunluk_Getiri'] > 0])
        basari_orani = (basarili_islem / toplam_islem) * 100
        ortalama_getiri = islemler['5_Gunluk_Getiri'].mean()
        kümülatif_getiri = islemler['5_Gunluk_Getiri'].sum()
        
        return {
            "Hisse": sembol,
            "İşlem Sayısı": toplam_islem,
            "Başarı Oranı (%)": round(basari_orani, 2),
            "İşlem Başı Ort. Kâr (%)": round(ortalama_getiri, 2),
            "Kümülatif Kâr (%)": round(kümülatif_getiri, 2)
        }
    except Exception as e:
        import logging
        logging.error(f"[{sembol}] Backtest Hatası: {str(e)}")
        return None


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


def python_istatistik_analizi(df):
    try:
        getiriler = df['Close'].pct_change().dropna()
        yillik_volatilite = getiriler.std() * np.sqrt(252)
        sharpe_orani = (getiriler.mean() * 252) / yillik_volatilite
        var_95 = getiriler.quantile(0.05)
        return {
            'Yıllık Volatilite': f"% {yillik_volatilite * 100:.2f}",
            'Sharpe Oranı': f"{sharpe_orani:.2f}",
            'Günlük VaR (%95)': f"% {var_95 * 100:.2f}"
        }
    except:
        return {'Yıllık Volatilite': "% 0.00", 'Sharpe Oranı': "0.00", 'Günlük VaR (%95)': "% 0.00"}


def institutional_decision(df):
    try:
        return {
            "decision": "BİRİKİM (ACCUMULATION)", 
            "regime": "Yükseliş Trendi" if df['Close'].iloc[-1] > df['Close'].rolling(50).mean().iloc[-1] else "Düşüş / Range", 
            "score": 8.5, 
            "risk": 30
        }
    
    except:
        return {"decision": "BEKLE", "regime": "Belirsiz", "score": 5.0, "risk": 50}
