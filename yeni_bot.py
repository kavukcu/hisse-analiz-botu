# ==========================================
# KÜTÜPHANELER (En üste taşındı ve hızlandırıldı)
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import requests
import concurrent.futures
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Yapay Zeka Kütüphaneleri
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import sqlite3
import optuna
from sklearn.metrics import mean_squared_error
# ==========================================
# SAYFA AYARLARI VE OTURUM
# ==========================================
st.set_page_config(layout="wide", page_title="God Mode Terminal v100")

oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})
# ==========================================
# VERİTABANI VE HAFIZA YÖNETİMİ
# ==========================================
def veritabani_baslat():
    """Yapay zekanın tahminlerini tutacağı yerel veritabanını oluşturur."""
    conn = sqlite3.connect('hisse_hafiza.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tahminler
                 (tarih TEXT, sembol TEXT, hedef_fiyat REAL, gerceklesme_fiyati REAL, durum TEXT)''')
    conn.commit()
    conn.close()

def tahmin_kaydet(sembol, hedef_fiyat):
    """Bugünün tahminlerini hafızaya yazar."""
    conn = sqlite3.connect('hisse_hafiza.db', timeout=10)
    c = conn.cursor()
    bugun = datetime.now().strftime("%Y-%m-%d")
    
    # Aynı gün aynı hisse için zaten kayıt yapıldıysa tekrar eklemeyi önle
    c.execute("SELECT * FROM tahminler WHERE tarih=? AND sembol=?", (bugun, sembol))
    if not c.fetchone():
        c.execute("INSERT INTO tahminler (tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum) VALUES (?, ?, ?, NULL, 'BEKLİYOR')", 
                  (bugun, sembol, hedef_fiyat))
    conn.commit()
    conn.close()

def tahminleri_degerlendir():
    """5 gün öncesinin tahminlerini bugünün gerçek fiyatlarıyla kıyaslar."""
    conn = sqlite3.connect('hisse_hafiza.db', timeout=10)
    c = conn.cursor()
    c.execute("SELECT rowid, tarih, sembol, hedef_fiyat FROM tahminler WHERE durum = 'BEKLİYOR'")
    bekleyenler = c.fetchall()
    
    for row in bekleyenler:
        rowid, tarih_str, sembol, hedef_fiyat = row
        kayit_tarihi = datetime.strptime(tarih_str, "%Y-%m-%d")
        
        # Eğer tahminin üzerinden 5 gün geçmişse kontrol et
        if (datetime.now() - kayit_tarihi).days >= 5:
            try:
                # Güncel fiyatı çek
                df = yf.download(sembol, period="1d", progress=False)
                if not df.empty:
                    gercek_fiyat = float(df['Close'].iloc[-1])
                    
                    # Hedef fiyat ile gerçek fiyat arasındaki sapmayı (hata payını) hesapla
                    sapma_orani = abs(gercek_fiyat - hedef_fiyat) / gercek_fiyat
                    
                    # %5'lik bir yanılma payını başarılı kabul ediyoruz
                    durum = "BAŞARILI ✅" if sapma_orani <= 0.05 else "BAŞARISIZ ❌"
                    
                    c.execute("UPDATE tahminler SET gerceklesme_fiyati = ?, durum = ? WHERE rowid = ?", 
                              (gercek_fiyat, durum, rowid))
            except Exception as e:
                logging.error(f"Tahmin değerlendirme hatası [{sembol}]: {e}")
                
    conn.commit()
    conn.close()

# Uygulama açıldığında veritabanını hazırla ve eski tahminleri kontrol et
veritabani_baslat()

# ==========================================
# 1. TEMEL VE İLERİ TEKNİK FONKSİYONLAR
# ==========================================
# ==========================================
# 1. GÜNLÜK VERİ ÇEKME FONKSİYONU
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def veri_yukle(ticker, start, end, interval="1d"):
    import yfinance as yf
    import pandas as pd
    import time
    import logging
    
    for _ in range(3):
        try:
            # 1. 'end' ve 'session' PARAMETRELERİNİ KALDIRDIK!
            # Böylece Yahoo Finance eski cache'den değil, canlı olarak o anki en son veriye kadar her şeyi getirmek ZORUNDA kalır.
            df = yf.download(
                ticker, 
                start=start, 
                interval=interval, 
                progress=False, 
                auto_adjust=True, 
                threads=True
            )
            
            if df.empty:
                time.sleep(1)
                continue
                
            # Yfinance MultiIndex hatası koruması
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            gerekli = ["Open", "High", "Low", "Close", "Volume"]
            if any(c not in df.columns for c in gerekli):
                raise ValueError("Eksik veya boş veri")
                
            df = df.dropna(subset=['Close'])
            
            # 2. Saat dilimlerini sıfırla ve Pandas ile kendi içimizde filtrele
            df.index = df.index.tz_localize(None)
            
            # Kullanıcının Streamlit'ten seçtiği bitiş tarihini baz alıyoruz
            bitis_tarihi = pd.to_datetime(end).date()
            df = df[df.index.date <= bitis_tarihi]
            
            if not df.empty:
                return df
        except Exception as e:
            logging.warning(f"{ticker} Günlük veri indirilemedi: {e}")
            time.sleep(1)
    return pd.DataFrame()

# ==========================================
# 2. 4 SAATLİK VERİ ÇEKME FONKSİYONU
# ==========================================
def veri_4saatlik_getir(ticker, start, end):
    import yfinance as yf
    import pandas as pd
    import time
    from datetime import datetime, timedelta
    import logging

    try:
        # Yahoo 1h veriyi en fazla 730 gün geriye dönük verir, sınırı aşıyorsa dinamik olarak kırp.
        start_dt = pd.to_datetime(start)
        if (datetime.now() - start_dt).days > 729:
            start_dt = datetime.now() - timedelta(days=729)
            start = start_dt.strftime('%Y-%m-%d')

        for deneme in range(3):
            try:
                time.sleep(0.1) # Anti-Ban beklemesi
                
                # Burada da 'end' parametresini sildik! En güncel saniyeye kadar çekecek.
                df_1h = yf.download(
                    ticker, 
                    start=start, 
                    interval="1h", 
                    progress=False
                )
                
                if df_1h.empty:
                    time.sleep(1)
                    continue

                if isinstance(df_1h.columns, pd.MultiIndex):
                    df_1h.columns = df_1h.columns.droplevel(1)

                df_1h.index = df_1h.index.tz_localize(None)
                bitis_tarihi = pd.to_datetime(end).date()
                
                # İndirilen ham veriyi, kullanıcının seçtiği güne kadar kırp
                df_1h = df_1h[df_1h.index.date <= bitis_tarihi]

                # 1 Saatliği 4 Saatliğe çevir
                df_4h = df_1h.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                   
                return df_4h
                
            except Exception as e:
                logging.debug(f"{ticker} 4H deneme {deneme+1} başarısız: {e}")
                time.sleep(2)
                
        return pd.DataFrame()
        
    except Exception as e:
        logging.error(f"{ticker} 4H Genel Hata: {e}")
        return pd.DataFrame()
def tilson_t3(close, period=5, vfactor=0.7):
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    ema4 = ema3.ewm(span=period, adjust=False).mean()
    ema5 = ema4.ewm(span=period, adjust=False).mean()
    ema6 = ema5.ewm(span=period, adjust=False).mean()
    
    c1 = -vfactor**3
    c2 = 3*vfactor**2 + 3*vfactor**3
    c3 = -6*vfactor**2 - 3*vfactor - 3*vfactor**3
    c4 = 1 + 3*vfactor + vfactor**3 + 3*vfactor**2
    
    return c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3

@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}
@st.cache_data(ttl=86400, show_spinner=False) # Veriyi 24 saatte bir günceller, API'yi yormaz
def sihirli_formul_skorla(sembol):
    """
    Şirketin temel çarpanlarını çeker ve hissenin 
    'ucuz', 'kârlı' ve 'güvenli' olup olmadığını 0-100 arası puanlar.
    """
    try:
        info = sirket_bilgisi_getir(sembol)
        if not info:
            return {'Puan': 0}
            
        skor = 0
        
        # 1. F/K Oranı (Fiyat/Kazanç) - Hisse ucuz mu? (Maksimum 25 Puan)
        fk = info.get('trailingPE', 999)
        if fk is None: fk = 999
        if 0 < fk <= 10: skor += 25
        elif 10 < fk <= 15: skor += 15
        elif 15 < fk <= 20: skor += 5
        
        # 2. PD/DD Oranı (Piyasa Değeri / Defter Değeri) - Özkaynaklarına göre ucuz mu? (Maks 25 Puan)
        pddd = info.get('priceToBook', 999)
        if pddd is None: pddd = 999
        if 0 < pddd <= 1.5: skor += 25
        elif 1.5 < pddd <= 3.0: skor += 15
        elif 3.0 < pddd <= 5.0: skor += 5
        
        # 3. ROE (Özsermaye Kârlılığı) - Parayı iyi yönetip kâr ediyor mu? (Maks 25 Puan)
        roe = info.get('returnOnEquity', -1)
        if roe is None: roe = -1
        if roe > 0.20: skor += 25  # %20 üzeri kârlılık
        elif roe > 0.10: skor += 15
        elif roe > 0.05: skor += 5
        
        # 4. Cari Oran (Dönen Varlıklar / Kısa Vadeli Yükümlülükler) - Batma/Borç riski var mı? (Maks 25 Puan)
        cari_oran = info.get('currentRatio', 0)
        if cari_oran is None: cari_oran = 0
        if cari_oran >= 1.5: skor += 25
        elif cari_oran >= 1.0: skor += 15
        
        return {'Puan': skor}
        
    except Exception as e:
        import logging
        logging.warning(f"[{sembol}] Temel veri puanlama hatası: {str(e)}")
        return {'Puan': 0}
def stokastik_hesapla(df, k_periyot=14, d_periyot=3):
    try:
        low_min = df['Low'].rolling(window=k_periyot).min()
        high_max = df['High'].rolling(window=k_periyot).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-9))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_periyot).mean()
        return df
    except Exception:
        df['Stoch_K'] = 50.0
        df['Stoch_D'] = 50.0
        return df

def smc_hesapla(df):
    df_smc = df.copy()
    df_smc['FVG_Bullish'] = (df_smc['Low'] > df_smc['High'].shift(2)) & (df_smc['Close'].shift(1) > df_smc['Open'].shift(1))
    df_smc['FVG_Bearish'] = (df_smc['High'] < df_smc['Low'].shift(2)) & (df_smc['Close'].shift(1) < df_smc['Open'].shift(1))
    return df_smc

@st.cache_data(ttl=3600, show_spinner=False)
def ileri_teknik_gostergeler(df):
    df_ta = df.copy()
    high_9 = df_ta['High'].rolling(window=9).max()
    low_9 = df_ta['Low'].rolling(window=9).min()
    df_ta['Tenkan_Sen'] = (high_9 + low_9) / 2
    high_26 = df_ta['High'].rolling(window=26).max()
    low_26 = df_ta['Low'].rolling(window=26).min()
    df_ta['Kijun_Sen'] = (high_26 + low_26) / 2
    df_ta['Senkou_Span_A'] = ((df_ta['Tenkan_Sen'] + df_ta['Kijun_Sen']) / 2).shift(26)
    high_52 = df_ta['High'].rolling(window=52).max()
    low_52 = df_ta['Low'].rolling(window=52).min()
    df_ta['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    df_ta['Chikou_Span'] = df_ta['Close'].shift(-26)
    
    prev_high = df_ta['High'].shift(1)
    prev_low = df_ta['Low'].shift(1)
    prev_close = df_ta['Close'].shift(1)
    range_hl = prev_high - prev_low
    
    df_ta['Cam_H4'] = prev_close + (range_hl * 1.1 / 2)
    df_ta['Cam_H3'] = prev_close + (range_hl * 1.1 / 4)
    df_ta['Cam_L3'] = prev_close - (range_hl * 1.1 / 4)
    df_ta['Cam_L4'] = prev_close - (range_hl * 1.1 / 2)

    df_ta['Ichimoku_Trend'] = np.where(df_ta['Close'] > df_ta['Senkou_Span_A'], 
                                       np.where(df_ta['Close'] > df_ta['Senkou_Span_B'], "GÜÇLÜ BOĞA", "NÖTR"), 
                                       np.where(df_ta['Close'] < df_ta['Senkou_Span_B'], "GÜÇLÜ AYI", "NÖTR"))
    return df_ta

def grafik_formasyon_bul(df, window=10, tolerans=0.03):
    try:
        df_form = df.copy()
        # Geleceği görme (look-ahead) engellendi. Tepe/Dip onayı 'window' gün sonra verilir.
        df_form['Roll_Max'] = df_form['High'].rolling(window=window*2+1).max()
        df_form['Roll_Min'] = df_form['Low'].rolling(window=window*2+1).min()
        
        df_form['Local_Max'] = df_form['High'].shift(window) == df_form['Roll_Max']
        df_form['Local_Min'] = df_form['Low'].shift(window) == df_form['Roll_Min']
        
        ikili_tepeler, ikili_dipler = [], []
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

def mum_formasyonlarini_bul(df):
    df_f = df.copy()
    govde = abs(df_f['Close'] - df_f['Open'])
    mum_boyu = df_f['High'] - df_f['Low']
    df_f['Doji'] = govde <= (mum_boyu * 0.1)
    df_f['Bullish_Engulfing'] = (df_f['Close'].shift(1) < df_f['Open'].shift(1)) & (df_f['Open'] < df_f['Close'].shift(1)) & (df_f['Close'] > df_f['Open'].shift(1))
    df_f['Bearish_Engulfing'] = (df_f['Close'].shift(1) > df_f['Open'].shift(1)) & (df_f['Open'] > df_f['Close'].shift(1)) & (df_f['Close'] < df_f['Open'].shift(1))
    return df_f
def dipten_donus_analizi(df):
    """Fiyatın dipten sekme ihtimalini kurumsal tekniklerle (Hacim, RSI Uyuşmazlığı, Spring) hesaplar."""
    if df is None or len(df) < 20:
        df_copy = df.copy() if df is not None else pd.DataFrame()
        df_copy['Hacim_Patlamasi'] = False
        df_copy['Pozitif_Uyusmazlik'] = False
        df_copy['Wyckoff_Spring'] = False
        return df_copy

    df_dip = df.copy()
   
    
    # 1. Hacim Patlaması (Son 20 günün ortalamasının en az 2 katı)
    df_dip['Vol_SMA_20'] = df_dip['Volume'].rolling(20).mean()
    df_dip['Hacim_Patlamasi'] = df_dip['Volume'] > (df_dip['Vol_SMA_20'] * 2)
    
    # 2. Wyckoff Spring (Ayı Tuzağı - Bollinger Alt Bandı İhlali ve Hızlı Dönüş)
    df_dip['SMA_20_Dip'] = df_dip['Close'].rolling(20).mean()
    df_dip['STD_20_Dip'] = df_dip['Close'].rolling(20).std()
    df_dip['Lower_Band'] = df_dip['SMA_20_Dip'] - (df_dip['STD_20_Dip'] * 2)
    
    # Gün içinde alt bandı kırmış ama günü bandın ve açılışın üzerinde (yeşil) kapatmış mı?
    df_dip['Wyckoff_Spring'] = (df_dip['Low'] < df_dip['Lower_Band']) & (df_dip['Close'] > df_dip['Lower_Band']) & (df_dip['Close'] > df_dip['Open'])
    
    # 3. RSI Pozitif Uyuşmazlık (Bullish Divergence)
    delta = df_dip['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df_dip['RSI_DIP'] = 100 - (100 / (1 + rs))
    
    # Son 20 günlük periyotta fiyat daha düşük dip yaparken, RSI daha yüksek dip yapıyorsa
    if len(df_dip) >= 20:
        son_5_fiyat = df_dip['Close'].iloc[-5:].min()
        eski_15_fiyat = df_dip['Close'].iloc[-20:-5].min()
        son_5_rsi = df_dip['RSI_DIP'].iloc[-5:].min()
        eski_15_rsi = df_dip['RSI_DIP'].iloc[-20:-5].min()
        
        # Uyuşmazlık şartı ve RSI'ın aşırı satım bölgesine yakın (45 altı) olması
        uyusmazlik = (son_5_fiyat < eski_15_fiyat) and (son_5_rsi > eski_15_rsi) and (son_5_rsi < 45)
        df_dip['Pozitif_Uyusmazlik'] = uyusmazlik
    else:
        df_dip['Pozitif_Uyusmazlik'] = False
        
    return df_dip
# --- MEVCUT KODUNUZ (BUNA KESİNLİKLE DOKUNMUYORUZ) ---
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

def haber_duygu_analizi(ticker):
    try:
        news_data = yf.Ticker(ticker, session=oturum).news
        if not news_data: return []
        olumlu = ["rekor", "artış", "büyüdü", "pozitif", "yüksel", "kazanç", "anlaşma"]
        olumsuz = ["düştü", "zarar", "azaldı", "negatif", "kayıp", "düşüş", "ceza"]
        sonuclar = []
        for n in news_data[:5]:
            metin = (str(n.get('title', '')) + " " + str(n.get('summary', ''))).lower()
            ol_skor = sum(1 for k in olumlu if k in metin)
            sz_skor = sum(1 for k in olumsuz if k in metin)
            duygu = "🟢 OLUMLU" if ol_skor > sz_skor else ("🔴 OLUMSUZ" if sz_skor > ol_skor else "🟡 NÖTR")
            sonuclar.append({"baslik": n.get('title'), "kaynak": n.get('publisher'), "link": n.get('link'), "duygu": duygu})
        return sonuclar
    except: return []
def asenkron_analiz_yap(sembol, baslangic, bitis, analiz_tipi="radar"):
    try:
        # 1. ÖNCE SADECE GÜNLÜK VERİYİ ÇEK (Çok Hızlı İşlem)
        df_gunluk = veri_yukle(sembol, baslangic, bitis, interval="1d")
        
        if df_gunluk.empty: 
            return None
            
        df_g_kapanmis = df_gunluk.copy()

        # Günlük bazda temel göstergeleri hemen hesapla
        df_g_kapanmis = stokastik_hesapla(df_g_kapanmis)
        df_g_kapanmis['Tilson_T3'] = tilson_t3(df_g_kapanmis['Close'])
        temp_g = dipten_donus_analizi(df_g_kapanmis)
        
        g_fiyat = df_g_kapanmis['Close'].iloc[-1]
        g_tilson = df_g_kapanmis['Tilson_T3'].iloc[-1]
        g_stoch_k = df_g_kapanmis['Stoch_K'].iloc[-1]
        g_stoch_d = df_g_kapanmis['Stoch_D'].iloc[-1]
        
        g_boga = g_fiyat > g_tilson
        g_stoch_al = (g_stoch_k < 35) and (g_stoch_k > g_stoch_d)
        g_hacim = temp_g['Hacim_Patlamasi'].iloc[-1]
        g_uyusmazlik = temp_g['Pozitif_Uyusmazlik'].iloc[-1]
        g_spring = temp_g['Wyckoff_Spring'].iloc[-1]

        # ==========================================
        # ⚡ AKILLI FİLTRE: ERKEN ÇIKIŞ (EARLY EXIT) 
        # ==========================================
        umut_var_mi = g_boga or g_stoch_al or g_hacim or g_uyusmazlik or g_spring
        
        if not umut_var_mi and analiz_tipi == "radar":
            return {
                "Varlık": sembol,
                "Kapanış Fiyatı": f"{g_fiyat:.2f}",
                "🎯 AL/SAT Kararı": "🐻 PAS GEÇİLDİ (Ölü Trend)",
                "Günlük T3": "🐻 AYI",
                "4S T3": "-",
                "📊 Temel Skor": "-",
                "💥 Hacim Analizi": "Normal",
                "📈 Pozitif Uyuşmazlık": "-",
                "🪤 Spring (Tuzak)": "-",
                "🤖 AI Kararı": "Zaman Tasarrufu",
                "🎯 AI Hedef": "-"
            }

        # ==========================================
        # 2. SADECE UMUT VAAT EDEN HİSSELER İÇİN AĞIR İŞLEMLER
        # ==========================================
        # Eğer hissede yukarı yönlü bir sinyal (hacim, boğa trendi vs.) varsa 
        # API'yi yoracak olan 4 Saatlik veri ve Temel Analiz bilgilerini ŞİMDİ çekiyoruz.
        df_4h = veri_4saatlik_getir(sembol, baslangic, bitis)
        
        # B) 4 SAATLİK KAPANIS ANALİZİ (Tilson + Stoch)
        if not df_4h.empty and len(df_4h) >= 20:
            df_4h = stokastik_hesapla(df_4h)
            df_4h['Tilson_T3'] = tilson_t3(df_4h['Close'])
            
            h4_fiyat = df_4h['Close'].iloc[-1]
            h4_tilson = df_4h['Tilson_T3'].iloc[-1]
            h4_stoch_k = df_4h['Stoch_K'].iloc[-1]
            h4_stoch_d = df_4h['Stoch_D'].iloc[-1]
            
            h4_boga = h4_fiyat > h4_tilson
            h4_stoch_al = (h4_stoch_k < 35) and (h4_stoch_k > h4_stoch_d)
        else:
            h4_boga, h4_stoch_al = g_boga, g_stoch_al

        # C) NİHAİ AL / SAT KARARI (Çift Onay Sistemi)
        if g_boga and h4_boga:
            if g_stoch_al and h4_stoch_al:
                al_sat_karari = "🚀 GÜÇLÜ AL (4S + Günlük Onaylı)"
            elif g_stoch_al or h4_stoch_al:
                al_sat_karari = "🟢 AL (Tek Zaman Dilimi Erken Sinyal)"
            else:
                al_sat_karari = "📈 BOĞA TRENDİ (Düzeltmede)"
        elif g_boga and not h4_boga:
            al_sat_karari = "⚠️ DÜZELTME (Günlük Boğa / 4S Ayı)"
        elif not g_boga and h4_boga:
            al_sat_karari = "⚡ TEPKİ YÜKSELİŞİ (4S Boğa / Günlük Ayı)"
        else:
            al_sat_karari = "🐻 GÜÇLÜ SAT / AYI (4S + Günlük Ayı)"

        if analiz_tipi == "radar":
            temp_4h = dipten_donus_analizi(df_4h) if not df_4h.empty else temp_g
            h4_hacim = temp_4h['Hacim_Patlamasi'].iloc[-1]
            
            if g_hacim and h4_hacim:
                hacim_durum = "🔥 GÜÇLÜ PATLAMA (4S+Günlük)"
            elif h4_hacim:
                hacim_durum = "⚡ 4S HACİM PATLAMASI (Erken)"
            elif g_hacim:
                hacim_durum = "💥 GÜNLÜK HACİM PATLAMASI"
            else:
                hacim_durum = "Normal"

            h4_uyusmazlik = temp_4h['Pozitif_Uyusmazlik'].iloc[-1]
            if g_uyusmazlik and h4_uyusmazlik:
                uyusmazlik_durum = "✅✅ ÇİFT UYUŞMAZLIK (4S+Günlük)"
            elif h4_uyusmazlik:
                uyusmazlik_durum = "⚡ 4S UYUŞMAZLIK (Erken Sinyal)"
            elif g_uyusmazlik:
                uyusmazlik_durum = "📈 GÜNLÜK UYUŞMAZLIK"
            else:
                uyusmazlik_durum = "-"

            spring_durum = "✅ VAR" if (g_spring or temp_4h['Wyckoff_Spring'].iloc[-1]) else "-"
            
            try:
                s_skor = sihirli_formul_skorla(sembol)['Puan']
            except:
                s_skor = 0

            # AI Kararı 
            if (g_boga or h4_boga) and (g_stoch_al or h4_stoch_al or h4_hacim or g_uyusmazlik or h4_uyusmazlik):
                ai_veri = ensemble_prediction(df_g_kapanmis, sembol)
            else:
                ai_veri = {'signal': "ZAYIF (AI Pas Geçti)", 'rf_prediction': 0.0, 'confidence': 0.0}

            return {
                "Varlık": sembol,
                "Kapanış Fiyatı": f"{g_fiyat:.2f}",
                "🎯 AL/SAT Kararı": al_sat_karari,
                "Günlük T3": "🚀 BOĞA" if g_boga else "🐻 AYI",
                "4S T3": "🚀 BOĞA" if h4_boga else "🐻 AYI",
                "📊 Temel Skor": s_skor,
                "💥 Hacim Analizi": hacim_durum,
                "📈 Pozitif Uyuşmazlık": uyusmazlik_durum,
                "🪤 Spring (Tuzak)": spring_durum,
                "🤖 AI Kararı": ai_veri['signal'],
                "🎯 AI Hedef": f"{ai_veri['rf_prediction']} TL"
            }

        elif analiz_tipi == "stoch":
            return {
                "Varlık": sembol,
                "Son Fiyat": f"{g_fiyat:.2f}",
                "Günlük Stoch %K": round(g_stoch_k, 2),
                "4S Stoch %K": round(h4_stoch_k, 2),
                "Durum": "🟢 Çift Dip/Al" if (g_stoch_al and h4_stoch_al) else ("↗️ Pozitif" if h4_stoch_al else "⚪ Nötr")
            }

        elif analiz_tipi == "tilson":
            return {
                "Varlık": sembol,
                "Son Fiyat": f"{g_fiyat:.2f}",
                "Günlük Tilson": f"{g_tilson:.2f} ({'🚀 BOĞA' if g_boga else '🐻 AYI'})",
                "4S Tilson": f"{h4_tilson:.2f} ({'🚀 BOĞA' if h4_boga else '🐻 AYI'})",
                "Trend Uyumu": "✅ ÇİFT BOĞA" if (g_boga and h4_boga) else ("⚠️ UYUMSUZ" if (g_boga != h4_boga) else "❌ ÇİFT AYI")
            }

    except Exception as e:
        logging.error(f"[{sembol}] Analiz Hatası: {str(e)}")
        return None
# ==========================================
# 2. YAPAY ZEKA VE KURUMSAL MOTORLAR
# ==========================================
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
@st.cache_data(ttl=86400) # Her hissenin en iyi ayarını 24 saat hafızada tut
def en_iyi_xgb_parametrelerini_bul(sembol, X_matrisi, y_vektoru):
    """Optuna ile hissenin o anki volatilitesine en uygun AI ayarlarını bulur."""
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Konsol kalabalığını önler
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        }
        # Geçmiş verinin %80'i ile çalışıp, %20'si ile kendini test eder
        # ✅ DOĞRU: Zamansal sıralı bölme
        # Zamansal Sıralı Kesme (Data Leakage Önlenir)
        tscv = TimeSeriesSplit(n_splits=5)
        # Sadece son split'i (en güncel eğitim/test ayrımını) alıyoruz
        for train_index, test_index in tscv.split(X_matrisi):
            X_train, X_test = X_matrisi[train_index], X_matrisi[test_index]
            y_train, y_test = y_vektoru[train_index], y_vektoru[test_index]
        model = XGBRegressor(**param, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse # Hatayı en aza indirmeye çalışır

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) # 10 farklı kombinasyon dener
    
    return study.best_params
@st.cache_data(ttl=3600, show_spinner=False)
def ensemble_prediction(df, sembol="Genel"):
    try:
        t_df = df.copy()
        
        # --- 1. Veri Hazırlığı ve Feature Engineering ---
        # --- 1. Veri Hazırlığı ve Feature Engineering ---
        if 'Stoch_K' not in t_df.columns:
            low_min = t_df['Low'].rolling(window=14).min()
            high_max = t_df['High'].rolling(window=14).max()
            t_df['Stoch_K'] = 100 * ((t_df['Close'] - low_min) / (high_max - low_min + 1e-9))
            
        # 1. Stoch_D (Sinyal Çizgisi) Hesaplaması (%K'nın 3 günlük ortalaması)
        t_df['Stoch_D'] = t_df['Stoch_K'].rolling(window=3).mean()

        # 2. Kesişim ve Momentum Farkı (%K - %D)
        t_df['Stoch_Diff'] = t_df['Stoch_K'] - t_df['Stoch_D']
        
        t_df['Tilson_T3'] = tilson_t3(t_df['Close'])
        t_df['Tilson_Dist'] = (t_df['Close'] - t_df['Tilson_T3']) / t_df['Close'].replace(0, 0.0001)
        
        delta = t_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        t_df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 0.0001)))

        macd = t_df['Close'].ewm(span=12, adjust=False).mean() - t_df['Close'].ewm(span=26, adjust=False).mean()
        t_df['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()

        bb_orta = t_df['Close'].rolling(window=20).mean()
        bb_std = t_df['Close'].rolling(window=20).std()
        bb_fark = (bb_std * 4).replace(0, 0.0001)
        t_df['BB_Pozisyon'] = (t_df['Close'] - (bb_orta - (bb_std * 2))) / bb_fark

        high_low = t_df['High'] - t_df['Low']
        high_close = (t_df['High'] - t_df['Close'].shift()).abs()
        low_close = (t_df['Low'] - t_df['Close'].shift()).abs()
        t_df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        t_df['Z_Score'] = (t_df['Close'] - t_df['Close'].rolling(20).mean()) / t_df['Close'].rolling(20).std().replace(0, 0.0001)
        t_df['Vol_Change'] = t_df['Volume'].pct_change()
        t_df['EMA_Trend'] = np.where(t_df['Close'] > t_df['Close'].ewm(span=20).mean(), 1, -1)

        t_df['Target_Return'] = ((t_df['Close'].shift(-5) - t_df['Close']) / t_df['Close']) * 100

        # --- YENİ: ZAMAN SERİSİ HAFIZASI (Lag Features) ---
        # Modelin son 3 günün hafızasını tutması için geçmiş verileri ekliyoruz
        t_df['Return_1d'] = t_df['Close'].pct_change(1)
        t_df['Return_2d'] = t_df['Close'].pct_change(2)
        t_df['Return_3d'] = t_df['Close'].pct_change(3)
        
        t_df['Vol_Lag1'] = t_df['Vol_Change'].shift(1)
        t_df['Vol_Lag2'] = t_df['Vol_Change'].shift(2)
        
        # Yeni 'Hafıza' verileri (Return_X ve Vol_LagX) eğitim matrisine eklendi
        # 3. Güncellenmiş Öznitelik (Features) Listesi
        features = [
            'RSI', 'MACD_Hist', 'BB_Pozisyon', 'ATR', 'Z_Score', 
            'Vol_Change', 'EMA_Trend', 'Stoch_K', 'Stoch_D', 'Stoch_Diff',
            'Tilson_Dist', 'Return_1d', 'Return_2d', 'Return_3d', 
            'Vol_Lag1', 'Vol_Lag2'
        ]
        # ----------------------------------------------------------------------------
        
        t_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        t_df[features] = t_df[features].ffill().bfill().fillna(0)
        ml_df = t_df.dropna(subset=['Target_Return'])

        if len(ml_df) < 50:
            return {"rf_prediction": float(t_df['Close'].iloc[-1]), "signal": "VERİ YETERSİZ", "confidence": 50.0, "expected_return_pct": 0.0, "feature_importances": {}}

        # --- 2. OPTUNA VE YAPAY ZEKA MODELLEME ---
        X = ml_df[features].values
        y = ml_df['Target_Return'].values
        son_veri = t_df[features].iloc[-1].values.reshape(1, -1)

        best_xgb_params = en_iyi_xgb_parametrelerini_bul(sembol, X, y)

        # Diğer modellerin tanımlandığı yer...
        # Diğer modellerin tanımlandığı yer...
        model_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        model_svr = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=1.5, epsilon=0.1, kernel='rbf'))
        ])
        
        # Gradient Boosting modelini tanımlıyoruz
        model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

        # YENİ: Ridge modelini standartlaştırma (scaler) ile tanımlıyoruz
        model_ridge = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])

        # YENİ: gb ve ridge'i de oylamaya (VotingRegressor) ekliyoruz
        ensemble = VotingRegressor(estimators=[
            ('xgb', model_xgb),
            ('rf', model_rf),
            ('svr', model_svr),
            ('gb', model_gb),
            ('ridge', model_ridge)
        ])


        ensemble.fit(X, y)

        # --- 3. ÇIKARIM VE KARAR ---
        beklenen_getiri_pct = float(ensemble.predict(son_veri)[0])
        anlik_fiyat = float(t_df['Close'].iloc[-1])
        hedef_fiyat = anlik_fiyat * (1 + (beklenen_getiri_pct / 100))
        
        sinyal = "🚀 GÜÇLÜ AL" if beklenen_getiri_pct > 2.0 else ("⚠️ SAT" if beklenen_getiri_pct < -1.0 else "NÖTR")
        guven_skoru = min(abs(beklenen_getiri_pct) * 8 + 50, 99.0)

        try:
            f_importances = ensemble.named_estimators_['xgb'].feature_importances_
            oznitelik_agirliklari = {f: float(imp) for f, imp in zip(features, f_importances)}
        except Exception:
            oznitelik_agirliklari = {}

        return {
            "rf_prediction": round(hedef_fiyat, 2),
            "signal": sinyal,
            "confidence": max(round(guven_skoru, 1), 0.0),
            "expected_return_pct": round(beklenen_getiri_pct, 2),
            "feature_importances": oznitelik_agirliklari
        }
        
    except Exception as e:
        import logging
        logging.error(f"AI Ensemble Hatası: {e}")
        return {"rf_prediction": 0.0, "signal": "Hata", "confidence": 0.0, "expected_return_pct": 0.0, "feature_importances": {}}

@st.cache_data(ttl=3600, show_spinner=False)
def gelismis_ai_tahmin(df, gelecek_gun=10):
    try:
        df_ml = df.copy()
        df_ml['Return'] = df_ml['Close'].pct_change()
        df_ml['Log_Return'] = np.log(df_ml['Close'] / df_ml['Close'].shift(1))
        df_ml['SMA_10_Dist'] = df_ml['Close'] / df_ml['Close'].rolling(10).mean() - 1
        df_ml['Volatilite_14'] = df_ml['Return'].rolling(14).std()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        
        df_ml.dropna(inplace=True)
        if len(df_ml) < 50:
            son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun

        features = ['Close', 'Volume', 'Log_Return', 'SMA_10_Dist', 'Volatilite_14']
        X = df_ml[features].values
        y = df_ml['Target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, objective='reg:squarederror', n_jobs=-1)
        model.fit(X_scaled, y)

        tahminler = []
        son_veri = X_scaled[-1].reshape(1, -1)
        
        # 1. Döngüye girmeden ÖNCE geçmiş kapanış verilerini hafızaya alıyoruz
        gecmis_kapanislar = df_ml['Close'].tail(20).tolist()
        
        # 2. Çok adımlı dinamik tahmin döngüsü
        for _ in range(gelecek_gun):
            pred = float(model.predict(son_veri)[0])
            tahminler.append(pred)
            
            # Kapanış listesini yeni tahminle güncelle
            gecmis_kapanislar.append(pred)
            gecmis_kapanislar = gecmis_kapanislar[-20:]  # Son 20 günü tut
            
            # İndikatörleri yeni tahmine göre dinamik hesapla
            yeni_log_ret = np.log(gecmis_kapanislar[-1] / gecmis_kapanislar[-2])
            yeni_sma_10 = np.mean(gecmis_kapanislar[-10:])
            yeni_sma_10_dist = (gecmis_kapanislar[-1] / yeni_sma_10) - 1
            
            getiriler = [np.log(gecmis_kapanislar[i] / gecmis_kapanislar[i-1]) for i in range(1, len(gecmis_kapanislar))]
            yeni_vol = np.std(getiriler[-14:]) if len(getiriler) >= 14 else np.std(getiriler)
            
            # Yeni veriyi ölçeklendirip (Scaler) bir sonraki gün için hazırlar
            yeni_ham_veri = np.array([[pred, son_veri[0, 1], yeni_log_ret, yeni_sma_10_dist, yeni_vol]])
            son_veri = scaler.transform(yeni_ham_veri)            
            
            
            
        tarihler = [df.index[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
        return tarihler, tahminler

    except Exception:
        son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
        return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun

# ==========================================
# 3. YAN MENÜ (SIDEBAR) & VERİ ÇEKME
# ==========================================
# ==========================================
# 3. YAN MENÜ (SIDEBAR) & VERİ ÇEKME
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def tum_bist_hisselerini_getir():
    """BIST'teki tüm hisseleri (yaklaşık 700+) dinamik olarak çeker."""
    try:
        url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/HisseOzet"
        res = requests.get(url, timeout=10)
        data = res.json()
        # Sembollerin sonuna .IS ekleyerek Yahoo Finance (yfinance) formatına uygun hale getiriyoruz
        return [f"{row['kod']}.IS" for row in data['value']]
    except Exception as e:
        import logging
        logging.error(f"BIST Hisseleri çekilemedi: {e}")
        # Bağlantı hatası olursa acil durum listesi (Fallback)
        return ["XU100.IS", "ACSEL.IS", "ADEL.IS", "ADESE.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS", "AGHOL.IS", "AHGAZ.IS", 
"AKBNK.IS", "AKCNS.IS", "AKENR.IS", "AKFGY.IS", "AKFYE.IS", "AKGRT.IS", "AKMGY.IS", "AKSA.IS", 
"AKSEN.IS", "AKSUE.IS", "AKYHO.IS", "ALARK.IS", "ALBRK.IS", "ALCAR.IS", "ALCTL.IS", "ALFAS.IS", 
"ALGYO.IS", "ALKA.IS", "ALKIM.IS", "ALMAD.IS", "ALTNY.IS", "ALVES.IS", "ANELE.IS", "ANGEN.IS", 
"ANHYT.IS", "ANSGR.IS", "ARADA.IS", "ARASE.IS", "ARCLK.IS", "ARDYZ.IS", "ARENA.IS", "ARSAN.IS", 
"ARTMS.IS", "ARZUM.IS", "ASELS.IS", "ASGYO.IS", "ASTOR.IS", "ASUZU.IS", "ATACP.IS", "ATAGY.IS", 
"ATATP.IS", "ATEKS.IS", "ATLAS.IS", "AVGYO.IS", "AVHOL.IS", "AVOD.IS", "AVTUR.IS", "AYCES.IS", 
"AYDEM.IS", "AYEN.IS", "AYGAZ.IS", "AZTEK.IS", "BAGFS.IS", "BAKAB.IS", "BALAT.IS", "BANVT.IS", 
"BARMA.IS", "BASCM.IS", "BASGZ.IS", "BAYRK.IS", "BEYAZ.IS", "BFREN.IS", "BIENY.IS", "BIGCH.IS", 
"BIMAS.IS", "BINHO.IS", "BIOEN.IS", "BIZIM.IS", "BJKAS.IS", "BLCYT.IS", "BMSCH.IS", "BMSTL.IS", 
"BNTAS.IS", "BOBET.IS", "BORSK.IS", "BOSSA.IS", "BOYP.IS", "BRISA.IS", "BRKO.IS", "BRKSN.IS", 
"BRKVY.IS", "BRLSM.IS", "BRMEN.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "BUCIM.IS", 
"BURCE.IS", "BURVA.IS", "BVSAN.IS", "BYDNR.IS", "CANTE.IS", "CASA.IS", "CATES.IS", "CCOLA.IS", 
"CELHA.IS", "CEMAS.IS", "CEMTS.IS", "CEOEM.IS", "CIMSA.IS", "CLEBI.IS", "CMBTN.IS", "CMENT.IS", 
"CONSE.IS", "COSMO.IS", "CRDFA.IS", "CRFSA.IS", "CUSAN.IS", "CVKMD.IS", "CWENE.IS", "DAGHL.IS", 
"DAGI.IS", "DAPGM.IS", "DARDL.IS", "DERHL.IS", "DERIM.IS", "DESA.IS", "DESPC.IS", "DEVA.IS", 
"DIRIT.IS", "DITAS.IS", "DMRGD.IS", "DOAS.IS", "DOCO.IS", "DOFER.IS", "DOGUB.IS", "DOHOL.IS", 
"DOKTA.IS", "DURDO.IS", "DYOBY.IS", "DZGYO.IS", "EBEBK.IS", "ECILC.IS", "ECZYT.IS", "EDATA.IS", 
"EFFE.IS", "EGCEY.IS", "EGCYO.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EKGYO.IS", 
"EKIZ.IS", "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "ENSRI.IS", 
"ENTGO.IS", "EPLAS.IS", "ERBOS.IS", "EREGL.IS", "ERSU.IS", "ESCAR.IS", "ESCOM.IS", "ESEN.IS", 
"ETILR.IS", "ETYAT.IS", "EUHOL.IS", "EUPWR.IS", "EUREN.IS", "EUYO.IS", "EYGYO.IS", "FADE.IS", 
"FENER.IS", "FLAP.IS", "FMIZP.IS", "FONET.IS", "FORMT.IS", "FORTE.IS", "FRIGO.IS", "FROTO.IS", 
"FZLGY.IS", "GARAN.IS", "GARFA.IS", "GEDIK.IS", "GEDZA.IS", "GENIL.IS", "GENTS.IS", "GEREL.IS", 
"GESAN.IS", "GIPTA.IS", "GLBMD.IS", "GLCVY.IS", "GLRYH.IS", "GLYHO.IS", "GMTAS.IS", "GOKNR.IS", 
"GOLTS.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS", "GRSEL.IS", "GRTRK.IS", "GSDDE.IS", "GSDHO.IS", 
"GSRAY.IS", "GUBRF.IS", "GWIND.IS", "GZNMI.IS", "HALKB.IS", "HATEK.IS", "HATSN.IS", "HDFGS.IS", 
"HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "HRKET.IS", "HTTBT.IS", "HUBVC.IS", "HUNER.IS", 
"HURGZ.IS", "ICBCT.IS", "IDEAS.IS", "IDGYO.IS", "IEYHO.IS", "IHAAS.IS", "IHEVA.IS", "IHGZT.IS", 
"IHLAS.IS", "IHLGM.IS", "IHYAY.IS", "IMASM.IS", "INDES.IS", "INFO.IS", "INGRM.IS", "INTEM.IS", 
"INVEO.IS", "INVES.IS", "IPEKE.IS", "ISBTR.IS", "ISCTR.IS", "ISDMR.IS", "ISFIN.IS", "ISGSY.IS", 
"ISGYO.IS", "ISKPL.IS", "ISKUR.IS", "ISMEN.IS", "ISSEN.IS", "ISYAT.IS", "ITTFH.IS", "IYISM.IS", 
"IZENR.IS", "IZFAS.IS", "IZINV.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS", "KAREL.IS", "KARSN.IS", 
"KARTN.IS", "KARYE.IS", "KATMR.IS", "KAYSE.IS", "KCAER.IS", "KCHOL.IS", "KENT.IS", "KERVN.IS", 
"KERVT.IS", "KFEIN.IS", "KGYO.IS", "KIMMR.IS", "KLGYO.IS", "KLKIM.IS", "KLMSN.IS", "KLNMA.IS", 
"KLRHO.IS", "KLSYN.IS", "KMPUR.IS", "KNFRT.IS", "KONKA.IS", "KONTR.IS", "KONYA.IS", "KOPOL.IS", 
"KORDS.IS", "KOZAA.IS", "KOZAL.IS", "KRDMA.IS", "KRDMB.IS", "KRDMD.IS", "KRGYO.IS", "KRONT.IS", 
"KRPLS.IS", "KRSTL.IS", "KRTEK.IS", "KRVGD.IS", "KSTUR.IS", "KTLEV.IS", "KTSKR.IS", "KUTPO.IS", 
"KUVVA.IS", "KUYAS.IS", "KZBGY.IS", "KZGYO.IS", "LIDER.IS", "LIDFA.IS", "LINK.IS", "LKMNH.IS", 
"LMOUR.IS", "LOGO.IS", "LRSHO.IS", "LUKSK.IS", "MAALT.IS", "MACKO.IS", "MACRO.IS", "MAGEN.IS", 
"MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MARBL.IS", "MARKA.IS", "MARTI.IS", "MAVI.IS", "MEDTR.IS", 
"MEGAP.IS", "MEKAG.IS", "MEPET.IS", "MERCN.IS", "MERIT.IS", "MERKO.IS", "METRO.IS", "METUR.IS", 
"MGROS.IS", "MHRGY.IS", "MIATK.IS", "MIPAZ.IS", "MMCAS.IS", "MNDRS.IS", "MNDTR.IS", "MOBTL.IS", 
"MOGAN.IS", "MPARK.IS", "MRGYO.IS", "MRSHL.IS", "MSGYO.IS", "MTRKS.IS", "MTRYO.IS", "MZHLD.IS", 
"NATEN.IS", "NETAS.IS", "NIBAS.IS", "NTGAZ.IS", "NTHOL.IS", "NUGYO.IS", "NUHCM.IS", "OBASE.IS", 
"OBAMS.IS", "ODAS.IS", "OFSYM.IS", "ONCSM.IS", "ORCAY.IS", "ORGE.IS", "ORMA.IS", "OSMEN.IS", 
"OSTIM.IS", "OTKAR.IS", "OUAKY.IS", "OYAKC.IS", "OYAYO.IS", "OYLUM.IS", "OYYAT.IS", "OZGYO.IS", 
"OZKGY.IS", "OZRDN.IS", "OZSUB.IS", "PAGYO.IS", "PAMEL.IS", "PAPIL.IS", "PARSN.IS", "PASEU.IS", 
"PATEK.IS", "PCILT.IS", "PEGYO.IS", "PEKGY.IS", "PENGD.IS", "PENTA.IS", "PETKM.IS", "PETUN.IS", 
"PGSUS.IS", "PINSU.IS", "PKART.IS", "PKENT.IS", "PLTUR.IS", "PNLSN.IS", "PNSUT.IS", "POLHO.IS", 
"POLTK.IS", "PRDGS.IS", "PRKAB.IS", "PRKME.IS", "PRZMA.IS", "PSDTC.IS", "PSGYO.IS", "QNBFB.IS", 
"QNBFL.IS", "QUAGR.IS", "RALYH.IS", "RAYSG.IS", "REEDR.IS", "RNPOL.IS", "RODRG.IS", "RTALB.IS", 
"RUBNS.IS", "RYGYO.IS", "RYSAS.IS", "SAHOL.IS", "SAMAT.IS", "SANEL.IS", "SANFM.IS", "SANKO.IS", 
"SARKY.IS", "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SEGYO.IS", "SEKFK.IS", "SEKUR.IS", "SELEC.IS", 
"SELGD.IS", "SELVA.IS", "SEYKM.IS", "SILVR.IS", "SISE.IS", "SKBNK.IS", "SKTAS.IS", "SMART.IS", 
"SMRTG.IS", "SNGYO.IS", "SNICA.IS", "SNKRN.IS", "SNPAM.IS", "SOKE.IS", "SOKM.IS", "SONME.IS", 
"SRVGY.IS", "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "SUWEN.IS", "TABGD.IS", "TARKM.IS", "TATEN.IS", 
"TATGD.IS", "TAVHL.IS", "TBORG.IS", "TCELL.IS", "TDGYO.IS", "TEKTU.IS", "TERA.IS", "TETMT.IS", 
"TEZOL.IS", "TGSAS.IS", "THYAO.IS", "TKFEN.IS", "TKNSA.IS", "TLMAN.IS", "TMPOL.IS", "TMSN.IS", 
"TOASO.IS", "TRCAS.IS", "TRGYO.IS", "TRILC.IS", "TSGYO.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS", 
"TTRAK.IS", "TUCLK.IS", "TUKAS.IS", "TUPRS.IS", "TUREX.IS", "TURGG.IS", "TURSG.IS", "UFUK.IS", 
"ULAS.IS", "ULUFA.IS", "ULUSE.IS", "ULUUN.IS", "UMPAS.IS", "UNLU.IS", "USAK.IS", "UZERB.IS", 
"VAKBN.IS", "VAKFN.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VERUS.IS", "VESBE.IS", "VESTL.IS", 
"VKGYO.IS", "VKING.IS", "VRGYO.IS", "YAPRK.IS", "YATAS.IS", "YAYLA.IS", "YBTAS.IS", "YEOTK.IS", 
"YESIL.IS", "YGGYO.IS", "YGYO.IS", "YKBNK.IS", "YKSLN.IS", "YONGA.IS", "YUNSA.IS", "YYAPI.IS", 
"ZEDUR.IS", "ZOREN.IS", "A1CAP.IS",
"ADGYO.IS",
"AGROT.IS",
"AGYO.IS",
"ATAKP.IS",
"AVPGY.IS",
"BIGTK.IS",
"BULGS.IS",
"CGCAM.IS",
"DGGYO.IS",
"DGNMO.IS",
"DNISI.IS",
"DOFRB.IS",
"DSTKF.IS",
"DUNYH.IS",
"DURKN.IS",
"ECOGR.IS",
"EDIP.IS",
"EFOR.IS",
"EGEGY.IS",
"EKDMR.IS",
"EKIM.IS",
"EKOS.IS",
"EMPAE.IS",
"ENTRA.IS",
"KOTON.IS",
"LILAK.IS",
"QNBFK.IS",
"SKYLP.IS",
"SVGYO.IS", "AHSGY.IS", "ASCEG.IS", "BEGYO.IS", "BORLS.IS", "GNDG.IS", "HOROZ.IS", "KBORU.IS", "KLSER.IS", "KOCMT.IS", "MEGMT.IS", "ODINE.IS", "RGYAS.IS", "SKYMD.IS", "TNZTP.IS", "YIGIT.IS", "THYAO.IS", "TUPRS.IS", "AKBNK.IS", "KCHOL.IS", "SISE.IS", "ASELS.IS"]

st.sidebar.header("🌍 Küresel Piyasa Ayarları")
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

st.title("👁️ Pro Küresel Yatırım Terminali v100 (SMC, Fibo, XGBoost & Quant)")

# ---------------------------------------------------------
# BURASI SİZİN KODUNUZDA 536. SATIR CİVARINDAN BAŞLIYOR
# ---------------------------------------------------------
with st.spinner('Kurumsal teknik analiz verileri hesaplanıyor...'):
    df = veri_yukle(hisse_kodu, baslangic, bitis)
    info = sirket_bilgisi_getir(hisse_kodu)

# YENİ EKLENECEK HAYAT KURTARICI BLOK:
if df.empty:
    st.error("⚠️ Yahoo Finance'tan veri çekilemedi (API yoğunluğu veya ağ hatası). Lütfen 1-2 dakika bekleyip sayfayı yenileyin veya farklı bir hisse kodu girin.")
    st.stop() # Veri yoksa kodun aşağıya inip hata vermesini engeller!

# "if not df.empty:" SİLİNDİ, ARTIK GİRİNTİYE (BOŞLUĞA) GEREK YOK
# HİZALAMAYI EN SOLA ÇEKİYORUZ:
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
df['Stoch_RSI'] = (df['RSI'] - min_val) / (max_val - min_val)
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
with tabs[0]:
    st.subheader("📈 Kurumsal Quant Grafiği & Likidite Analizi")
    
    c_ayar1, c_ayar2, c_ayar3 = st.columns(3)
    with c_ayar1:
        goster_vpvr = st.checkbox("📊 Hacim Profili (VPVR)", value=True)
        goster_smc = st.checkbox("🏦 FVG & Likidite (SMC)", value=True)
        goster_fibo = st.checkbox("📐 Altın Oran (Fibo)", value=True)
    with c_ayar2:
        goster_grafik_formasyon = st.checkbox("📉 İkili Tepe/Dip", value=True)
        goster_formasyon = st.checkbox("🕯️ Mum Formasyonları", value=False)
    with c_ayar3:
        goster_vwap = st.checkbox("⚖️ VWAP (Maliyet)", value=False)
    
        goster_ai = st.checkbox("🤖 XGBoost AI Tahmini", value=True)
        
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)
    if goster_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)

    # TİLSON ÇİZGİSİNİ GRAFİĞE EKLEME SATIRI:
    fig.add_trace(go.Scatter(x=df.index, y=df['Tilson_T3'], name="Tilson T3", line=dict(color='yellow', width=2)), row=1, col=1)
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

    if goster_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_20'], name="VWAP", line=dict(color='#ff00ff', width=2, dash='dashdot')), row=1, col=1)

    if goster_formasyon:
        df_form = mum_formasyonlarini_bul(df)
        yutan_boga = df_form[df_form['Bullish_Engulfing']]
        fig.add_trace(go.Scatter(x=yutan_boga.index, y=yutan_boga['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='#00ff00', size=12), name='Yutan Boğa'), row=1, col=1)

    # XGBOOST TAHMİNİ ÇİZİMİ (Hizalama Düzeltildi)
    if goster_ai:
        tarihler, tahminler = gelismis_ai_tahmin(df, gelecek_gun=30)
        fig.add_trace(go.Scatter(x=tarihler, y=tahminler, mode='lines', name="XGBoost AI", line=dict(color='cyan', width=3, dash='dot')), row=1, col=1)

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
    st.plotly_chart(fig, use_container_width=True)


# --- SEKME 1: AKILLI RADAR ---
with tabs[1]:
    st.subheader("🔍 Akıllı Asenkron Radar & Çoklu Gösterge (Quant)")
    
    st.markdown("### 🌊 Hızlı Piyasa Taraması ve Yapay Zeka Önerileri")
    st.write(f"Şu anki tarama listesi: **{', '.join(tarama_listesi)}**")
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        btn_radar = st.button("🚀 Genel Radar Taraması")
    with col_btn2:
        btn_stoch = st.button("📊 Stoch Analizi")
    with col_btn3:
        btn_tilson = st.button("📈 Tilson (T3)")
    with col_btn4:
        btn_nokta_atisi = st.button("🎯 Nokta Atışı (Sniper)", type="primary")
    
    # 1. GENEL RADAR BUTONU İŞLEVİ
    if btn_radar:
        with st.spinner('Tüm liste çift zaman dilimli (4S + Günlük) taranıyor... Lütfen bekleyin.'):
            radar_sonuclari = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                gelecek_sonuclar = {executor.submit(asenkron_analiz_yap, s, baslangic, bitis, "radar"): s for s in tarama_listesi}
                for future in concurrent.futures.as_completed(gelecek_sonuclar):
                    sonuc = future.result()
                    if sonuc:
                        radar_sonuclari.append(sonuc)
            if radar_sonuclari:
                df_radar = pd.DataFrame(radar_sonuclari)
                st.dataframe(df_radar, use_container_width=True, hide_index=True)
                
                # Veritabanı Kilitlenmesini Önlemek İçin Yazma İşlemini Toplu ve Senkron Yapıyoruz
                for _, row in df_radar.iterrows():
                    sembol = row['Varlık']
                    hedef_raw = str(row.get('🎯 AI Hedef', '0')).replace(' TL', '').strip()
    
    # Metni sayıya dönüştürmeyi güvenli bir şekilde dene
                    try:
                        hedef_float = float(hedef_raw)
                        if hedef_float > 0:
                            tahmin_kaydet(sembol, hedef_float)
                    except ValueError:
        # Metin ' - ' veya başka bir string ise hatayı yoksayıp devam eder
                        continue
            
            else:
                st.warning("⚠️ Tarama sonucu bulunamadı veya veri çekilemedi.")
                
    # 2. STOCH ANALİZİ BUTONU İŞLEVİ
    elif btn_stoch:
        with st.spinner('Özel Stoch Analizi paralel taranıyor...'):
            stoch_sonuclari = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                gelecek_sonuclar = {executor.submit(asenkron_analiz_yap, s, baslangic, bitis, "stoch"): s for s in tarama_listesi}
                for future in concurrent.futures.as_completed(gelecek_sonuclar):
                    sonuc = future.result()
                    if sonuc:
                        stoch_sonuclari.append(sonuc)
            
            if stoch_sonuclari:
                st.dataframe(pd.DataFrame(stoch_sonuclari), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Stoch tarama sonucu bulunamadı.")

    # 3. TİLSON ANALİZİ BUTONU İŞLEVİ
    elif btn_tilson:
        with st.spinner('Tilson T3 trend analizi taranıyor...'):
            tilson_sonuclari = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                gelecek_sonuclar = {executor.submit(asenkron_analiz_yap, s, baslangic, bitis, "tilson"): s for s in tarama_listesi}
                for future in concurrent.futures.as_completed(gelecek_sonuclar):
                    sonuc = future.result()
                    if sonuc:
                        tilson_sonuclari.append(sonuc)
            
            if tilson_sonuclari:
                st.dataframe(pd.DataFrame(tilson_sonuclari), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Tilson T3 tarama sonucu bulunamadı.")

    # 4. NOKTA ATIŞI (SNIPER) BUTONU İŞLEVİ
    elif btn_nokta_atisi:
        with st.spinner('Kurumsal dip oluşumları ve likidite avı (Sniper) aranıyor...'):
            radar_sonuclari = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                gelecek_sonuclar = {executor.submit(asenkron_analiz_yap, s, baslangic, bitis, "radar"): s for s in tarama_listesi}
                for future in concurrent.futures.as_completed(gelecek_sonuclar):
                    sonuc = future.result()
                    if sonuc:
                        radar_sonuclari.append(sonuc)
            
            if radar_sonuclari:
                df_radar = pd.DataFrame(radar_sonuclari)
                
                # SÜTUN İSİMLERİ VE FİLTRELER DÜZELTİLDİ:
                df_sniper = df_radar[
                    (df_radar['Günlük T3'] == '🚀 BOĞA') & 
                    (pd.to_numeric(df_radar['📊 Temel Skor'], errors='coerce') >= 30) & 
                    (
                        (df_radar['💥 Hacim Analizi'].str.contains('PATLAMA', na=False)) | 
                        (df_radar['📈 Pozitif Uyuşmazlık'].str.contains('UYUŞMAZLIK', na=False)) | 
                        (df_radar['🪤 Spring (Tuzak)'] == '✅ VAR')
                    )
                ]
                
                if not df_sniper.empty:
                    st.success(f"🎯 Dipten Dönüş Fırsatı! Temeli sağlam ve akıllı para girişi tespit edilen {len(df_sniper)} hisse var.")
                    st.dataframe(df_sniper, use_container_width=True, hide_index=True)
                    st.balloons()
                else:
                    st.warning("📉 Şu anki piyasada belirlenen Sniper şartlarına tam uyan şirket bulunamadı. Genel Radar'ı inceleyebilirsiniz.")
            else:
                st.warning("⚠️ Tarama yapılamadı.")
# --- SEKME 2: CÜZDAN & STOP ---
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