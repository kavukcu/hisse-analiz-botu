"""
patterns.py — Grafik formasyonu / harmonik desen tespiti ve dipten dönüş
(Wyckoff Spring, pozitif uyumsuzluk, süper sinyal) analizleri.
"""
import numpy as np
import pandas as pd


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


def yapay_zeka_icin_formasyon_bul(df):
    """
    Mum formasyonlarını yapay zekanın anlayacağı sayısal değerlere (-1, 0, 1) dönüştürür.
    1: Yükseliş Formasyonu (Boğa)
    -1: Düşüş Formasyonu (Ayı)
    0: Nötr
    """
    df_f = df.copy()
    
    # Mum gövdesi ve gölgeleri hesaplama
    govde = abs(df_f['Close'] - df_f['Open'])
    mum_boyu = df_f['High'] - df_f['Low']
    ust_golge = df_f['High'] - df_f[['Close', 'Open']].max(axis=1)
    alt_golge = df_f[['Close', 'Open']].min(axis=1) - df_f['Low']
    
    # 1. Doji (Kararsızlık)
    df_f['Doji'] = np.where(govde <= (mum_boyu * 0.1), 1, 0)
    
    # 2. Yutan Boğa / Ayı (Engulfing)
    bullish_engulfing = (df_f['Close'].shift(1) < df_f['Open'].shift(1)) & (df_f['Open'] < df_f['Close'].shift(1)) & (df_f['Close'] > df_f['Open'].shift(1))
    bearish_engulfing = (df_f['Close'].shift(1) > df_f['Open'].shift(1)) & (df_f['Open'] > df_f['Close'].shift(1)) & (df_f['Close'] < df_f['Open'].shift(1))
    
    # 3. Çekiç (Hammer) ve Kayan Yıldız (Shooting Star)
    hammer = (alt_golge > (2 * govde)) & (ust_golge < (govde * 0.2)) & (df_f['Close'] > df_f['Close'].rolling(10).mean()) # Dipten sekme
    shooting_star = (ust_golge > (2 * govde)) & (alt_golge < (govde * 0.2)) & (df_f['Close'] < df_f['Close'].rolling(10).mean()) # Tepeden dönüş
    
    # Yapay Zeka İçin Tekil Skor Sütunları
    df_f['P_Engulfing'] = np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))
    df_f['P_Pinbar'] = np.where(hammer, 1, np.where(shooting_star, -1, 0))
    
    # Toplam Formasyon Skoru (AI'ın genel piyasa hissiyatını anlaması için)
    df_f['AI_Formasyon_Skoru'] = df_f['P_Engulfing'] + df_f['P_Pinbar']
    
    return df_f


def makro_formasyonlari_bul(df, window=20):
    """
    OBO, TOBO, İkili Tepe/Dip, Üçgenler ve Bayrak formasyonlarının 
    matematiksel ayak izlerini AI için hesaplar.
    """
    df_f = df.copy()
    
    # 1. Yerel Zirve ve Dipler (Destek ve Dirençler)
    df_f['Rolling_Max'] = df_f['High'].rolling(window=window).max()
    df_f['Rolling_Min'] = df_f['Low'].rolling(window=window).min()
    
    # 2. İkili Tepe ve İkili Dip (Double Top / Bottom)
    # Fiyatın son 20 günün zirvesine/dibine %1 toleransla yaklaşması ve dönüş mumu yakması
    df_f['Ikili_Tepe'] = np.where((df_f['High'] >= df_f['Rolling_Max'] * 0.99) & (df_f['Close'] < df_f['Open']), -1, 0)
    df_f['Ikili_Dip'] = np.where((df_f['Low'] <= df_f['Rolling_Min'] * 1.01) & (df_f['Close'] > df_f['Open']), 1, 0)
    
    # 3. Üçgenler, Kama (Wedge) ve Flama İçin Eğim Hesaplaması
    # Son 10 bar içindeki zirve ve diplerin değişim yönü (Türev/Eğim)
    df_f['High_Slope'] = df_f['High'].diff(3).rolling(10).mean() 
    df_f['Low_Slope'] = df_f['Low'].diff(3).rolling(10).mean()   
    
    # Simetrik Üçgen / Flama (Symmetrical / Pennant): Zirveler düşüyor, dipler yükseliyor (Sıkışma)
    df_f['Simetrik_Ucgen'] = np.where((df_f['High_Slope'] < 0) & (df_f['Low_Slope'] > 0), 1, 0)
    
    # Yükselen Üçgen (Ascending Triangle): Zirveler yatay (direnç kırılamıyor), dipler yükseliyor
    df_f['Yukselen_Ucgen'] = np.where((abs(df_f['High_Slope']) < (df_f['Close'] * 0.002)) & (df_f['Low_Slope'] > 0), 1, 0)
    
    # Alçalan Üçgen (Descending Triangle): Dipler yatay, zirveler düşüyor
    df_f['Alcalan_Ucgen'] = np.where((df_f['High_Slope'] < 0) & (abs(df_f['Low_Slope']) < (df_f['Close'] * 0.002)), -1, 0)
    
    # 4. Bayrak (Flag) Formasyonu
    # Sert bir fiyat hareketi (Bayrak Direği) sonrası dar aralıkta zıt yönlü dinlenme
    df_f['Sert_Yukselis'] = df_f['Close'].pct_change(5) > 0.06 # 5 günde %6 üstü artış
    df_f['Dar_Bant_Konsolidasyon'] = (df_f['High'].rolling(4).max() - df_f['Low'].rolling(4).min()) < (df_f['Close'] * 0.02)
    df_f['Bayrak_Formasyonu'] = np.where(df_f['Sert_Yukselis'].shift(4) & df_f['Dar_Bant_Konsolidasyon'], 1, 0)
    
    # 5. OBO, TOBO ve Çanak İçin Derinlik / Kavis Sensörleri
    # AI bu Z-Score ve mesafe verilerini alıp kendi içindeki ağaçlarda OBO/TOBO'yu tanıyacaktır.
    df_f['Tepe_Uzakligi_Z'] = (df_f['Rolling_Max'] - df_f['Close']) / df_f['Close'].rolling(window).std()
    df_f['Dip_Uzakligi_Z'] = (df_f['Close'] - df_f['Rolling_Min']) / df_f['Close'].rolling(window).std()
    
    # Toplam Makro Formasyon Gücü
    df_f['Makro_Guc_Skoru'] = df_f['Ikili_Dip'] + df_f['Yukselen_Ucgen'] + df_f['Simetrik_Ucgen'] + df_f['Bayrak_Formasyonu'] - abs(df_f['Ikili_Tepe']) - abs(df_f['Alcalan_Ucgen'])
    
    # Temizlik (AI'a NaN veri gitmemesi için)
    df_f.fillna(0, inplace=True)
    
    return df_f


def trend_ve_harmonik_bul(df):
    """
    Golden Cross, Death Cross (Hareketli Ortalama Kesişimleri) 
    ve ABCD Harmonik formasyonunu AI için hesaplar.
    """
    df_f = df.copy()
    
    # --- 1. GOLDEN CROSS & DEATH CROSS ---
    # 50 ve 200 periyotluk Basit Hareketli Ortalamalar (SMA)
    df_f['SMA_50'] = df_f['Close'].rolling(window=50).mean()
    df_f['SMA_200'] = df_f['Close'].rolling(window=200).mean()
    
    # Altın Kesişim (Golden Cross): 50 SMA, 200 SMA'yı YUKARI keserse (Boğa Piyasası Başlangıcı)
    golden_cross = (df_f['SMA_50'] > df_f['SMA_200']) & (df_f['SMA_50'].shift(1) <= df_f['SMA_200'].shift(1))
    
    # Ölüm Kesişimi (Death Cross): 50 SMA, 200 SMA'yı AŞAĞI keserse (Ayı Piyasası Başlangıcı)
    death_cross = (df_f['SMA_50'] < df_f['SMA_200']) & (df_f['SMA_50'].shift(1) >= df_f['SMA_200'].shift(1))
    
    # Yapay Zeka Sinyali (Kesişim anında 1 veya -1, trend devam ederken gücünü göstermesi için mesafe)
    df_f['Cross_Sinyali'] = np.where(golden_cross, 1, np.where(death_cross, -1, 0))
    df_f['SMA_50_200_Farki'] = (df_f['SMA_50'] - df_f['SMA_200']) / df_f['SMA_200'] # Trendin gücü
    
    
    # --- 2. ABCD FORMASYONU (Basitleştirilmiş Harmonik Yaklaşım) ---
    # Fiyatın 3 ana dalgası: AB (Dalga 1), BC (Dalga 2), CD (Dalga 3)
    # AI'ın anlayabilmesi için yaklaşık 5'er günlük swing (salınım) periyotları kullanıyoruz.
    swing_ab = df_f['Close'].shift(10) - df_f['Close'].shift(15) 
    swing_bc = df_f['Close'].shift(5) - df_f['Close'].shift(10)  
    swing_cd = df_f['Close'] - df_f['Close'].shift(5)            
    
    ab_boyu = abs(swing_ab)
    cd_boyu = abs(swing_cd)
    
    # Yükseliş Beklenen (Bullish) ABCD: AB düşer, BC yükselir (tepki), CD tekrar düşer.
    # Kural: AB ve CD dalgalarının boyları birbirine yakın olmalıdır (AB = CD %70 ile %130 tolerans)
    bullish_abcd = (swing_ab < 0) & (swing_bc > 0) & (swing_cd < 0) & (cd_boyu > ab_boyu * 0.7) & (cd_boyu < ab_boyu * 1.3)
    
    # Düşüş Beklenen (Bearish) ABCD: AB yükselir, BC düşer (düzeltme), CD tekrar yükselir.
    bearish_abcd = (swing_ab > 0) & (swing_bc < 0) & (swing_cd > 0) & (cd_boyu > ab_boyu * 0.7) & (cd_boyu < ab_boyu * 1.3)
    
    df_f['ABCD_Formasyonu'] = np.where(bullish_abcd, 1, np.where(bearish_abcd, -1, 0))
    
    # Eksik verileri 0 ile doldur
    df_f.fillna(0, inplace=True)
    
    return df_f


def formasyon_tespit_et_ve_hedefle(df):
    """
    Son güncel verileri analiz ederek tespit edilen formasyonu ve
    teknik hedef yüzde (%) değişimini döndürür.
    """
    if df is None or len(df) < 20:
        return "Yok", "% 0.00"

    df_f = df.copy()
    df_f = yapay_zeka_icin_formasyon_bul(df_f)
    df_f = makro_formasyonlari_bul(df_f, window=20)
    df_f = trend_ve_harmonik_bul(df_f)
    
    son = df_f.iloc[-1]
    fiyat = son['Close'] if son['Close'] > 0 else 1.0
    
    formasyon_adi = "Yok"
    hedef_yuzde = 0.0

    # 1. Makro / Grafik Formasyonları
    if son.get('Ikili_Dip', 0) == 1:
        formasyon_adi = "📐 İkili Dip"
        derinlik = (son['Rolling_Max'] - son['Rolling_Min']) / son['Rolling_Min'] * 100
        hedef_yuzde = round(derinlik, 2)
    elif son.get('Ikili_Tepe', 0) == -1:
        formasyon_adi = "📐 İkili Tepe"
        derinlik = (son['Rolling_Max'] - son['Rolling_Min']) / son['Rolling_Max'] * 100
        hedef_yuzde = round(-derinlik, 2)
    elif son.get('Bayrak_Formasyonu', 0) == 1:
        formasyon_adi = "🚩 Bayrak Formasyonu"
        direk_boyu = df_f['Close'].pct_change(5).iloc[-4] * 100 if len(df_f) > 5 else 7.5
        hedef_yuzde = round(abs(direk_boyu), 2)
    elif son.get('Yukselen_Ucgen', 0) == 1:
        formasyon_adi = "🔺 Yükselen Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(yukseklik, 2)
    elif son.get('Alcalan_Ucgen', 0) == -1:
        formasyon_adi = "🔻 Alçalan Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(-yukseklik, 2)
    elif son.get('Simetrik_Ucgen', 0) == 1:
        formasyon_adi = "📐 Simetrik Üçgen"
        yukseklik = (son['Rolling_Max'] - son['Rolling_Min']) / fiyat * 100
        hedef_yuzde = round(yukseklik / 2, 2)

    # 2. Harmonik ve Trend Kesişimleri
    elif son.get('ABCD_Formasyonu', 0) == 1:
        formasyon_adi = "⚡ Bullish ABCD"
        hedef_yuzde = 6.5
    elif son.get('ABCD_Formasyonu', 0) == -1:
        formasyon_adi = "⚡ Bearish ABCD"
        hedef_yuzde = -6.5
    elif son.get('Cross_Sinyali', 0) == 1:
        formasyon_adi = "🌟 Golden Cross"
        hedef_yuzde = 12.0
    elif son.get('Cross_Sinyali', 0) == -1:
        formasyon_adi = "💀 Death Cross"
        hedef_yuzde = -12.0

    # 3. Mikro Mum Formasyonları
    elif son.get('P_Engulfing', 0) == 1:
        formasyon_adi = "🕯️ Yutan Boğa"
        hedef_yuzde = 4.0
    elif son.get('P_Engulfing', 0) == -1:
        formasyon_adi = "🕯️ Yutan Ayı"
        hedef_yuzde = -4.0
    elif son.get('P_Pinbar', 0) == 1:
        formasyon_adi = "🔨 Çekiç (Pinbar)"
        hedef_yuzde = 3.5
    elif son.get('P_Pinbar', 0) == -1:
        formasyon_adi = "🏹 Kayan Yıldız"
        hedef_yuzde = -3.5
    elif son.get('Doji', 0) == 1:
        formasyon_adi = "⚖️ Doji (Kararsızlık)"
        hedef_yuzde = 0.0

    hedef_str = f"% {hedef_yuzde:+.2f}" if hedef_yuzde != 0 else "% 0.00"
    return formasyon_adi, hedef_str


def dipten_donus_analizi(df):
    """
    Fiyatın dipten sekme ihtimalini kurumsal tekniklerle hesaplar:
    1. Hacim Patlaması (20 Günlük Ortalamanın 2 Katı)
    2. Wyckoff Spring (Bollinger Alt Bandı Ayı Tuzağı)
    3. Gelişmiş Pozitif Uyumsuzluk (Yerel Dipler üzerinden RSI, MACD, Stokastik)
    """
    # Eksik veya yetersiz veri güvenliği
    if df is None or len(df) < 20:
        df_copy = df.copy() if df is not None else pd.DataFrame()
        df_copy['Hacim_Patlamasi'] = False
        df_copy['Wyckoff_Spring'] = False
        df_copy['RSI_Uyumsuzluk'] = False
        df_copy['MACD_Uyumsuzluk'] = False
        df_copy['Stokastik_Uyumsuzluk'] = False
        df_copy['Pozitif_Uyusmazlik'] = False
        df_copy['Super_Sinyal'] = False
        return df_copy

    df_dip = df.copy()

    # ---------------------------------------------------------
    # 1. HACİM PATLAMASI
    # ---------------------------------------------------------
    df_dip['Vol_SMA_20'] = df_dip['Volume'].rolling(20).mean()
    df_dip['Hacim_Patlamasi'] = df_dip['Volume'] > (df_dip['Vol_SMA_20'] * 2)

    # ---------------------------------------------------------
    # 2. WYCKOFF SPRING (AYI TUZAĞI)
    # ---------------------------------------------------------
    df_dip['SMA_20_Dip'] = df_dip['Close'].rolling(20).mean()
    df_dip['STD_20_Dip'] = df_dip['Close'].rolling(20).std()
    df_dip['Lower_Band'] = df_dip['SMA_20_Dip'] - (df_dip['STD_20_Dip'] * 2)

    df_dip['Wyckoff_Spring'] = (df_dip['Low'] < df_dip['Lower_Band']) & \
                               (df_dip['Close'] > df_dip['Lower_Band']) & \
                               (df_dip['Close'] > df_dip['Open'])

    # ---------------------------------------------------------
    # 3. İNDİKATÖR HESAPLAMALARI (Eksikse Otomatik Eklenir)
    # ---------------------------------------------------------
    # A. RSI (14)
    if 'RSI' not in df_dip.columns:
        delta = df_dip['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, 1e-9))
        df_dip['RSI'] = 100 - (100 / (1 + rs))

    # B. MACD Histogram (12, 26, 9)
    if 'MACD_Hist' not in df_dip.columns:
        ema12 = df_dip['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_dip['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df_dip['MACD_Hist'] = macd - macd_signal

    # C. Stokastik %K (14)
    if 'Stoch_K' not in df_dip.columns:
        low_min = df_dip['Low'].rolling(window=14).min()
        high_max = df_dip['High'].rolling(window=14).max()
        df_dip['Stoch_K'] = 100 * ((df_dip['Close'] - low_min) / (high_max - low_min + 1e-9))

    # Başlangıç değerleri (Varsayılan: False)
    df_dip['RSI_Uyumsuzluk'] = False
    df_dip['MACD_Uyumsuzluk'] = False
    df_dip['Stokastik_Uyumsuzluk'] = False
    df_dip['Pozitif_Uyusmazlik'] = False
    df_dip['Super_Sinyal'] = False

    # ---------------------------------------------------------
    # 4. GELİŞMİŞ UYUMSUZLUK (DIVERGENCE) ANALİZİ
    # ---------------------------------------------------------
    # Fiyatın gerçek yerel dip noktalarını (Local Minima) tespit et
    dip_mask = (df_dip['Low'].shift(1) > df_dip['Low']) & (df_dip['Low'].shift(-1) > df_dip['Low'])
    dipler = df_dip[dip_mask]['Low'].tail(2)

    # Grafikte en az 2 adet yerel dip oluşmuşsa analiz yap
    if len(dipler) == 2:
        eski_idx, yeni_idx = dipler.index[0], dipler.index[1]
        
        eski_fiyat = df_dip['Low'].loc[eski_idx]
        yeni_fiyat = df_dip['Low'].loc[yeni_idx]

        # ŞART: Fiyat yeni bir daha düşük dip yapmış olmalı (Lower Low)
        if yeni_fiyat < eski_fiyat:
            rsi_uyum = df_dip['RSI'].loc[yeni_idx] > df_dip['RSI'].loc[eski_idx]
            macd_uyum = df_dip['MACD_Hist'].loc[yeni_idx] > df_dip['MACD_Hist'].loc[eski_idx]
            stoch_uyum = df_dip['Stoch_K'].loc[yeni_idx] > df_dip['Stoch_K'].loc[eski_idx]

            super_sinyal = rsi_uyum and macd_uyum and stoch_uyum
            herhangi_uyum = rsi_uyum or macd_uyum or stoch_uyum

            # Oluşan sinyalleri son dip noktası ve sonrasındaki mumlara yansıt
            df_dip.loc[yeni_idx:, 'RSI_Uyumsuzluk'] = rsi_uyum
            df_dip.loc[yeni_idx:, 'MACD_Uyumsuzluk'] = macd_uyum
            df_dip.loc[yeni_idx:, 'Stokastik_Uyumsuzluk'] = stoch_uyum
            df_dip.loc[yeni_idx:, 'Pozitif_Uyusmazlik'] = herhangi_uyum
            df_dip.loc[yeni_idx:, 'Super_Sinyal'] = super_sinyal
            destek = df_dip['Low'].rolling(window=20).min()
            destege_yakinlik = (df_dip['Close'] - destek) / df_dip['Close']
    
            df_dip['Kesin_Dip_Donusu'] = (
                (df_dip['RSI'] < 40) & 
                (df_dip['Hacim_Patlamasi'] | df_dip['Wyckoff_Spring']) & 
                (df_dip['Super_Sinyal'] | df_dip['Pozitif_Uyusmazlik']) &
                (destege_yakinlik < 0.03) # Desteğe maksimum %3 uzaklıkta
    )
    return df_dip
