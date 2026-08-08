"""
data_fetching.py — Tüm harici veri kaynaklarına (Yahoo Finance, TradingView,
İş Yatırım) erişim katmanı. Ağ isteği atan HER fonksiyon burada toplanır.
"""
import time as tm
import logging
import asyncio

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
import isyatirimhisse
import aiohttp

# --- Ortak HTTP oturumu (yfinance.Ticker çağrılarında User-Agent taklidi için) ---
oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})


@st.cache_resource(show_spinner=False)
def get_tv_datafeed():
    """TradingView bağlantısını bir kez kurar ve hafızada (cache) tutar."""
    try:
        # Eğer premium hesabın varsa username ve password parametrelerini girebilirsin.
        # Yoksa anonim olarak bağlanır.
        tv = TvDatafeed() 
        return tv
    except Exception as e:
        logging.error(f"TradingView Bağlantı Hatası: {e}")
        return None


def sembol_formatla(hisse_kodu):
    # Eğer gelen kodda '.IS' veya 'BIST:' varsa temizleyip ana sembolü (örneğin THYAO) bulalım
    ana_sembol = hisse_kodu.replace('.IS', '').replace('BIST:', '').strip().upper()
    
    # 3 farklı platform için doğru formatları döndürüyoruz
    formatlar = {
        'yfinance': f"{ana_sembol}.IS",
        'isyatirim': ana_sembol,
        'tradingview': f"BIST:{ana_sembol}", # tvDatafeed kütüphanesi için sadece ana_sembol yeterli olabilir
        'saf_sembol': ana_sembol
    }
    
    return formatlar


@st.cache_data(ttl=300, show_spinner=False)
def veri_yukle(ticker, start, end, interval="1d", kaynak="Yahoo Finance (yfinance)"):
    # 1. Sembolü temizle
    ticker = ticker.replace("$", "").strip() 
    
    # 2. Hangi piyasa olduğunu tespit et (BIST mi, Kripto mu, ABD mi?)
    is_bist = ".IS" in ticker
    is_crypto = "-" in ticker
    
    # 3. Öncelikli kaynak listesi oluştur
    tum_kaynaklar = ["Yahoo Finance (yfinance)", "TradingView (tvdatafeed)"]
    if is_bist:
        tum_kaynaklar.append("İş Yatırım (Sadece BIST)")
    
    # Seçilen kaynağı en başa al
    if kaynak in tum_kaynaklar:
        tum_kaynaklar.remove(kaynak)
        tum_kaynaklar.insert(0, kaynak)

    # 4. Kaynakları sırayla dene (Biri çökerse diğeri devreye girer)
    for aktif_kaynak in tum_kaynaklar:
        
        # --- YAHOO FINANCE DENEMESİ ---
        if aktif_kaynak == "Yahoo Finance (yfinance)":
            for _ in range(2):
                try:
                    # 👇 EKLENEN KISIM BAŞLANGICI 👇
                    if end is not None:
                        yf_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    else:
                        yf_end = None
                    # 👆 EKLENEN KISIM BİTİŞİ 👆

                    # Sadece yfinance formatına uygun orijinal ticker string'ini veriyoruz
                    df = yf.download(
                        ticker, 
                        start=start, 
                        end=yf_end,  # <--- BURASI DEĞİŞTİ: end=end yerine end=yf_end oldu
                        interval=interval, 
                        progress=False, 
                        auto_adjust=True, 
                        threads=True
                    )
                    
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(1)
                        gerekli = ["Open", "High", "Low", "Close", "Volume"]
                        if all(c in df.columns for c in gerekli):
                            df = df.dropna(subset=['Close'])
                            df.index = df.index.tz_localize(None)
                            df.index = pd.to_datetime(df.index).normalize()
                            return df
                except Exception as e:
                    logging.debug(f"Yahoo deneme hatası ({ticker}): {e}")
                tm.sleep(0.5)

        # --- TRADINGVIEW DENEMESİ ---
        elif aktif_kaynak == "TradingView (tvdatafeed)":
            try:
                tv = get_tv_datafeed()
                if tv:
                    # TradingView için sembolü formatla
                    if is_bist:
                        exchange = 'BIST'
                        tv_symbol = ticker.replace(".IS", "")
                    elif is_crypto:
                        exchange = 'CRYPTO'
                        tv_symbol = ticker.replace("-", "")
                    else:
                        exchange = 'NASDAQ'
                        tv_symbol = ticker
                        
                    df = tv.get_hist(symbol=tv_symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)
                    if df is not None and not df.empty:
                        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                        df.index = df.index.tz_localize(None)
                        if end:
                            df = df[df.index.date <= pd.to_datetime(end).date()]
                        if start:
                            df = df[df.index.date >= pd.to_datetime(start).date()]
                        if not df.empty:
                            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"TradingView deneme hatası ({ticker}): {e}")

        # --- İŞ YATIRIM DENEMESİ (Sadece BIST) ---
        elif aktif_kaynak == "İş Yatırım (Sadece BIST)" and is_bist:
            try:
                sembol = ticker.replace(".IS", "")
                start_str = pd.to_datetime(start).strftime('%d-%m-%Y') if start else None
                end_str = pd.to_datetime(end).strftime('%d-%m-%Y') if end else pd.Timestamp.today().strftime('%d-%m-%Y')
                
                df = isyatirimhisse.fetch_data(symbol=sembol, start_date=start_str, end_date=end_str)
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        'TARIH': 'Date', 'ACILIS_FIYATI': 'Open', 
                        'EN_YUKSEK_FIYAT': 'High', 'EN_DUSUK_FIYAT': 'Low', 
                        'KAPANIS_FIYATI': 'Close', 'ISLEM_ADEDI': 'Volume'
                    })
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    if not df.empty:
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"İş Yatırım deneme hatası ({ticker}): {e}")

    # Hiçbir kaynaktan veri gelmezse boş DataFrame döndür
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def veri_4saatlik_getir(ticker, start, end, kaynak="Yahoo Finance (yfinance)"):
    import yfinance as yf
    import pandas as pd
    import time
    from datetime import datetime, timedelta
    import logging
    from tvDatafeed import TvDatafeed, Interval
    

    kaynaklar = ["TradingView (tvdatafeed)", "Yahoo Finance (yfinance)"]
    if kaynak in kaynaklar:
        kaynaklar.remove(kaynak)
        kaynaklar.insert(0, kaynak)

    for aktif_kaynak in kaynaklar:
        # 1. TRADINGVIEW 4 SAATLİK DENEMESİ
        # 1. TRADINGVIEW 4 SAATLİK DENEMESİ
        if aktif_kaynak == "TradingView (tvdatafeed)":
            try:
                tv = get_tv_datafeed()  # <--- SADECE BU SATIR DEĞİŞTİ
                if ".IS" in ticker:
                    exchange = 'BIST'
                    tv_symbol = ticker.replace(".IS", "")
                elif "-" in ticker:
                    exchange = 'CRYPTO'
                    tv_symbol = ticker.replace("-", "")
                else:
                    exchange = 'NASDAQ'
                    tv_symbol = ticker
                    
                df_4h = tv.get_hist(symbol=tv_symbol, exchange=exchange, interval=Interval.in_4_hour, n_bars=1000)
                if df_4h is not None and not df_4h.empty:
                    df_4h = df_4h.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df_4h.index = df_4h.index.tz_localize(None)
                    if end:
                        df_4h = df_4h[df_4h.index.date <= pd.to_datetime(end).date()]
                    if not df_4h.empty:
                        return df_4h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception as e:
                logging.debug(f"TV 4H deneme hatası ({ticker}): {e}")

        # 2. YAHOO FINANCE 1 SAATLİK -> 4 SAATLİK RESAMPLE DENEMESİ
        elif aktif_kaynak == "Yahoo Finance (yfinance)":
            try:
                start_dt = pd.to_datetime(start)
                if (datetime.now() - start_dt).days > 729:
                    start_dt = datetime.now() - timedelta(days=729)
                    s_str = start_dt.strftime('%Y-%m-%d')
                else:
                    s_str = start

                for _ in range(2):
                    df_1h = yf.download(ticker, start=s_str, interval="1h", progress=False)
                    if not df_1h.empty:
                        if isinstance(df_1h.columns, pd.MultiIndex):
                            df_1h.columns = df_1h.columns.droplevel(1)
                        df_1h.index = df_1h.index.tz_localize(None)
                        if end:
                            df_1h = df_1h.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                        
                        df_4h = df_1h.resample('4h').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                        
                        if not df_4h.empty:
                            return df_4h
                    time.sleep(0.5)
            except Exception as e:
                logging.debug(f"Yahoo 4H resample deneme hatası ({ticker}): {e}")

    return pd.DataFrame()


def borsa_endeks_verisini_ekle(t_df):
    try:
        # BIST100 verisini çek (Son 1 yıllık günlük veri yeterli olacaktır)
        xu100 = yf.download("XU100.IS", period="1y", interval="1d", progress=False)
        
        # BIST100'ün günlük getirisini hesapla
        xu100['XU100_Return'] = xu100['Close'].pct_change()
        xu100['XU100_Trend'] = np.where(xu100['Close'] > xu100['Close'].rolling(20).mean(), 1, -1)
        
        # Sadece ihtiyacımız olan sütunları al ve tarih indeksini hisse tablosuyla eşleştir
        xu100 = xu100[['XU100_Return', 'XU100_Trend']]
        t_df = t_df.merge(xu100, left_index=True, right_index=True, how='left')
        
        # Boşlukları doldur (Eğer endeks kapalıysa, bir önceki günün verisini kullan)
        t_df[['XU100_Return', 'XU100_Trend']] = t_df[['XU100_Return', 'XU100_Trend']].ffill().fillna(0)
        
        return t_df
    except Exception as e:
        print(f"Endeks verisi çekilemedi: {e}")
        t_df['XU100_Return'] = 0
        t_df['XU100_Trend'] = 0
        return t_df


@st.cache_data(show_spinner=False)
def sirket_bilgisi_getir(ticker):
    try: 
        return yf.Ticker(ticker, session=oturum).info
    except: 
        return {}


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


@st.cache_data(ttl=60, show_spinner=False)
def toplu_guncel_fiyat_getir(sembol_tuple, kaynak="Yahoo Finance (yfinance)"):
    """
    Tarama listesindeki TÜM hisselerin anlık (1 dakikalık) son fiyatını TEK bir
    toplu istekte çeker. Önceden her hisse için ayrı ayrı yf.download çağrısı
    yapılıyordu (700 hisse = 700 ayrı, önbelleksiz istek) — bu hem Yahoo
    Finance'i rate-limit'e sokuyor hem de istek başarısız olduğunda sessizce
    dünün kapanışına düşülmesine (bayat fiyat) yol açıyordu. Sonuç 60 saniye
    önbelleklenir, böylece aynı tarama içindeki tekrar çağrılar ek istek atmaz.
    """
    fiyatlar = {}
    if not sembol_tuple:
        return fiyatlar
    try:
        veri = yf.download(
            list(sembol_tuple),
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True,
        )
        if veri is None or veri.empty:
            return fiyatlar

        for sembol in sembol_tuple:
            try:
                if isinstance(veri.columns, pd.MultiIndex):
                    if sembol not in veri.columns.get_level_values(0):
                        continue
                    kapanislar = veri[sembol]['Close'].dropna()
                else:
                    # Tek sembol istenmişse MultiIndex oluşmayabilir
                    kapanislar = veri['Close'].dropna()
                if not kapanislar.empty:
                    fiyatlar[sembol] = float(kapanislar.iloc[-1])
            except Exception:
                continue
    except Exception as e:
        logging.debug(f"Toplu anlık fiyat çekme hatası: {e}")
    return fiyatlar


@st.cache_data(ttl=300, show_spinner=False)
def toplu_gecmis_veri_getir(sembol_tuple, start, end, kaynak="Yahoo Finance (yfinance)"):
    """
    Tarama listesindeki TÜM hisselerin günlük geçmiş verisini TEK toplu
    istekte çeker (536 ayrı istek yerine yfinance'in kendi iç gruplamasıyla
    birkaç istek). Sadece Yahoo Finance toplu indirmeyi destekler; başka bir
    kaynak seçiliyse boş sözlük döner ve paralel_tara/asenkron_analiz_yap
    otomatik olarak eski tek-tek veri_yukle() yöntemine (TV/İş Yatırım dahil
    çoklu-kaynak deneme mantığı) düşer.
    """
    sonuc = {}
    if not sembol_tuple or kaynak != "Yahoo Finance (yfinance)":
        return sonuc
    try:
        if end is not None:
            yf_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            yf_end = None

        veri = yf.download(
            list(sembol_tuple),
            start=start,
            end=yf_end,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True,
        )
        if veri is None or veri.empty:
            return sonuc

        gerekli = ["Open", "High", "Low", "Close", "Volume"]
        for sembol in sembol_tuple:
            try:
                if isinstance(veri.columns, pd.MultiIndex):
                    if sembol not in veri.columns.get_level_values(0):
                        continue
                    df = veri[sembol].copy()
                else:
                    # Tek sembol istenmişse MultiIndex oluşmayabilir
                    df = veri.copy()

                if not all(c in df.columns for c in gerekli):
                    continue
                df = df.dropna(subset=['Close'])
                if df.empty or len(df) < 20:
                    continue
                df.index = df.index.tz_localize(None)
                df.index = pd.to_datetime(df.index).normalize()
                sonuc[sembol] = df
            except Exception:
                continue
    except Exception as e:
        logging.debug(f"Toplu geçmiş veri çekme hatası: {e}")
    return sonuc


async def tek_hisse_getir(session, sem, hisse_kodu):
    """
    Tek bir hissenin verisini asenkron olarak çeker.
    Yahoo Finance (veya kullandığın API) için örnek bir endpoint.
    """
    # Aynı anda API'ye yüklenmemek için kilit mekanizması
    async with sem:
        pass # Buraya hisse verisi çekme (request) işlemleriniz gelecek...
    async with sem:
        # BIST hisseleri için Yahoo Finance formatı genelde 'ISCTR.IS' şeklindedir
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{hisse_kodu}.IS?interval=1d&range=1y"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Gelen JSON'ı okuyup basit bir DataFrame'e çevirme işlemi
                    timestamps = data['chart']['result'][0]['timestamp']
                    quotes = data['chart']['result'][0]['indicators']['quote'][0]
                    
                    df = pd.DataFrame({
                        'Date': pd.to_datetime(timestamps, unit='s'),
                        'Close': quotes['close'],
                        'Volume': quotes['volume']
                    })
                    df.set_index('Date', inplace=True)
                    return hisse_kodu, df
                else:
                    return hisse_kodu, None
        except Exception as e:
            return hisse_kodu, None


async def tum_piyasayi_tara_async(hisse_listesi):
    """
    Tüm hisse listesini asenkron motorla saniyeler içinde tarar.
    """
    # Aynı anda maksimum 50 istek atacak şekilde sınırlandırıyoruz (API ban yememek için)
    sem = asyncio.Semaphore(10) 
    
    async with aiohttp.ClientSession() as session:
        # Tüm görevleri (task) hazırlıyoruz
        gorevler = [tek_hisse_getir(session, sem, hisse) for hisse in hisse_listesi]
        
        # Görevleri çalıştır ve sonuçları bekle
        sonuclar = await asyncio.gather(*gorevler)
        
        # Başarılı çekilen verileri bir sözlükte topla
        basarili_veriler = {hisse: df for hisse, df in sonuclar if df is not None}
        return basarili_veriler
