# ==========================================
# KÜTÜPHANELER (En üste taşındı ve hızlandırıldı)
# ==========================================
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import requests
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})
import concurrent.futures
import logging
import os
import pytz
import threading
import json
import time
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Yapay Zeka Kütüphaneleri
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import sqlite3
import joblib
import optuna
from sklearn.metrics import mean_squared_error
from tvDatafeed import TvDatafeed, Interval
import isyatirimhisse
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import asyncio
import aiohttp
import pandas as pd

# --- TENSORFLOW / KERAS (OPSİYONEL) ---
# Bu kütüphane sadece LSTM tahmin özelliği için kullanılıyor. Streamlit Cloud'da
# kurulamazsa veya sürüm uyuşmazlığı olursa TÜM uygulamanın çökmesini engellemek için
# import hatası burada yakalanıp bayrağa çevriliyor; ilgili fonksiyon bu bayrağı kontrol eder.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    import tensorflow.keras.backend as K
    TENSORFLOW_AVAILABLE = True
except Exception as _tf_err:
    TENSORFLOW_AVAILABLE = False
    logging.warning(f"TensorFlow/Keras yüklenemedi, LSTM özelliği devre dışı: {_tf_err}")
# --- TRADINGVIEW BAĞLANTISINI HAFIZADA TUTAN BLOK ---
st.set_page_config(layout="wide", page_title="God Mode Terminal v100")
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
# ==========================================
# SAYFA AYARLARI VE OTURUM
# ==========================================

oturum = requests.Session()
oturum.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
})
# ==========================================
# VERİTABANI VE HAFIZA YÖNETİMİ
# ==========================================
def tum_bist_hisselerini_getir():
    """BIST hisselerini sabit listeden hızlıca getirir. API engellerine takılmaz."""
    return ["A1CAP.IS", "ACSEL.IS", "ADEL.IS", "ADESE.IS", "ADGYO.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS",
  "AGHOL.IS", "AGROT.IS", "AGYO.IS", "AHGAZ.IS", "AHSGY.IS", "AKBNK.IS", "AKCNS.IS", "AKENR.IS",
  "AKFGY.IS", "AKFYE.IS", "AKGRT.IS", "AKMGY.IS", "AKSA.IS", "AKSEN.IS", "AKSUE.IS", "AKYHO.IS",
  "ALARK.IS", "ALBRK.IS", "ALCAR.IS", "ALCTL.IS", "ALFAS.IS", "ALGYO.IS", "ALKA.IS", "ALKIM.IS",
  "ALTNY.IS", "ALVES.IS", "ANELE.IS", "ANGEN.IS", "ANHYT.IS", "ANSGR.IS", "ARASE.IS", "ARCLK.IS",
  "ARDYZ.IS", "ARENA.IS", "ARSAN.IS", "ARTMS.IS", "ARZUM.IS", "ASELS.IS", "ASGYO.IS", "ASTOR.IS",
  "ASUZU.IS", "ATAGY.IS", "ATAKP.IS", "ATATP.IS", "ATEKS.IS", "ATLAS.IS", "AVGYO.IS", "AVHOL.IS",
  "AVOD.IS", "AVPGY.IS", "AVTUR.IS", "AYCES.IS", "AYDEM.IS", "AYEN.IS", "AYGAZ.IS", "AZTEK.IS",
  "BAGFS.IS", "BAKAB.IS", "BALAT.IS", "BANVT.IS", "BARMA.IS", "BASCM.IS", "BASGZ.IS", "BAYRK.IS",
  "BEGYO.IS", "BEYAZ.IS", "BFREN.IS", "BIENY.IS", "BIGCH.IS", "BIGTK.IS", "BIMAS.IS", "BINHO.IS",
  "BIOEN.IS", "BIZIM.IS", "BJKAS.IS", "BLCYT.IS", "BMSCH.IS", "BMSTL.IS", "BNTAS.IS", "BOBET.IS",
  "BORLS.IS", "BORSK.IS", "BOSSA.IS", "BRISA.IS", "BRKO.IS", "BRKSN.IS", "BRKVY.IS", "BRLSM.IS",
  "BRMEN.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "BUCIM.IS", "BULGS.IS", "BURCE.IS",
  "BURVA.IS", "BVSAN.IS", "BYDNR.IS", "CANTE.IS", "CASA.IS", "CATES.IS", "CCOLA.IS", "CELHA.IS",
  "CEMAS.IS", "CEMTS.IS", "CEOEM.IS", "CGCAM.IS", "CIMSA.IS", "CLEBI.IS", "CMBTN.IS", "CMENT.IS",
  "CONSE.IS", "COSMO.IS", "CRDFA.IS", "CRFSA.IS", "CUSAN.IS", "CVKMD.IS", "CWENE.IS", "DAGI.IS",
  "DAPGM.IS", "DARDL.IS", "DERHL.IS", "DERIM.IS", "DESA.IS", "DESPC.IS", "DEVA.IS", "DGGYO.IS",
  "DGNMO.IS", "DIRIT.IS", "DITAS.IS", "DMRGD.IS", "DNISI.IS", "DOAS.IS", "DOCO.IS", "DOFER.IS",
  "DOFRB.IS", "DOGUB.IS", "DOHOL.IS", "DOKTA.IS", "DSTKF.IS", "DUNYH.IS", "DURDO.IS", "DURKN.IS",
  "DYOBY.IS", "DZGYO.IS", "EBEBK.IS", "ECILC.IS", "ECOGR.IS", "ECZYT.IS", "EDATA.IS", "EDIP.IS",
  "EFOR.IS", "EGEEN.IS", "EGEGY.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EKDMR.IS", "EKGYO.IS",
  "EKIM.IS", "EKIZ.IS", "EKOS.IS", "EKSUN.IS", "ELITE.IS", "EMKEL.IS", "EMPAE.IS", "ENERY.IS",
  "ENJSA.IS", "ENKAI.IS", "ENSRI.IS", "ENTRA.IS", "EPLAS.IS", "ERBOS.IS", "EREGL.IS", "ERSU.IS",
  "ESCAR.IS", "ESCOM.IS", "ESEN.IS", "ETILR.IS", "ETYAT.IS", "EUHOL.IS", "EUPWR.IS", "EUREN.IS",
  "EUYO.IS", "EYGYO.IS", "FADE.IS", "FENER.IS", "FLAP.IS", "FMIZP.IS", "FONET.IS", "FORMT.IS",
  "FORTE.IS", "FRIGO.IS", "FROTO.IS", "FZLGY.IS", "GARAN.IS", "GARFA.IS", "GEDIK.IS", "GEDZA.IS",
  "GENIL.IS", "GENTS.IS", "GEREL.IS", "GESAN.IS", "GIPTA.IS", "GLBMD.IS", "GLCVY.IS", "GLRYH.IS",
  "GLYHO.IS", "GMTAS.IS", "GOKNR.IS", "GOLTS.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS", "GRSEL.IS",
  "GSDDE.IS", "GSDHO.IS", "GSRAY.IS", "GUBRF.IS", "GWIND.IS", "GZNMI.IS", "HALKB.IS", "HATEK.IS",
  "HATSN.IS", "HDFGS.IS", "HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "HOROZ.IS", "HRKET.IS",
  "HTTBT.IS", "HUBVC.IS", "HUNER.IS", "HURGZ.IS", "ICBCT.IS", "IDGYO.IS", "IEYHO.IS", "IHAAS.IS",
  "IHEVA.IS", "IHGZT.IS", "IHLAS.IS", "IHLGM.IS", "IHYAY.IS", "IMASM.IS", "INDES.IS", "INFO.IS",
  "INGRM.IS", "INTEM.IS", "INVEO.IS", "INVES.IS", "ISBTR.IS", "ISCTR.IS", "ISDMR.IS", "ISFIN.IS",
  "ISGSY.IS", "ISGYO.IS", "ISKPL.IS", "ISKUR.IS", "ISMEN.IS", "ISSEN.IS", "ISYAT.IS", "IZENR.IS",
  "IZFAS.IS", "IZINV.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS", "KAREL.IS", "KARSN.IS", "KARTN.IS",
  "KATMR.IS", "KAYSE.IS", "KBORU.IS", "KCAER.IS", "KCHOL.IS", "KENT.IS", "KERVN.IS", "KFEIN.IS",
  "KGYO.IS", "KIMMR.IS", "KLGYO.IS", "KLKIM.IS", "KLMSN.IS", "KLNMA.IS", "KLRHO.IS", "KLSER.IS",
  "KLSYN.IS", "KMPUR.IS", "KNFRT.IS", "KOCMT.IS", "KONKA.IS", "KONTR.IS", "KONYA.IS", "KOPOL.IS",
  "KORDS.IS", "KOTON.IS", "KRDMA.IS", "KRDMB.IS", "KRDMD.IS", "KRGYO.IS", "KRONT.IS", "KRPLS.IS",
  "KRSTL.IS", "KRTEK.IS", "KRVGD.IS", "KSTUR.IS", "KTLEV.IS", "KTSKR.IS", "KUTPO.IS", "KUVVA.IS",
  "KUYAS.IS", "KZBGY.IS", "KZGYO.IS", "LIDER.IS", "LIDFA.IS", "LILAK.IS", "LINK.IS", "LKMNH.IS",
  "LOGO.IS", "LRSHO.IS", "LUKSK.IS", "MAALT.IS", "MACKO.IS", "MAGEN.IS", "MAKIM.IS", "MAKTK.IS",
  "MANAS.IS", "MARBL.IS", "MARKA.IS", "MARTI.IS", "MAVI.IS", "MEDTR.IS", "MEGAP.IS", "MEGMT.IS",
  "MEKAG.IS", "MEPET.IS", "MERCN.IS", "MERIT.IS", "MERKO.IS", "METRO.IS", "MGROS.IS", "MHRGY.IS",
  "MIATK.IS", "MMCAS.IS", "MNDRS.IS", "MNDTR.IS", "MOBTL.IS", "MOGAN.IS", "MPARK.IS", "MRGYO.IS",
  "MRSHL.IS", "MSGYO.IS", "MTRKS.IS", "MTRYO.IS", "MZHLD.IS", "NATEN.IS", "NETAS.IS", "NIBAS.IS",
  "NTGAZ.IS", "NTHOL.IS", "NUGYO.IS", "NUHCM.IS", "OBASE.IS", "OBAMS.IS", "ODAS.IS", "ODINE.IS",
  "OFSYM.IS", "ONCSM.IS", "ORCAY.IS", "ORGE.IS", "ORMA.IS", "OSMEN.IS", "OSTIM.IS", "OTKAR.IS",
  "OYAKC.IS", "OYAYO.IS", "OYLUM.IS", "OYYAT.IS", "OZGYO.IS", "OZKGY.IS", "OZRDN.IS", "OZSUB.IS",
  "PAGYO.IS", "PAMEL.IS", "PAPIL.IS", "PARSN.IS", "PASEU.IS", "PATEK.IS", "PCILT.IS", "PEKGY.IS",
  "PENGD.IS", "PENTA.IS", "PETKM.IS", "PETUN.IS", "PGSUS.IS", "PINSU.IS", "PKART.IS", "PKENT.IS",
  "PLTUR.IS", "PNLSN.IS", "PNSUT.IS", "POLHO.IS", "POLTK.IS", "PRDGS.IS", "PRKAB.IS", "PRKME.IS",
  "PRZMA.IS", "PSDTC.IS", "PSGYO.IS", "QNBFB.IS", "QNBFK.IS", "QUAGR.IS", "RALYH.IS", "RAYSG.IS",
  "REEDR.IS", "RGYAS.IS", "RNPOL.IS", "RODRG.IS", "RTALB.IS", "RUBNS.IS", "RYGYO.IS", "RYSAS.IS",
  "SAHOL.IS", "SAMAT.IS", "SANEL.IS", "SANFM.IS", "SANKO.IS", "SARKY.IS", "SASA.IS", "SAYAS.IS",
  "SDTTR.IS", "SEGYO.IS", "SEKFK.IS", "SEKUR.IS", "SELEC.IS", "SELVA.IS", "SEYKM.IS", "SILVR.IS",
  "SISE.IS", "SKBNK.IS", "SKTAS.IS", "SKYLP.IS", "SKYMD.IS", "SMART.IS", "SMRTG.IS", "SNGYO.IS",
  "SNICA.IS", "SNPAM.IS", "SOKE.IS", "SOKM.IS", "SONME.IS", "SRVGY.IS", "SUMAS.IS", "SUNTK.IS",
  "SURGY.IS", "SUWEN.IS", "SVGYO.IS", "TABGD.IS", "TARKM.IS", "TATEN.IS", "TATGD.IS", "TAVHL.IS",
  "TBORG.IS", "TCELL.IS", "TDGYO.IS", "TEKTU.IS", "TERA.IS", "TEZOL.IS", "TGSAS.IS", "THYAO.IS",
  "TKFEN.IS", "TKNSA.IS", "TLMAN.IS", "TMPOL.IS", "TMSN.IS", "TNZTP.IS", "TOASO.IS", "TRCAS.IS",
  "TRGYO.IS", "TRILC.IS", "TSGYO.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS", "TTRAK.IS", "TUCLK.IS",
  "TUKAS.IS", "TUPRS.IS", "TUREX.IS", "TURGG.IS", "TURSG.IS", "UFUK.IS", "ULAS.IS", "ULUFA.IS",
  "ULUSE.IS", "ULUUN.IS", "UMPAS.IS", "UNLU.IS", "USAK.IS", "VAKBN.IS", "VAKFN.IS", "VAKKO.IS",
  "VANGD.IS", "VBTYZ.IS", "VERUS.IS", "VESBE.IS", "VESTL.IS", "VKGYO.IS", "VKING.IS", "VRGYO.IS",
  "YAPRK.IS", "YATAS.IS", "YAYLA.IS", "YBTAS.IS", "YEOTK.IS", "YESIL.IS", "YGGYO.IS", "YIGIT.IS",
  "YKBNK.IS", "YKSLN.IS", "YONGA.IS", "YUNSA.IS", "YYAPI.IS", "ZEDUR.IS", "ZOREN.IS", "XU100.IS"
  ]

    """Yapay zekanın tahminlerini tutacağı yerel veritabanını oluşturur."""
    conn = sqlite3.connect('hisse_hafiza.db', timeout=10, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tahminler
                 (tarih TEXT, sembol TEXT, hedef_fiyat REAL, gerceklesme_fiyati REAL, durum TEXT)''')
    conn.commit()
    conn.close()

def tahmin_kaydet(sembol, hedef_fiyat):
    bugun = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect('hisse_hafiza.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM tahminler WHERE tarih=? AND sembol=?", (bugun, sembol))
        if not c.fetchone():
            c.execute("INSERT INTO tahminler (tarih, sembol, hedef_fiyat, gerceklesme_fiyati, durum) VALUES (?, ?, ?, NULL, 'BEKLİYOR')", 
                      (bugun, sembol, hedef_fiyat))
        conn.commit()
def tahminleri_degerlendir():
    """5 gün öncesinin tahminlerini bugünün gerçek fiyatlarıyla kıyaslar (Optimize Edilmiş Sürüm)."""
    try:
        conn = sqlite3.connect('hisse_hafiza.db', timeout=10, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT rowid, tarih, sembol, hedef_fiyat FROM tahminler WHERE durum = 'BEKLİYOR'")
        bekleyenler = c.fetchall()
        
        if not bekleyenler:
            conn.close()
            return

        bugun = datetime.now()
        degerlendirilecekler = []

        # 1. Aşama: Sadece süresi (5 gün) dolmuş tahminleri filtrele
        for row in bekleyenler:
            rowid, tarih_str, sembol, hedef_fiyat = row
            try:
                kayit_tarihi = datetime.strptime(tarih_str, "%Y-%m-%d")
                if (bugun - kayit_tarihi).days >= 5:
                    degerlendirilecekler.append((rowid, sembol, hedef_fiyat))
            except Exception as e:
                logging.error(f"Tarih format hatası [{sembol}]: {e}")

        # Eğer süresi dolmuş tahmin yoksa veritabanını kapatıp çık
        if not degerlendirilecekler:
            conn.close()
            return

        # 2. Aşama: Aranacak benzersiz hisse kodlarını topla
        sembol_listesi = list(set([item[1] for item in degerlendirilecekler]))
        
        # 3. Aşama: Tüm hisselerin fiyatını TEK BİR İSTEKLE indir
        df_guncel = yf.download(sembol_listesi, period="1d", progress=False)

        if df_guncel.empty:
            conn.close()
            return

        # Yfinance MultiIndex kolon yapısını düzelt (Kapanış fiyatlarını al)
        if "Close" in df_guncel.columns:
            kapanislar = df_guncel["Close"]
        else:
            kapanislar = df_guncel

        # 4. Aşama: Tahminleri değerlendir ve DB'yi güncelle
        for rowid, sembol, hedef_fiyat in degerlendirilecekler:
            try:
                # Fiyatı tablodan çek (Tek hisse mi çoklu hisse mi kontrol et)
                if isinstance(kapanislar, pd.DataFrame) and sembol in kapanislar.columns:
                    gercek_fiyat = float(kapanislar[sembol].dropna().iloc[-1])
                elif isinstance(kapanislar, pd.Series):
                    gercek_fiyat = float(kapanislar.dropna().iloc[-1])
                else:
                    continue # Fiyat bulunamadıysa bu hisseyi atla

                # Sapma oranını hesapla (%5 tolerans)
                sapma_orani = abs(gercek_fiyat - hedef_fiyat) / gercek_fiyat
                durum = "BAŞARILI ✅" if sapma_orani <= 0.05 else "BAŞARISIZ ❌"

                # Veritabanında güncelle
                c.execute(
                    "UPDATE tahminler SET gerceklesme_fiyati = ?, durum = ? WHERE rowid = ?", 
                    (gercek_fiyat, durum, rowid)
                )
            except Exception as e:
                logging.error(f"Tahmin kıyaslama hatası [{sembol}]: {e}")

        conn.commit()
        conn.close()

    except Exception as e:
        logging.error(f"tahminleri_degerlendir genel hatası: {e}")
def veritabani_baslat():
    # Veritabanı (SQLite vs.) bağlantı kodlarınız burada olmalı
    pass
# Uygulama açıldığında veritabanını hazırla ve eski tahminleri kontrol et
veritabani_baslat()
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

# Örnek Kullanım:
semboller = sembol_formatla("THYAO.IS")
print(f"Yahoo için: {semboller['yfinance']}")
print(f"İş Yatırım için: {semboller['isyatirim']}")
# Kodun en üst kısımlarına ekle:

# ==========================================
# 1. TEMEL VE İLERİ TEKNİK FONKSİYONLAR


import time as tm
import yfinance as yf
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import isyatirimhisse

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
# ==========================================
# 2. 4 SAATLİK VERİ ÇEKME FONKSİYONU
# ==========================================
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
import yfinance as yf

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
def anomali_tespit_et(df):
    """
    Son 100 günlük veriye bakarak son gündeki hareketin anomali olup olmadığını test eder.
    """
    # İlgili özellikleri seç (Örn: Hacim değişimi, Kapanış değişimi, RSI vb.)
    features = df[['Close', 'Volume']].pct_change().dropna()
    
    if len(features) < 20:
        return "Veri yetersiz"
        
    # contamination=0.02: Verinin %2'sini "anormal" kabul edecek şekilde eğit
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    features['Anomaly'] = iso_forest.fit_predict(features)
    
    # Son günün anomali durumu (-1 = Anomali, 1 = Normal)
    son_durum = features['Anomaly'].iloc[-1]
    
    if son_durum == -1:
        return "⚠️ RİSKLİ: Fiyat/Hacim hareketlerinde anomali tespit edildi!"
    return "✅ Piyasaya uygun, normal hareket."
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
    # ==========================================
    # YENİ EKLENEN: 5, 8, 13 HAREKETLİ ORTALAMALAR
    # ==========================================
    # Basit Hareketli Ortalamalar (SMA)
    df_ta['SMA_5'] = df_ta['Close'].rolling(window=5).mean()
    df_ta['SMA_8'] = df_ta['Close'].rolling(window=8).mean()
    df_ta['SMA_13'] = df_ta['Close'].rolling(window=13).mean()
    df_ta['EMA_21'] = df_ta['Close'].ewm(span=21, adjust=False).mean()
    df_ta['EMA_34'] = df_ta['Close'].ewm(span=34, adjust=False).mean()
    df_ta['EMA_52'] = df_ta['Close'].ewm(span=52, adjust=False).mean()
    df_ta['EMA_89'] = df_ta['Close'].ewm(span=89, adjust=False).mean()
    df_ta['EMA_144'] = df_ta['Close'].ewm(span=144, adjust=False).mean()
    # Üstel Hareketli Ortalamalar (EMA) - (Yapay zeka için daha etkilidir)
    df_ta['EMA_5'] = df_ta['Close'].ewm(span=5, adjust=False).mean()
    df_ta['EMA_8'] = df_ta['Close'].ewm(span=8, adjust=False).mean()
    df_ta['EMA_13'] = df_ta['Close'].ewm(span=13, adjust=False).mean()
    df_ta['Destek'] = df_ta['Low'].rolling(window=20).min()
    df_ta['Direnc'] = df_ta['High'].rolling(window=20).max()
    df_ta['Destege_Uzaklik'] = (df_ta['Close'] - df_ta['Destek']) / df_ta['Close'].replace(0, 1e-9)
    df_ta['Dirence_Uzaklik'] = (df_ta['Direnc'] - df_ta['Close']) / df_ta['Close'].replace(0, 1e-9)
    # Kısa Vadeli Fibonacci Kesişim Trend Sinyali (5 > 8 > 13)
    df_ta['Fibo_MA_Trend'] = np.where(
        (df_ta['EMA_5'] > df_ta['EMA_8']) & (df_ta['EMA_8'] > df_ta['EMA_13']), "🚀 GÜÇLÜ YÜKSELİŞ",
        np.where((df_ta['EMA_5'] < df_ta['EMA_8']) & (df_ta['EMA_8'] < df_ta['EMA_13']), "🔻 GÜÇLÜ DÜŞÜŞ", "⚖️ YATAY NÖTR")
    )
    return df_ta


def genisletilmis_indikatorler_hesapla(df, period=14):
    """
    RSI(14), MACD(12,26,9), Bollinger Bantları(20,2), ADX/+DI/-DI(14) ve
    Wilder ATR(14) hesaplayıp DataFrame'e ekler. Tüm hesaplamalar sadece
    OHLCV verisi üzerinden yapılır, ek ağ isteği gerektirmez.
    """
    d = df.copy()
    if d is None or len(d) < period + 1:
        for kol in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Orta', 'BB_Ust',
                    'BB_Alt', 'BB_Yuzde', 'ATR', 'Plus_DI', 'Minus_DI', 'ADX']:
            d[kol] = np.nan
        return d

    # --- RSI (Wilder) ---
    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    d['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['MACD_Signal']

    # --- Bollinger Bantları (20, 2) ---
    d['BB_Orta'] = d['Close'].rolling(window=20).mean()
    bb_std = d['Close'].rolling(window=20).std()
    d['BB_Ust'] = d['BB_Orta'] + (bb_std * 2)
    d['BB_Alt'] = d['BB_Orta'] - (bb_std * 2)
    bb_genislik = (d['BB_Ust'] - d['BB_Alt']).replace(0, 1e-9)
    # %B: 0 = alt bant, 0.5 = orta bant, 1 = üst bant
    d['BB_Yuzde'] = (d['Close'] - d['BB_Alt']) / bb_genislik

    # --- ATR (Wilder, True Range üzerinden) ---
    prev_close = d['Close'].shift(1)
    tr = pd.concat([
        d['High'] - d['Low'],
        (d['High'] - prev_close).abs(),
        (d['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    d['ATR'] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # --- ADX / +DI / -DI (Wilder) ---
    up_move = d['High'].diff()
    down_move = -d['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_di = d['ATR'].replace(0, 1e-9)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_di
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_di
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    d['Plus_DI'] = plus_di
    d['Minus_DI'] = minus_di
    d['ADX'] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return d


def fibonacci_seviyeleri_hesapla(df, lookback=60):
    """
    Son `lookback` bar içindeki en yüksek/en düşük noktalar arasında
    Fibonacci geri çekilme seviyelerini (0, %23.6, %38.2, %50, %61.8, %78.6,
    %100) hesaplar ve fiyata en yakın seviyeyi döner.
    """
    if df is None or df.empty:
        return None
    pencere = df.tail(lookback) if len(df) >= 10 else df
    if len(pencere) < 10:
        return None

    tepe = float(pencere['High'].max())
    dip = float(pencere['Low'].min())
    fark = tepe - dip
    if fark <= 0:
        return None

    fiyat = float(df['Close'].iloc[-1])
    tepe_idx = pencere['High'].idxmax()
    dip_idx = pencere['Low'].idxmin()
    # Dip, tepeden önce oluştuysa bu bir YÜKSELİŞ bacağıdır -> tepeden geri çekilme
    yukselis_bacagi = dip_idx < tepe_idx

    oranlar = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    seviyeler = {}
    for o in oranlar:
        seviyeler[o] = (tepe - fark * o) if yukselis_bacagi else (dip + fark * o)

    en_yakin_oran = min(seviyeler, key=lambda o: abs(seviyeler[o] - fiyat))
    en_yakin_fiyat = seviyeler[en_yakin_oran]
    uzaklik_pct = (abs(fiyat - en_yakin_fiyat) / fiyat * 100) if fiyat else 0.0

    return {
        "tepe": tepe, "dip": dip, "yukselis_bacagi": yukselis_bacagi,
        "en_yakin_oran": en_yakin_oran, "en_yakin_fiyat": en_yakin_fiyat,
        "uzaklik_pct": uzaklik_pct, "seviyeler": seviyeler,
    }


def haftalik_veri_turet(df_gunluk):
    """
    Ayrı bir ağ isteği ATMADAN, zaten elde olan günlük veriyi haftalığa
    (Pazartesi-Cuma) resample ederek türetir. Haftalık teyit için ekstra
    API çağrısı gerekmediğinden tarama hızını etkilemez.
    """
    if df_gunluk is None or df_gunluk.empty:
        return pd.DataFrame()
    try:
        gerekli = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df_gunluk.columns for c in gerekli):
            return pd.DataFrame()
        df_w = df_gunluk[gerekli].resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        return df_w
    except Exception:
        return pd.DataFrame()
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
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

def tahmini_logla(sembol, anlik_fiyat, tahmin_edilen_fiyat, sinyal, guven_skoru):
    try:
        conn = sqlite3.connect('ai_performans.db')
        c = conn.cursor()
        
        # Tablo yoksa otomatik oluşturur
        c.execute('''CREATE TABLE IF NOT EXISTS tahmin_loglari
                     (tarih TEXT, sembol TEXT, anlik_fiyat REAL, tahmin_fiyat REAL, sinyal TEXT, guven REAL, gerceklesen_fiyat REAL)''')
        
        bugun = datetime.now().strftime("%Y-%m-%d")
        
        c.execute("INSERT INTO tahmin_loglari (tarih, sembol, anlik_fiyat, tahmin_fiyat, sinyal, guven, gerceklesen_fiyat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (bugun, sembol, anlik_fiyat, tahmin_edilen_fiyat, sinyal, guven_skoru, 0.0))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Loglama hatası: {e}")
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
import numpy as np
import pandas as pd

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
# ==========================================
# YENİ EKLENEN: FORMASYON TESPİT VE HEDEF HESAPLAMA
# ==========================================
def hedef_degiskeni_olustur(df: pd.DataFrame) -> pd.DataFrame:
    # 5 Gün sonrasının net getirisi
    df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # 5 Gün içindeki MAKSİMUM düşüş (Drawdown)
    df['Future_Min'] = df['Low'].rolling(window=5).min().shift(-5)
    df['Max_Drawdown'] = df['Future_Min'] / df['Close'] - 1
    
    # Kural: Getiri %3'ten büyük OLMALI VE aradaki sarkma %1.5'u GEÇMEMELİ
    # XGBoost ve Random Forest Classifier için hedef belirliyoruz
    df['Target_Class'] = np.where(
        (df['Future_Return'] > 0.03) & (df['Max_Drawdown'] > -0.015), 
        1, 
        0
    )
    
    return df
def ozellikleri_zenginlestir(df: pd.DataFrame) -> pd.DataFrame:
    # 1. ATR (Average True Range) Hesaplaması
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 2. Tilson Uzaklığını Normalize Etmek
    # Sıfıra bölünme hatasını önlemek için küçük bir değer (1e-5) ekliyoruz
    df['Tilson_Dist_Norm'] = df['Tilson_Dist'] / (df['ATR'] + 1e-5)
    
    # 3. Hacim Teyidi: Fiyat ve Hacim Çarpımı (Basit VWAP Yaklaşımı)
    df['Volume_Trend'] = df['Volume'].rolling(window=5).mean() / df['Volume'].rolling(window=20).mean()
    
    # Gereksiz geçici kolonları temizle
    df.drop(['H-L', 'H-C', 'L-C', 'TR'], axis=1, inplace=True)
    
    return df
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

def modeli_degerlendir(X, y):
    # n_splits=5 ile veriyi zaman ekseninde 5 parçaya böler, 
    # her seferinde sadece geçmiş verilerle eğitip GELECEK veride test eder.
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Yeni sınıflandırma modelimiz
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    
    # Doğrulama skorları (Accuracy veya F1 Skoru kullanılabilir)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')
    print(f"TimeSeriesSplit F1 Skorları: {scores}")
    print(f"Ortalama F1 Skoru: {scores.mean():.4f}")
    
    return model
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
import pandas as pd
import numpy as np

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
def stacking_model_olustur(xgb_model, rf_model, svr_model):
    estimators = [
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('svr', svr_model)
    ]
    
    # Meta-model olarak Ridge Regresyon kullanıyoruz (aşırı öğrenmeyi engeller)
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    )
    
    return stack_model
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


def asenkron_analiz_yap(sembol, baslangic, bitis, analiz_tipi="radar", veri_kaynagi="Yahoo Finance (yfinance)", guncel_fiyat_override=None, df_gunluk_override=None):
    try:
        # 1. Günlük Veriyi Çek — toplu taramadan (paralel_tara) hazır veri geldiyse
        # onu kullan, gelmediyse (tek hisse görüntüleme ya da toplu indirmede
        # bulunamayan sembol) eski çoklu-kaynak yöntemiyle tek tek çek.
        if df_gunluk_override is not None and not df_gunluk_override.empty:
            df_gunluk = df_gunluk_override
        else:
            df_gunluk = veri_yukle(sembol, baslangic, bitis, interval="1d", kaynak=veri_kaynagi)
        if df_gunluk is None or df_gunluk.empty or len(df_gunluk) < 20: 
            return None
            
        df_g = df_gunluk.copy()
        
        # --- A. ANLIK FİYAT ---
        # Toplu taramadan (paralel_tara) bir fiyat verildiyse onu kullan; verilmediyse
        # (örn. tek hisse görüntüleme gibi doğrudan çağrılarda) eski tek-tek yöntemle dene.
        if guncel_fiyat_override is not None:
            guncel_fiyat = float(guncel_fiyat_override)
        else:
            try:
                guncel_df = yf.download(
                    sembol,
                    period="1d",
                    interval="1m",
                    progress=False,
                    auto_adjust=True
                )
                if not guncel_df.empty:
                    guncel_fiyat = float(guncel_df["Close"].iloc[-1])
                else:
                    guncel_fiyat = float(df_g["Close"].iloc[-1])
            except:
                guncel_fiyat = float(df_g["Close"].iloc[-1])


# --------------------------------------------------
# Kapanış fiyatı (veriye bakarak - saat tahminine dayanmadan)
# --------------------------------------------------
        # df_g'nin son barının tarihi bugünse, o bar muhtemelen canlı/tamamlanmamıştır;
        # "resmi kapanış" bir önceki tamamlanmış bardır. Son bar bugünden eskiyse
        # (piyasa henüz bugünün barını oluşturmadıysa), zaten en son tamamlanmış kapanıştır.
        tz = pytz.timezone("Europe/Istanbul")
        bugun_tarih = datetime.now(tz).date()
        son_bar_tarihi = df_g.index[-1].date() if len(df_g) > 0 else None

        if son_bar_tarihi == bugun_tarih and len(df_g) >= 2:
            kapanis_fiyati = float(df_g["Close"].iloc[-2])
        else:
            kapanis_fiyati = float(df_g["Close"].iloc[-1])

        # İndikatörler anlık fiyata göre hesaplansın diye son barın kapanışını canlı fiyatla güncelle
        if not df_g.empty and guncel_fiyat is not None:
            last_idx = df_g.index[-1]
            df_g.loc[last_idx, 'Close'] = guncel_fiyat
            df_g.loc[last_idx, 'High'] = max(df_g.loc[last_idx, 'High'], guncel_fiyat)
            df_g.loc[last_idx, 'Low'] = min(df_g.loc[last_idx, 'Low'], guncel_fiyat)

        # --- C. TEMEL İNDİKATÖRLER & İLERİ TEKNİK ANALİZ ---
        df_g = stokastik_hesapla(df_g)
        df_g['Tilson_T3'] = tilson_t3(df_g['Close'])
        df_g = ileri_teknik_gostergeler(df_g)
        # RSI / MACD / Bollinger / ADX / ATR — ek ağ isteği gerektirmez
        df_g = genisletilmis_indikatorler_hesapla(df_g)

        # Yenilenen dipten dönüş analizi çağrılıyor
        temp_g = dipten_donus_analizi(df_g)
        
        g_fiyat = guncel_fiyat
        g_tilson = df_g['Tilson_T3'].iloc[-1]
        g_stoch_k = df_g['Stoch_K'].iloc[-1]
        g_stoch_d = df_g['Stoch_D'].iloc[-1]
        g_ema5 = df_g['EMA_5'].iloc[-1]
        g_ema8 = df_g['EMA_8'].iloc[-1]
        g_ema13 = df_g['EMA_13'].iloc[-1]
        
        g_boga = g_fiyat > g_tilson
        g_stoch_al = (g_stoch_k < 35) and (g_stoch_k > g_stoch_d)
        g_hacim = temp_g['Hacim_Patlamasi'].iloc[-1]
        g_uyusmazlik = temp_g['Pozitif_Uyusmazlik'].iloc[-1]
        # 🌟 YENİ EKLENEN SATIR: Süper Sinyal Durumu
        g_super_sinyal = temp_g.get('Super_Sinyal', pd.Series([False])).iloc[-1]
        g_spring = temp_g['Wyckoff_Spring'].iloc[-1]
        g_ma_kestimi = (g_ema5 > g_ema8) and (g_ema8 > g_ema13)

        # --- C2. HAFTALIK ZAMAN DİLİMİ TEYİDİ (ek ağ isteği YOK — günlükten türetilir) ---
        df_w = haftalik_veri_turet(df_g)
        w_boga = g_boga  # veri yetersizse günlük yönle aynı kabul edilir (nötr varsayım)
        if not df_w.empty and len(df_w) >= 10:
            try:
                df_w['Tilson_T3'] = tilson_t3(df_w['Close'])
                w_fiyat = float(df_w['Close'].iloc[-1])
                w_tilson = float(df_w['Tilson_T3'].iloc[-1])
                if pd.notna(w_tilson):
                    w_boga = w_fiyat > w_tilson
            except Exception as e:
                logging.debug(f"[{sembol}] Haftalık analiz hatası: {e}")

        # --- C3. RSI / MACD / BOLLINGER / ADX METİNLERİ ---
        g_rsi = df_g['RSI'].iloc[-1] if 'RSI' in df_g.columns else np.nan
        if pd.notna(g_rsi):
            if g_rsi >= 70:
                rsi_metin = f"🔴 {g_rsi:.1f} (Aşırı Alım)"
            elif g_rsi <= 30:
                rsi_metin = f"🟢 {g_rsi:.1f} (Aşırı Satım)"
            else:
                rsi_metin = f"{g_rsi:.1f}"
        else:
            rsi_metin = "-"

        g_macd_hist = df_g['MACD_Hist'].iloc[-1] if 'MACD_Hist' in df_g.columns else np.nan
        g_macd_hist_onceki = df_g['MACD_Hist'].iloc[-2] if ('MACD_Hist' in df_g.columns and len(df_g) > 1) else np.nan
        if pd.notna(g_macd_hist):
            if g_macd_hist > 0 and (pd.isna(g_macd_hist_onceki) or g_macd_hist_onceki <= 0):
                macd_metin = "🟢 Yeni Pozitif Kesişim"
            elif g_macd_hist < 0 and (pd.isna(g_macd_hist_onceki) or g_macd_hist_onceki >= 0):
                macd_metin = "🔴 Yeni Negatif Kesişim"
            elif g_macd_hist > 0:
                macd_metin = "🟢 Pozitif"
            else:
                macd_metin = "🔴 Negatif"
        else:
            macd_metin = "-"

        g_bb_yuzde = df_g['BB_Yuzde'].iloc[-1] if 'BB_Yuzde' in df_g.columns else np.nan
        if pd.notna(g_bb_yuzde):
            if g_bb_yuzde >= 1:
                bb_metin = "🔴 Üst Bandın Üstünde"
            elif g_bb_yuzde <= 0:
                bb_metin = "🟢 Alt Bandın Altında"
            elif g_bb_yuzde >= 0.8:
                bb_metin = "Üst Banda Yakın"
            elif g_bb_yuzde <= 0.2:
                bb_metin = "Alt Banda Yakın"
            else:
                bb_metin = "Orta Bant"
        else:
            bb_metin = "-"

        g_adx = df_g['ADX'].iloc[-1] if 'ADX' in df_g.columns else np.nan
        if pd.notna(g_adx):
            if g_adx >= 25:
                adx_metin = f"💪 {g_adx:.1f} (Güçlü Trend)"
            elif g_adx >= 15:
                adx_metin = f"〰️ {g_adx:.1f} (Orta)"
            else:
                adx_metin = f"😴 {g_adx:.1f} (Zayıf/Yatay)"
        else:
            adx_metin = "-"

        # --- C4. FİBONACCİ SEVİYELERİ (son 60 bar) ---
        fib_bilgi = fibonacci_seviyeleri_hesapla(df_g, lookback=60)
        if fib_bilgi:
            fib_metin = f"%{fib_bilgi['en_yakin_oran'] * 100:.1f} ({fib_bilgi['en_yakin_fiyat']:.2f} TL, %{fib_bilgi['uzaklik_pct']:.1f} uzak)"
        else:
            fib_metin = "-"

        # --- C5. ATR TABANLI STOP-LOSS / KAR-AL HEDEFİ (1.5x ATR stop, 2x/3x ATR hedef) ---
        g_atr = df_g['ATR'].iloc[-1] if 'ATR' in df_g.columns else np.nan
        if pd.notna(g_atr) and g_atr > 0 and g_fiyat:
            atr_stop = round(g_fiyat - 1.5 * g_atr, 2)
            atr_hedef1 = round(g_fiyat + 2 * g_atr, 2)
            atr_hedef2 = round(g_fiyat + 3 * g_atr, 2)
            risk = g_fiyat - atr_stop
            odul = atr_hedef1 - g_fiyat
            rr_orani = round(odul / risk, 2) if risk > 0 else None
            stop_metin = f"{atr_stop} TL (%{((g_fiyat - atr_stop) / g_fiyat * 100):.1f})"
            hedef_metin = f"{atr_hedef1} TL / {atr_hedef2} TL"
            rr_metin = f"1:{rr_orani}" if rr_orani else "-"
        else:
            stop_metin = "-"
            hedef_metin = "-"
            rr_metin = "-"

        # --- C6. TEKNİK TEYİT PUANI (dahili — ekranda gösterilmez, AI Kararı ve
        # AL/SAT Kararı'nı etkilemek için kullanılır) ---
        teknik_puan = 0
        teknik_notlar = []

        if pd.notna(g_rsi):
            if g_rsi <= 35:
                teknik_puan += 15
                teknik_notlar.append("RSI aşırı satım bölgesinde")
            elif g_rsi >= 70:
                teknik_puan -= 15
                teknik_notlar.append("RSI aşırı alım bölgesinde (risk)")

        if macd_metin.startswith("🟢 Yeni Pozitif"):
            teknik_puan += 15
            teknik_notlar.append("MACD yeni pozitif kesişim verdi")
        elif macd_metin.startswith("🔴 Yeni Negatif"):
            teknik_puan -= 15
            teknik_notlar.append("MACD yeni negatif kesişim verdi")
        elif macd_metin == "🟢 Pozitif":
            teknik_puan += 5
        elif macd_metin == "🔴 Negatif":
            teknik_puan -= 5

        if pd.notna(g_bb_yuzde):
            if g_bb_yuzde <= 0.05:
                teknik_puan += 10
                teknik_notlar.append("Fiyat Bollinger alt bandında (tepki potansiyeli)")
            elif g_bb_yuzde >= 0.95:
                teknik_puan -= 10
                teknik_notlar.append("Fiyat Bollinger üst bandında (aşırı genişleme riski)")

        if pd.notna(g_adx) and g_adx >= 25:
            if g_boga:
                teknik_puan += 10
                teknik_notlar.append(f"ADX ({g_adx:.1f}) güçlü boğa trendini onaylıyor")
            else:
                teknik_puan -= 10
                teknik_notlar.append(f"ADX ({g_adx:.1f}) güçlü ayı trendini onaylıyor")

        if fib_bilgi and fib_bilgi['uzaklik_pct'] < 1.5 and fib_bilgi['en_yakin_oran'] in (0.5, 0.618, 0.786):
            teknik_puan += 10
            teknik_notlar.append(f"Fiyat %{fib_bilgi['en_yakin_oran']*100:.1f} Fibonacci seviyesine çok yakın")

        if w_boga and g_boga:
            teknik_puan += 5
        elif (not w_boga) and (not g_boga):
            teknik_puan -= 5

        teknik_detay = {
            "Haftalık T3": "🚀 BOĞA" if w_boga else "🐻 AYI",
            "RSI(14)": rsi_metin,
            "MACD Durumu": macd_metin,
            "Bollinger Pozisyonu": bb_metin,
            "Trend Gücü (ADX)": adx_metin,
            "Fibonacci Seviyesi": fib_metin,
            "ATR Stop-Loss": stop_metin,
            "ATR Kar Hedefi": hedef_metin,
            "Risk/Ödül": rr_metin,
            "Teknik Teyit Puanı": teknik_puan,
            "Teknik Notlar": "; ".join(teknik_notlar) if teknik_notlar else "Belirgin bir teknik teyit/uyarı yok",
        }

        # ⚡ DOĞRULANMIŞ AKILLI FİLTRE (g_super_sinyal eklendi)
        umut_var_mi = g_boga or g_stoch_al or g_hacim or g_uyusmazlik or g_super_sinyal or g_spring or g_ma_kestimi
        
        if not umut_var_mi and analiz_tipi == "radar":
            return {
                "Varlık": sembol,
                "Güncel Fiyat": f"{guncel_fiyat:.2f}",
                "Kapanış Fiyatı": f"{kapanis_fiyati:.2f}",
                "Günlük T3": "🐻 AYI",
                "4S T3": "-",
                "🤖 AI Kararı": "Zaman Tasarrufu",
                "🎯 AI Hedef": "-",
                "🎯 AL/SAT Kararı": "🐻 PAS GEÇİLDİ (Ölü Trend)",
                "🎯 Kesin Dip Onayı": "-",
                "🔍 Tespit Edilen Formasyon": "Yok",
                "🎯 Formasyon Hedefi (%)": "% 0.00",
                "📈 Pozitif Uyuşmazlık": "-",
                "📊 Temel Skor": "-",
                "💥 Hacim Analizi": "Normal",
                "🪤 Spring (Tuzak)": "-",
                "_teknik_detay": teknik_detay,
            }

        # --- D. 4 SAATLİK VERİ ÇEKME & ANALİZ ---
        df_4h = veri_4saatlik_getir(sembol, baslangic, bitis, kaynak=veri_kaynagi)
        
        h4_fiyat, h4_tilson = g_fiyat, g_tilson
        h4_stoch_k, h4_stoch_d = g_stoch_k, g_stoch_d
        h4_boga, h4_stoch_al = g_boga, g_stoch_al

        if not df_4h.empty and len(df_4h) >= 20:
            try:
                df_4h = stokastik_hesapla(df_4h)
                df_4h['Tilson_T3'] = tilson_t3(df_4h['Close'])
                
                h4_fiyat = df_4h['Close'].iloc[-1]
                h4_tilson = df_4h['Tilson_T3'].iloc[-1]
                h4_stoch_k = df_4h['Stoch_K'].iloc[-1]
                h4_stoch_d = df_4h['Stoch_D'].iloc[-1]
                
                h4_boga = h4_fiyat > h4_tilson
                h4_stoch_al = (h4_stoch_k < 35) and (h4_stoch_k > h4_stoch_d)
            except Exception as e:
                logging.error(f"[{sembol}] 4S Analiz Hatası: {e}")

        # --- E. KARAR MEKANİZMASI (Günlük + 4S + Haftalık 3'lü Teyit) ---
        if g_boga and h4_boga and w_boga:
            al_sat_karari = "🚀🚀 SÜPER GÜÇLÜ AL (Haftalık+4S+Günlük Onaylı)" if (g_stoch_al and h4_stoch_al) else "🚀 GÜÇLÜ AL (3 Zaman Dilimi Onaylı)"
        elif g_boga and h4_boga and not w_boga:
            al_sat_karari = "🟢 AL (Günlük+4S Onaylı / Haftalık Zayıf)" if (g_stoch_al and h4_stoch_al) else "🟢 AL (Trend Onaylı)"
        elif g_boga and not h4_boga:
            al_sat_karari = "⚠️ DÜZELTME (Günlük Boğa / 4S Ayı)"
        elif not g_boga and h4_boga:
            al_sat_karari = "⚡ TEPKİ YÜKSELİŞİ (4S Boğa / Günlük Ayı)"
        elif not g_boga and not h4_boga and w_boga:
            al_sat_karari = "🟡 ANA TREND HALA BOĞA (Haftalık) — Kısa Vade Düzeltmede"
        else:
            al_sat_karari = "🐻 GÜÇLÜ SAT / AYI"

        # Teknik teyit puanı (RSI/MACD/Bollinger/ADX/Fibonacci) kararı destekliyor/zayıflatıyorsa etiketle
        if teknik_puan >= 25:
            al_sat_karari += " ⭐ (Teknik Güçlü Teyit)"
        elif teknik_puan <= -20:
            al_sat_karari += " ⚠️ (Teknik Zayıflama Sinyali)"

        if analiz_tipi == "radar":
            temp_4h = dipten_donus_analizi(df_4h) if not df_4h.empty else temp_g
            h4_hacim = temp_4h['Hacim_Patlamasi'].iloc[-1] if not temp_4h.empty else False
            
            hacim_durum = "🔥 GÜÇLÜ PATLAMA" if (g_hacim or h4_hacim) else "Normal"
            
            # 🌟 YENİ UYUŞMAZLIK / SÜPER SİNYAL METNİ HESAPLAMA
            h4_super = temp_4h.get('Super_Sinyal', pd.Series([False])).iloc[-1] if not temp_4h.empty else False
            h4_uyusmazlik = temp_4h.get('Pozitif_Uyusmazlik', pd.Series([False])).iloc[-1] if not temp_4h.empty else False
            
            if g_super_sinyal or h4_super:
                uyusmazlik_durum = "🌟 SÜPER SİNYAL (RSI+MACD+Stoch)"
            elif g_uyusmazlik or h4_uyusmazlik:
                uyusmazlik_durum = "✅ POZİTİF UYUŞMAZLIK"
            else:
                uyusmazlik_durum = "-"

            spring_durum = "✅ VAR" if (g_spring or temp_4h.get('Wyckoff_Spring', pd.Series([False])).iloc[-1]) else "-"
            
            # Formasyon tespiti ve hedef hesaplama
            formasyon_adi, formasyon_hedef = formasyon_tespit_et_ve_hedefle(df_g)
            
            # 1. KRİTİK DÜZELTME: AI Verisini Veto'dan ÖNCE Hesapla!
            # 1. KRİTİK DÜZELTME: AI Verisini Veto'dan ÖNCE Hesapla!
            ai_veri = ensemble_prediction(df_g, sembol) if umut_var_mi else {'signal': "ZAYIF", 'rf_prediction': 0.0}

            # Teknik teyit puanı AI kararına da yansıtılır (RSI/MACD/Bollinger/ADX/Fibonacci)
            if umut_var_mi:
                if teknik_puan >= 25:
                    ai_veri['signal'] = f"{ai_veri.get('signal', 'NÖTR')} ⭐"
                elif teknik_puan <= -20:
                    ai_veri['signal'] = f"{ai_veri.get('signal', 'NÖTR')} ⚠️"
            
            # 👇 YENİ EKLENECEK BLOK 👇
            # Eğer yapay zeka bir tahmin ürettiyse bunu SQLite veritabanına kaydet
            if umut_var_mi and ai_veri.get('rf_prediction', 0.0) > 0:
                tahmin_kaydet(sembol, ai_veri['rf_prediction'])
            # 👆 YENİ EKLENECEK BLOK BİTİŞ 👆

            # --- 🚨 FORMASYON VETO (RİSK KONTROL) MEKANİZMASI ---
            
            # --- 🚨 FORMASYON VETO (RİSK KONTROL) MEKANİZMASI ---
            try:
                hedef_str = str(formasyon_hedef).replace('%', '').replace(' ', '').strip()
                if hedef_str != '-' and hedef_str != '0.00':
                    hedef_oran = float(hedef_str)
                    
                    if hedef_oran < -4.0:
                        if "AL" in al_sat_karari:
                            al_sat_karari = f"⚠️ RİSKLİ: Trend Pozitif ama {formasyon_adi} Tehdidi!"
                        
                        ai_sinyal = ai_veri.get('signal', 'NÖTR')
                        if "AL" in ai_sinyal:
                            ai_veri['signal'] = f"🛑 AI İPTAL ({formasyon_adi})"
            except Exception as e:
                logging.warning(f"[{sembol}] Veto mekanizmasında hata: {e}")

            try:
                sihirli_veri = sihirli_formul_skorla(sembol, df=df_g)
                s_skor = sihirli_veri.get('Puan', 0) if isinstance(sihirli_veri, dict) else 0
            except Exception:
                s_skor = 0
            
            kesin_dip_mi = temp_4h.get('Kesin_Dip_Donusu', pd.Series([False])).iloc[-1] if not temp_4h.empty else False
            dip_durum = "🔥 KESİN DÖNÜŞ ONAYI!" if kesin_dip_mi else "-"

            return {
                "Varlık": sembol,
                "Güncel Fiyat": f"{guncel_fiyat:.2f}",
                "Kapanış Fiyatı": f"{kapanis_fiyati:.2f}",
                "Günlük T3": "🚀 BOĞA" if g_boga else "🐻 AYI",
                "4S T3": "🚀 BOĞA" if h4_boga else "🐻 AYI",
                "🤖 AI Kararı": ai_veri.get('signal', 'NÖTR'), # İçinde gün tahmini de yazacak
                "🎯 AI Hedef": f"{ai_veri.get('rf_prediction', 0.0)} TL",
                "🎯 AL/SAT Kararı": al_sat_karari,
                "🎯 Kesin Dip Onayı": dip_durum,
                "🔍 Tespit Edilen Formasyon": formasyon_adi,
                "🎯 Formasyon Hedefi (%)": formasyon_hedef,
                "📈 Pozitif Uyuşmazlık": uyusmazlik_durum,
                "📊 Temel Skor": s_skor,
                "💥 Hacim Analizi": hacim_durum,
                "🪤 Spring (Tuzak)": spring_durum,
                "_teknik_detay": teknik_detay,
            }

        elif analiz_tipi == "stoch":
            return {
                "Varlık": sembol,
                "Son Fiyat": f"{guncel_fiyat:.2f}",
                "Günlük Stoch %K": round(g_stoch_k, 2),
                "4S Stoch %K": round(h4_stoch_k, 2),
                "Durum": "🟢 Çift Dip/Al" if (g_stoch_al and h4_stoch_al) else ("↗️ Pozitif" if h4_stoch_al else "⚪ Nötr")
            }

        elif analiz_tipi == "tilson":
            g_fark_pct = ((g_fiyat - g_tilson) / g_tilson * 100) if g_tilson else 0.0
            h4_fark_pct = ((h4_fiyat - h4_tilson) / h4_tilson * 100) if h4_tilson else 0.0
            if g_boga and h4_boga:
                tilson_durum = "🚀 ÇİFT BOĞA (4S + Günlük)"
            elif g_boga and not h4_boga:
                tilson_durum = "⚠️ Günlük Boğa / 4S Ayı"
            elif not g_boga and h4_boga:
                tilson_durum = "⚡ 4S Boğa / Günlük Ayı"
            else:
                tilson_durum = "🐻 ÇİFT AYI (4S + Günlük)"
            return {
                "Varlık": sembol,
                "Son Fiyat": f"{guncel_fiyat:.2f}",
                "Günlük T3": round(float(g_tilson), 2),
                "Günlük Fark (%)": round(g_fark_pct, 2),
                "4S T3": round(float(h4_tilson), 2),
                "4S Fark (%)": round(h4_fark_pct, 2),
                "Durum": tilson_durum
            }

    except Exception as e:
        logging.error(f"[{sembol}] Analiz Hatası: {str(e)}")
        return None
# ==========================================
# 2. YAPAY ZEKA VE KURUMSAL MOTORLAR
# ==========================================
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
        model = XGBRegressor(**param, random_state=42, n_jobs=1)
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
        # 🟢 DÜZELTME: Orijinal dataframe'i bozmamak için doğrudan kopyalıyoruz
        t_df = df.copy()
        t_df = borsa_endeks_verisini_ekle(t_df)
        # --- 1. Veri Hazırlığı ve Feature Engineering ---
        t_df = yapay_zeka_icin_formasyon_bul(t_df)
        t_df = makro_formasyonlari_bul(t_df, window=20)
        t_df = trend_ve_harmonik_bul(t_df)
        
        if 'Stoch_K' not in t_df.columns:
            low_min = t_df['Low'].rolling(window=14).min()
            high_max = t_df['High'].rolling(window=14).max()
            t_df['Stoch_K'] = 100 * ((t_df['Close'] - low_min) / (high_max - low_min + 1e-9))
            
        t_df['Stoch_D'] = t_df['Stoch_K'].rolling(window=3).mean()
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

        # Hafıza Verileri
        t_df['Return_1d'] = t_df['Close'].pct_change(1)
        t_df['Return_2d'] = t_df['Close'].pct_change(2)
        t_df['Return_3d'] = t_df['Close'].pct_change(3)
        t_df['Vol_Lag1'] = t_df['Vol_Change'].shift(1)
        t_df['Vol_Lag2'] = t_df['Vol_Change'].shift(2)
        
        # EMA
        t_df['EMA_5'] = t_df['Close'].ewm(span=5, adjust=False).mean()
        t_df['EMA_8'] = t_df['Close'].ewm(span=8, adjust=False).mean()
        t_df['EMA_13'] = t_df['Close'].ewm(span=13, adjust=False).mean()
        
        t_df['EMA_5_Dist'] = (t_df['Close'] - t_df['EMA_5']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_8_Dist'] = (t_df['Close'] - t_df['EMA_8']) / t_df['Close'].replace(0, 0.0001)
        t_df['EMA_13_Dist'] = (t_df['Close'] - t_df['EMA_13']) / t_df['Close'].replace(0, 0.0001)
        
        t_df['Trend_5_8'] = np.where(t_df['EMA_5'] > t_df['EMA_8'], 1, -1)
        t_df['Trend_8_13'] = np.where(t_df['EMA_8'] > t_df['EMA_13'], 1, -1)

        # --- LSTM ENTEGRASYONU (XGBoost'a Öznitelik Olarak Verilmek Üzere) ---
        # LSTM'in ihtiyaç duyduğu Target_Class, Tilson_Dist_Norm ve Volume_Trend
        # sütunlarını burada üretiyoruz (daha önce hiç çağrılmadıkları için LSTM
        # sessizce hep nötr/0.5 dönüyor ve gerçek bir entegrasyon oluşmuyordu).
        try:
            t_df = ozellikleri_zenginlestir(t_df)   # Tilson_Dist_Norm, Volume_Trend (ATR de tazelenir)
            t_df = hedef_degiskeni_olustur(t_df)    # Target_Class (LSTM'in eğitim etiketi)
            t_df = lstm_tahmin_yap(t_df, lookback_days=40)  # Tüm seri için LSTM_Score üretir
        except Exception as e:
            logging.warning(f"[{sembol}] LSTM öznitelik üretimi başarısız, nötr skorla devam: {e}")
            t_df['LSTM_Score'] = 0.5

        # Ham Öznitelik Listesi
        raw_features = [
            'RSI', 'MACD_Hist', 'BB_Pozisyon', 'ATR', 'Z_Score', 
            'Vol_Change', 'EMA_Trend', 'Stoch_K', 'Stoch_D', 'Stoch_Diff',
            'Tilson_Dist', 'Return_1d', 'Return_2d', 'Return_3d', 
            'Vol_Lag1', 'Vol_Lag2', 'EMA_5_Dist', 'EMA_8_Dist', 'EMA_13_Dist', 
            'Trend_5_8', 'Trend_8_13', 'Doji', 'P_Engulfing', 'P_Pinbar', 
            'AI_Formasyon_Skoru', 'EMA_21', 'EMA_34', 'EMA_52', 'EMA_89', 
            'EMA_144', 'Destege_Uzaklik', 'Dirence_Uzaklik', 'Ikili_Tepe', 
            'Ikili_Dip', 'Simetrik_Ucgen', 'Yukselen_Ucgen', 'Alcalan_Ucgen', 
            'Bayrak_Formasyonu', 'Tepe_Uzakligi_Z', 'Dip_Uzakligi_Z', 
            'High_Slope', 'Low_Slope', 'Makro_Guc_Skoru', 'Cross_Sinyali', 
            'SMA_50_200_Farki', 'ABCD_Formasyonu', 'F_K', 'PD_DD', 'ROE', 'Cari_Oran', 'Temel_Skor', 
            'XU100_Return', 'XU100_Trend', 'LSTM_Score'
        ]
        
        # Sadece tabloda var olan özellikleri güvenli şekilde filtreliyoruz
        features = [f for f in raw_features if f in t_df.columns]
       
        t_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        t_df[features] = t_df[features].ffill().bfill().fillna(0)
        ml_df = t_df.dropna(subset=['Target_Return'])

        if len(ml_df) < 50:
            return {"rf_prediction": float(t_df['Close'].iloc[-1]), "signal": "VERİ YETERSİZ", "confidence": 50.0, "expected_return_pct": 0.0, "feature_importances": {}}

        X = ml_df[features].values
        y = ml_df['Target_Return'].values
        son_veri = t_df[features].iloc[-1].values.reshape(1, -1)

        # --- MODEL YÜKLEME VEYA EĞİTME BLOĞU ---
        model_klasoru = "ai_modeller"
        os.makedirs(model_klasoru, exist_ok=True)
        model_dosyasi = os.path.join(model_klasoru, f"{sembol}_ai_model.pkl")

        def model_egit_ve_kaydet():
            # 1. Optuna ile en iyi XGBoost parametrelerini bul
            best_xgb_params = en_iyi_xgb_parametrelerini_bul(sembol, X, y)
            
            # 2. Gürültü temizleme için Özellik Seçici (Feature Selector) tanımla
            feature_selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold="median")
            
            # 3. Tüm modelleri Pipeline içine alarak gereksiz özellikleri (noise) eliyoruz
            m_xgb = Pipeline([
                ('fs', feature_selector), 
                ('xgb', XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1))
            ])
            
            m_rf = Pipeline([
                ('fs', feature_selector), 
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1))
            ])
            
            m_svr = Pipeline([
                ('scaler', StandardScaler()), 
                ('fs', feature_selector), 
                ('svr', SVR(C=1.5, epsilon=0.1, kernel='rbf'))
            ])
            
            m_gb = Pipeline([
                ('fs', feature_selector), 
                ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42))
            ])
            
            m_ridge = Pipeline([
                ('scaler', StandardScaler()), 
                ('fs', feature_selector), 
                ('ridge', Ridge(alpha=1.0))
            ])

            # 4. Topluluk modelini (Voting Regressor) oluştur ve eğit
            ens = VotingRegressor(estimators=[
                ('xgb', m_xgb), ('rf', m_rf), ('svr', m_svr), 
                ('gb', m_gb), ('ridge', m_ridge)
            ])
            
            ens.fit(X, y)
            joblib.dump(ens, model_dosyasi)
            return ens

        if os.path.exists(model_dosyasi):
            try:
                ensemble = joblib.load(model_dosyasi)
                # Özellik sayısı değişmişse veya model bozuksa yeniden eğit
                if not hasattr(ensemble, "n_features_in_") or ensemble.n_features_in_ != X.shape[1]:
                    ensemble = model_egit_ve_kaydet()
            except Exception:
                ensemble = model_egit_ve_kaydet()
        else:
            ensemble = model_egit_ve_kaydet()

        # --- 3. TAHMİN VE KARAR ---
        # --- 3. TAHMİN VE KARAR ---
        # LSTM_Score artık yukarıda gerçek bir öznitelik olarak eğitime dahil edildi;
        # ensemble.predict() çağrısı LSTM'in sinyalini de içeren tam tahmini üretir.
        beklenen_getiri_pct = float(ensemble.predict(son_veri).item())

        # LSTM'in son gündeki ham skorunu (0-1 arası olasılık) güven skoruna küçük bir
        # katkı olarak yansıtıyoruz — tahminin kendisini artık değiştirmiyor (çifte sayımı önlemek için)
        lstm_guven_katkisi = 0.0
        try:
            if 'LSTM_Score' in t_df.columns:
                lstm_skoru = float(t_df['LSTM_Score'].iloc[-1])
                # 0.5 nötr kabul edilir; sapma ne kadar büyükse güvene katkısı o kadar artar (±10 puan)
                lstm_guven_katkisi = (lstm_skoru - 0.5) * 20
        except Exception:
            pass

        anlik_fiyat = float(t_df['Close'].iloc[-1])
        hedef_fiyat = anlik_fiyat * (1 + (beklenen_getiri_pct / 100))
        guven_skoru = min(max(abs(beklenen_getiri_pct) * 8 + 50 + lstm_guven_katkisi, 0.0), 99.0)
        
        # Grafik Okuma
        makro_skor = int(t_df['Makro_Guc_Skoru'].iloc[-1]) if 'Makro_Guc_Skoru' in t_df.columns else 0
        mikro_skor = int(t_df['AI_Formasyon_Skoru'].iloc[-1]) if 'AI_Formasyon_Skoru' in t_df.columns else 0
        grafik_okuma_skoru = makro_skor + mikro_skor
        
        if abs(beklenen_getiri_pct) < 1.5 or guven_skoru < 65:
            if grafik_okuma_skoru >= 2:
                sinyal = "🟢 TEKNİK AL (Formasyon Keşfi)"
                beklenen_getiri_pct = 3.0 + grafik_okuma_skoru
                guven_skoru = 70.0
                hedef_fiyat = anlik_fiyat * (1 + (beklenen_getiri_pct / 100))
            elif grafik_okuma_skoru <= -2:
                sinyal = "🔴 TEKNİK SAT (Ayı Formasyonu)"
                beklenen_getiri_pct = -3.0 + grafik_okuma_skoru
                guven_skoru = 70.0
                hedef_fiyat = anlik_fiyat * (1 + (beklenen_getiri_pct / 100))
            else:
                sinyal = "🟡 NÖTR (Yön Belirsiz)"
        else:
            sinyal = "🚀 GÜÇLÜ AL" if beklenen_getiri_pct >= 2.0 else ("⚠️ SAT" if beklenen_getiri_pct < -1.0 else "NÖTR")

        gunluk_atr_pct = (t_df['ATR'].iloc[-1] / anlik_fiyat) * 100
        if beklenen_getiri_pct > 0 and gunluk_atr_pct > 0:
            tahmini_gun = int(np.ceil(beklenen_getiri_pct / gunluk_atr_pct))
            tahmini_gun = max(1, min(tahmini_gun, 15))
            hedef_sure_metni = f"{tahmini_gun} ile {tahmini_gun + 3} gün içinde"
        else:
            hedef_sure_metni = "Belirsiz"

        if "AL" in sinyal:
            ai_karar_metni = f"{sinyal} (Hedef %{beklenen_getiri_pct:.1f} | Süre: {hedef_sure_metni})"
        else:
            ai_karar_metni = sinyal

        # --- Ensemble Modellerinin Ortak Öznitelik Ağırlıklarını Hesaplama ---
        # --- Evrensel ve Güvenli Öznitelik Ağırlıkları Hesaplama ---
        # --- KESİN ÇÖZÜMLÜ ÖZNİTELİK AĞIRLIKLARI ---
        # --- ÖZNİTELİK AĞIRLIKLARI (BOYUT UYUŞMAZLIĞI GİDERİLMİŞ) ---
        oznitelik_agirliklari = {}
        try:
            importances_list = []
            estimators = []
            
            if hasattr(ensemble, 'named_estimators_'):
                estimators = list(ensemble.named_estimators_.values())
            elif hasattr(ensemble, 'estimators_'):
                estimators = [est[1] if isinstance(est, tuple) else est for est in ensemble.estimators_]

            for pipe in estimators:
                if isinstance(pipe, Pipeline):
                    fs = pipe.named_steps.get('fs') # Feature Selector
                    
                    # Pipeline içindeki ana tahmin modelini bul
                    model_obj = None
                    for name, step in pipe.named_steps.items():
                        if name != 'fs' and name != 'scaler':
                            model_obj = step
                            break
                    
                    if model_obj is not None:
                        imp = None
                        if hasattr(model_obj, 'feature_importances_'):
                            imp = model_obj.feature_importances_
                        elif hasattr(model_obj, 'coef_'):
                            imp = np.abs(model_obj.coef_)
                            if imp.ndim > 1:
                                imp = np.mean(imp, axis=0)

                        if imp is not None:
                            # Seçilen öznitelikleri orijinal feature boyutuna haritala
                            full_imp = np.zeros(len(features))
                            if fs is not None and hasattr(fs, 'get_support'):
                                mask = fs.get_support() # Hangi öznitelikler seçildi (True/False)
                                full_imp[mask] = imp
                            elif len(imp) == len(features):
                                full_imp = imp
                            
                            importances_list.append(full_imp)

            if importances_list:
                avg_importances = np.mean(importances_list, axis=0)
                total = np.sum(avg_importances)
                if total > 0:
                    avg_importances = avg_importances / total
                oznitelik_agirliklari = {f: float(imp) for f, imp in zip(features, avg_importances) if imp > 0}
            else:
                # Güvenli yedek plan
                equal_val = 1.0 / len(features)
                oznitelik_agirliklari = {f: float(equal_val) for f in features}

        except Exception as e:
            print(f"Öznitelik hesaplama hatası: {e}")
            equal_val = 1.0 / len(features)
            oznitelik_agirliklari = {f: float(equal_val) for f in features}
        try:
            tahmini_logla(sembol, anlik_fiyat, hedef_fiyat, ai_karar_metni, guven_skoru)
        except Exception:
            pass

        # Sonuçları arayüze döndür
        return {
            "rf_prediction": round(hedef_fiyat, 2),
            "signal": ai_karar_metni,
            "confidence": max(round(guven_skoru, 1), 0.0),
            "expected_return_pct": round(beklenen_getiri_pct, 2),
            "feature_importances": oznitelik_agirliklari,
            "estimated_days": hedef_sure_metni
        }
        
    except Exception as e:
        import logging
        logging.error(f"AI Ensemble Hatası: {e}")
        return {"rf_prediction": 0.0, "signal": f"Hata: {e}", "confidence": 0.0, "expected_return_pct": 0.0, "feature_importances": {}}

@st.cache_data(ttl=3600, show_spinner=False)
def gelismis_ai_tahmin(df, gelecek_gun=10, temel_veriler=None, temel_skor=0):
    try:
        df_ml = df.copy()
        
        # --- 1. ADIM: Temel Analiz Verilerini Dahil Etme ---
        if temel_veriler is None:
            temel_veriler = {}
            
        fk = float(temel_veriler.get('F/K', 0))
        pd_dd = float(temel_veriler.get('PD/DD', 0))
        roe = float(temel_veriler.get('ROE', 0))
        cari_oran = float(temel_veriler.get('Cari_Oran', 0))
        
        # Temel verileri DataFrame'e ekliyoruz
        df_ml['F_K'] = fk
        df_ml['PD_DD'] = pd_dd
        df_ml['ROE'] = roe
        df_ml['Cari_Oran'] = cari_oran
        df_ml['Temel_Skor'] = temel_skor
        
        # Boş ve sonsuz değerleri temizliyoruz
        df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_ml.fillna(0, inplace=True)

        # --- Teknik Metrikler ---
        df_ml['Return'] = df_ml['Close'].pct_change()
        df_ml['Log_Return'] = np.log(df_ml['Close'] / df_ml['Close'].shift(1))
        df_ml['SMA_10_Dist'] = df_ml['Close'] / df_ml['Close'].rolling(10).mean() - 1
        df_ml['Volatilite_14'] = df_ml['Return'].rolling(14).std()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        
        df_ml.dropna(inplace=True)
        if len(df_ml) < 50:
            son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun

        # --- 2. ADIM: Özellik (Features) Listesine Temel Verileri Ekleme ---
        features = [
            'Close', 'Volume', 'Log_Return', 'SMA_10_Dist', 'Volatilite_14',
            'F_K', 'PD_DD', 'ROE', 'Cari_Oran', 'Temel_Skor'  # Temel Analiz Sütunları
        ]
        
        X = df_ml[features].values
        y = df_ml['Target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, objective='reg:squarederror', n_jobs=-1)
        model.fit(X_scaled, y)

        tahminler = []
        son_veri = X_scaled[-1].reshape(1, -1)
        
        gecmis_kapanislar = df_ml['Close'].tail(20).tolist()
        
        # Çok adımlı dinamik tahmin döngüsü
        for _ in range(gelecek_gun):
            pred = model.predict(son_veri).item()
            tahminler.append(pred)
            
            gecmis_kapanislar.append(pred)
            gecmis_kapanislar = gecmis_kapanislar[-20:]
            
            # Teknik göstergelerin dinamik hesabı
            yeni_log_ret = np.log(gecmis_kapanislar[-1] / gecmis_kapanislar[-2])
            yeni_sma_10 = np.mean(gecmis_kapanislar[-10:])
            yeni_sma_10_dist = (gecmis_kapanislar[-1] / yeni_sma_10) - 1
            
            getiriler = [np.log(gecmis_kapanislar[i] / gecmis_kapanislar[i-1]) for i in range(1, len(gecmis_kapanislar))]
            yeni_vol = np.std(getiriler[-14:]) if len(getiriler) >= 14 else np.std(getiriler)
            
            # --- 3. ADIM: Gelecek Günler İçin Temel Verileri Sabit Tutup Diziye Ekleme ---
            yeni_ham_veri = np.array([[
                pred, 
                son_veri[0, 1],  # Hacim (Volume) sabit tutuluyor
                yeni_log_ret, 
                yeni_sma_10_dist, 
                yeni_vol,
                fk, pd_dd, roe, cari_oran, temel_skor  # Temel analiz verileri her gün için ekleniyor
            ]])
            
            son_veri = scaler.transform(yeni_ham_veri)            
            
        tarihler = [df.index[-1] + timedelta(days=i) for i in range(1, gelecek_gun + 1)]
        return tarihler, tahminler

    except Exception:
        son_fiyat = float(df['Close'].iloc[-1]) if not df.empty else 0.0
        return [pd.Timestamp.now() + timedelta(days=i) for i in range(1, gelecek_gun + 1)], [son_fiyat] * gelecek_gun
def lstm_tahmin_yap(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    if not TENSORFLOW_AVAILABLE:
        t_df = df.copy()
        t_df['LSTM_Score'] = 0.5  # TensorFlow yok, nötr skor ile devam
        return t_df
    try:
        t_df = df.copy()
        
        # 1. Yeni Stratejiye Uygun Özellik Listesi (Normalize Edilmiş İndikatörler)
        yapay_zeka_ozellikleri = [
            'Close', 'Volume', 
            'Tilson_Dist_Norm', 'Volume_Trend',  # Adım 1'de eklediğimiz güçlü özellikler
            'Stoch_K', 'Stoch_D', 'RSI'
        ]
        
        # Tabloda var olan özellikleri filtrele
        kullanilacak_ozellikler = [col for col in yapay_zeka_ozellikleri if col in t_df.columns]
        
        # Veri seti yetersizse veya Target_Class (Adım 2) yoksa işlem yapma
        if len(t_df) <= lookback_days or 'Target_Class' not in t_df.columns:
            t_df['LSTM_Score'] = 0.5 # Nötr skor atıyoruz
            return t_df

        X_raw = t_df[kullanilacak_ozellikler].values
        y_raw = t_df['Target_Class'].values

        # Ölçeklendirme
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler_X.fit_transform(X_raw)
        
        X_train, y_train = [], []
        for i in range(lookback_days, len(scaled_X)):
            X_train.append(scaled_X[i-lookback_days:i, :])
            y_train.append(y_raw[i])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        if len(X_train) == 0:
            t_df['LSTM_Score'] = 0.5
            return t_df

        # 2. Sınıflandırma Odaklı LSTM Mimarısi
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') # 0 ile 1 arasında olasılık skoru üretir
        ])
        
        # Sigmoid çıkışı için Binary Crossentropy kullanıyoruz
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, batch_size=32, epochs=8, verbose=0)
        
        # 3. Tüm Veri Seti İçin Tahmin Üretme (XGBoost'a Özellik Olarak Vermek İçin)
        tahminler = model(X_train, training=False).numpy().flatten()
        
        # İlk 'lookback_days' kadar satır diziye giremediği için baş tarafı 0.5 (Nötr) ile dolduruyoruz
        dolgu = np.full(lookback_days, 0.5)
        t_df['LSTM_Score'] = np.concatenate([dolgu, tahminler])
        
        return t_df
        
    except Exception as e:
        print(f"LSTM Çalıştırılamadı: {e}")
        df['LSTM_Score'] = 0.5
        return df
def temel_verileri_temizle(df):
    """
    Temel analiz sütunlarındaki eksik (NaN) veya sonsuz (inf) değerleri temizler.
    """
    df_temiz = df.copy()
    temel_sutunlar = ['F_K', 'PD_DD', 'ROE', 'Cari_Oran', 'Temel_Skor']

    # Boş değerleri 0 ile doldur
    for sutun in temel_sutunlar:
        if sutun in df_temiz.columns:
            df_temiz[sutun] = df_temiz[sutun].fillna(0)

    # Sonsuz değerleri (inf, -inf) 0 ile değiştir
    df_temiz.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_temiz
# ==========================================
# 4. YAN MENÜ (SIDEBAR) & VERİ ÇEKME
# ==========================================
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
        st.dataframe(df, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
        Tabloyu ekrana basar; '_teknik_detay' kolonu varsa (RSI/MACD/Bollinger/
        ADX/Fibonacci/ATR Stop-Hedef/Haftalık T3/Teknik Teyit Puanı — bunlar
        artık ana tabloda GÖSTERİLMİYOR, sadece dahili AI/AL-SAT kararını
        etkiliyorlar) ekrandan gizler ve kullanıcı bir satıra tıkladığında
        (destekleyen Streamlit sürümlerinde) veya bir hisse seçtiğinde (eski
        sürümlerde fallback) altında ayrı bir 'Gelişmiş Teknik Analiz Paneli'
        açar.
        """
        detaylar_var = '_teknik_detay' in df_tablo.columns
        df_gorunen = df_tablo.drop(columns=['_teknik_detay']) if detaylar_var else df_tablo

        secili_sembol = None
        if detaylar_var:
            try:
                event = st.dataframe(
                    df_gorunen, width="stretch", hide_index=True,
                    on_select="rerun", selection_mode="single-row",
                    key=f"tablo_{anahtar}_{len(df_gorunen)}"
                )
                secim_nesnesi = getattr(event, "selection", None)
                secili_satirlar = getattr(secim_nesnesi, "rows", None) if secim_nesnesi is not None else None
                if not secili_satirlar and isinstance(event, dict):
                    secili_satirlar = event.get("selection", {}).get("rows", [])
                if secili_satirlar:
                    secili_sembol = df_gorunen.iloc[secili_satirlar[0]]['Varlık']
            except TypeError:
                # Eski Streamlit sürümü satır tıklama seçimini (on_select) desteklemiyor
                st.dataframe(df_gorunen, width="stretch", hide_index=True)
                secim = st.selectbox(
                    "📊 Detaylı teknik paneli görmek için bir hisse seç:",
                    options=["-"] + df_gorunen['Varlık'].tolist(),
                    key=f"secim_{anahtar}_{len(df_gorunen)}"
                )
                secili_sembol = None if secim == "-" else secim
        else:
            st.dataframe(df_gorunen, width="stretch", hide_index=True)

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
            
            st.plotly_chart(fig_imp, width="stretch")
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
            st.dataframe(gecmis_df, width="stretch", hide_index=True)
            
            basarili_sayisi = len(gecmis_df[gecmis_df['durum'] == 'BAŞARILI ✅'])
            degerlendirilen_sayisi = len(gecmis_df[gecmis_df['durum'] != 'BEKLİYOR'])
            
            if degerlendirilen_sayisi > 0:
                basari_orani = (basarili_sayisi / degerlendirilen_sayisi) * 100
                st.metric(label="Net Başarı Oranı", value=f"% {basari_orani:.1f}")
        else:
            st.info("Henüz kaydedilmiş tahmin yok. Radar veya AI analizi çalıştırıldığında veriler buraya akacaktır.")
    except Exception as e:
        st.error(f"Veritabanı erişim hatası: {e}")