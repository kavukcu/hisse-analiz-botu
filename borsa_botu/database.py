"""
database.py — SQLite tabanlı tahmin geçmişi ve başarı karnesi yönetimi.
"""
import sqlite3
import logging
from datetime import datetime

import yfinance as yf
import pandas as pd


def veritabani_baslat():
    """Yapay zekanın tahminlerini tutacağı yerel veritabanını oluşturur.

    NOT: Bu fonksiyon eskiden (monolitik yeni_bot.py'de) boş bir `pass` idi ve
    asıl CREATE TABLE kodu, aynı dosyada tum_bist_hisselerini_getir()
    fonksiyonunun içinde bir `return`'den SONRA (yani hiçbir zaman
    çalışmayan, ölü kodda) duruyordu. Modülerleştirme sırasında bu kod
    buraya, ait olduğu yere taşındı.
    """
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


# Modül import edildiğinde veritabanını/tabloyu hazırla (orijinal davranışla aynı)
veritabani_baslat()

