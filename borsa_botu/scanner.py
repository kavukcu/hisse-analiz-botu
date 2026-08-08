"""
scanner.py — Tek bir hisse için tam kapsamlı radar/stoch/tilson analizini
yürüten çekirdek fonksiyon (asenkron_analiz_yap). Paralel/arka-plan tarama
orkestrasyonu (thread yönetimi, ilerleme dosyası, Streamlit widget'ları)
kasıtlı olarak burada DEĞİL, ui/tab_tarama.py içinde tutulur — böylece bu
modül tamamen Streamlit'ten bağımsız, test edilebilir kalır.
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime, timedelta

from data_fetching import veri_yukle, veri_4saatlik_getir
from indicators import (
    stokastik_hesapla, tilson_t3, ileri_teknik_gostergeler,
    genisletilmis_indikatorler_hesapla, fibonacci_seviyeleri_hesapla,
    haftalik_veri_turet,
)
from patterns import dipten_donus_analizi, formasyon_tespit_et_ve_hedefle
from ml_models import ensemble_prediction
from scoring import sihirli_formul_skorla
from database import tahmin_kaydet


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
