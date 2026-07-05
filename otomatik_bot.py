import yfinance as yf
import requests
from datetime import datetime

# === AYARLAR ===
TELEGRAM_TOKEN = "8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk"
TELEGRAM_CHAT_ID = "1634044181"
BIST_LISTE = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS", "GARAN.IS", "AKBNK.IS", "TUPRS.IS"]

def telegram_gonder(mesaj):
    if TELEGRAM_TOKEN == "8868337575:AAE4TUSI-PtXfwWn-zmzjpEv2kZ-t59_mRk":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={mesaj}"
    try:
        requests.get(url)
    except:
        pass

def otomatik_tarama():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tarama başlatıldı...")
    for ticker in BIST_LISTE:
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if len(df) < 200: continue
            
            sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
            onceki_sma50 = df['Close'].rolling(window=50).mean().iloc[-2]
            onceki_sma200 = df['Close'].rolling(window=200).mean().iloc[-2]

            if onceki_sma50 <= onceki_sma200 and sma50 > sma200:
                mesaj = f"🚀 OTOMATİK SİNYAL!\n\n📈 Hisse: {ticker}\n🔔 Formasyon: Golden Cross\n⏱ Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                telegram_gonder(mesaj)
                print(f"Sinyal bulundu ve gönderildi: {ticker}")
        except:
            continue
    print("Tarama tamamlandı.")

# Dosya çalıştırıldığında fonksiyonu tetikle
if __name__ == "__main__":
    otomatik_tarama()