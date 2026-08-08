"""
borsa_botu — BIST teknik analiz / AI tahmin botu.

Modüler mimari (eski yeni_bot.py monolitinden ayrıştırıldı):

    config          BIST hisse listesi ve sabitler
    database        SQLite tahmin geçmişi (tahmin_kaydet, tahminleri_degerlendir)
    data_fetching   Yahoo/TradingView/İş Yatırım veri çekme katmanı
    indicators      Saf teknik indikatörler (T3, RSI, MACD, Bollinger, ADX, Fibonacci)
    patterns        Formasyon tespiti ve dipten dönüş analizi
    ml_models       Ensemble/LSTM makine öğrenmesi tahmin modelleri
    scoring         Skorlama, backtest, Monte Carlo simülasyonu
    scanner         Tek hisse için uçtan uca analiz (asenkron_analiz_yap)
    app             Streamlit arayüzü (çalıştırılabilir giriş noktası)

Kullanım: `streamlit run -m borsa_botu.app` yerine, borsa_botu/ paketinin
BULUNDUĞU klasörden `streamlit run borsa_botu/app.py` çalıştırın (app.py
paket-göreli importlar kullandığı için doğrudan python ile değil, paketin
bir parçası olarak çalıştırılmalıdır — bkz. README.md).
"""
