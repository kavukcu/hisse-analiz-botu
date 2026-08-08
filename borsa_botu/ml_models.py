"""
ml_models.py — Makine öğrenmesi tahmin katmanı (RandomForest/XGBoost/SVR
ensemble, stacking, LSTM). En çok dış modüle bağımlı katmandır: veri
zenginleştirme için data_fetching + indicators + patterns + database
fonksiyonlarını kullanır.
"""
import logging
import os

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import optuna
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier

from data_fetching import borsa_endeks_verisini_ekle
from patterns import makro_formasyonlari_bul, trend_ve_harmonik_bul, yapay_zeka_icin_formasyon_bul
from database import tahmini_logla
from indicators import tilson_t3

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
