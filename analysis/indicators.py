import pandas as pd
import numpy as np
import logging

# Hata loglaması
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rsi(data: pd.DataFrame, column: str = 'Close', period: int = 14) -> pd.Series:
    """Klasik RSI (Relative Strength Index) hesaplaması."""
    try:
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"RSI hesaplanırken hata: {e}")
        return pd.Series(dtype='float64')

def calculate_macd(data: pd.DataFrame, column: str = 'Close', fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, Sinyal Hattı ve Histogram hesaplaması."""
    try:
        exp1 = data[column].ewm(span=fast, adjust=False).mean()
        exp2 = data[column].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    except Exception as e:
        logging.error(f"MACD hesaplanırken hata: {e}")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')

def detect_fvg(data: pd.DataFrame) -> pd.Series:
    """
    SMC: Fair Value Gap (Adil Değer Boşluğu) tespiti.
    1: Bullish FVG (Alış yönlü boşluk)
    -1: Bearish FVG (Satış yönlü boşluk)
    0: Boşluk yok
    """
    try:
        fvg = pd.Series(0, index=data.index)
        
        # Geçmişteki 1. mum ve 3. mum arasındaki boşlukları kontrol etmek için kaydırma (shift) işlemi
        high_prev = data['High'].shift(2)
        low_curr = data['Low']
        
        low_prev = data['Low'].shift(2)
        high_curr = data['High']
        
        # Bullish FVG: 1. mumun tepesi, 3. mumun dibinden düşükse ve arada güçlü yeşil mum varsa
        bullish_condition = low_curr > high_prev
        fvg.loc[bullish_condition] = 1
        
        # Bearish FVG: 1. mumun dibi, 3. mumun tepesinden yüksekse ve arada güçlü kırmızı mum varsa
        bearish_condition = high_curr < low_prev
        fvg.loc[bearish_condition] = -1
        
        return fvg
    except Exception as e:
        logging.error(f"FVG hesaplanırken hata: {e}")
        return pd.Series(0, index=data.index)

def apply_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Tüm indikatörleri tek seferde DataFrame'e uygulayan ana fonksiyon."""
    if data.empty:
        return data
        
    df = data.copy()
    
    # İndikatörleri dataframe'e kolon olarak ekleme
    df['RSI_14'] = calculate_rsi(df)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)
    df['FVG'] = detect_fvg(df)
    
    # Eksik verileri (NaN) temizleme - ML modelleri için önemlidir
    df.dropna(inplace=True)
    
    return df