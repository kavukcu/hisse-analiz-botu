import yfinance as yf
import pandas as pd
import streamlit as st
import logging

# Hata loglaması için temel ayar
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data(ttl=900) # Veriyi 15 dakika (900 saniye) boyunca hafızada tutar
def fetch_bist_data(tickers, period="1y", interval="1d"):
    """
    Belirtilen BIST hisselerinin verilerini hızlıca çeker.
    Tek bir hisse (str) veya hisse listesi (list) alabilir.
    """
    # Eğer tek bir hisse girildiyse listeye çevir (yfinance uyumluluğu için)
    if isinstance(tickers, str):
        tickers = [tickers]
        
    # BIST hisselerinin sonuna otomatik '.IS' ekleme kontrolü
    formatted_tickers = [t if t.endswith('.IS') else f"{t}.IS" for t in tickers]
    
    try:
        # threads=True parametresi ile asenkron/çoklu indirme yaparak hızı artırıyoruz
        data = yf.download(
            formatted_tickers, 
            period=period, 
            interval=interval, 
            threads=True,
            progress=False
        )
        
        if data.empty:
            logging.warning(f"Veri çekilemedi: {formatted_tickers}")
            return pd.DataFrame()
            
        return data
        
    except Exception as e:
        logging.error(f"Veri çekme hatası ({formatted_tickers}): {e}")
        return pd.DataFrame()