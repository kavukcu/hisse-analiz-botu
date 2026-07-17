import streamlit as st
from data.data_engine import fetch_bist_data

st.set_page_config(page_title="BIST Analiz Terminali", layout="wide")

st.title("🚀 Profesyonel BIST Analiz Terminali")

# Kullanıcıdan hisse sembolü alma
hisse = st.text_input("Hisse Sembolü Giriniz (Örn: THYAO, ASELS, FROTO):", "THYAO")

if st.button("Verileri Getir"):
    with st.spinner("Veriler çekiliyor..."):
        # Modüler dosyamızdan fonksiyonu çağırıyoruz
        df = fetch_bist_data(hisse, period="6mo", interval="1d")
        
        if not df.empty:
            st.success(f"{hisse} verileri başarıyla yüklendi!")
            st.dataframe(df.tail())
        else:
            st.error("Veri çekilirken bir sorun oluştu. Sembolü kontrol edin.")