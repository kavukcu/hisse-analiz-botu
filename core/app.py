# Grafik

fig = draw_chart(df, symbol)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# AI ANALİZİ
# ===============================

ai = calculate_ai_score(df)

st.subheader("🤖 AI ANALİZİ")

col1, col2 = st.columns(2)

with col1:
    st.metric("AI SCORE", f"{ai['score']}/100")

with col2:
    st.metric("Sinyal", ai["signal"])

st.write("### Analiz Sonucu")

for item in ai["reasons"]:
    st.success(item)
    import streamlit as st
# from ai_engine import LSTMPredictor # Yukarıdaki sınıfı import et

st.subheader("🧠 Derin Öğrenme (LSTM) ile Fiyat Tahmini")

# Varsayılan olarak kullanıcının indirdiği veri seti (örneğin df) olduğunu varsayıyoruz
if st.button("LSTM Modelini Eğit ve Tahmin Et"):
    with st.spinner("Yapay Zeka geçmiş verileri öğreniyor... (Bu biraz sürebilir)"):
        # 1. Sınıfı başlat
        lstm_engine = LSTMPredictor(look_back=60)
        
        # 2. Veriyi Hazırla
        X, y, raw_data = lstm_engine.prepare_data(df, feature_col='Close')
        
        # 3. Modeli Kur ve Eğit (Hızlı test için son %80 veriyi kullanıyoruz)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        
        lstm_engine.build_model((X_train.shape[1], 1))
        lstm_engine.train(X_train, y_train, epochs=5, batch_size=32)
        
        # 4. Geleceği Tahmin Et
        next_price = lstm_engine.predict_next(raw_data)
        
        st.success("Eğitim Tamamlandı!")
        st.metric(label="Tahmin Edilen Bir Sonraki Kapanış Fiyatı", value=f"${next_price:,.2f}")