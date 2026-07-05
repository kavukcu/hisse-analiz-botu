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