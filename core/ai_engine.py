import pandas as pd

def calculate_ai_score(df):
    last = df.iloc[-1]

    score = 0
    reasons = []

    # EMA Trend
    if last["EMA20"] > last["EMA50"]:
        score += 10
        reasons.append("EMA20 > EMA50")

    if last["EMA50"] > last["EMA200"]:
        score += 15
        reasons.append("EMA50 > EMA200")

    # RSI
    if 50 <= last["RSI"] <= 70:
        score += 10
        reasons.append("RSI Güçlü")

    elif 30 <= last["RSI"] < 50:
        score += 5

    # MACD
    if last["MACD"] > last["SIGNAL"]:
        score += 15
        reasons.append("MACD AL")

    # Bollinger
    if last["Close"] > last["BB_MID"]:
        score += 10

    # ATR
    atr_percent = last["ATR"] / last["Close"] * 100

    if atr_percent < 3:
        score += 10
        reasons.append("Volatilite Normal")

    # Hacim
    if last["Volume"] > last["VOL20"]:
        score += 10
        reasons.append("Hacim Güçlü")

    # Trend
    if last["Close"] > last["EMA200"]:
        score += 20
        reasons.append("Uzun Vadeli Yükseliş")

    score = min(score, 100)

    if score >= 80:
        signal = "🟢 GÜÇLÜ AL"
    elif score >= 60:
        signal = "🔵 AL"
    elif score >= 40:
        signal = "🟡 BEKLE"
    else:
        signal = "🔴 SAT"

    return {
        "score": score,
        "signal": signal,
        "reasons": reasons
    }