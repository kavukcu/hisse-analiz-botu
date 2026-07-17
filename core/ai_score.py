import numpy as np


class AIScore:

    def __init__(self, df):

        self.df = df

        self.score = 0

        self.detail = {}

    def calculate(self):

        self.score = 0
        self.detail = {}

        self._trend_score()
        self._momentum_score()
        self._volume_score()
        self._volatility_score()

        self.score = min(100, self.score)

        return {
            "total": self.score,
            "detail": self.detail,
            "rating": self.rating()
        }

    def rating(self):

        if self.score >= 90:
            return "★★★★★ GÜÇLÜ AL"

        if self.score >= 75:
            return "★★★★ AL"

        if self.score >= 55:
            return "★★★ NÖTR"

        if self.score >= 35:
            return "★★ SAT"

        return "★ GÜÇLÜ SAT"

    ###################################

    def _trend_score(self):

        s = 0

        last = self.df.iloc[-1]

        if last["Close"] > last["EMA20"]:
            s += 5

        if last["EMA20"] > last["EMA50"]:
            s += 5

        if last["EMA50"] > last["EMA200"]:
            s += 5

        if last["ADX"] > 25:
            s += 5

        if last["Supertrend"]:
            s += 5

        self.score += s

        self.detail["Trend"] = s

    ###################################

    def _momentum_score(self):

        s = 0

        last = self.df.iloc[-1]

        if 50 < last["RSI"] < 70:
            s += 5

        if last["MACD"] > last["MACD_SIGNAL"]:
            s += 5

        if last["STOCH_K"] > last["STOCH_D"]:
            s += 5

        if last["CCI"] > 100:
            s += 5

        self.score += s

        self.detail["Momentum"] = s

    ###################################

    def _volume_score(self):

        s = 0

        last = self.df.iloc[-1]

        if last["OBV"] > self.df["OBV"].rolling(20).mean().iloc[-1]:
            s += 5

        if last["CMF"] > 0:
            s += 5

        if last["MFI"] > 50:
            s += 5

        if last["Volume"] > self.df["Volume"].rolling(20).mean().iloc[-1]:
            s += 5

        self.score += s

        self.detail["Volume"] = s

    ###################################

    def _volatility_score(self):

        s = 0

        last = self.df.iloc[-1]

        if last["ATR"] > 0:
            s += 5

        if last["Close"] > last["BB_MID"]:
            s += 5

        self.score += s

        self.detail["Volatility"] = s