import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

class LSTMPredictor:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, df, feature_col='Close'):
        """Veriyi LSTM'in anlayacağı 3 boyutlu formata çevirir."""
        data = df.filter([feature_col]).values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        # LSTM için 3 boyutlu hale getirme: (Örnek Sayısı, Zaman Adımı, Özellik Sayısı)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, data

    def build_model(self, input_shape):
        """LSTM Mimarisini kurar."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(units=25))
        model.add(Dense(units=1)) # Tek bir fiyat tahmini
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Modeli eğitir (Streamlit'te donmayı önlemek için düşük epoch ile başlanabilir)."""
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict_next(self, recent_data):
        """Eğitilmiş model ile bir sonraki adımı (mumu) tahmin eder."""
        scaled_recent = self.scaler.transform(recent_data.reshape(-1, 1))
        X_test = []
        X_test.append(scaled_recent[-self.look_back:, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_scaled = self.model.predict(X_test)
        pred_price = self.scaler.inverse_transform(pred_scaled)
        
        return pred_price[0][0]