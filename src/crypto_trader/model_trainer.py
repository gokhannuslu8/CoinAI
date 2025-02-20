from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        
    def prepare_features(self, data, symbol=None):
        """Veriyi eğitim için hazırla"""
        features = pd.DataFrame()
        
        # Teknik indikatörler
        features['rsi'] = data['RSI']
        features['macd'] = data['MACD']
        features['macd_signal'] = data['MACD_Signal']
        features['bb_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        features['adx'] = data['ADX']
        
        # Fiyat değişimleri
        features['price_change'] = data['close'].pct_change()
        features['volume_change'] = data['volume'].pct_change()
        
        # Trend özellikleri
        features['trend'] = (data['close'] > data['MA20']).astype(int)
        features['ma_cross'] = (data['MA20'] > data['MA50']).astype(int)
        
        # NaN değerleri temizle
        features = features.dropna()
        
        return features
        
    def prepare_targets(self, data, lookahead=24):
        """Hedef değerleri hazırla"""
        future_returns = data['close'].pct_change(periods=lookahead).shift(-lookahead)
        return future_returns.dropna()
        
    def train(self, data, symbol=None):
        """Modeli eğit"""
        X = self.prepare_features(data, symbol)
        y = self.prepare_targets(data)
        
        # Veri boyutlarını eşitle
        X = X[:len(y)]
        
        # Veriyi normalize et
        X_scaled = self.scaler.fit_transform(X)
        
        # Modeli eğit
        self.model.fit(X_scaled, y)
        
        return self.model
        
    def predict(self, data):
        """Tahmin yap"""
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) 