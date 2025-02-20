from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class AdaptiveTrader:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.trade_history = pd.DataFrame()
        self.min_samples = 50  # Minimum eğitim örneği sayısı
        
    def prepare_features(self, data):
        """İndikatörlerden özellikler oluştur"""
        return {
            'rsi': data['RSI'].iloc[-1],
            'macd': data['MACD'].iloc[-1],
            'macd_signal': data['MACD_Signal'].iloc[-1],
            'bb_position': (data['close'].iloc[-1] - data['BB_lower'].iloc[-1]) / (data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1]),
            'trend': 1 if data['close'].iloc[-1] > data['MA20'].iloc[-1] else -1,
            'volume_change': data['volume'].pct_change().iloc[-1],
            'price_change': data['close'].pct_change().iloc[-1]
        }
        
    def add_trade_result(self, trade_data):
        """İşlem sonucunu kaydet"""
        features = self.prepare_features(trade_data['data'])
        result = 1 if trade_data['profit_loss'] > 0 else 0
        
        new_data = pd.DataFrame({
            'features': [features],
            'result': result,
            'profit_loss': trade_data['profit_loss']
        })
        
        self.trade_history = pd.concat([self.trade_history, new_data])
        self._update_model()
        
    def _update_model(self):
        """Modeli güncelle"""
        if len(self.trade_history) > self.min_samples:
            X = pd.DataFrame(self.trade_history['features'].tolist())
            y = self.trade_history['result']
            self.model.fit(X, y)
            
    def get_signal_confidence(self, current_data):
        """Sinyal güvenilirliğini hesapla"""
        if len(self.trade_history) < self.min_samples:
            return 1.0  # Yeterli veri yoksa normal güven skorunu kullan
            
        features = self.prepare_features(current_data)
        X = pd.DataFrame([features])
        
        # Modelin tahmin olasılıkları
        proba = self.model.predict_proba(X)[0]
        confidence = proba[1]  # Karlı işlem olasılığı
        
        # 0.5-1.0 arasını 0.8-1.2 aralığına dönüştür
        scaled_confidence = 0.8 + (confidence - 0.5) * 0.8
        return max(0.5, min(1.2, scaled_confidence)) 