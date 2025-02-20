from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class AdaptiveTrader:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.trade_history = pd.DataFrame()
        self.min_samples = 50  # Minimum eğitim örneği sayısı
        
        # Trading results klasörünü oluştur
        self.results_dir = "trading_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Bugünün tarihiyle dosya adı oluştur
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.results_file = f"{self.results_dir}/trades_{self.current_date}.json"
        
        # Eğer dosya yoksa boş bir işlem geçmişiyle başlat
        if not os.path.exists(self.results_file):
            self.trade_history = []
            self.save_trade_history()
        else:
            self.load_trade_history()
        
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
        """
        Yeni işlem sonucunu kaydet
        """
        trade_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': trade_data.get('symbol'),
            'timeframe': trade_data.get('timeframe'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('price'),
            'profit_loss': trade_data.get('profit_loss'),
            'entry_signal': trade_data.get('entry_signal'),
            'exit_reason': trade_data.get('exit_reason'),
            'indicators': {
                'RSI': trade_data['data']['RSI'].iloc[-1],
                'MACD': trade_data['data']['MACD'].iloc[-1],
                'ADX': trade_data['data']['ADX'].iloc[-1]
            }
        }
        
        self.trade_history.append(trade_result)
        self.save_trade_history()
        
        print(f"\n💾 İşlem kaydedildi - {trade_result['symbol']}")
        print(f"Kar/Zarar: %{trade_result['profit_loss']:.2f}")
        
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

    def save_trade_history(self):
        """İşlem geçmişini JSON dosyasına kaydet"""
        with open(self.results_file, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def load_trade_history(self):
        """İşlem geçmişini JSON dosyasından yükle"""
        try:
            with open(self.results_file, 'r') as f:
                self.trade_history = json.load(f)
        except:
            self.trade_history = [] 