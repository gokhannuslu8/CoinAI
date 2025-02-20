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
        self.trade_history = {}  # Her sembol için işlem geçmişi
        self.pattern_history = {}  # Benzer pattern'ların başarı oranı
        self.min_trades_for_stats = 10  # İstatistik için minimum işlem sayısı
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

    def calculate_model_boost(self, symbol, indicators):
        """
        Model boost değerini hesaplar
        """
        try:
            # Pattern oluştur
            current_pattern = self.create_pattern(indicators)
            
            # Benzer pattern'ların başarı oranını hesapla
            pattern_success = self.get_pattern_success_rate(current_pattern)
            
            # Sembol bazlı başarı oranını hesapla
            symbol_success = self.get_symbol_success_rate(symbol)
            
            # Model boost hesaplama
            boost = 0
            
            if pattern_success > 0.6:  # %60'tan yüksek başarı
                boost += (pattern_success - 0.6) * 20  # Max +8%
                
            if symbol_success > 0.5:  # %50'den yüksek başarı
                boost += (symbol_success - 0.5) * 10  # Max +5%
                
            return round(boost, 1)
            
        except Exception as e:
            print(f"Model boost hesaplama hatası: {str(e)}")
            return 0
            
    def create_pattern(self, indicators):
        """
        Mevcut market durumundan bir pattern oluşturur
        """
        return {
            'rsi_zone': self.get_rsi_zone(indicators['rsi']),
            'trend': indicators['trend'],
            'adx_strength': self.get_adx_strength(indicators['adx']),
            'bb_position': indicators['bollinger']['position'],
            'macd_trend': indicators['macd']['trend']
        }
        
    def record_trade_result(self, symbol, pattern, success):
        """
        İşlem sonucunu kaydeder
        """
        # Sembol bazlı işlem geçmişi
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        self.trade_history[symbol].append(success)
        
        # Pattern bazlı işlem geçmişi
        pattern_key = str(pattern)
        if pattern_key not in self.pattern_history:
            self.pattern_history[pattern_key] = []
        self.pattern_history[pattern_key].append(success)
        
    def get_trade_statistics(self, symbol):
        """
        İşlem istatistiklerini döndürür
        """
        stats = {
            'total_trades': 0,
            'success_rate': 0,
            'pattern_success': 0
        }
        
        # Sembol bazlı istatistikler
        if symbol in self.trade_history:
            trades = self.trade_history[symbol]
            stats['total_trades'] = len(trades)
            if trades:
                stats['success_rate'] = round((sum(trades) / len(trades)) * 100, 1)
                
        return stats
        
    def get_pattern_success_rate(self, pattern):
        """
        Benzer pattern'ların başarı oranını hesaplar
        """
        pattern_key = str(pattern)
        if pattern_key in self.pattern_history:
            results = self.pattern_history[pattern_key]
            if len(results) >= 5:  # En az 5 benzer işlem
                return sum(results) / len(results)
        return 0.5  # Varsayılan oran
        
    @staticmethod
    def get_rsi_zone(rsi):
        if rsi < 30: return 'oversold'
        if rsi > 70: return 'overbought'
        return 'neutral'
        
    @staticmethod
    def get_adx_strength(adx):
        if adx > 35: return 'strong'
        if adx > 25: return 'moderate'
        return 'weak' 