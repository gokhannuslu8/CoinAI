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
        self.trade_history = pd.DataFrame(columns=[
            'symbol',
            'entry_date',
            'exit_date', 
            'signal_type',  # AL/SAT
            'entry_price',
            'exit_price',
            'profit_loss',  # Yüzdelik kar/zarar
            'confidence',   # Sinyal güven skoru
            'timeframe',
            'indicators',   # Girişteki indikatör değerleri
            'exit_reason'   # stop/target/trend_change
        ])
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
        try:
            # Başarı oranını al
            success_rate = self.get_symbol_success_rate(symbol)  # Bu fonksiyon eksik
            
            # Model boost hesapla
            boost = min(15, success_rate * 0.2)  # Maximum %15 boost
            return boost
            
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
        Sembol için istatistikleri getir
        """
        return {
            "total_trades": len(self.trade_history),
            "success_rate": 0,  # Henüz işlem yok
            "pattern_success": 0  # Henüz pattern analizi yok
        }
        
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

    def get_symbol_success_rate(self, symbol):
        """
        Sembol için başarı oranını hesapla
        """
        if len(self.trade_history) == 0:
            return 0
        
        symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
        if len(symbol_trades) == 0:
            return 0
        
        successful = len(symbol_trades[symbol_trades['profit'] > 0])
        return (successful / len(symbol_trades)) * 100 

    def record_trade(self, trade_data):
        """
        İşlem sonucunu kaydet
        """
        try:
            # Mevcut kayıtları yükle
            try:
                with open(self.results_file, 'r') as f:
                    trades = json.load(f)
            except:
                trades = []
            
            # Yeni işlemi ekle
            trades.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': trade_data['symbol'],
                'signal_type': trade_data['signal_type'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'profit_loss': trade_data['profit_loss'],
                'confidence': trade_data['confidence'],
                'timeframe': trade_data['timeframe'],
                'indicators': trade_data['indicators'],
                'exit_reason': trade_data['exit_reason']
            })
            
            # Dosyaya kaydet
            with open(self.results_file, 'w') as f:
                json.dump(trades, f, indent=4)
            
            print(f"İşlem kaydedildi: {trade_data['symbol']} - {trade_data['profit_loss']:.2f}%")
            
            # İstatistikleri güncelle
            self.update_statistics(trade_data['symbol'])
            
        except Exception as e:
            print(f"İşlem kayıt hatası: {str(e)}")
            
    def update_statistics(self, symbol):
        """
        Symbol bazlı istatistikleri güncelle
        """
        try:
            # Trade history bir liste olduğu için önce DataFrame'e çevirelim
            df = pd.DataFrame(self.trade_history)
            
            # Symbol'e göre filtrele
            symbol_trades = df[df['symbol'] == symbol]
            
            if len(symbol_trades) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'avg_profit': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'success_rate': 0
                }
            
            stats = {
                'total_trades': len(symbol_trades),
                'winning_trades': len(symbol_trades[symbol_trades['profit_loss'] > 0]),
                'avg_profit': symbol_trades['profit_loss'].mean(),
                'max_profit': symbol_trades['profit_loss'].max(),
                'max_loss': symbol_trades['profit_loss'].min(),
                'success_rate': (len(symbol_trades[symbol_trades['profit_loss'] > 0]) / len(symbol_trades)) * 100
            }
            
            print(f"\n=== {symbol} İstatistikleri ===")
            print(f"Toplam İşlem: {stats['total_trades']}")
            print(f"Kazanan İşlem: {stats['winning_trades']}")
            print(f"Ortalama Kar: %{stats['avg_profit']:.2f}")
            print(f"En Yüksek Kar: %{stats['max_profit']:.2f}")
            print(f"En Yüksek Zarar: %{stats['max_loss']:.2f}")
            print(f"Başarı Oranı: %{stats['success_rate']:.1f}")
            
            return stats
            
        except Exception as e:
            print(f"İstatistik güncelleme hatası: {str(e)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'success_rate': 0
            } 